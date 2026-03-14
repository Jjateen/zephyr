/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * model_runner.cpp — tiny_conv_v4 classifier + MFSC audio preprocessor.
 *
 * Two-model pipeline:
 *   1. Audio preprocessor (INT8 or float32): 480-sample PCM → 40-bin MFSC
 *   2. tiny_conv_v4 classifier (INT8):       49×40 features → 10-class logits
 *
 * Op set for tiny_conv_v4 (standard conv, BN folded at quantization):
 *   Conv2D × 3, MaxPool2D × 2, Reshape, FullyConnected, Softmax
 */

#include "model_runner.h"

#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(model_runner);

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#if defined(MICRO_SPEECH_FLOAT_PREPROCESSOR)
#include "audio_preprocessor_float_model.hpp"
#define g_audio_preprocessor_model_data g_audio_preprocessor_float_model
#else
#include "audio_preprocessor_int8_model.hpp"
#define g_audio_preprocessor_model_data g_audio_preprocessor_int8_model
#endif

#include "tiny_conv_v4_model.hpp"
#include "micro_model_settings.h"

#include <zephyr/kernel.h>
#include <algorithm>
#include <string.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Op resolvers
 * ----------------------------------------------------------------------- */
/* tiny_conv_v4 uses standard Conv2D (not depthwise) + MaxPool2D */
using MicroSpeechOpResolver      = tflite::MicroMutableOpResolver<5>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

using Features = int8_t[kFeatureCount][kFeatureSize];

/* -----------------------------------------------------------------------
 * Tensor arenas
 * ----------------------------------------------------------------------- */
namespace {

#if defined(MICRO_SPEECH_FLOAT_PREPROCESSOR)
constexpr size_t kAudioPreprocessorArenaSize = 32768;
#else
constexpr size_t kAudioPreprocessorArenaSize = 16384;
#endif

/* tiny_conv_v4 peak activation: Conv2 output [1,25,20,48] = 24 000 B.
 * Allocator reported needing 34240 B; use 40 KB for headroom. */
constexpr size_t kMicroSpeechArenaSize = 40960;

alignas(16) uint8_t g_audio_preprocessor_arena[kAudioPreprocessorArenaSize];
alignas(16) uint8_t g_micro_speech_arena[kMicroSpeechArenaSize];

constexpr int kAudioSampleDurationCount =
	kFeatureDurationMs * kAudioSampleFrequency / 1000;
constexpr int kAudioSampleStrideCount =
	kFeatureStrideMs * kAudioSampleFrequency / 1000;

static const tflite::Model *g_audio_preprocessor_model;
static const tflite::Model *g_micro_speech_model;
static AudioPreprocessorOpResolver *g_audio_preprocessor_op_resolver;
static MicroSpeechOpResolver       *g_micro_speech_op_resolver;
static tflite::MicroInterpreter    *g_audio_preprocessor_interpreter;
static tflite::MicroInterpreter    *g_micro_speech_interpreter;
static bool g_initialized;

static Features g_features;

/* -----------------------------------------------------------------------
 * Op registration
 * ----------------------------------------------------------------------- */
static TfLiteStatus register_micro_speech_ops(MicroSpeechOpResolver &r)
{
	TF_LITE_ENSURE_STATUS(r.AddReshape());
	TF_LITE_ENSURE_STATUS(r.AddConv2D());
	TF_LITE_ENSURE_STATUS(r.AddMaxPool2D());
	TF_LITE_ENSURE_STATUS(r.AddFullyConnected());
	TF_LITE_ENSURE_STATUS(r.AddSoftmax());
	return kTfLiteOk;
}

static TfLiteStatus register_audio_preprocessor_ops(AudioPreprocessorOpResolver &r)
{
	TF_LITE_ENSURE_STATUS(r.AddReshape());
	TF_LITE_ENSURE_STATUS(r.AddCast());
	TF_LITE_ENSURE_STATUS(r.AddStridedSlice());
	TF_LITE_ENSURE_STATUS(r.AddConcatenation());
	TF_LITE_ENSURE_STATUS(r.AddMul());
	TF_LITE_ENSURE_STATUS(r.AddAdd());
	TF_LITE_ENSURE_STATUS(r.AddDiv());
	TF_LITE_ENSURE_STATUS(r.AddMinimum());
	TF_LITE_ENSURE_STATUS(r.AddMaximum());
	TF_LITE_ENSURE_STATUS(r.AddWindow());
	TF_LITE_ENSURE_STATUS(r.AddFftAutoScale());
	TF_LITE_ENSURE_STATUS(r.AddRfft());
	TF_LITE_ENSURE_STATUS(r.AddEnergy());
	TF_LITE_ENSURE_STATUS(r.AddFilterBank());
	TF_LITE_ENSURE_STATUS(r.AddFilterBankSquareRoot());
	TF_LITE_ENSURE_STATUS(r.AddFilterBankSpectralSubtraction());
	TF_LITE_ENSURE_STATUS(r.AddPCAN());
	TF_LITE_ENSURE_STATUS(r.AddFilterBankLog());
	return kTfLiteOk;
}

/* -----------------------------------------------------------------------
 * Interpreter initialisation
 * ----------------------------------------------------------------------- */
static TfLiteStatus initialize_interpreters(void)
{
	if (g_initialized) {
		return kTfLiteOk;
	}

#if defined(MICRO_SPEECH_FLOAT_PREPROCESSOR)
	LOG_INF("Initializing TFLM [tiny_conv_v4, preprocessor: float32]");
#else
	LOG_INF("Initializing TFLM [tiny_conv_v4, preprocessor: INT8]");
#endif

	g_audio_preprocessor_model =
		tflite::GetModel(g_audio_preprocessor_model_data);
	if (g_audio_preprocessor_model->version() != TFLITE_SCHEMA_VERSION) {
		LOG_ERR("Audio preprocessor model schema mismatch");
		return kTfLiteError;
	}

	static AudioPreprocessorOpResolver audio_preprocessor_op_resolver;
	g_audio_preprocessor_op_resolver = &audio_preprocessor_op_resolver;
	if (register_audio_preprocessor_ops(*g_audio_preprocessor_op_resolver)
	    != kTfLiteOk) {
		LOG_ERR("Failed to register audio preprocessor ops");
		return kTfLiteError;
	}

	static tflite::MicroInterpreter audio_preprocessor_interpreter(
		g_audio_preprocessor_model,
		*g_audio_preprocessor_op_resolver,
		g_audio_preprocessor_arena,
		kAudioPreprocessorArenaSize);
	g_audio_preprocessor_interpreter = &audio_preprocessor_interpreter;
	if (g_audio_preprocessor_interpreter->AllocateTensors() != kTfLiteOk) {
		LOG_ERR("Audio preprocessor tensor allocation failed");
		return kTfLiteError;
	}

	g_micro_speech_model = tflite::GetModel(g_tiny_conv_v4_model);
	if (g_micro_speech_model->version() != TFLITE_SCHEMA_VERSION) {
		LOG_ERR("tiny_conv_v4 model schema mismatch");
		return kTfLiteError;
	}

	static MicroSpeechOpResolver micro_speech_op_resolver;
	g_micro_speech_op_resolver = &micro_speech_op_resolver;
	if (register_micro_speech_ops(*g_micro_speech_op_resolver) != kTfLiteOk) {
		LOG_ERR("Failed to register micro speech ops");
		return kTfLiteError;
	}

	static tflite::MicroInterpreter micro_speech_interpreter(
		g_micro_speech_model,
		*g_micro_speech_op_resolver,
		g_micro_speech_arena,
		kMicroSpeechArenaSize);
	g_micro_speech_interpreter = &micro_speech_interpreter;
	if (g_micro_speech_interpreter->AllocateTensors() != kTfLiteOk) {
		LOG_ERR("tiny_conv_v4 tensor allocation failed — try increasing kMicroSpeechArenaSize");
		return kTfLiteError;
	}

	g_initialized = true;
	LOG_INF("TFLM interpreters initialized (10-class: silence/unknown/up/down/go/stop/left/right/on/off)");
	return kTfLiteOk;
}

/* -----------------------------------------------------------------------
 * Feature generation
 * ----------------------------------------------------------------------- */
static TfLiteStatus generate_single_feature(const int16_t *audio_data,
					    int audio_data_size,
					    int8_t *feature_output)
{
	TfLiteTensor *input =
		g_audio_preprocessor_interpreter->input(0);
	if (!input) {
		return kTfLiteError;
	}
	if (audio_data_size != kAudioSampleDurationCount) {
		LOG_ERR("Feature window size mismatch: %d vs %d",
			audio_data_size, kAudioSampleDurationCount);
		return kTfLiteError;
	}

	TfLiteTensor *output =
		g_audio_preprocessor_interpreter->output(0);
	if (!output) {
		return kTfLiteError;
	}

	std::copy_n(audio_data, audio_data_size,
		    tflite::GetTensorData<int16_t>(input));

	if (g_audio_preprocessor_interpreter->Invoke() != kTfLiteOk) {
		LOG_ERR("Audio preprocessor invoke failed");
		return kTfLiteError;
	}

#if defined(MICRO_SPEECH_FLOAT_PREPROCESSOR)
	{
		const float *f = tflite::GetTensorData<float>(output);
		TfLiteTensor *cls_in = g_micro_speech_interpreter->input(0);
		const float scale      = cls_in->params.scale;
		const int   zero_point = cls_in->params.zero_point;

		for (int i = 0; i < kFeatureSize; i++) {
			int32_t q = static_cast<int32_t>(roundf(f[i] / scale))
				    + zero_point;
			q = std::max(-128, std::min(127, q));
			feature_output[i] = static_cast<int8_t>(q);
		}
	}
#else
	std::copy_n(tflite::GetTensorData<int8_t>(output),
		    kFeatureSize, feature_output);
#endif
	return kTfLiteOk;
}

static TfLiteStatus generate_features(const int16_t *audio_data,
				      size_t audio_data_size,
				      Features *features_output)
{
	memset(features_output, 0, sizeof(Features));

	size_t remaining   = audio_data_size;
	size_t feature_idx = 0;
	const int16_t *ptr = audio_data;

	while (remaining >= (size_t)kAudioSampleDurationCount &&
	       feature_idx < kFeatureCount) {
		if (generate_single_feature(ptr, kAudioSampleDurationCount,
					    (*features_output)[feature_idx])
		    != kTfLiteOk) {
			LOG_ERR("Feature %zu generation failed", feature_idx);
			return kTfLiteError;
		}
		feature_idx++;
		ptr       += kAudioSampleStrideCount;
		remaining -= kAudioSampleStrideCount;
	}

	return kTfLiteOk;
}

/* -----------------------------------------------------------------------
 * Inference
 * ----------------------------------------------------------------------- */
static TfLiteStatus run_inference(const Features &features)
{
	TfLiteTensor *input  = g_micro_speech_interpreter->input(0);
	TfLiteTensor *output = g_micro_speech_interpreter->output(0);

	if (!input || !output) {
		return kTfLiteError;
	}
	if (output->dims->data[output->dims->size - 1] != kCategoryCount) {
		LOG_ERR("Output tensor size mismatch: got %d, want %d",
			output->dims->data[output->dims->size - 1],
			kCategoryCount);
		return kTfLiteError;
	}

	std::copy_n(&features[0][0], kFeatureElementCount,
		    tflite::GetTensorData<int8_t>(input));

	int64_t t_infer = k_uptime_get();
	if (g_micro_speech_interpreter->Invoke() != kTfLiteOk) {
		LOG_ERR("tiny_conv_v4 invoke failed");
		return kTfLiteError;
	}
	int64_t infer_ms = k_uptime_delta(&t_infer);

	float output_scale      = output->params.scale;
	int   output_zero_point = output->params.zero_point;

	float scores[kCategoryCount];
	const int8_t *raw = tflite::GetTensorData<int8_t>(output);

	for (int i = 0; i < kCategoryCount; i++) {
		scores[i] = (raw[i] - output_zero_point) * output_scale;
	}

	int best = (int)(std::max_element(scores, scores + kCategoryCount) -
			 scores);
	LOG_INF("Detected: %s (%.2f) [infer %lld ms]", kCategoryLabels[best],
		(double)scores[best], infer_ms);
	/* Debug: dump all class scores to find correct label order */
	for (int i = 0; i < kCategoryCount; i++) {
		LOG_INF("  [%d] %s = %.3f", i, kCategoryLabels[i], (double)scores[i]);
	}

	return kTfLiteOk;
}

} /* namespace */

/* -----------------------------------------------------------------------
 * Public C API
 * ----------------------------------------------------------------------- */
extern "C" {

void model_runner_init(void)
{
	if (initialize_interpreters() != kTfLiteOk) {
		LOG_ERR("Failed to initialize TFLM interpreters");
	}
}

int micro_speech_process_audio(const int16_t *audio_data,
			       size_t audio_data_size)
{
	int64_t t_feat = k_uptime_get();
	if (generate_features(audio_data, audio_data_size,
			      &g_features) != kTfLiteOk) {
		LOG_ERR("Feature generation failed");
		return -1;
	}
	int64_t feat_ms = k_uptime_delta(&t_feat);
	LOG_INF("Feature gen: %lld ms (%d windows)", feat_ms, kFeatureCount);

	if (run_inference(g_features) != kTfLiteOk) {
		LOG_ERR("Inference failed");
		return -2;
	}
	return 0;
}

} /* extern "C" */
