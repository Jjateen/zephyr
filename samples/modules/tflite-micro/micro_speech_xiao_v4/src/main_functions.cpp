/*
 * Copyright 2025 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "main_functions.hpp"

#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(micro_speech_xiao);

#include "audio_provider.h"
#include "model_runner.h"
#include "micro_model_settings.h"

#include <zephyr/kernel.h>
#include <string.h>

/* Audio constants */
#define SAMPLES_PER_SECOND      (16000)
/* Trigger every 500 ms — slides the 1-second inference window forward */
#define TRIGGER_SAMPLES         (SAMPLES_PER_SECOND / 2)   /* 8000 */
#define TRIGGER_BYTES           (TRIGGER_SAMPLES * sizeof(int16_t))
/* Inference always operates on a full 1-second window */
#define INFERENCE_SAMPLES       (SAMPLES_PER_SECOND)
#define INFERENCE_BYTES         (INFERENCE_SAMPLES * sizeof(int16_t))

/* Thread stack sizes */
#define CAPTURE_STACK_SIZE      (2048)
#define INFERENCE_STACK_SIZE    (4096)

/*
 * Sliding-window buffers:
 *   capture_a / capture_b  – double-buffered 500 ms chunks
 *   inference_buf_a / b    – double-buffered 1-second windows passed to TFLM
 *
 * Layout of each inference buffer (updated every 500 ms):
 *   [  old 500 ms  |  new 500 ms  ]
 *        ↑ shifted left each trigger
 */
static int16_t capture_a[TRIGGER_SAMPLES];
static int16_t capture_b[TRIGGER_SAMPLES];
static int16_t *write_buffer   = capture_a;
static int16_t *new_chunk      = capture_b;   /* alias: filled chunk ready to slide in */

static int16_t infer_a[INFERENCE_SAMPLES];
static int16_t infer_b[INFERENCE_SAMPLES];
static int16_t *inference_buf  = infer_a;     /* being read by inference thread */
static int16_t *staging_buf    = infer_b;     /* being built by capture thread  */

static size_t g_samples_in_write_buffer;

K_THREAD_STACK_DEFINE(capture_stack, CAPTURE_STACK_SIZE);
K_THREAD_STACK_DEFINE(inference_stack, INFERENCE_STACK_SIZE);
static struct k_thread capture_thread_data;
static struct k_thread inference_thread_data;

static K_SEM_DEFINE(buffer_ready_sem, 0, 1);
static K_SEM_DEFINE(inference_done_sem, 1, 1);
static K_MUTEX_DEFINE(buffer_mutex);

/*
 * PDM capture thread — sliding-window approach.
 *
 * Accumulates 500 ms (TRIGGER_SAMPLES) of audio into write_buffer.
 * When full, it slides the 1-second staging_buf forward:
 *   staging_buf = [ old second half | new 500 ms chunk ]
 * then hands staging_buf off to the inference thread.
 *
 * Because the capture window is only 500 ms, even if one trigger is
 * dropped (inference busy), the next trigger fires 500 ms later —
 * keeping worst-case latency under ~2 seconds.
 */
extern "C" void audio_capture_thread(void *arg1, void *arg2, void *arg3)
{
	ARG_UNUSED(arg1);
	ARG_UNUSED(arg2);
	ARG_UNUSED(arg3);

	const size_t block_samples =
		(size_t)(kFeatureDurationMs * kAudioSampleFrequency / 1000);
	static int16_t block_buf[kFeatureDurationMs * kAudioSampleFrequency / 1000];

	LOG_INF("Audio capture thread started");

	while (1) {
		int ret = audio_provider_read(block_buf, block_samples, 200);

		if (ret < 0) {
			LOG_ERR("PDM read error: %d", ret);
			continue;
		}

		k_mutex_lock(&buffer_mutex, K_FOREVER);

		size_t space    = TRIGGER_SAMPLES - g_samples_in_write_buffer;
		size_t to_copy  = (block_samples < space) ? block_samples : space;

		memcpy(write_buffer + g_samples_in_write_buffer,
		       block_buf, to_copy * sizeof(int16_t));
		g_samples_in_write_buffer += to_copy;

		if (g_samples_in_write_buffer >= TRIGGER_SAMPLES) {
			/*
			 * Slide the staging buffer:
			 *   [ discard first 500ms | keep second 500ms ]
			 * then append the freshly captured 500ms chunk.
			 */
			memmove(staging_buf,
				staging_buf + TRIGGER_SAMPLES,
				TRIGGER_BYTES);
			memcpy(staging_buf + TRIGGER_SAMPLES,
			       write_buffer, TRIGGER_BYTES);

			/* Swap capture buffers regardless */
			int16_t *tmp  = write_buffer;
			write_buffer  = new_chunk;
			new_chunk     = tmp;
			g_samples_in_write_buffer = 0;

			if (k_sem_take(&inference_done_sem, K_NO_WAIT) == 0) {
				/* Hand staging_buf to inference */
				int16_t *itmp  = inference_buf;
				inference_buf  = staging_buf;
				staging_buf    = itmp;
				k_sem_give(&buffer_ready_sem);
			}
			/* If inference is busy: drop this trigger, keep
			 * updating staging_buf so it stays fresh. */
		}

		k_mutex_unlock(&buffer_mutex);
	}
}

/*
 * Inference thread.
 *
 * Waits for a full second of audio, runs the TFLM pipeline and logs the
 * detected keyword.
 */
extern "C" void audio_inference_thread(void *arg1, void *arg2, void *arg3)
{
	ARG_UNUSED(arg1);
	ARG_UNUSED(arg2);
	ARG_UNUSED(arg3);

	LOG_INF("Inference thread started");

	while (1) {
		k_sem_take(&buffer_ready_sem, K_FOREVER);

		if (micro_speech_process_audio(inference_buf,
					       INFERENCE_SAMPLES) != 0) {
			LOG_ERR("Inference failed");
		}

		/* Buffer is handed back to staging — no memset needed */
		k_sem_give(&inference_done_sem);
	}
}

extern "C" {

void setup(void)
{
	LOG_INF("Starting Micro Speech XIAO application");

	/* Power on the PDM microphone (MSM261D3526HICPM-C, enable pin P1.10) */
	if (audio_provider_init() != 0) {
		LOG_ERR("Failed to init audio provider");
		return;
	}

	/* Initialize TFLM interpreters */
	model_runner_init();

	/* Capture thread – higher priority */
	k_thread_create(&capture_thread_data, capture_stack,
			CAPTURE_STACK_SIZE,
			audio_capture_thread, NULL, NULL, NULL,
			5, 0, K_NO_WAIT);
	k_thread_name_set(&capture_thread_data, "pdm_capture");

	/* Inference thread – lower priority */
	k_thread_create(&inference_thread_data, inference_stack,
			INFERENCE_STACK_SIZE,
			audio_inference_thread, NULL, NULL, NULL,
			6, 0, K_NO_WAIT);
	k_thread_name_set(&inference_thread_data, "ml_inference");
}

void loop(void)
{
	k_sleep(K_FOREVER);
}

} /* extern "C" */
