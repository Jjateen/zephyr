/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Model settings for the tiny_conv_v4 10-class keyword spotter.
 * Trained on Google Speech Commands v2 with PREPROCESS=micro,
 * WINDOW_SIZE_MS=30, WINDOW_STRIDE=20, FEATURE_BIN_COUNT=40.
 */

#ifndef MICRO_SPEECH_XIAO_V4_MODEL_SETTINGS_H_
#define MICRO_SPEECH_XIAO_V4_MODEL_SETTINGS_H_

constexpr int kAudioSampleFrequency = 16000;
constexpr int kFeatureSize          = 40;
constexpr int kFeatureCount         = 49;
constexpr int kFeatureElementCount  = (kFeatureSize * kFeatureCount);
constexpr int kFeatureStrideMs      = 20;
constexpr int kFeatureDurationMs    = 30;

/* 10-class output: silence + unknown + 8 wanted words in training order.
 * WANTED_WORDS = "up,down,go,stop,left,right,on,off"
 * prepare_words_list() prepends silence/unknown without sorting. */
constexpr int kCategoryCount = 10;
constexpr const char *kCategoryLabels[kCategoryCount] = {
	"silence",  /* 0 */
	"unknown",  /* 1 */
	"up",       /* 2 */
	"down",     /* 3 */
	"go",       /* 4 */
	"stop",     /* 5 */
	"left",     /* 6 */
	"right",    /* 7 */
	"on",       /* 8 */
	"off",      /* 9 */
};

#endif /* MICRO_SPEECH_XIAO_V4_MODEL_SETTINGS_H_ */
