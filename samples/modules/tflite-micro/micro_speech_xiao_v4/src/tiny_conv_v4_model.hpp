/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * tiny_conv_v4 INT8 quantized classifier — 10-class keyword spotter.
 * Trained on Speech Commands v2: silence, unknown, down, go, left,
 * off, on, right, stop, up.
 * Input:  [1, 49, 40, 1] INT8   (49 MFSC frames × 40 bins)
 * Output: [1, 10]        INT8   (class logits with softmax)
 */

#ifndef MICRO_SPEECH_XIAO_V4_TINY_CONV_V4_MODEL_H_
#define MICRO_SPEECH_XIAO_V4_TINY_CONV_V4_MODEL_H_

extern const unsigned char g_tiny_conv_v4_model[];
extern const unsigned int  g_tiny_conv_v4_model_len;

#endif /* MICRO_SPEECH_XIAO_V4_TINY_CONV_V4_MODEL_H_ */
