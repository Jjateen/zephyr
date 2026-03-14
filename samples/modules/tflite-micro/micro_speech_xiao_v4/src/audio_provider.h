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

#ifndef MICRO_SPEECH_XIAO_AUDIO_PROVIDER_H_
#define MICRO_SPEECH_XIAO_AUDIO_PROVIDER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Power on the PDM mic, configure and start the DMIC driver.
 *
 * Must be called once before audio_provider_read().
 *
 * @return 0 on success, negative errno on failure.
 */
int audio_provider_init(void);

/**
 * @brief Read exactly @p num_samples mono int16 samples from the PDM mic.
 *
 * Blocks until enough samples have been collected or @p timeout_ms expires.
 *
 * @param buffer      Destination buffer (must hold at least num_samples int16).
 * @param num_samples Number of samples to read.
 * @param timeout_ms  Maximum time to wait (milliseconds).
 *
 * @return 0 on success, negative errno on failure or timeout.
 */
int audio_provider_read(int16_t *buffer, size_t num_samples,
			uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* MICRO_SPEECH_XIAO_AUDIO_PROVIDER_H_ */
