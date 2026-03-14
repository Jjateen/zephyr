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

#include "audio_provider.h"

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(audio_provider);

/* -----------------------------------------------------------------------
 * Hardware constants for the XIAO BLE Sense
 * PDM mic power enable: GPIO1 pin 10, active-high
 * (MSM261D3526HICPM-C regulator in the board DTS has no label so we
 *  drive the GPIO directly from code.)
 * ----------------------------------------------------------------------- */
#define MIC_PWR_PORT  DT_NODELABEL(gpio1)
#define MIC_PWR_PIN   10

/* -----------------------------------------------------------------------
 * DMIC / PDM configuration
 * 16 kHz, 16-bit, mono – matching the TFLM model requirements.
 * Block size: 30 ms = 480 samples = 960 bytes.
 * We keep 4 blocks in the slab so the driver never stalls.
 * ----------------------------------------------------------------------- */
#define SAMPLE_RATE_HZ   16000
#define SAMPLE_BIT_WIDTH 16
#define NUM_CHANNELS     1
#define BLOCK_MS         30
#define BLOCK_SAMPLES    (SAMPLE_RATE_HZ * BLOCK_MS / 1000)   /* 480 */
#define BLOCK_BYTES      (BLOCK_SAMPLES * sizeof(int16_t))     /* 960 */
#define SLAB_BLOCK_COUNT 8

K_MEM_SLAB_DEFINE_STATIC(audio_slab, BLOCK_BYTES, SLAB_BLOCK_COUNT, 4);

static const struct device *dmic_dev;
static bool g_initialized;

int audio_provider_init(void)
{
	if (g_initialized) {
		return 0;
	}

	/* Power on the PDM microphone */
	const struct device *mic_pwr = DEVICE_DT_GET(MIC_PWR_PORT);

	if (!device_is_ready(mic_pwr)) {
		LOG_ERR("GPIO1 not ready");
		return -ENODEV;
	}
	gpio_pin_configure(mic_pwr, MIC_PWR_PIN, GPIO_OUTPUT_ACTIVE);
	/* Datasheet requires ≥ 1 ms after power-on before the first clock */
	k_msleep(5);

	/* Get the dmic device – aliased as "dmic_dev" in the overlay */
	dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));
	if (!device_is_ready(dmic_dev)) {
		LOG_ERR("DMIC device not ready");
		return -ENODEV;
	}

	struct pcm_stream_cfg stream = {
		.pcm_width = SAMPLE_BIT_WIDTH,
		.mem_slab  = &audio_slab,
	};

	struct dmic_cfg cfg = {
		.io = {
			.min_pdm_clk_freq = 1000000,
			.max_pdm_clk_freq = 3500000,
			.min_pdm_clk_dc   = 40,
			.max_pdm_clk_dc   = 60,
		},
		.streams          = &stream,
		.channel = {
			.req_num_chan    = NUM_CHANNELS,
			.req_num_streams = 1,
		},
	};

	cfg.channel.req_chan_map_lo =
		dmic_build_channel_map(0, 0, PDM_CHAN_LEFT);
	cfg.streams[0].pcm_rate  = SAMPLE_RATE_HZ;
	cfg.streams[0].block_size = BLOCK_BYTES;

	int ret = dmic_configure(dmic_dev, &cfg);

	if (ret < 0) {
		LOG_ERR("dmic_configure failed: %d", ret);
		return ret;
	}

	ret = dmic_trigger(dmic_dev, DMIC_TRIGGER_START);
	if (ret < 0) {
		LOG_ERR("DMIC START trigger failed: %d", ret);
		return ret;
	}

	g_initialized = true;
	LOG_INF("Audio provider initialized (%d Hz, %d-bit, %d ch)",
		SAMPLE_RATE_HZ, SAMPLE_BIT_WIDTH, NUM_CHANNELS);
	return 0;
}

int audio_provider_read(int16_t *buffer, size_t num_samples,
			uint32_t timeout_ms)
{
	if (!g_initialized) {
		return -EAGAIN;
	}

	size_t samples_collected = 0;

	while (samples_collected < num_samples) {
		void *block;
		uint32_t block_size;

		int ret = dmic_read(dmic_dev, 0, &block, &block_size,
				    timeout_ms);
		if (ret < 0) {
			LOG_ERR("dmic_read failed: %d", ret);
			return ret;
		}

		size_t block_samples = block_size / sizeof(int16_t);
		size_t remaining     = num_samples - samples_collected;
		size_t to_copy       = (block_samples < remaining)
					? block_samples : remaining;

		memcpy(buffer + samples_collected, block,
		       to_copy * sizeof(int16_t));
		samples_collected += to_copy;

		k_mem_slab_free(&audio_slab, block);
	}

	return 0;
}
