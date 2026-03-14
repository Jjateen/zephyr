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

#include <zephyr/kernel.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/devicetree.h>

int main(int argc, char *argv[])
{
#if defined(CONFIG_UART_LINE_CTRL)
	/* Wait for the USB CDC-ACM host to open the port (DTR asserted)
	 * before starting so that early log messages are not lost. */
	const struct device *console =
		DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
	if (device_is_ready(console)) {
		uint32_t dtr = 0;
		while (!dtr) {
			uart_line_ctrl_get(console,
					   UART_LINE_CTRL_DTR, &dtr);
			k_sleep(K_MSEC(100));
		}
	}
#endif

	setup();
	while (1) {
		loop();
	}
	return 0;
}
