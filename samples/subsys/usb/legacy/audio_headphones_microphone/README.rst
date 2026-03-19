.. zephyr:code-sample:: usb-audio-headphones-microphone
   :name: Legacy USB Audio microphone & headphones
   :relevant-api: _usb_device_core_api

   Implement a USB Audio microphone + headphones device with audio IN/OUT loopback.

Overview
********

This sample app demonstrates use of a USB Audio driver by the Zephyr
project. This very simple sample that performs loopback over IN/OUT
ISO endpoints. The device will show up as two audio devices. One
Input (Microphone) and one Output (Headphones) device.

.. note::
   This samples demonstrate deprecated :ref:`usb_device_stack`.

Building and Running
********************

In order to build the sample an overlay file with required options
must be provided. By default app.overlay is added. An overlay contains
software and hardware specific information which allow to fully
describe the device.

After you have built and flashed the sample app image to your board, plug the
board into a host device.

Testing
*******

Steps to test the sample:

- Build and flash the sample as described above.
- Connect to the HOST.
- Chose default Audio IN/OUT.
- Start streaming audio (for example by playing an audio file on the HOST).
- Start recording audio stream (for example using Audacity).
- Verify the recorded audio stream.

Measuring CPU load
******************

To measure per-thread CPU usage, add the following to your build configuration:

.. code-block:: cfg

   CONFIG_THREAD_ANALYZER=y
   CONFIG_THREAD_ANALYZER_AUTO=y
   CONFIG_THREAD_ANALYZER_AUTO_INTERVAL=5
   CONFIG_THREAD_ANALYZER_USE_LOG=y
   CONFIG_THREAD_NAME=y
   CONFIG_SCHED_THREAD_USAGE=y
   CONFIG_SCHED_THREAD_USAGE_ALL=y

.. note::
   ``CONFIG_TRACING_CPU_STATS`` referenced in older reports is no longer
   available. Use the thread analyzer as shown above.

With :github:`64174` applied (Zephyr 3.6+), the ``usbd_workq`` thread CPU
usage during audio playback is reduced from ~33% to ~2.4% on nRF52840/nRF5340,
as the EasyDMA busy-wait loop was replaced with a semaphore in
``drivers/usb/common/nrf_usbd_common/nrf_usbd_common.c``.
