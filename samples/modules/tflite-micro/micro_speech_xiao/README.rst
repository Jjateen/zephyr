.. zephyr:code-sample:: tflite-micro-speech-xiao
   :name: Micro Speech XIAO BLE Sense

   Recognize "yes" and "no" speech commands in real time using the onboard
   PDM microphone of the Seeed XIAO BLE Sense (nRF52840) and TensorFlow Lite
   for Microcontrollers.

Overview
********

This sample runs a two-stage keyword-spotting pipeline entirely on the
Seeed XIAO BLE Sense board using its onboard MEMS PDM microphone
(MSM261D3526HICPM-C).  No host computer, no external microphone, and no
wireless connection are required at inference time.

Pipeline stages
---------------

.. code-block:: text

   +---[ PDM Mic ]---+     +---[ Audio Preprocessor ]---+     +---[ Keyword Classifier ]---+
   |  16 kHz mono    |---->|  30 ms windows, 20 ms stride|---->|  INT8 quantized            |
   |  INT16 PCM      |     |  40-band MFSC features      |     |  4 classes                 |
   +-----------------+     |  INT8 or float32 model      |     |  silence / unknown         |
                           +----------------------------+     |  yes / no                  |
                                                              +----------------------------+

1. **Audio capture** — the PDM driver streams 16 kHz, 16-bit, mono PCM into a
   double-buffered ring.  A sliding window triggers inference every 500 ms on
   the latest 1-second of audio, keeping worst-case detection latency under
   approximately 2 seconds.

2. **Audio preprocessor** — a TFLite Micro model computes 40-band Mel-filterbank
   spectral coefficients (MFSC) over 49 overlapping 30 ms windows (20 ms stride),
   producing a ``49 × 40`` feature map.

3. **Keyword classifier** — a second TFLite Micro model (depthwise-separable CNN)
   classifies the feature map into one of four categories:

   * ``silence`` — no speech detected
   * ``unknown`` — speech detected but not a trained keyword
   * ``yes``
   * ``no``

Results are printed over USB CDC-ACM serial at log level INFO.

Hardware
********

+----------------------------+----------------------------------------------+
| Item                       | Details                                      |
+============================+==============================================+
| Board                      | Seeed XIAO BLE Sense (``xiao_ble/nrf52840/sense``) |
+----------------------------+----------------------------------------------+
| SoC                        | Nordic nRF52840 (Cortex-M4F, 64 MHz)        |
+----------------------------+----------------------------------------------+
| Flash                      | 1 MB                                         |
+----------------------------+----------------------------------------------+
| RAM                        | 256 KB                                       |
+----------------------------+----------------------------------------------+
| Microphone                 | MSM261D3526HICPM-C (PDM, onboard)           |
+----------------------------+----------------------------------------------+
| Mic power enable           | GPIO1 pin 10, active-high                    |
+----------------------------+----------------------------------------------+
| PDM CLK pin                | P1.00                                        |
+----------------------------+----------------------------------------------+
| PDM DIN pin                | P0.16                                        |
+----------------------------+----------------------------------------------+
| Serial output              | USB CDC-ACM (``/dev/ttyACM0`` on Linux)      |
+----------------------------+----------------------------------------------+

Memory footprint (approximate)
-------------------------------

+----------------------------+----------+--------------+
| Component                  | Flash    | RAM          |
+============================+==========+==============+
| INT8 audio preprocessor    | ~2.4 KB  | 16 KB arena  |
+----------------------------+----------+--------------+
| Float32 audio preprocessor | ~7.8 KB  | 32 KB arena  |
+----------------------------+----------+--------------+
| Keyword classifier (INT8)  | ~20 KB   | 12 KB arena  |
+----------------------------+----------+--------------+
| Audio ring buffers (2×500ms)| —       | ~32 KB       |
+----------------------------+----------+--------------+
| Inference buffers (2×1s)   | —        | ~64 KB       |
+----------------------------+----------+--------------+
| Total (INT8 preprocessor)  | ~192 KB  | ~130 KB      |
+----------------------------+----------+--------------+

Models
******

Both models are derived from the
`TensorFlow Lite Micro micro_speech example <https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech>`_.

Audio preprocessor
------------------

Two variants are provided and selectable at build time (see
`Building and Running`_):

* **INT8 quantized** (default) — ``audio_preprocessor_int8_model.cpp``
  Runs the MFSC feature extraction entirely in 8-bit integer arithmetic.
  Faster and uses half the tensor arena compared to the float variant.

* **Float32** — ``audio_preprocessor_float_model.cpp``
  Runs MFSC feature extraction in single-precision floating point.
  Produces slightly higher-quality features at the cost of speed and
  memory.  The float output is requantized to INT8 before being fed to
  the keyword classifier using the classifier input tensor's own
  scale / zero-point parameters.

Keyword classifier
------------------

``micro_speech_quantized_model.cpp`` — always INT8 quantized regardless of
which preprocessor is selected.

+------------------+-------+
| Category label   | Index |
+==================+=======+
| silence          | 0     |
+------------------+-------+
| unknown          | 1     |
+------------------+-------+
| yes              | 2     |
+------------------+-------+
| no               | 3     |
+------------------+-------+

Software Architecture
*********************

.. code-block:: text

   main()
    ├── audio_provider_init()   — powers mic GPIO, configures & starts PDM driver
    ├── model_runner_init()     — allocates TFLM interpreters (both models)
    ├── pdm_capture thread (priority 5)
    │     Reads 30 ms PDM blocks → accumulates into 500 ms write_buffer
    │     Every 500 ms:
    │       slide staging_buf:  [old 500 ms | new 500 ms]
    │       if inference idle → swap & signal inference thread
    │       else              → drop trigger, keep staging_buf fresh
    └── ml_inference thread (priority 6)
          Waits for buffer_ready_sem
          micro_speech_process_audio(inference_buf, 16000)
            ├── generate_features()          — 49 × preprocessor invocations
            └── run_inference()              — classifier invocation
                  LOG_INF("Detected: yes/no/silence/unknown")

Key design decisions
--------------------

* **Sliding-window capture** — inference fires every 500 ms on the latest 1 s
  of audio rather than on non-overlapping 1 s blocks.  This halves worst-case
  keyword detection latency compared to a simple double-buffer scheme.

* **Non-blocking capture** — the capture thread never waits for inference to
  finish.  If inference is still running when a new 500 ms chunk is ready,
  the chunk is incorporated into the staging buffer and the trigger is
  silently dropped.  This prevents PDM DMA slab starvation (``-ENOMEM``) that
  would otherwise occur because a 4-block × 30 ms slab is exhausted in
  under 150 ms when the capture thread stalls for a full inference cycle (~1 s).

* **Float→INT8 bridge** — when the float32 preprocessor is selected, each
  feature value is requantized as
  ``q = clamp(round(f / scale) + zero_point, -128, 127)``
  using the classifier input tensor's quantization parameters, ensuring the
  INT8 classifier receives correctly-scaled inputs.

Building and Running
********************

Prerequisites
-------------

Ensure the following west modules are present in your workspace:

.. code-block:: console

   west update hal_nordic
   cd modules/hal/nordic && git submodule update --init nrfx

The ``tflite-micro`` optional module must also be available under
``$ZEPHYR_BASE/../optional/modules/lib/tflite-micro``.

USB serial configuration
------------------------

A ``usb_serial.conf`` file is included in the sample directory.  It routes
log output over USB CDC-ACM instead of the hardware UART — required for the
XIAO BLE Sense which does not expose a UART on its USB connector:

.. code-block:: ini

   CONFIG_USB_DEVICE_STACK_NEXT=y
   CONFIG_USBD_CDC_ACM_CLASS=y
   CONFIG_UART_LINE_CTRL=y
   CONFIG_LOG=y
   CONFIG_PRINTK=y

Pass it via ``EXTRA_CONF_FILE`` as shown in the build commands below.

Build — INT8 preprocessor (default)
------------------------------------

.. code-block:: console

   west build -p always -b xiao_ble/nrf52840/sense \
     samples/modules/tflite-micro/micro_speech_xiao \
     -- -DEXTRA_CONF_FILE=usb_serial.conf

Build — Float32 preprocessor
------------------------------

.. code-block:: console

   west build -p always -b xiao_ble/nrf52840/sense \
     samples/modules/tflite-micro/micro_speech_xiao \
     -- -DEXTRA_CONF_FILE=usb_serial.conf \
        -DMICRO_SPEECH_FLOAT_PREPROCESSOR=y

The selected preprocessor is confirmed in the CMake output:

.. code-block:: text

   -- micro_speech_xiao: preprocessor = INT8 quantized (default)
   -- micro_speech_xiao: preprocessor = float32

Flashing
--------

Put the XIAO BLE Sense into UF2 bootloader mode by double-pressing the reset
button until the LED pulses.  Then flash:

.. code-block:: console

   west flash -r uf2

The board mounts as a USB mass-storage device (``XIAO-SENSE``).  The UF2
file is written automatically.

Sample output
*************

Connect to the USB CDC-ACM serial port (``/dev/ttyACM0`` on Linux,
``COMx`` on Windows) at any baud rate and observe:

.. code-block:: none

   *** Booting Zephyr OS build v4.2.0-... ***
   [00:00:00.310] <inf> dmic_nrfx_pdm: PDM clock frequency: 1280000, actual PCM rate: 16000
   [00:00:00.310] <inf> audio_provider: Audio provider initialized (16000 Hz, 16-bit, 1 ch)
   [00:00:00.310] <inf> model_runner: Initializing TFLM interpreters [preprocessor: INT8 quantized]
   [00:00:00.351] <inf> model_runner: TFLM interpreters initialized
   [00:00:00.351] <inf> micro_speech_xiao: Audio capture thread started
   [00:00:00.351] <inf> micro_speech_xiao: Inference thread started
   [00:00:02.359] <inf> model_runner: Detected: silence
   [00:00:03.394] <inf> model_runner: Detected: silence
   [00:00:04.428] <inf> model_runner: Detected: yes
   [00:00:05.462] <inf> model_runner: Detected: no

Speak clearly into the top of the board (the microphone is on the underside,
near the USB connector).  Detection fires approximately every 1 second in
steady state.

Preprocessor comparison
***********************

+------------------------------+-----------------------+-----------------------+
| Property                     | INT8 preprocessor     | Float32 preprocessor  |
+==============================+=======================+=======================+
| Preprocessor model size      | ~2.4 KB               | ~7.8 KB               |
+------------------------------+-----------------------+-----------------------+
| Preprocessor arena           | 16 KB                 | 32 KB                 |
+------------------------------+-----------------------+-----------------------+
| Feature precision            | 8-bit quantized       | 32-bit floating point |
+------------------------------+-----------------------+-----------------------+
| Classifier                   | INT8 (same)           | INT8 (same)           |
+------------------------------+-----------------------+-----------------------+
| Float→INT8 bridge needed     | No                    | Yes (requantization)  |
+------------------------------+-----------------------+-----------------------+
| Build flag                   | *(default)*           | ``-DMICRO_SPEECH_FLOAT_PREPROCESSOR=y`` |
+------------------------------+-----------------------+-----------------------+

File Structure
**************

.. code-block:: text

   micro_speech_xiao/
   ├── CMakeLists.txt                          # Build system; MICRO_SPEECH_FLOAT_PREPROCESSOR option
   ├── prj.conf                                # Zephyr Kconfig: TFLM, PDM, USB CDC-ACM
   ├── usb_serial.conf                         # Extra conf: routes logging over USB CDC-ACM
   ├── sample.yaml                             # Twister test descriptor
   ├── README.rst                              # This file
   ├── boards/
   │   └── xiao_ble_nrf52840_sense.overlay     # Enables pdm0 as dmic_dev
   └── src/
       ├── main.cpp                            # Entry point: setup() + loop()
       ├── main_functions.hpp / .cpp           # Thread creation; sliding-window logic
       ├── audio_provider.h / .cpp             # PDM mic init and block read
       ├── model_runner.h / .cpp               # TFLM pipeline (preprocessor + classifier)
       ├── micro_model_settings.h              # Model constants (sample rate, feature dims)
       ├── audio_preprocessor_int8_model.hpp / .cpp   # INT8 preprocessor flatbuffer
       ├── audio_preprocessor_float_model.hpp / .cpp  # Float32 preprocessor flatbuffer
       ├── micro_speech_quantized_model.hpp / .cpp    # INT8 keyword classifier flatbuffer
       └── micro_model_settings.h             # kFeatureSize, kCategoryLabels, etc.

Troubleshooting
***************

PDM slab exhaustion (``-ENOMEM`` / ``-EAGAIN``)
-------------------------------------------------

If you see repeated ``dmic_read failed: -11`` errors, the PDM DMA slab has
been exhausted.  This happens when the capture thread stalls for longer than
``SLAB_BLOCK_COUNT × 30 ms`` (currently 8 × 30 = 240 ms).  The capture
thread is designed never to block for more than one 30 ms PDM read, so this
should not occur in normal operation.  If it does reappear after code changes,
increase ``SLAB_BLOCK_COUNT`` in ``audio_provider.cpp``.

No output on serial
-------------------

Ensure ``usb_serial.conf`` was passed via ``-DEXTRA_CONF_FILE=usb_serial.conf``.
On Linux, check ``dmesg | grep tty`` after plugging in the board.  The device
appears as ``/dev/ttyACM0`` (or ``ACM1`` if another ACM device is present).

Float32 preprocessor always outputs "no"
-----------------------------------------

This occurs when the float output tensor is read as INT8 raw bytes (type
mismatch).  The ``model_runner.cpp`` requantization block (guarded by
``#if defined(MICRO_SPEECH_FLOAT_PREPROCESSOR)``) must be present and the
project must be built with ``-DMICRO_SPEECH_FLOAT_PREPROCESSOR=y``.

hal_nordic / nrfx MDK missing
------------------------------

.. code-block:: console

   Cannot find source file: .../nrfx/mdk/system_nrf52840.c

Run:

.. code-block:: console

   cd $ZEPHYR_BASE/..
   west update hal_nordic
   cd modules/hal/nordic && git submodule update --init nrfx

References
**********

* `TensorFlow Lite for Microcontrollers — micro_speech example
  <https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech>`_
* `Seeed XIAO BLE Sense product page
  <https://wiki.seeedstudio.com/XIAO_BLE/>`_
* `Zephyr audio/dmic API
  <https://docs.zephyrproject.org/latest/hardware/peripherals/audio/dmic.html>`_
* `Nordic nRF52840 PDM driver
  <https://docs.zephyrproject.org/latest/build/dts/api/bindings/audio/nordic,nrf-pdm.html>`_
* `Zephyr micro_speech OpenAMP sample (i.MX8MP)
  <../micro_speech/README.rst>`_
