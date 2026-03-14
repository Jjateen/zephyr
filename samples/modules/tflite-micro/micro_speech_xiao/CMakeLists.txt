cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(micro_speech_xiao)

# These samples use local static initialization. Since Zephyr doesn't support the
# C++ ABI for thread-safe initialization of local statics and the constructors don't
# appear to require thread safety, we turn it off in the C++ compiler.
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# Paths
set(OPTIONAL_TFLITE_DIR $ENV{ZEPHYR_BASE}/../optional/modules/lib/tflite-micro/signal)

# Model selection: pass -DMICRO_SPEECH_FLOAT_PREPROCESSOR=y to use float32
# audio preprocessor instead of the default INT8 quantized one.
if(MICRO_SPEECH_FLOAT_PREPROCESSOR)
  message(STATUS "micro_speech_xiao: preprocessor = float32")
  zephyr_compile_definitions(MICRO_SPEECH_FLOAT_PREPROCESSOR=1)
else()
  message(STATUS "micro_speech_xiao: preprocessor = INT8 quantized (default)")
endif()

file(GLOB app_sources src/*.cpp src/*.c)

target_include_directories(
  app
  PRIVATE
  src
  ${ZEPHYR_TFLITE_MICRO_MODULE_DIR}/signal
  ${ZEPHYR_TFLITE_MICRO_MODULE_DIR}/signal/micro/kernels/
  ${ZEPHYR_TFLITE_MICRO_MODULE_DIR}/signal/src
)

target_sources(
  app
  PRIVATE
  ${app_sources}

  # Signal processing
  ${OPTIONAL_TFLITE_DIR}/src/fft_auto_scale.cc
  ${OPTIONAL_TFLITE_DIR}/src/max_abs.cc
  ${OPTIONAL_TFLITE_DIR}/src/msb_32.cc
  ${OPTIONAL_TFLITE_DIR}/src/energy.cc
  ${OPTIONAL_TFLITE_DIR}/src/filter_bank.cc
  ${OPTIONAL_TFLITE_DIR}/src/filter_bank_square_root.cc
  ${OPTIONAL_TFLITE_DIR}/src/square_root_64.cc
  ${OPTIONAL_TFLITE_DIR}/src/square_root_32.cc
  ${OPTIONAL_TFLITE_DIR}/src/msb_64.cc
  ${OPTIONAL_TFLITE_DIR}/src/pcan_argc_fixed.cc
  ${OPTIONAL_TFLITE_DIR}/src/filter_bank_log.cc
  ${OPTIONAL_TFLITE_DIR}/src/log.cc
  ${OPTIONAL_TFLITE_DIR}/src/filter_bank_spectral_subtraction.cc

  # Micro kernels
  ${OPTIONAL_TFLITE_DIR}/micro/kernels/energy.cc
  ${OPTIONAL_TFLITE_DIR}/micro/kernels/fft_auto_scale_common.cc
  ${OPTIONAL_TFLITE_DIR}/micro/kernels/fft_auto_scale_kernel.cc
  ${OPTIONAL_TFLITE_DIR}/micro/kernels/filter_bank_square_root_common.cc
  ${OPTIONAL_TFLITE_DIR}/micro/kernels/filter_bank_square_root.cc
  ${OPTIONAL_TFLITE_DIR}/micro/kernels/filter_bank_spectral_subtraction.cc
  ${OPTIONAL_TFLITE_DIR}/micro/kernels/filter_bank.cc
  ${OPTIONAL_TFLITE_DIR}/micro/kernels/pcan.cc
  ${OPTIONAL_TFLITE_DIR}/micro/kernels/filter_bank_log.cc
)
