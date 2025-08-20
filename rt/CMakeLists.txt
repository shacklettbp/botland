define_property(TARGET PROPERTY BOT_GPU_SOURCES)

list(APPEND BOT_RT_COMMON_SOURCES
  macros.hpp span.hpp types.hpp
  utils.hpp utils.inl utils.cpp
  rand.hpp rand.inl rand.cpp
  math.hpp math.inl
  global_alloc.hpp global_alloc.cpp
  rt.hpp rt.inl rt.cpp
  err.hpp err.inl err.cpp
)

add_library(bot-rt STATIC
  ${BOT_RT_COMMON_SOURCES}
  rt_cpu.cpp
  job.hpp
  os.hpp os.cpp
  io.hpp io.cpp
  opnewdel.cpp
  stack_alloc.hpp stack_alloc.inl stack_alloc.cpp
)

target_link_libraries(bot-rt
  PUBLIC
    bot-common-flags
    bot-libcxx
)

if (BOT_CUDA)
  list(APPEND BOT_GPU_SOURCES
    ${BOT_RT_COMMON_SOURCES}
    rt_gpu.hpp rt_gpu.cpp
    rt_gpu_impl.hpp
    gpu_alloc.cpp
    cuda_comm.hpp cuda_comm.inl
    host_print.hpp host_print.inl host_print.cpp
    log.hpp log.inl
  )

  option(BOT_CUDA_DISABLE_OPTIMIZATION "Disable Optimization" OFF)
  set(BOT_CUDA_ARCH "native" CACHE STRING "GPU Architecture")

  target_sources(bot-rt PRIVATE
    cuda_utils.hpp cuda_utils.inl cuda_utils.cpp
    cuda_comm.hpp cuda_comm.inl
    cuda_mgr.hpp cuda_mgr.inl cuda_mgr.cpp
    cuda_comm.hpp cuda_comm.inl
    host_print.hpp host_print.inl host_print.cpp
    log.hpp log.inl
  )

  set_target_properties(bot-rt PROPERTIES BOT_GPU_SOURCES "${BOT_GPU_SOURCES}")

  target_compile_definitions(bot-rt PUBLIC
    BOT_CUDA_SUPPORT=1
  )

  target_link_libraries(bot-rt
    PUBLIC
      CUDA::cudart
      CUDA::cuda_driver
      bot-common-flags
  )

  add_executable(bot-cuda-cc
    cuda_cc.cpp
  )

  target_link_libraries(bot-cuda-cc
    bot-rt
    CUDA::nvrtc
    ${CUDA_NVJITLINK_LIBRARY}
  )

  function(bot_cuda_target cuda_target target)
    if (NOT TARGET ${target})
      message(FATAL_ERROR "Target '${target}' does not exist.")
    endif()

    # set(CCCL_OVERRIDE_INCLUDE_DIRS "")
    # set(CCCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps/cccl")
    # list(APPEND CCCL_OVERRIDE_INCLUDE_DIRS
    #   "${CCCL_DIR}/libcudacxx/include/"
    #   "${CCCL_DIR}/cub"
    #   "${CCCL_DIR}/thrust"
    # )

    list(APPEND BOT_INCLUDE_DIRS
      "${CMAKE_SOURCE_DIR}/rt/cuda"
      "${CMAKE_SOURCE_DIR}"
      "${CUDAToolkit_INCLUDE_DIRS}"
      # "${CCCL_OVERRIDE_INCLUDE_DIRS}"
    )

    list(APPEND BOT_DEFINES
      BOT_GPU=1
    )

    list(APPEND BOT_CUDA_COMPILE_FLAGS
      -std=c++20
      -default-device
      -rdc=true
      -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
      -DCCCL_DISABLE_BF16_SUPPORT=1
      -DCUB_DISABLE_BF16_SUPPORT=1
    )

    list(APPEND BOT_CUDA_LINK_FLAGS
      -split-compile=0
    )

    if (BOT_CUDA_DISABLE_OPTIMIZATION)
      list(APPEND BOT_CUDA_COMPILE_FLAGS
        -G
        -maxrregcount=40
      )
      list(APPEND BOT_CUDA_LINK_FLAGS
        -g
        -O0
        -maxrregcount=40
      )
    else ()
      list(APPEND BOT_CUDA_COMPILE_FLAGS
        -dopt=on
        -dlto
        -lineinfo
        -G
        --extra-device-vectorization
        -use_fast_math
      )

      list(APPEND BOT_CUDA_LINK_FLAGS
        -ftz=1
        -prec-div=0
        -prec-sqrt=0
        -fma=1
        -optimize-unused-variables
        -O3
        -lineinfo
        -lto
      )
    endif ()

    function (get_defines_and_includes_for_target tgt out_defns out_incs)
      get_target_property(TGT_DEFNS ${tgt} COMPILE_DEFINITIONS)
      if (NOT TGT_DEFNS)
        set(TGT_DEFNS)
      endif()

      get_target_property(TGT_INCS ${tgt} INCLUDE_DIRECTORIES)
      if (NOT TGT_INCS)
        set(TGT_INCS)
      endif()

      get_target_property(CHECK_TGTS ${tgt} LINK_LIBRARIES)

      while (CHECK_TGTS)
        list(POP_BACK CHECK_TGTS CHECK_TGT)
        if ("${CHECK_TGT}" STREQUAL "bot-rt" OR NOT TARGET ${CHECK_TGT})
          continue()
        endif()

        get_target_property(DEP_DEFNS ${CHECK_TGT} INTERFACE_COMPILE_DEFINITIONS)
        if (DEP_DEFNS)
          list(APPEND TGT_DEFNS ${DEP_DEFNS})
        endif()

        get_target_property(DEP_INCS ${CHECK_TGT} INTERFACE_INCLUDE_DIRECTORIES)
        if (DEP_INCS)
          list(APPEND TGT_INCS ${DEP_INCS})
        endif()

        get_target_property(LINK_LIBS ${TGT} INTERFACE_LINK_LIBRARIES)
        foreach (LINK_TGT IN LISTS LINK_LIBS)
          list(APPEND CHECK_TGTS ${LINK_TGT})
        endforeach ()
      endwhile ()

      set(${out_defns} ${TGT_DEFNS} PARENT_SCOPE)
      set(${out_incs} ${TGT_INCS} PARENT_SCOPE)
    endfunction ()

    function (compile_target_sources TGT TGT_DEFNS TGT_INCS OUT_OBJS)
      set(DEFINES ${BOT_DEFINES})
      set(INCLUDE_DIRS ${BOT_INCLUDE_DIRS})
      list(APPEND DEFINES ${TGT_DEFNS})
      list(APPEND INCLUDE_DIRS ${TGT_INCS})

      get_target_property(TARGET_DIR ${TGT} SOURCE_DIR)

      # Prefer custom property for sources to allow overriding
      get_target_property(TGT_SRCS ${TGT} BOT_GPU_SOURCES)
      if (NOT TGT_SRCS)
        get_target_property(TGT_SRCS ${TGT} SOURCES)
        if (NOT TGT_SRCS)
          set(${OUT_OBJS} "" PARENT_SCOPE)
          return()
        endif()
      endif()

      set(ABS_SOURCE_FILES)
      foreach (SRC IN LISTS TGT_SRCS)
        cmake_path(ABSOLUTE_PATH SRC BASE_DIRECTORY "${TARGET_DIR}"
          OUTPUT_VARIABLE SRC_ABSOLUTE)
        list(APPEND ABS_SOURCE_FILES "${SRC_ABSOLUTE}")
      endforeach ()

      set(INCLUDE_ARGS)
      set(DEFINE_ARGS)

      foreach (INCLUDE_DIR IN LISTS INCLUDE_DIRS)
        list(APPEND INCLUDE_ARGS
          -I${INCLUDE_DIR}
        )
      endforeach ()

      foreach (DEFINE IN LISTS DEFINES)
        list(APPEND DEFINE_ARGS 
          -D${DEFINE}
        )
      endforeach ()

      set(OBJS)
      foreach (ABS_SOURCE_FILE REL_SOURCE_FILE 
          IN ZIP_LISTS ABS_SOURCE_FILES TGT_SRCS)
        if(NOT EXISTS "${ABS_SOURCE_FILE}")
          message(FATAL_ERROR "Source file '${ABS_SOURCE_FILE}' does not exist.")
        endif()

        get_filename_component(REL_SOURCE_DIR "${REL_SOURCE_FILE}" DIRECTORY)
        get_filename_component(SOURCE_NAME "${REL_SOURCE_FILE}" NAME)
        get_filename_component(SOURCE_EXT "${REL_SOURCE_FILE}" LAST_EXT)

        if (NOT ("${SOURCE_EXT}" STREQUAL ".cpp" OR 
                 "${SOURCE_EXT}" STREQUAL ".c" OR 
                 "${SOURCE_EXT}" STREQUAL ".cc" OR 
                 "${SOURCE_EXT}" STREQUAL ".cxx"))
          continue()
        endif()

        set(OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${cuda_target}/${REL_SOURCE_DIR}")

        file(MAKE_DIRECTORY "${OUT_DIR}")

        set(DEP_FILE "${OUT_DIR}/${SOURCE_NAME}.dep")

        if (BOT_CUDA_DISABLE_OPTIMIZATION)
          set(BOT_CUDA_CC_MODE "cc")
          set(OBJ_FILE "${OUT_DIR}/${SOURCE_NAME}.ptx")
        else()
          set(BOT_CUDA_CC_MODE "lto-cc")
          set(OBJ_FILE "${OUT_DIR}/${SOURCE_NAME}.ltoir")
        endif()

        add_custom_command(
          OUTPUT "${DEP_FILE}" "${OBJ_FILE}"
          COMMAND ${CMAKE_CXX_COMPILER} "${ABS_SOURCE_FILE}"
            -undef -nostdinc -M
            -std=c++20
            -D__CUDACC_RTC__
            -D__CUDA_ARCH__
            -D__CUDACC_VER_MAJOR__=${CUDAToolkit_VERSION_MAJOR}
            -D__CUDACC_VER_MINOR__=${CUDAToolkit_VERSION_MINOR}
            -D__CUDA_ARCH__=700
            ${DEFINE_ARGS}
            ${INCLUDE_ARGS}
            > "${DEP_FILE}"
            COMMAND $<TARGET_FILE:bot-cuda-cc> ${BOT_CUDA_CC_MODE} ${BOT_CUDA_ARCH} 
            "${OBJ_FILE}" "${ABS_SOURCE_FILE}" --
            ${DEFINE_ARGS}
            ${INCLUDE_ARGS}
            ${BOT_CUDA_COMPILE_FLAGS}
          DEPENDS "${ABS_SOURCE_FILE}" bot-cuda-cc
          DEPFILE ${DEP_FILE}
          VERBATIM
        )

        # Append the .ltoir file to the list
        list(APPEND OBJS "${OBJ_FILE}")
      endforeach()

      set(${OUT_OBJS} ${OBJS} PARENT_SCOPE)
    endfunction()

    if (ARGN)
      get_target_property(MAIN_TGT_SOURCES ${target} SOURCES)
      list(APPEND MAIN_TGT_SOURCES ${ARGN})
      set_target_properties(${target} PROPERTIES BOT_GPU_SOURCES 
        "${MAIN_TGT_SOURCES}")
    endif()

    set(ALL_OBJS)
    set(PROCESS_TGTS ${target})
    while (PROCESS_TGTS)
      list(POP_BACK PROCESS_TGTS TGT)
      if ("${TGT}" STREQUAL "bot-rt" OR NOT TARGET ${TGT})
        continue()
      endif()

      get_defines_and_includes_for_target(${TGT} TGT_DEFNS TGT_INC_DIRS)

      compile_target_sources(${TGT} "${TGT_DEFNS}" "${TGT_INC_DIRS}" TGT_OBJS)
      list(APPEND ALL_OBJS ${TGT_OBJS})

      get_target_property(LINK_LIBS ${TGT} LINK_LIBRARIES)
      foreach (LINK_TGT IN LISTS LINK_LIBS)
        list(APPEND PROCESS_TGTS ${LINK_TGT})
      endforeach ()
    endwhile ()

    compile_target_sources(bot-rt "" "" BOT_OBJS)
    list(APPEND ALL_OBJS ${BOT_OBJS})

    set(CUBIN_OUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${cuda_target}.cubin")

    if (BOT_CUDA_DISABLE_OPTIMIZATION)
      set(BOT_CUDA_LD_MODE "ld")
    else ()
      set(BOT_CUDA_LD_MODE "lto-ld")
    endif ()

    add_custom_command(
      OUTPUT "${CUBIN_OUT}"
      COMMAND $<TARGET_FILE:bot-cuda-cc> ${BOT_CUDA_LD_MODE} ${BOT_CUDA_ARCH}
      "${CUBIN_OUT}" ${ALL_OBJS} -- ${BOT_CUDA_LINK_FLAGS}
      DEPENDS ${ALL_OBJS}
      VERBATIM
    )

    add_custom_target(${cuda_target} ALL DEPENDS ${CUBIN_OUT})
  endfunction()
endif()
