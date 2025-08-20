function(fetch_git_dependency NAME)
  set(ONE_VALUE_ARGS URL TAG SUBMODULES)
  cmake_parse_arguments(A "" "${ONE_VALUE_ARGS}" "" ${ARGN})

  if (NOT A_URL)
    message(FATAL_ERROR "fetch_git_dependency: you must specify URL")
  endif()
  if (NOT A_TAG)
    message(FATAL_ERROR "fetch_git_dependency: you must specify URL")
  endif()

  if (NOT DEFINED A_USE_SUBMODULES)
    set(A_USE_SUBMODULES ON)
  endif()

  set(TIMESTAMP_FILE "${BOT_BUNDLE_TMP_DIR}/${NAME}-stamp")
  set(CONFIG_HASH_FILE "${BOT_BUNDLE_TMP_DIR}/${NAME}-config-hash")

  file(SHA512 "${CMAKE_CURRENT_LIST_FILE}" CONFIG_FILE_HASH)

  if (EXISTS "${TIMESTAMP_FILE}")
    file(READ "${TIMESTAMP_FILE}" CUR_BUILD_TIMESTAMP)
  else()
    set(CUR_BUILD_TIMESTAMP "")
  endif()

  if (EXISTS "${CONFIG_HASH_FILE}")
    file(READ "${CONFIG_HASH_FILE}" CUR_CONFIG_HASH)
  else()
    set(CUR_CONFIG_HASH "")
  endif()

  set(NEED_FETCH FALSE)
  if (NOT "${CUR_CONFIG_HASH}" MATCHES "${CONFIG_FILE_HASH}")
    set(NEED_FETCH_GTEST TRUE)
  endif()

  if (NOT EXISTS "${TIMESTAMP_FILE}")
    set(NEED_FETCH TRUE)
  else()
    if ("${CMAKE_CURRENT_LIST_FILE}" IS_NEWER_THAN "${TIMESTAMP_FILE}")
      set(NEED_FETCH TRUE)
    endif()
  endif()

  if (NOT NEED_FETCH)
    return()
  endif()

  if (A_USE_SUBMODULES)
    set(GIT_SUBMODULES_ARG)
    set(GIT_SUBMODULES_RECURSE ON)
  else()
    set(GIT_SUBMODULES_ARG "GIT_SUBMODULES \"\"")
    set(GIT_SUBMODULES_RECURSE OFF)
  endif()

  FetchContent_Populate(${NAME}-bundled
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}"
    GIT_REPOSITORY "${A_URL}"
    GIT_TAG "${A_TAG}"
    GIT_PROGRESS ON
    ${GIT_SUBMODULES_ARG}
    GIT_SUBMODULES_RECURSE ${GIT_SUBMODULES_RECURSE}
  )

  file(SHA512 "${CMAKE_CURRENT_LIST_FILE}" CONFIG_FILE_HASH)

  file(TOUCH "${TIMESTAMP_FILE}")
  file(WRITE "${CONFIG_HASH_FILE}" "${CONFIG_FILE_HASH}")
endfunction()
