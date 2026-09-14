# Optional, target-only compiler experiments. The default build remains unchanged.
option(ASTRA_AUTO_INLINE "Enable clang-cl automatic inlining in RelWithDebInfo" OFF)
option(ASTRA_THINLTO "Enable target-only clang-cl ThinLTO" OFF)
set(ASTRA_PGO_GENERATE_DIR "" CACHE PATH "Directory for target-only clang-cl IR PGO raw profiles")
set(ASTRA_PGO_USE_FILE "" CACHE FILEPATH "Indexed target-only clang-cl IR PGO profile")

if(ASTRA_PGO_GENERATE_DIR AND ASTRA_PGO_USE_FILE)
    message(FATAL_ERROR "ASTRA_PGO_GENERATE_DIR and ASTRA_PGO_USE_FILE are mutually exclusive")
endif()

if(ASTRA_AUTO_INLINE OR ASTRA_THINLTO OR ASTRA_PGO_GENERATE_DIR OR ASTRA_PGO_USE_FILE)
    if(NOT MSVC OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "Astra compiler experiments currently require clang-cl")
    endif()
endif()

if(ASTRA_AUTO_INLINE)
    # CMake's RelWithDebInfo defaults to /Ob1 (inline-hint-functions). /Ob2 is
    # the normal Release inlining policy. Leave optimization level, floating
    # point rules, debug information and dependency targets unchanged.
    target_compile_options(shadps4 PRIVATE
        "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CXX>>:/Ob2>")
endif()

if(ASTRA_THINLTO)
    get_filename_component(_astra_linker "${CMAKE_LINKER}" NAME_WE)
    if(NOT _astra_linker STREQUAL "lld-link")
        message(FATAL_ERROR "ASTRA_THINLTO requires lld-link")
    endif()
    # lld-link recognizes LLVM bitcode directly. -flto belongs to the compile
    # step, not the native COFF linker command. Keep dependency libraries native.
    target_compile_options(shadps4 PRIVATE
        "$<$<COMPILE_LANGUAGE:C,CXX>:-flto=thin>")
    target_link_options(shadps4 PRIVATE
        /opt:lldltojobs=12
        "/lldltocache:${CMAKE_CURRENT_BINARY_DIR}/astra-thinlto-cache")
endif()

if(ASTRA_PGO_GENERATE_DIR)
    # IR PGO is enabled only on the main executable. The profile runtime is
    # referenced by compiler-generated object directives; direct lld-link
    # therefore needs the matching clang resource directory, but must not be
    # given -fprofile-generate itself.
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" --print-resource-dir
        OUTPUT_VARIABLE _astra_pgo_resource_dir
        ERROR_VARIABLE _astra_pgo_resource_error
        RESULT_VARIABLE _astra_pgo_resource_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _astra_pgo_resource_result EQUAL 0 OR
       NOT IS_DIRECTORY "${_astra_pgo_resource_dir}/lib/windows")
        message(FATAL_ERROR
            "Could not query clang resource lib/windows: ${_astra_pgo_resource_error}")
    endif()
    if(NOT EXISTS "${_astra_pgo_resource_dir}/lib/windows/clang_rt.profile-x86_64.lib")
        message(FATAL_ERROR "Missing clang_rt.profile-x86_64.lib in clang resource directory")
    endif()

    target_compile_options(shadps4 PRIVATE
        "$<$<COMPILE_LANGUAGE:C,CXX>:-fprofile-generate=${ASTRA_PGO_GENERATE_DIR}>")
    target_link_options(shadps4 PRIVATE "/libpath:${_astra_pgo_resource_dir}/lib/windows")
endif()

if(ASTRA_PGO_USE_FILE)
    if(NOT EXISTS "${ASTRA_PGO_USE_FILE}")
        message(FATAL_ERROR "ASTRA_PGO_USE_FILE does not exist: ${ASTRA_PGO_USE_FILE}")
    endif()
    target_compile_options(shadps4 PRIVATE
        "$<$<COMPILE_LANGUAGE:C,CXX>:-fprofile-use=${ASTRA_PGO_USE_FILE}>")
endif()
