# Optional, target-only compiler experiments. The default build remains unchanged.
option(ASTRA_AUTO_INLINE "Enable clang-cl automatic inlining in RelWithDebInfo" OFF)
option(ASTRA_THINLTO "Enable target-only clang-cl ThinLTO" OFF)

if(ASTRA_AUTO_INLINE OR ASTRA_THINLTO)
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
