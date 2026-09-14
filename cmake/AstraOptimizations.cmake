# Optional, target-only compiler experiments. The default build remains unchanged.
option(ASTRA_AUTO_INLINE "Enable clang-cl automatic inlining in RelWithDebInfo" OFF)

if(ASTRA_AUTO_INLINE)
    if(NOT MSVC OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "ASTRA_AUTO_INLINE currently requires clang-cl")
    endif()
    # CMake's RelWithDebInfo defaults to /Ob1 (inline-hint-functions). /Ob2 is
    # the normal Release inlining policy. Leave optimization level, floating
    # point rules, debug information and dependency targets unchanged.
    target_compile_options(shadps4 PRIVATE
        "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CXX>>:/Ob2>")
endif()
