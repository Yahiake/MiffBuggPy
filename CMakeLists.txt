cmake_minimum_required(VERSION 3.22)
project(MiffBuggPy VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(modules/JUCE)

# ---------------------------------------------------------------------------
# RTNeural backend selection.
#
# Backend choice is delegated to RTNeural's own cmake/ChooseBackend.cmake, which
# modules/RTNeural's CMakeLists.txt includes. That matters: setting
# RTNEURAL_USE_XSIMD by hand on top of it does not replace the default, it adds to
# it. ChooseBackend defaults to Eigen, so a manual xsimd define leaves BOTH
# macros defined at once and the maths provider no longer matches the layer
# headers. The option has to be set before the subdirectory is added.
#
# This is not a micro-optimisation. The shipped network is dense 4->128, GRU
# 128->128, dense 128->1, which is roughly 99k MACs per sample per channel. At
# 48 kHz that is about 4.8 GMAC/s per channel, or 9.5 GMAC/s for a stereo pair.
# Scalar float cannot carry that in real time. A SIMD backend is a requirement
# for this plugin to run at all, not a nicety.
# ---------------------------------------------------------------------------
set(MIFFBUGGPY_BACKEND "xsimd" CACHE STRING "SIMD backend for RTNeural: xsimd, eigen or stl")
set_property(CACHE MIFFBUGGPY_BACKEND PROPERTY STRINGS xsimd eigen stl)

# ChooseBackend acts on three options that all default to OFF, and falls through
# to Eigen when none is set. Selecting one has to force the other two OFF, or a
# stale value from a previous configure leaves two backends defined at once.
if(MIFFBUGGPY_BACKEND STREQUAL "xsimd")
    set(RTNEURAL_XSIMD ON  CACHE BOOL "" FORCE)
    set(RTNEURAL_EIGEN OFF CACHE BOOL "" FORCE)
    set(RTNEURAL_STL   OFF CACHE BOOL "" FORCE)
elseif(MIFFBUGGPY_BACKEND STREQUAL "eigen")
    set(RTNEURAL_XSIMD OFF CACHE BOOL "" FORCE)
    set(RTNEURAL_EIGEN ON  CACHE BOOL "" FORCE)
    set(RTNEURAL_STL   OFF CACHE BOOL "" FORCE)
else()
    set(RTNEURAL_XSIMD OFF CACHE BOOL "" FORCE)
    set(RTNEURAL_EIGEN OFF CACHE BOOL "" FORCE)
    set(RTNEURAL_STL   ON  CACHE BOOL "" FORCE)
endif()

add_subdirectory(modules/RTNeural)

if(MIFFBUGGPY_BACKEND STREQUAL "stl")
    message(WARNING "MiffBuggPy -- RTNeural backend: scalar STL. Not real-time safe for this network.")
endif()

# 32-byte alignment plus /arch:AVX2 lets the SIMD backend use full 256-bit vectors.
# RTNEURAL_DEFAULT_ALIGNMENT stays at 16 without this, which silently halves the
# usable vector width on a network that is almost entirely matrix-vector products.
#
# Applied to RTNeural only, not the whole plugin, so JUCE's own code keeps the
# compiler's default baseline. This project targets Windows/MSVC x64 only, so
# there is no portability reason to feature-detect: /arch:AVX2 is guaranteed.
if(MSVC)
    target_compile_options(RTNeural PUBLIC /arch:AVX2)
    target_compile_definitions(RTNeural PUBLIC RTNEURAL_AVX_ENABLED=1 RTNEURAL_DEFAULT_ALIGNMENT=32)
    message(STATUS "MiffBuggPy -- AVX2 enabled on RTNeural (MSVC)")
elseif(APPLE)
    message(STATUS "MiffBuggPy -- macOS build configured")
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(RTNeural PUBLIC -mavx2 -mfma)
        target_compile_definitions(RTNeural PUBLIC RTNEURAL_AVX_ENABLED=1 RTNEURAL_DEFAULT_ALIGNMENT=32)
    endif()
    message(STATUS "MiffBuggPy -- Linux build configured")
endif()

# ---------------------------------------------------------------------------
# The plugin. VST3, Standalone, and AU on macOS.
# ---------------------------------------------------------------------------
if(APPLE)
    set(MIFFBUGGPY_FORMATS VST3 AU Standalone)
else()
    set(MIFFBUGGPY_FORMATS VST3 Standalone)
endif()

juce_add_plugin(MiffBuggPy
    COMPANY_NAME "Yahia Kemari"
    PLUGIN_MANUFACTURER_CODE YhKm
    PLUGIN_CODE MbfB
    FORMATS ${MIFFBUGGPY_FORMATS}
    ProductName "MiffBuggPy"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
)

target_compile_definitions(MiffBuggPy PUBLIC
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_STANDALONE_OWN_WRAPPER=1
    # v1 scope: one pedal, done properly.
    JUCE_MODAL_LOOPS_PERMITTED=0
)

target_sources(MiffBuggPy PRIVATE
    src/Plugin.cpp
    src/Editor.cpp
    modules/r8brain-free-src/r8bbase.cpp
)

# The exported network is embedded so the built plugin is self-contained and does
# not depend on model.json sitting next to it on disk.
juce_add_binary_data(BinaryData SOURCES
    model.json
)
set_target_properties(BinaryData PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(MiffBuggPy PUBLIC
    BinaryData
    juce::juce_audio_utils
    juce::juce_dsp
    # LookAndFeel_V4, the concrete base the editor's knob skin derives from.
    juce::juce_gui_extra
    RTNeural
)

# RTNeural.h lives one level below the RTNeural project root, which the
# RTNeural target does not add to its interface include path.
target_include_directories(MiffBuggPy PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/modules/RTNeural/RTNeural"
    "${CMAKE_CURRENT_SOURCE_DIR}/modules/r8brain-free-src"
)

# Must come after target_link_libraries. The generated header includes exactly the
# modules linked into the target, so generating it earlier yields a header missing
# juce_audio_formats, juce_core and the binary-data header.
juce_generate_juce_header(MiffBuggPy)

# ---------------------------------------------------------------------------
# Bundle repair.
#
# JUCE 8.0.12 writes the VST3 bundle's moduleinfo.json with trailing commas, which
# is not valid JSON. Hosts read that file to enumerate classes before loading the
# module, so a strict parser rejects the bundle outright and the plugin silently
# never appears in the host at all. The file is generated, so it has to be fixed
# after the generator rather than in the source tree.
#
# A custom target rather than POST_BUILD on MiffBuggPy_VST3, because JUCE
# assembles the bundle in its own post-build step and POST_BUILD commands do not
# reliably run after it. A custom target's commands are guaranteed to run once
# its dependencies finish.
# ---------------------------------------------------------------------------
add_custom_target(MiffBuggPyFixBundle ALL
    COMMAND ${CMAKE_COMMAND}
            "-DSTART=$<TARGET_FILE_DIR:MiffBuggPy_VST3>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixModuleInfoJson.cmake"
    DEPENDS MiffBuggPy_VST3
    COMMENT "Normalising moduleinfo.json to strict JSON"
    VERBATIM
)

# ---------------------------------------------------------------------------
# Release package.
#
# Depends on MiffBuggPyFixBundle rather than MiffBuggPy_VST3, because packaging a
# bundle whose moduleinfo.json is still invalid JSON produces something no host
# will list, and the entire point of the fix step is to make that impossible to
# ship quietly.
# ---------------------------------------------------------------------------
if(WIN32)
    set(miffbuggpyPackageDir "${CMAKE_CURRENT_SOURCE_DIR}/dist")

    add_custom_target(MiffBuggPyPackage ALL
        COMMAND ${CMAKE_COMMAND}
                "-DSTAGE=$<TARGET_FILE_DIR:MiffBuggPy>"
                "-DDEST=${miffbuggpyPackageDir}"
                "-DREADMESRC=${CMAKE_CURRENT_SOURCE_DIR}/packaging/README.txt"
                "-DBUNDLE=${PROJECT_NAME}.vst3"
                "-DEXE=${PROJECT_NAME}.exe"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Package.cmake"
        DEPENDS MiffBuggPyFixBundle MiffBuggPy_Standalone
        COMMENT "Packaging MiffBuggPy into dist/"
        VERBATIM
    )
endif()
