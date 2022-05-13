cmake_minimum_required(VERSION 3.12)
cmake_policy(VERSION ${CMAKE_VERSION})

project(BLAKE3
	VERSION 0.3.7
	DESCRIPTION "BLAKE3 C implementation"
	HOMEPAGE_URL "https://github.com/BLAKE3-team/BLAKE3"
	LANGUAGES C)

add_library(blake3)

set_target_properties(blake3 PROPERTIES
	PUBLIC_HEADER "blake3.h"
	SOVERSION ${PROJECT_VERSION_MAJOR}
	VERSION ${PROJECT_VERSION})

target_sources(blake3 PRIVATE
	blake3.c
	blake3_dispatch.c
	blake3_portable.c)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
	enable_language(ASM)
	if(MSVC)
		set(SUFFIX "windows_msvc.asm")
	elseif(WIN32)
		set(SUFFIX "windows_gnu.S")
	else()
		set(SUFFIX "unix.S")
	endif()
	target_sources(blake3 PRIVATE
		blake3_avx2_x86-64_${SUFFIX}
		blake3_avx512_x86-64_${SUFFIX}
		blake3_sse2_x86-64_${SUFFIX}
		blake3_sse41_x86-64_${SUFFIX})
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL i686)
	target_compile_options(blake3 PRIVATE
		-mavx2
		-mavx512f -mavx512vl
		-msse2
		-msse4.1)
	target_sources(blake3 PRIVATE
		blake3_avx2.c
		blake3_avx512.c
		blake3_sse2.c
		blake3_sse41.c)
elseif((CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64) OR
       (ANDROID_ABI STREQUAL armeabi-v7a))
	target_compile_definitions(blake3 PRIVATE BLAKE3_USE_NEON)
	target_compile_options(blake3 PRIVATE -mfpu=neon)
	target_sources(blake3 PRIVATE blake3_neon.c)
endif()

install(TARGETS blake3
        EXPORT blake3_targets
        PUBLIC_HEADER)
install(EXPORT blake3_targets
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
        FILE "${PROJECT_NAME}Targets.cmake"
        NAMESPACE "${PROJECT_NAME}::")