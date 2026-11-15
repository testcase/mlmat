cmake_minimum_required(VERSION 3.19)
project(Min-DevKit)

enable_testing()

string(REGEX REPLACE "(.*)/" "" THIS_PACKAGE_NAME "${CMAKE_CURRENT_SOURCE_DIR}")

if (APPLE)
    if (${CMAKE_GENERATOR} MATCHES "Xcode")
            if (${XCODE_VERSION} VERSION_LESS 10)
                message(STATUS "Xcode 10 or higher is required. Please install from the Mac App Store.")
                return ()
            elseif(${XCODE_VERSION} VERSION_GREATER_EQUAL 12)
                set(C74_BUILD_FAT YES)
            endif ()
    endif ()

    if (NOT CMAKE_OSX_ARCHITECTURES)
        if(C74_BUILD_FAT)
            set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "macOS architecture" FORCE)
        else()
            set(CMAKE_OSX_ARCHITECTURES ${CMAKE_SYSTEM_PROCESSOR} CACHE STRING "macOS architecture" FORCE)
        endif()
        message("CMAKE_OSX_ARCHITECTURES set to ${CMAKE_OSX_ARCHITECTURES}")
    endif()
endif()


# Misc setup and subroutines

# max-package.cmake only reads the version when package-info.json.in
# exists; this repo configures package-info.json at release time
# instead (see the release workflows), so supply the version here from
# the single source of truth. Without it PACKAGE_VERSION is empty and
# the Windows verinfo generation fails in max-package.cmake
# (string REPLACE with a missing argument).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/package-info.json" PKGINFOFILE)
string(JSON PACKAGE_VERSION ERROR_VARIABLE PKG_ERROR GET ${PKGINFOFILE} "version")
if (PACKAGE_VERSION MATCHES "-NOTFOUND")
    set(PACKAGE_VERSION "0.0.0")
endif ()

include(${CMAKE_CURRENT_SOURCE_DIR}/source/min-api/script/min-package.cmake)

# Centralized third-party dependency targets (mlmat::deps, mlmat::fft)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cmake/mlmat_deps.cmake)

# Add the Lib, if it exists
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/source/min-lib/CMakeLists.txt")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/min-lib)
endif ()


# Generate a project for every folder in the "source/projects" folder
SUBDIRLIST(PROJECT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/source/projects)
foreach (project_dir ${PROJECT_DIRS})
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/source/projects/${project_dir}/CMakeLists.txt")
        message("Generating: ${project_dir}")
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/projects/${project_dir})
    endif ()
endforeach ()

# Regression tests (tests/conversions, tests/som, tests/spectral). These are small compared
# to the externals themselves, so they are built by default; disable with
# -DMLMAT_BUILD_TESTS=OFF.
option(MLMAT_BUILD_TESTS "Build the regression tests" ON)
if (MLMAT_BUILD_TESTS)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/conversions/CMakeLists.txt")
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/conversions)
    endif ()
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/som/CMakeLists.txt")
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/som)
    endif ()
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/spectral/CMakeLists.txt")
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/spectral)
    endif ()
endif ()

# mlpack 4.x uses C++17 <filesystem>, which requires macOS 10.15 or later.
# (The min-api scripts force the deployment target back to 10.11 in each
# project subdirectory, so this must be set again after them.)
if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version" FORCE)
endif ()

# Comment the line below if you want automatic cmake regneration enabled
set(CMAKE_SUPPRESS_REGENERATION true)


