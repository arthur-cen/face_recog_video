cmake_minimum_required(VERSION 2.8)
project(capstone_track3)

set(CMAKE_C_FLAGS "-std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11")
set(CMAKE_BUILD_TYPE Release)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

### 3rd party ###
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# LIBUSB
find_package(PkgConfig)
pkg_check_modules(LIBUSB libusb-1.0)

set(EXT_LIBS
     ${OpenCV_LIBS}
     ${LIBUSB_LIBRARIES}
     -lpthread
     -lblas
     -llapack
)


# Set library and headers.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)


# Build the sample apps.
set(APPS
	"src/train.cpp"
	"src/online_face_recog.cpp"
    "src/simple_model_tester.cpp"
)

foreach(app ${APPS})
    get_filename_component(EXE_NAME ${app} NAME_WE)
    add_executable(${EXE_NAME} ${app})
    target_link_libraries(${EXE_NAME} -lPerceptInPIRVS ${EXT_LIBS})
endforeach(app)
