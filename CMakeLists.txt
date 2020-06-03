cmake_minimum_required(VERSION 2.8)

# ---[ WCaffe project
project(WCaffeTest)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)


# ---[ OpenCV
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)


# ---[ caffe
find_package(Caffe REQUIRED)
add_definitions(-DCPU_ONLY)
#set(Caffe_INCLUDE_DIRS "/opt/caffe/src;/usr/include;/opt/caffe/bin/include;/usr/include/opencv;/usr/include/atlas;/opt/caffe/include")
#set(Caffe_LIBRARIES "/opt/caffe/bin/lib/libcaffe.so")
include_directories(${Caffe_INCLUDE_DIRS})


# ---[ Boost
set(BOOST_COMPONENTS filesystem system thread)
find_package(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})


# ---[ Glog
find_package(Glog REQUIRED)


# ---[ TBB
find_package(TBB REQUIRED)


# ---[ ITK
set(ITK_DIR "/opt/ITK/Bin/")

find_package(ITK REQUIRED)
if(ITK_FOUND)
        include(${ITK_USE_FILE})
endif(ITK_FOUND)


# ---[ VTK
set(VTK_DIR "/opt/VTK/Bin/")

find_package(VTK REQUIRED)
if(VTK_FOUND)
        include(${VTK_USE_FILE})
endif(VTK_FOUND)


# ---[
#add_executable(main main.cpp)

#target_link_libraries(main ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIBRARIES} ${OpenCV_LIBS})

add_executable(trainer trainer.cpp database.hpp classify.hpp)

target_link_libraries(trainer ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIBRARIES} ${OpenCV_LIBS})

