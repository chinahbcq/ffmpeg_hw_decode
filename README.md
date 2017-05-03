# brief
This project demonstrate how to decode video stream and convert color space by using GPU.
When hard decoding video stream, the project use ffmpeg(version: 3.3) and `h264_cuvid` codec which based on NVIDA GPU.
When converting NV12 to bgr24 color space, the project use CUDA(version: 8.0) api where readers can find source conversion codes.

As this is a part of `Video Monitor` project focus on decoding RTSP video stream to OpenCV cv::gpu::GpuMat frame, high performance is 
a must. By using multiple GPUs, we can reduce stress of CPUs and most important, we can increase processing rate.

# build
make 

# run
./server

# dependency
* ffmpeg 3.3
* cuda 8.0
* opencv 2.4.13

```
export CMAKE_PREFIX_PATH=/path/to/opencv/dir
export LD_LIBRARY_PATH=$CMAKE_PREFIX_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/ffmpeg/lib:$LD_LIBRARY_PATH
```
