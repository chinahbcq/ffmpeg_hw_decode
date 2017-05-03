# build
make 
make clean
# run
./server
#dependency
ffmpeg 3.3
cuda 8.0
opencv 2.4.13


```
export CMAKE_PREFIX_PATH=/path/to/opencv/dir
export LD_LIBRARY_PATH=$CMAKE_PREFIX_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/ffmpeg/lib:$LD_LIBRARY_PATH
```
