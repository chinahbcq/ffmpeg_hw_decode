#include <stdio.h>
#include <stdlib.h>
#include "decode.h"
#include <pthread.h> 
//#define TEST
int cpu_callback(cv::Mat &image) {
#ifdef TEST
	printf("get cpu image, image.width:%d, image.height:%d\n", image.cols, image.rows);
	//cv::imshow("press ESC to exit", image);
	//cv::waitKey(0);
#endif
	return 0;
}

int gpu_callback(cv::gpu::GpuMat &image) {
#ifdef TEST
	printf("get gpu image, image.width:%d, image.height:%d\n", image.cols, image.rows);
	cv::Mat mat(image.cols, image.rows, CV_8UC3);
	image.download(mat);

	//cv::imshow("press ESC to exit", mat);
	//cv::waitKey(0);
#endif
	return 0;
}

void *videoHandle1(void *) {
	std::string rtsp_addr = "/home/sensetime/test.h264";
	printf("open video:%s\n", rtsp_addr.c_str());
	
	bool use_hw_decode = true; //使用硬解码
	bool only_key_frame = false; //是否只使用关键帧
	ffmpeg_video_decode(rtsp_addr, cpu_callback, gpu_callback, use_hw_decode, only_key_frame);
}

void *videoHandle2(void *) {
	std::string rtsp_addr = "rtsp://admin:12345@10.0.3.137:554";
	printf("open video:%s\n", rtsp_addr.c_str());
	
	bool use_hw_decode = true; //使用硬解码
	bool only_key_frame = false; //是否只使用关键帧
	ffmpeg_video_decode(rtsp_addr, cpu_callback, gpu_callback, use_hw_decode, only_key_frame);
}

int main() {
	//ffmpeg初始化
	ffmpeg_global_init();
	
	//thread mode
	pthread_t t1,t2;
	pthread_create(&t1, NULL, videoHandle1, NULL);
	pthread_create(&t2, NULL, videoHandle2, NULL);

	getchar();
	
	return 0;
}
