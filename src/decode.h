/**  
 *  Copyright (c) 2017 sensetime.com, Inc. All Rights Reserved
 *  @author Chen Qian (chenqian@sensetime.com)
 *  @date 2017.04.18 11:32:44
 *  @brief 解码
 **/
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#ifdef __cplusplus
extern "C"{
#endif

#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libswscale/swscale.h"
#include "libavutil/opt.h"
#include "libavutil/error.h"
#include "libavutil/parseutils.h"
#include "libavutil/samplefmt.h"
#include "libavutil/fifo.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/dict.h"
#include "libavutil/mathematics.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avstring.h"
#include "libavutil/imgutils.h"
#include "libswresample/swresample.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/avfiltergraph.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"

/**
 * 本函数执行ffmpeg的初始化
 *
 * @return 失败返回非0, 成功返回0
 */
int ffmpeg_global_init();

/**
 * 本函数提供视频转码功能
 * 
 * @param addr 视频流的地址
 * @param cpu_cb cpu 回调函数,原型 int cb_name(cv::Mat &mat),由使用方设计实现; 当解码出一
 *                   帧图像后,首先会转化为cv::Mat格式,然后再调用cb将转化后的mat回调给使用
 *                   者.
 * @param gpu_cb gpu 回调函数,原型 int cb_name(cv::gpu::GpuMat &mat),由使用方设计实现; 当解码出一
 *                   帧图像后,首先会转化为cv::gpu::GpuMat格式,然后再调用cb将转化后的mat回调给使用
 *                   者.
 * @param use_hw_decode 若该参数设置为true,则使用gpu硬解码; 若设置为false,则使用
 *                      软解码;
 *                      默认false
 * @param only_key_frame 若该参数设置为true, 则只解码视频流中的关键帧并输出, 能够
 *                       显著的减少重复帧的影响,降低对系统的压力,提高整体的性能;
 *                       若该参数设置为false, 则视频流中的每一帧会解码输出.
 *                       默认 false 
 *
 * @return 失败返回非0, 成功返回0
 */
int ffmpeg_video_decode(const std::string &addr, 
		int (*cpu_cb)(cv::Mat&),
		int (*gpu_cb)(cv::gpu::GpuMat &),
		bool use_hw_decode = false, 
		bool only_key_frame = false);

#ifdef __cplusplus
}
#endif

