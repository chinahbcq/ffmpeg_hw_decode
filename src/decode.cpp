/**  
 *  Copyright (c) 2017 LGPL, Inc. All Rights Reserved
 *  @author Chen Qian (chinahbcq@qq.com)
 *  @date 2017.04.18 11:33:21
 *  @brief 解码
 */
#include "decode.h"
#include "yuv2bgr.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand.h>
static volatile int DISTURBE_SIGNALS   = 0;
static int decode_interrupt_cb(void *ctx) {
	return DISTURBE_SIGNALS > 0;
}
static const AVIOInterruptCB int_cb = { decode_interrupt_cb, NULL };

int ffmpeg_decode(AVFormatContext *ic, 
		const int type, 
		const int stream_id, 
		int (*cpu_cb)(const int type, cv::Mat&),
		int (*gpu_cb)(const int type, cv::gpu::GpuMat &),
		bool use_hw_decode,
		bool only_key_frame);

int ffmpeg_global_init() {
	avcodec_register_all();
	avdevice_register_all(); 
	avfilter_register_all(); 
	av_register_all();
	avformat_network_init();
	av_log_set_level(AV_LOG_INFO);
	return 0;
}

int ffmpeg_video_decode(const std::string &addr,
		const int type, 
		int (*cpu_cb)(const int type, cv::Mat&),
		int (*gpu_cb)(const int type, cv::gpu::GpuMat &),
		bool use_hw_decode, 
		bool only_key_frame) {
	av_log(NULL, AV_LOG_INFO, "stream path:%s, \nuse_hw_decode:%s, \nonly_key_frame:%s,\n", addr.c_str(), 
			use_hw_decode ? "true" : "false",
			only_key_frame ? "true" : "false");
	AVFormatContext *ic;
	ic = avformat_alloc_context();
	if (!ic) {
		return -1;
	}
	//设置ic的初始化参数
	ic->video_codec_id = AV_CODEC_ID_NONE;
	ic->audio_codec_id = AV_CODEC_ID_NONE;
	ic->flags |= AVFMT_FLAG_NONBLOCK;
	ic->interrupt_callback = int_cb;
	
	//设置opts参数
	AVDictionary *fmt_opts = NULL;
	av_dict_set(&fmt_opts, "max_req_times", "2", 0);

	int err = avformat_open_input(&ic, addr.c_str(), NULL, &fmt_opts);
	if (err < 0) {
		av_dict_free(&fmt_opts);
		return err;
	} 

	err = avformat_find_stream_info(ic, NULL);
	if (err < 0) {
		av_dict_free(&fmt_opts);
		avformat_close_input(&ic);
		return err;
	} 
	
	//选出分辨率最大的的video stream
	bool has_video_stream = false;
	unsigned int video_stream_id = 0;
	int max_video_resolution = 0;

	for (unsigned int i = 0; i < ic->nb_streams; i++) {
		AVStream *st = ic->streams[i];
		if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
			if ((st->disposition & AV_DISPOSITION_ATTACHED_PIC) &&
					(st->discard < AVDISCARD_ALL)) {
				//audio cover stream
			} else {
				int resolution = st->codecpar->width * st->codecpar->height;
				if (resolution > max_video_resolution &&
						st->codecpar->codec_id != AV_CODEC_ID_NONE) {
					has_video_stream = true;
					max_video_resolution = resolution;
					video_stream_id = i;
				}
			}
		}
	}
	
	//若没有video_stream, 返回错误
	if (!has_video_stream) {
		av_dict_free(&fmt_opts);
		avformat_close_input(&ic);
		return -1;
	}
	
	//开始对视频流解码	
	err = ffmpeg_decode(ic, type, video_stream_id, cpu_cb, gpu_cb, use_hw_decode, only_key_frame);
	if (err != 0) {
		av_log(NULL, AV_LOG_FATAL, "ffmpeg decode failed\n");
		return err;
	}
	
	//释放内存
	av_dict_free(&fmt_opts);
	avformat_close_input(&ic);

	return 0;
}

int ffmpeg_decode(AVFormatContext *ic,
		const int type, 
		const int stream_id,
		int (*cpu_cb)(const int type, cv::Mat&),
		int (*gpu_cb)(const int type, cv::gpu::GpuMat &),
		bool use_hw_decode,
		bool only_key_frame) {
	AVCodecParameters *codecpar = NULL;
	AVCodec *codec = NULL;
	AVCodecContext *codec_ctx = NULL;

	//duration
	int64_t duration = ic->duration / AV_TIME_BASE;
	//fps
	int fps_num = ic->streams[stream_id]->r_frame_rate.num;
	int fps_den = ic->streams[stream_id]->r_frame_rate.den;
	double fps = 0.0;
	if (fps_den > 0) {
		fps = fps_num / fps_den;	
	} 
	av_log(NULL, AV_LOG_INFO, "duration:%ld,\nfps:%f,\n", duration, fps);
	codecpar = ic->streams[stream_id]->codecpar;
	if (use_hw_decode) {
		//Nvidia硬解码, AV_CODEC_ID_H264 cuvid.c
		codec = avcodec_find_decoder_by_name("h264_cuvid");
	} else {
		//软解码
		codec = avcodec_find_decoder(codecpar->codec_id);
	}
	if (NULL == codec) {
		av_log(NULL, AV_LOG_FATAL, "avcodec_find_decoder_by_name h264_cuvid failed\n");
		return -1;
	}
	
	codec_ctx = avcodec_alloc_context3(codec);
	if (NULL == codec_ctx) {
		av_log(NULL, AV_LOG_FATAL, "avcodec_alloc_context3 failed\n");
		return -2;
	}
	
	//time_base 不需要赋值
	//codec_ctx->time_base = codecpar->sample_aspect_ratio; 
	//codec_id不需要赋值 
	//codec_ctx->codec_id = codecpar->codec_id; 
	//硬解码不需要赋值,从硬件读取, AV_PIX_FMT_NV12;
	//软编码需要赋值
	codec_ctx->pix_fmt = AVPixelFormat(codecpar->format);  
	codec_ctx->height = codecpar->height;
	codec_ctx->width = codecpar->width;
	codec_ctx->thread_count = 8; //设置解码线程数目
	codec_ctx->thread_type  = FF_THREAD_FRAME; //设置解码type
	//codec_ctx->thread_type  = FF_THREAD_SLICE; //设置解码type

	int err = avcodec_open2(codec_ctx, codec, NULL);
	if (err < 0) {
		av_log(NULL, AV_LOG_FATAL, "avcodec_open2 failed\n");
		return -3;
	}

	//视频流中解码的frame
	AVFrame *frame = av_frame_alloc();	
	//sws frame
	AVFrame *frame_bgr = av_frame_alloc();	
	/**
	 * AV_PIX_FMT_BGRA 对应opencv中的CV_8UC4, 
	 * AV_PIX_FMT_BGR24对应opencv中的CV_8UC3
	 */
	AVPixelFormat format = AV_PIX_FMT_BGR24; 
	int cv_format = CV_8UC3;
	int align = 1;
	av_log(NULL, AV_LOG_INFO, "width:%d,\nheight:%d,\n", codec_ctx->width, codec_ctx->height);
	int buffer_size = av_image_get_buffer_size(format, codec_ctx->width, codec_ctx->height, align);
	unsigned char *buffer = (unsigned char*) av_malloc(buffer_size);
	if (NULL == buffer) {
		av_log(NULL, AV_LOG_FATAL, "allocate buffer failed!\n");
		return -4;
	}
	av_image_fill_arrays(frame_bgr->data, frame_bgr->linesize, buffer, format, codec_ctx->width, codec_ctx->height, align);
	
	//初始化图像pixer format进行转换的context	
	struct SwsContext * sws_ctx;
	int dest_width = codec_ctx->width;
	int dest_height = codec_ctx->height;
	av_log(NULL, AV_LOG_INFO, "src_codec_id:%d,\ncodec_id:%d,\n", codecpar->codec_id, codec_ctx->codec_id);
	av_log(NULL, AV_LOG_INFO, "src_pix_format:%d,\ncodec_pix_format:%d,\ndest_pix_format:%d,\n", codecpar->format, codec_ctx->pix_fmt, format);
	//软解码
	sws_ctx = sws_getContext(codec_ctx->width, 
			codec_ctx->height,
			codec_ctx->pix_fmt, 
			dest_width, 
			dest_height, 
			format, 
			SWS_FAST_BILINEAR,
			NULL, NULL,NULL);

	//硬解码
	int is_first_frame = false;
	int bufsize0, bufsize1, resolution;
	cv::gpu::GpuMat reqMat, resMat;
	//开始解码
	AVPacket pkt;
	while (av_read_frame(ic, &pkt) >= 0) {
		if (pkt.stream_index != stream_id ||
				(only_key_frame && !(pkt.flags & AV_PKT_FLAG_KEY)) || /*不是关键帧就会跳过*/
				pkt.size < 1) {
			goto discard_packet;
		}

		//解码packet
		err = avcodec_send_packet(codec_ctx, &pkt);
		if (err != AVERROR(EAGAIN) && err != AVERROR_EOF && err < 0) {
			goto discard_packet;
		}
		err = avcodec_receive_frame(codec_ctx, frame);
		if (err == AVERROR(EAGAIN)) {
			goto discard_packet;
		} else if (err == AVERROR_EOF) {
			av_packet_unref(&pkt);
			break;
		}
		else if (err < 0) {
			goto discard_packet;
		}

		//use gpu将frame转化为cv::Mat 格式
		if (use_hw_decode) {
			
			if (!is_first_frame) {
				bufsize0 = frame->height * frame->linesize[0];
				bufsize1 = frame->height * frame->linesize[1] / 2;
				resolution = frame->height * frame->width;
				//硬解码		
				reqMat.create(frame->height, frame->width, cv_format);
				resMat.create(frame->height, frame->width, cv_format);
				resMat.step = frame_bgr->linesize[0];
				is_first_frame = true;
			}
			cudaMemcpy(reqMat.data, frame->data[0], bufsize0, cudaMemcpyHostToDevice);
			cudaMemcpy(reqMat.data + bufsize0, frame->data[1], bufsize1, cudaMemcpyHostToDevice);

			cvtColor(reqMat.data, resMat.data, resolution, frame->height, frame->width, frame->linesize[0]);
			//回调
			gpu_cb(type, resMat);
		} else { //软解码
			memset(buffer, 0, buffer_size);
			sws_scale(sws_ctx, frame->data,
					frame->linesize, 0, frame->height, frame_bgr->data,
					frame_bgr->linesize);

			/*show 出来的img出现问题,是因为linesize设置错误, 因为是bgr24,
			 * 所以linesize应该为 width * 3
			 * */	
			cv::Mat image(frame->height, frame->width, cv_format, frame_bgr->data[0], frame_bgr->linesize[0]);
			//回调
			cpu_cb(type, image);
		}

discard_packet:
		av_frame_unref(frame);
		av_packet_unref(&pkt);
	}
	sws_freeContext(sws_ctx);
	av_frame_unref(frame);
	av_frame_unref(frame_bgr);
	av_freep(&frame);
	av_freep(&frame_bgr);
	av_freep(&buffer);
	avcodec_close(codec_ctx);
	av_freep(&codec_ctx);
	return 0;
}
