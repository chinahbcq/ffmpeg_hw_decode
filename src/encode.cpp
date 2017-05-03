/**  
 *  Copyright (c) 2017 LGPL, Inc. All Rights Reserved
 *  @author Chen Qian (chinahbcq@qq.com)
 *  @date 2017.04.19 10:13:51
 *  @brief 编码
 */
#include "encode.h"

//对frame进行编码, 编码成jpeg格式
int ffmpeg_encode_jpeg(AVFrame *frame, std::string &filename) {
	AVCodec *codec = NULL;
	AVCodecContext *codec_ctx = NULL;
	codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
	//硬编码
	//codec = avcodec_find_encoder_by_name("h264_nvenc");
	//codec = avcodec_find_encoder_by_name("hevc_nvenc");
	if (NULL == codec) {
		return -1;
	}
	codec_ctx = avcodec_alloc_context3(codec);
	if (NULL == codec_ctx) {
		return -2;
	}

	AVRational rate;
	rate.num = 1;
	rate.den = 25;
	codec_ctx->time_base = rate;
	//codec_ctx->thread_count = 1;
	codec_ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;
	//codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
	//codec_ctx->pix_fmt = AV_PIX_FMT_NV12;
	codec_ctx->height = frame->height;
	codec_ctx->width = frame->width;
	
	int err = avcodec_open2(codec_ctx, codec, NULL);
	if (err < 0) {
		av_log(NULL, AV_LOG_INFO, "avcodec_open2 failed:%d", err);
		return -3;
	}

	AVPacket pkt;
	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;
	
	avcodec_send_frame(codec_ctx, frame);
	avcodec_receive_packet(codec_ctx, &pkt);
	printf("pkt.size: %d\n", pkt.size);
	
	FILE *fp = fopen(filename.c_str(), "wb");
	if (fp) {
		fwrite(pkt.data, 1, pkt.size, fp);
		fclose(fp);
	}
	
	av_packet_unref(&pkt);
	avcodec_close(codec_ctx);
	av_freep(&codec_ctx);

	return 0;
}
