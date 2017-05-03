/**  
 *  Copyright (c) 2017 sensetime.com, Inc. All Rights Reserved
 *  @author Chen Qian (chenqian@sensetime.com)
 *  @date 2017.04.19 10:13:05
 *  @brief 编码
 */
#include <string>

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
 * 本函数提供编码功能
 */
int ffmpeg_encode_jpeg(AVFrame *frame, std::string &filename);

#ifdef __cplusplus
}
#endif
