/**  
 *  Copyright (c) 2017 LGPL, Inc. All Rights Reserved
 *  @author Chen Qian (chinahbcq@qq.com)
 *  @date 2017.04.22 14:32:02
 *  @brief gpu颜色空间转换
 */
#ifdef __cplusplus
extern "C"{
#endif

int cvtColor(unsigned char *d_req,
		unsigned char *d_res,
		int resolution,
		int height, 
		int width, 
		int linesize);

#ifdef __cplusplus
}
#endif
