#define _CRT_SECURE_NO_WARNINGS
#include "windows.h"
#include <string.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <iostream>
#include"ArmorDetector.h"
#include <time.h>
#include"CameraApi.h"
//#include <sys/time.h>
//#include"config.h"
//#include"serialport.h"

#ifdef _WIN64
#pragma comment(lib, "MVCAMSDK_X64.lib")
#else
#pragma comment(lib, "MVCAMSDK.lib")
#endif

using namespace cv;
using namespace std;

ArmorDetector detector;//装甲识别类初始化
ArmorDetector detector2;//装甲识别类初始化
unsigned char* g_pRgbBuffer;
unsigned char* g_pRgbBuffer2;

int main() {
	//////////////////////////////////串口通信初始化//////////////////////////////////
	//Serialport serp("/dev/ttyTHS2");
	//serp.set_opt(115200, 8, 'N', 1);

	///////////////////////////////////变量初始化/////////////////////////////////////
	double e1, e2, time;

	//////////////////////////////工业相机参数初始化//////////////////////////////////
	int                     iCameraCounts = 10;
	int                     iStatus = -1;
	int						iStatus2 = -1;
	tSdkCameraDevInfo       tCameraEnumList;
	tSdkCameraDevInfo       tCameraEnumList2;
	int                     lCamera;//左边摄像头
	int						rCamera;//右边摄像头
	tSdkCameraCapbility     tCapability;
	tSdkCameraCapbility     tCapability2;
	tSdkFrameHead           sFrameInfo = { 0 };
	tSdkFrameHead           sFrameInfo2 = { 0 };
	BYTE* m_pbyBuffer;
	BYTE* m_pbyBuffer2;

	int                     iDisplayFrames = 10000;
	int                     iDisplayFrames2 = 10000;
	IplImage *iplImage = NULL;
	IplImage *iplImage2 = NULL;
	int                     channel = 3;

	CameraSdkInit(1);
	iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
	//iStatus2 = CameraEnumerateDevice(&tCameraEnumList2, &iCameraCounts);
	printf("state = %d\n", iStatus);
	printf("state2 = %d\n", iStatus2);
	printf("count = %d\n", iCameraCounts);
	if (iCameraCounts == 0) {
		return -1;
	}
	iStatus = CameraInit(&tCameraEnumList, -1, -1, &lCamera);
	iStatus2 = CameraInit(&tCameraEnumList+1, -1, -1, &rCamera);
	printf("state = %d\n", iStatus);
	printf("state2 = %d\n", iStatus2);
	if (iStatus != CAMERA_STATUS_SUCCESS) {
		return -1;
	}

	if (iStatus2 != CAMERA_STATUS_SUCCESS) {
		return -1;
	}

	//iStatus = CameraReadParameterFromFile(hCamera, "./camera.Config");
	//printf("state = %d\n", iStatus);
	if (iStatus != CAMERA_STATUS_SUCCESS) {
		return -1;
	}
	CameraGetCapability(lCamera, &tCapability);
	CameraGetCapability(rCamera, &tCapability2);

	g_pRgbBuffer = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);
	g_pRgbBuffer2 = (unsigned char*)malloc(tCapability2.sResolutionRange.iHeightMax * tCapability2.sResolutionRange.iWidthMax * 3);
	CameraPlay(lCamera);
	CameraPlay(rCamera);
	CameraSetIspOutFormat(lCamera, CAMERA_MEDIA_TYPE_BGR8);
	CameraSetIspOutFormat(rCamera, CAMERA_MEDIA_TYPE_BGR8);

	//////////////////////////////////主循环///////////////////////////////////////
	
	VideoCapture c;
	c.open("./red5.mp4");

#ifdef DEBUG
	namedWindow("DEBUG");
	createTrackbar("color_thresh", "DEBUG", &detector.color_thresh, 255, 0);
	createTrackbar("gray_thresh", "DEBUG", &detector.gray_thresh, 255, 0);
#endif // DEBUG


	
	

	while (true) {
		//相机开始采集
		//if (CameraGetImageBuffer(hCamera, &sFrameInfo, &m_pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS)
		if (CameraGetImageBufferPriority(lCamera, &sFrameInfo, &m_pbyBuffer, 1000, CAMERA_GET_IMAGE_PRIORITY_NEWEST) == CAMERA_STATUS_SUCCESS && CameraGetImageBufferPriority(rCamera, &sFrameInfo2, &m_pbyBuffer2, 1000, CAMERA_GET_IMAGE_PRIORITY_NEWEST) == CAMERA_STATUS_SUCCESS) {
			iStatus = CameraImageProcess(lCamera, m_pbyBuffer, g_pRgbBuffer, &sFrameInfo);
			iStatus2 = CameraImageProcess(rCamera, m_pbyBuffer2, g_pRgbBuffer2, &sFrameInfo2);

			e1 = getTickCount();

			if (iStatus == CAMERA_STATUS_SUCCESS && iStatus2 == CAMERA_STATUS_SUCCESS) {
				Mat dstImage(cvSize(sFrameInfo.iWidth, sFrameInfo.iHeight), sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3, g_pRgbBuffer);
				//Mat dstImage2, dstImage;
				//c >> dstImage;
				//c >> dstImage2;
				Mat dstImage2(cvSize(sFrameInfo2.iWidth, sFrameInfo2.iHeight), sFrameInfo2.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3, g_pRgbBuffer2);
				flip(dstImage, dstImage, 0);//若图像颠倒，请注释本行
				flip(dstImage2, dstImage2, 0);//若图像颠倒，请注释本行
				//imshow("left", dstImage);
				//imshow("right", dstImage2);
				detector.getResult(dstImage);
				detector2.getResult(dstImage2);

				imshow("bin_left", detector.binary);
				imshow("bin_right", detector2.binary);

				imshow("last_left", detector.src);
				imshow("last_right", detector2.src);
				
			/*	if (!detector.roiimg.empty())
					imshow("roi", detector.roiimg);
				if (!detector2.roiimg.empty())
					imshow("roi2", detector2.roiimg);*/
				//waitKey(0);
			}
			else {
				return -1;
			}

			float xy[2];
			float xy2[2];
			if (detector.islost == false) {
				cout << "find the armor successfully!" << endl;
				xy[0] = detector.target.center.x;
				xy[1] = detector.target.center.y;

				xy2[0] = detector2.target.center.x;
				xy2[1] = detector2.target.center.y;

				cout << "Distance:" << detector.measureDistance(xy[0], xy2[0]) << " m" << endl;

				//cout << "x:" << xy[0] - 640 << "  y:" << 512 - xy[1] << endl;
			}
			else {
				//cout << "lost the armor!" << endl;
				xy[0] = 640;
				xy[1] = 512;
			}
			//serp.sendXY(xy);

			if (waitKey(1) == 27)exit(0);

			e2 = getTickCount();
			time = (e2 - e1) / getTickFrequency();
			cout << "fps:" << int(1 / time) << endl;

			CameraReleaseImageBuffer(lCamera, m_pbyBuffer);
			CameraReleaseImageBuffer(rCamera, m_pbyBuffer2);
		}
	}
	CameraUnInit(lCamera);
	CameraUnInit(rCamera);
	//释放相机
	free(g_pRgbBuffer);
	free(m_pbyBuffer);
	free(g_pRgbBuffer2);
	free(m_pbyBuffer2);

	waitKey(0);
}