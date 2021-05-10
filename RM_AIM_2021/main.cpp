#define _CRT_SECURE_NO_WARNINGS
#include "windows.h"
#include <string.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <pthread.h>
#include <time.h>
#include"ArmorDetector.h"
#include"CameraApi.h"
//#include <sys/time.h>
//#include"config.h"
//#include"serialport.h"

#define PTHREAD_CREATE_SUCCESS 0
#define RED 0
#define BLUE 1

#ifdef _WIN64
#pragma comment(lib, "MVCAMSDK_X64.lib")
#else
#pragma comment(lib, "MVCAMSDK.lib")
#endif

using namespace cv;
using namespace std;

ArmorDetector detector;//装甲识别类初始化
unsigned char* g_pRgbBuffer;

static void* Get_Armor(void* dstImage);
static void* Get_Video(void* arg);

typedef struct thread_data {
	Mat dstImage;
	pthread_mutex_t lock;
	bool run = FALSE;
}thread_data;

int main() {
	//////////////////////////////////串口通信初始化//////////////////////////////////
	//Serialport serp("/dev/ttyTHS2");
	//serp.set_opt(115200, 8, 'N', 1);

	///////////////////////////////////变量初始化/////////////////////////////////////
	int Ret = 0;
	pthread_t nThreadID[2];
	thread_data TD;
	pthread_mutex_init(&(TD.lock), NULL);

	/////////////////////////////////////多线程///////////////////////////////////////
	Ret = pthread_create(&nThreadID[0], NULL, Get_Video, (void*)&TD);
	if (Ret != PTHREAD_CREATE_SUCCESS) {
		printf("%s\n", "VIDEO_PTHREAD_CREATE_FILED");
		return -1;
	}
	while (TD.run == FALSE);//等待相机
	Ret = pthread_create(&nThreadID[1], NULL, Get_Armor, (void*)&TD);
	if (Ret != PTHREAD_CREATE_SUCCESS) {
		printf("%s\n", "ARMOR_PTHREAD_CREATE_FILED");
		return -1;
	}

	pthread_join(nThreadID[0], NULL);
	pthread_join(nThreadID[1], NULL);

	pthread_mutex_destroy(&(TD.lock));
	return 0;
}

static void* Get_Armor(void* threadarg) {
	thread_data* self_data = (thread_data*)threadarg;
	double e1, e2, time;

#ifdef DEBUG
	namedWindow("DEBUG");
	createTrackbar("color_thresh", "DEBUG", &detector.color_thresh, 255, 0);
	createTrackbar("gray_thresh", "DEBUG", &detector.gray_thresh, 255, 0);
#endif // DEBUG

	while (self_data->run) {
		e1 = getTickCount();

		pthread_mutex_lock(&(self_data->lock));//线程锁
		detector.getSrcImage(self_data->dstImage);
		pthread_mutex_unlock(&(self_data->lock));
		detector.getResult(RED);

		float xy[2];
		if (detector.islost == false) {
			cout << "find the armor successfully!" << endl;
			xy[0] = detector.target.center.x;
			xy[1] = detector.target.center.y;
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
	}

	pthread_exit(NULL);
	return 0;
}

static void* Get_Video(void* threadarg) {
	thread_data* self_data = (thread_data*)threadarg;

	//////////////////////////////工业相机参数初始化//////////////////////////////////
	int                     iCameraCounts = 1;//接入设备数目上限
	int                     iStatus = -1;
	tSdkCameraDevInfo       tCameraEnumList;
	int                     hCamera;
	tSdkCameraCapbility     tCapability;
	tSdkFrameHead           sFrameInfo = { 0 };
	BYTE* m_pbyBuffer = nullptr;
	int                     iDisplayFrames = 10000;
	IplImage* iplImage = NULL;
	int                     channel = 3;

	CameraSdkInit(1);
	iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
	printf("state = %d\n", iStatus);
	printf("count = %d\n", iCameraCounts);

	CameraInit(&tCameraEnumList, -1, -1, &hCamera);

	iStatus = CameraReadParameterFromFile(hCamera, "./camera.Config");//读取相机配置文件，请选择相机相应的配置文件
	if (iStatus == CAMERA_STATUS_SUCCESS) {
		printf("%s\n", "CAMERA_CONFIG_READ_SUCCESS");
	}

	CameraGetCapability(hCamera, &tCapability);
	g_pRgbBuffer = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);
	CameraPlay(hCamera);
	CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);

	//相机开始采集
	while (true) {
		//Liunx用第一个，Windows用第二个
		//if (CameraGetImageBuffer(hCamera, &sFrameInfo, &m_pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS)
		if (CameraGetImageBufferPriority(hCamera, &sFrameInfo, &m_pbyBuffer, 1000, CAMERA_GET_IMAGE_PRIORITY_NEWEST) == CAMERA_STATUS_SUCCESS) {
			iStatus = CameraImageProcess(hCamera, m_pbyBuffer, g_pRgbBuffer, &sFrameInfo);

			//获取图像
			if (iStatus == CAMERA_STATUS_SUCCESS) {
				Mat Image(cvSize(sFrameInfo.iWidth, sFrameInfo.iHeight), sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3, g_pRgbBuffer);
				flip(Image, Image, 0);//若图像颠倒，请注释本行

				pthread_mutex_lock(&(self_data->lock));//线程锁
				Image.copyTo(self_data->dstImage);
				pthread_mutex_unlock(&(self_data->lock));
				self_data->run = TRUE;

				//imshow("x", Image);
				//waitKey(1);
			}
			else {
				printf("%s\n", "CAMERA_READ_FILED");
				self_data->run = FALSE;
			}

			CameraReleaseImageBuffer(hCamera, m_pbyBuffer);
		}
	}
	CameraUnInit(hCamera);
	//释放相机
	free(g_pRgbBuffer);
	free(m_pbyBuffer);

	pthread_exit(NULL);
	return 0;
}