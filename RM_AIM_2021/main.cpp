#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
//#include <sys/time.h>
#include "windows.h"
#include <string.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
//#include"config.h"
#include"ArmorDetector.h"
#include"CameraApi.h"
//#include"serialport.h"
#include <time.h>

using namespace cv;
using namespace std;

#ifdef _WIN64
#pragma comment(lib, "C:\\Program Files (x86)\\MindVision\\SDK\\X64\\MVCAMSDK_X64.lib")
#else
#pragma comment(lib, "MVCAMSDK.lib")
#endif

unsigned char           * g_pRgbBuffer;     //���������ݻ�����

ArmorDetector detector;//װ��ʶ���ࡣ

int main() {
    //Serialport serp("/dev/ttyTHS2");
    //serp.set_opt(115200, 8, 'N', 1);
    int                     iCameraCounts = 1;
    int                     iStatus=-1;
    tSdkCameraDevInfo       tCameraEnumList;
    int                     hCamera=1;
    tSdkCameraCapbility     tCapability;      //�豸������Ϣ
    tSdkFrameHead           sFrameInfo;
    BYTE*			        pbyBuffer;
    int                     iDisplayFrames = 10000;
    //IplImage *iplImage = NULL;
    int                     channel=3;
    float e1,e2,time;
    //Mat m = Mat(1,1,CV_8UC3);

    CameraSdkInit(1);

    //ö���豸���������豸�б�
    iStatus = CameraEnumerateDevice(&tCameraEnumList,&iCameraCounts);
    printf("state = %d\n", iStatus);

    printf("count = %d\n", iCameraCounts);
    //û�������豸
    if(iCameraCounts==0){
        return -1;
    }

    //������ʼ������ʼ���ɹ��㬲��ܵ����κ������������صĲ����ӿ�
    iStatus = CameraInit(&tCameraEnumList,1,-1,&hCamera);

    //��ʼ��ʧ��
    printf("state = %d\n", iStatus);
    if(iStatus!=CAMERA_STATUS_SUCCESS){
        return -1;
    }

    //�������������������ṹ�塣�ýṹ���а��������������õĸ��ֲ����ķ�Χ��Ϣ�����������غ����Ĳ���
    CameraGetCapability(hCamera,&tCapability);

    //string filename = "/Camera/Configs/MV-UB31-Group0.config";
    //CameraReadParameterFromFile(hCamera,"/Camera/Configs/MV-UB31-Group0.config");

    //
    g_pRgbBuffer = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);
    //g_readBuf = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

    /*��SDK���빤��ģʽ����ʼ���������������͵�ͼ��
    ���ݡ�������ǰ�����Ǵ���ģʽ������Ҫ���յ�
    ����֡�Ժ��Ż�����ͼ����    */
    CameraPlay(hCamera);

    if(tCapability.sIspCapacity.bMonoSensor){
        channel=1;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_MONO8);
    }else{
        channel=3;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_BGR8);
    }

    while (true) {
        e1=getTickCount();

        if(CameraGetImageBuffer(hCamera,&sFrameInfo,&pbyBuffer,1000) == CAMERA_STATUS_SUCCESS)
        {
            CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer,&sFrameInfo);

            Mat dstImage(
                        Size(sFrameInfo.iWidth,sFrameInfo.iHeight),
                        sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
                        g_pRgbBuffer
                        );

            detector.getResult(dstImage);
            //Sdetector.getResult(dstImage);

            float xy[2];
            if (detector.islost == false) {
                cout << "find the armor successfully!" << endl;
                xy[0] = detector.target.center.x;
                xy[1] = detector.target.center.y;
                cout << "x:" << xy[0] -640<< "  y:" << 512-xy[1] << endl;
            }
            else{
                cout << "lost the armor!" << endl;
                xy[0]=640;
                xy[1]=512;


            }
            //serp.sendXY(xy);

            //imshow("0",m);
            //if(waitKey(1)==27)exit(0);


            CameraReleaseImageBuffer(hCamera,pbyBuffer);

        }
        e2=getTickCount();
        time=(e2-e1)/getTickFrequency();
        cout<<"fps:"<<int(1/time)<<endl;
    }
    CameraUnInit(hCamera);
    //ע�⣬�ַ���ʼ������free
    free(g_pRgbBuffer);

}

