#ifndef _ARMORDETECTOR_H_
#define _ARMORDETECTOR_H_
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<vector>
//#include"config.h"
#include<queue>

//#define DEBUG

using namespace cv;
using namespace std;

typedef struct ARMOR
{
	Point2f center;
	Point2f rect[4];
	float d_x, d_z, d_y;
}Armor;

typedef struct ROI
{
	Point2f lefttop;
	int rwidth;
	int rheight;
}Roi;

typedef struct ANGLE
{
	float yaw;
	float pitch;
}Angle;

class ArmorDetector
{
private:
	Mat binary;//binary image
	Mat outline;//outline image
	Roi roi;
	int num;
	vector< vector<Point> > contours;
	float matchrank[1500][1500];

public:
	Mat src;//source image
	bool islost;//1代表丢失
	Armor target;
	Mat roiimg;
	Mat roinimg;
	int color_thresh = 20;//通道相减二值化阈值
	int gray_thresh = 20;//灰度图二值化阈值

private:
	Mat pointProcess(Mat srcImg, int enemyColor, int color_threshold, int gry_threshold);
	Mat imgProcess(Mat tempBinary);
	/////////////////////////////装甲板//////////////////////////////////
	int a(RotatedRect box, int high, int low);
	int isArmorPattern(Mat& front);

public:
	ArmorDetector();
	ArmorDetector(Mat src0);
	void getResult(bool color);
	void getSrcImage(Mat src0);
	void getBinaryImage(int color);
	void getContours();
	void getTarget();
	void getPnp();
};
#endif
