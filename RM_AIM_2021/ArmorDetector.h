#ifndef _ARMORDETECTOR_H_
#define _ARMORDETECTOR_H_
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/types_c.h>
#include<vector>
//#include"config.h"
#include<queue>

using namespace cv;
using namespace std;

typedef struct ARMOR
{
	Point2f center;
	Point2f rect[4];
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
	Mat roiimg;
	Mat roinimg;
	int num;
	vector< vector<Point> > contours;
	float matchrank[1500][1500];

public:
	Mat src;//source image
	bool islost;//1代表丢失
	Armor target;

private:
	Mat pointProcess(Mat srcImg, int enemyColor, int color_threshold);
	Mat imgProcess(Mat tempBinary);
public:
	ArmorDetector();
	ArmorDetector(Mat src0);
	void getResult(Mat src0);
	void getSrcImage(Mat src0);
	void getBinaryImage();
	void getContours();
	void getTarget();
};
#endif
