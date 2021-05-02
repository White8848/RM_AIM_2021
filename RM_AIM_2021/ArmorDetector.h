#ifndef _ARMORDETECTOR_H_
#define _ARMORDETECTOR_H_
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
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
	Mat outline;//outline image
	Roi roi;
	//Mat roiimg;
	Mat roinimg;
	int num;
	vector< vector<Point> > contours;
	float matchrank[1500][1500];

public:
	Mat binary;//binary image
	Mat src;//source image
	bool islost;//1代表丢失
	Armor target;
	Mat roiimg;

private:
	int ArmorDetector::isArmorPattern(Mat &front);
	Mat pointProcess(Mat srcImg, int enemyColor, int color_threshold, int gry_threshold);
	Mat imgProcess(Mat tempBinary);
	/////////////////////////////装甲板//////////////////////////////////	
	int a(RotatedRect box, int high, int low);

public:
	ArmorDetector();
	ArmorDetector(Mat src0);
	void getResult(Mat src0);
	void getSrcImage(Mat src0);
	void getBinaryImage(int color);
	void getContours();
	void getTarget();
	float measureDistance(float x1, float x2);
};
#endif
