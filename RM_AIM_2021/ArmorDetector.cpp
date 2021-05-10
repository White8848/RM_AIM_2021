#define _CRT_SECURE_NO_WARNINGS
#include"ArmorDetector.h"
#include<string.h>
#include<algorithm>
#include<iostream>
#include<stdio.h>
#include<math.h>
#include<time.h>
#define PI 3.14159265
#define BIG 0
#define SMALL 1

using namespace cv;

ArmorDetector::ArmorDetector()
{
	islost = true;
	roi.lefttop = Point2f(0, 0);
	roi.rwidth = src.cols;
	roi.rheight = src.rows;
	roiimg = src(Rect(roi.lefttop.x, roi.lefttop.y, roi.rwidth, roi.rheight));
	roinimg = roiimg;
}

ArmorDetector::ArmorDetector(Mat src0)
{
	src0.copyTo(src);
	islost = true;
	roi.lefttop = Point2f(0, 0);
	roi.rwidth = src.cols;
	roi.rheight = src.rows;
	roiimg = src(Rect(roi.lefttop.x, roi.lefttop.y, roi.rwidth, roi.rheight));
}

/////////////////////////////////////PUBLIC//////////////////////////////////////////
void ArmorDetector::getResult(bool color)
{
	//getSrcImage(src0);
	//if (!roiimg.empty())
		//imshow("roi",roiimg);
	int result = -1;
	if (!roinimg.empty()) {
		//imshow("number", roinimg);
		//result = isArmorPattern(roinimg);
	}
	cout << "预测结果：" << result << endl;
	getBinaryImage(color);
	//imshow("bin", binary);
	getContours();
	getTarget();
	getPnp();
	//imshow("out",outline);
	//imshow("last", src);
}

//原图
void ArmorDetector::getSrcImage(Mat src0)
{
	src0.copyTo(src);
}

//二值化
void ArmorDetector::getBinaryImage(int color)
{
	Mat gry;
	src.copyTo(gry);
	//roi
	/*
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols*3; col++)
		{
			if (row <= roi.lefttop.y || row >= roi.lefttop.y + roi.rheight || col <= roi.lefttop.x*3 || col >= roi.lefttop.x*3 + roi.rwidth*3)
			{
				gry.at<uchar>(row, col) = 0;
			}
		}
	}
	*/
	//imshow("gry", gry);
	if (color == 0) {
#ifndef DEBUG
		color_thresh = 20;
		gray_thresh = 20;
#endif // !DEBUG
		binary = pointProcess(gry, color, color_thresh, gray_thresh);//RED
	}
	else {
#ifndef DEBUG
		color_thresh = 50;
		gray_thresh = 3;
#endif // !DEBUG
		binary = pointProcess(gry, color, color_thresh, gray_thresh);//BLUE
	}
}

void ArmorDetector::getContours()
{
	vector<Vec4i> hierarcy;
	Point2f rect[4];
	src.copyTo(outline);
	findContours(binary, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //CV_CHAIN_APPROX_NONE

	vector<vector<Point> >::iterator itc = contours.begin();
	while (itc != contours.end()) {
		if (itc->size() < 35) {
			itc = contours.erase(itc);
		}
		else {
			itc++;
		}
	}
	vector<Rect> boundRect(contours.size());
	vector<RotatedRect> box(contours.size());//最小外接矩形集合
	num = contours.size();//轮廓的数量
	for (int i = 0; i < num; i++)
	{
		box[i] = minAreaRect(Mat(contours[i]));//计算每个轮廓的最小外接矩形
		boundRect[i] = boundingRect(Mat(contours[i]));
		circle(outline, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);
		box[i].points(rect);
		rectangle(outline, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);
		for (int j = 0; j < 4; j++)
		{
			line(outline, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边（非必要）
		}
	}/*
	for (int i=0;i<num;i++)
	{
		cout<<"num "<<i<<": area="<<box[i].size.area()<<" angle="<<box[i].angle<<" a="<<box[i].size.height<<" b="
			<<box[i].size.width<<endl;
	}*/
	memset(matchrank, 0, sizeof(matchrank));
	for (int i = 0; i < num; i++)
	{
		for (int j = i + 1; j < num; j++)
		{
			//去掉太斜的矩形
			if ((box[i].size.width > box[i].size.height && box[i].angle > -77) || (box[i].size.width < box[i].size.height && box[i].angle < -13)) matchrank[i][j] -= 100000;
			if ((box[j].size.width > box[j].size.height && box[j].angle > -77) || (box[j].size.width < box[j].size.height && box[j].angle < -13)) matchrank[i][j] -= 100000;
			//根据长宽筛选
			double longi, shorti, longj, shortj;
			longi = box[i].size.height;
			shorti = box[i].size.width;
			if (longi < shorti)
			{
				int temp = longi;
				longi = shorti;
				shorti = temp;
			}
			longj = box[j].size.height;
			shortj = box[j].size.width;
			if (longj < shortj)
			{
				int temp = longj;
				longj = shortj;
				shortj = temp;
			}

			if ((longi / shorti) >= 0.7 && (longi / shorti) <= 1.8 && (longj / shortj) >= 0.7 && (longj / shortj) <= 1.8) matchrank[i][j] -= 10000;
			if ((longi / shorti) >= 1.8 && (longi / shorti) <= 2.8 && (longj / shortj) >= 1.8 && (longj / shortj) <= 2.8) //两个轮廓的长宽比
				matchrank[i][j] += 100;
			//相对位置筛选
			if ((box[i].center.y - box[j].center.y) > 0.5 * longi || (box[i].center.y - box[j].center.y) > 0.5 * longj) matchrank[i][j] -= 10000;
			if (abs(box[i].center.x - box[j].center.x) < 0.8 * longi || abs(box[i].center.x - box[j].center.x) < 0.8 * longj) matchrank[i][j] -= 10000;
			//根据角度筛选
			double anglei, anglej;
			anglei = box[i].angle;
			anglej = box[j].angle;
			if (abs(anglei - anglej) <= 10 || abs(anglei - anglej) >= 80) matchrank[i][j] += 100;
			else matchrank[i][j] -= 10000;
			//面积比
			double areai = box[i].size.area();
			double areaj = box[j].size.area();
			if (areai < 7 || areaj < 7) matchrank[i][j] -= 20000;
			if (areai / areaj >= 5 || areaj / areai >= 5) matchrank[i][j] -= 10000;
			if (areai / areaj >= 2 || areaj / areai >= 2) matchrank[i][j] -= 100;
			if (areai / areaj > 0.8 && areai - areaj < 1.2) matchrank[i][j] += 100;
			//连线长
			double d = sqrt((box[i].center.x - box[j].center.x) * (box[i].center.x - box[j].center.x) + (box[i].center.y - box[j].center.y) * (box[i].center.y - box[j].center.y));
			if (d >= longi * 4.5 || d >= longj * 4.5 || d < 2 * longi || d < 2 * longj) matchrank[i][j] -= 10000;
			//cout<<"i j d:"<<i<<" "<<j<<" "<<d<<endl;
		}
	}
}

void ArmorDetector::getTarget()
{
	int maxpoint = -10000;
	int besti = -1;
	int bestj = -1;
	for (int i = 0; i < num; i++)
		for (int j = i + 1; j < num; j++)
		{
			//cout<<"mathrank "<<i<<" "<<j<<" :"<<matchrank[i][j]<<endl;
			if (maxpoint < matchrank[i][j])
			{
				maxpoint = matchrank[i][j];
				besti = i;
				bestj = j;
			}
		}

	if (besti == -1 || bestj == -1)
	{
		islost = true;
		roi.lefttop = Point2f(0, 0);
		roi.rwidth = src.cols;
		roi.rheight = src.rows;
		roiimg = src(Rect(roi.lefttop.x, roi.lefttop.y, roi.rwidth, roi.rheight));
		return;
	}
	islost = false;
	RotatedRect boxi;
	RotatedRect boxj;
	//获取最优匹配
	boxi = minAreaRect(Mat(contours[besti]));
	boxj = minAreaRect(Mat(contours[bestj]));
	//获取中心点
	target.center = Point2f((boxi.center.x + boxj.center.x) / 2, (boxi.center.y + boxj.center.y) / 2);

	//cout << "i " << besti << " :x=" << boxi.center.x << " y=" << boxi.center.y << endl;
	//cout << "j " << bestj << " :x=" << boxj.center.x << " y=" << boxj.center.y << endl;
	//cout<<"target : x="<<target.center.x<<" y="<<target.center.y<<endl;
	circle(src, Point(target.center.x, target.center.y), 5, Scalar(255, 0, 0), -1, 8);
	//circle(outline,Point(target.center.x,target.center.y),5,Scalar(255,0,0),-1,8);
	char tam[100];
	sprintf(tam, "(%0.0f,%0.0f)", target.center.x, target.center.y);
	putText(src, tam, Point(target.center.x, target.center.y), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 255), 1);

	//获取周围四个点的坐标
	if (boxi.center.x > boxj.center.x)
	{
		RotatedRect temp;
		temp = boxi;
		boxi = boxj;
		boxj = temp;
	}
	Point2f rect1[4];
	Point2f rect2[4];
	Point2f rect3[4] = { Point2f(0,0) };
	boxi.points(rect1);
	boxj.points(rect2);
	for (int i = 0; i < 4; i++)
	{
		if (rect1[i].y < boxi.center.y)
		{
			rect3[0].x += rect1[i].x;   //左上
			rect3[0].y += rect1[i].y;
		}
		if (rect1[i].y > boxi.center.y)
		{
			rect3[1].x += rect1[i].x;   //左下
			rect3[1].y += rect1[i].y;
		}
	}
	for (int i = 0; i < 4; i++)
	{
		if (rect2[i].y < boxj.center.y)
		{
			rect3[2].x += rect2[i].x;   //右上
			rect3[2].y += rect2[i].y;
		}
		if (rect2[i].y > boxj.center.y)
		{
			rect3[3].x += rect2[i].x;   //右下
			rect3[3].y += rect2[i].y;
		}
	}
	//绘制周围四个点的坐标

	Scalar color4[4] = { Scalar(255,0,255),Scalar(255,0,0),Scalar(0,255,0),Scalar(0,255,255) };
	//左上紫色 左下蓝色 右上绿色 右下黄色
	for (int i = 0; i < 4; i++)
	{
		target.rect[i] = Point2f(rect3[i].x / 2, rect3[i].y / 2);
		circle(src, Point(target.rect[i].x, target.rect[i].y), 5, color4[i], -1, 8);
		sprintf(tam, "(%0.0f,%0.0f)", target.rect[i].x, target.rect[i].y);
		putText(src, tam, Point(target.rect[i].x, target.rect[i].y), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 255), 1);
	}
	char tam4[100];
	sprintf(tam4, "x=%0.2f   y=%0.2f", target.center.x, target.center.y);
	putText(src, tam4, Point(15, 60), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1);

	float s = (target.rect[2].x - target.rect[0].x) / (target.rect[3].y - target.rect[0].y);
	if (s <= 4.5 && s >= 2.8) { 
		target.size = BIG; 
		cout << "BIG_ARMOR" << endl;
	}
	else if (s < 2.8 && s >= 1.3) {
		target.size = SMALL;
		cout << "SMALL_ARMOR" << endl;
	}

	//装甲识别roi
	/*
	float x, y;
	if (target.rect[0].x - 100 < 0) x = 0; else x = target.rect[0].x - 100;
	if (target.rect[0].y - 100 < 0) y = 0; else y = target.rect[0].y - 100;
	roi.lefttop = Point2f(x, y);
	int h, w;
	w = target.rect[2].x - target.rect[0].x + 200;
	h = target.rect[3].y - target.rect[0].y + 200;
	if (roi.lefttop.x + w > src.cols) roi.rwidth = src.cols - roi.lefttop.x; else roi.rwidth = w;
	if (roi.lefttop.y + h > src.rows) roi.rheight = src.rows - roi.lefttop.y; else roi.rheight = h;
	roiimg = src(Rect(roi.lefttop.x, roi.lefttop.y, roi.rwidth, roi.rheight));
	*/
	//数字识别roi
	float xn, yn;
	xn = target.rect[0].x;
	if (target.rect[0].y - (target.rect[3].y - target.rect[0].y) / 2 < 0) yn = 0; else yn = target.rect[0].y - (target.rect[3].y - target.rect[0].y) / 2;
	int hn, wn;
	wn = target.rect[2].x - target.rect[0].x;
	hn = 2 * abs(target.rect[3].y - target.rect[0].y);
	if (hn < 0)hn = 0;
	if (xn + wn > src.cols) wn = src.cols - xn; else wn = wn;
	if (yn + hn > src.rows) hn = src.rows - yn; else hn = hn;
	roinimg = src(Rect(xn, yn, wn, hn));
}

void ArmorDetector::getPnp()
{
	double camera1[9] = { 1000 ,0  ,320,  0,  1000  ,240,  0 , 0,  1 };
	double theerror1[5] = { 0 ,0 , 0.0 , 0.00 , 0.0 };
	double camera2[9] = {2105.39 ,0 , 640 , 0 , 2105.39 , 512 , 0 , 0 , 1 };//相机内参
	double theerror2[5] = { -0.110008 ,2.220571 , 0.0 , 0.0 , -14.069738 };
	//Rect ans = trackBox;
	double width_target, height_target;
	if (target.size == BIG) width_target = 22.5;
	else width_target = 13;
	height_target = 6;

	vector <Point2d> point2d;
	point2d.push_back(target.rect[0]);
	point2d.push_back(target.rect[2]);
	point2d.push_back(target.rect[3]);
	point2d.push_back(target.rect[1]);
	point2d.push_back(target.center);

	std::vector<cv::Point3d> point3d;
	double half_x = width_target / 2.0;
	double half_y = height_target / 2.0;
	point3d.push_back(Point3d(-half_x, -half_y, 0));
	point3d.push_back(Point3d(half_x, -half_y, 0));
	point3d.push_back(Point3d(half_x, half_y, 0));
	point3d.push_back(Point3d(-half_x, half_y, 0));
	point3d.push_back(Point3d(0, 0, 0));
	Mat cam(Size(3, 3), CV_64F, camera2);
	Mat dist(Size(5, 1), CV_64F, theerror2);
	Mat rvec(3, 3, CV_64F);
	Mat trec(3, 1, CV_64F);

	solvePnP(point3d, point2d, cam, dist, rvec, trec, false, CV_EPNP);
	double ans_x, ans_y, ans_z;
	//cout<<trec<<endl;
	ans_x = trec.at<double>(0, 0);
	ans_y = trec.at<double>(1, 0);
	ans_z = trec.at<double>(2, 0);
	double dis_x = 0, dis_y = 0;

	target.d_y = (acos(35.0 / sqrt((ans_x + 35.0) * (ans_x + 35.0) + (ans_z + 127) * (ans_z + 127))) - acos((ans_x + 35.0) / sqrt((ans_x + 35.0) * (ans_x + 35.0) + (ans_z + 127) * (ans_z + 127)))) * 180 / 3.1415926;
	target.d_z = ans_z;
	//cout<<ans_x<<" "<<ans_y<<endl<<endl;;
	target.d_x = atan((ans_y) / (ans_z)) * 180 / 3.1415926;
	//d_y=atan(ans_x/ans_z)*180/3.1415926;
	//d_z=ans_z;
	//cout << d_x << " " << d_y << " " << d_z << endl << endl;;
	//cout<<"  ans "<<x<<"   "<<last_x<<endl;
	cout<<"Distance = "<<target.d_z - 10<<endl;
}

/////////////////////////////////////PRIVATE//////////////////////////////////////////
//判断数字
int ArmorDetector::isArmorPattern(Mat& front)
{
	Mat gray;
	cvtColor(front, gray, CV_BGR2GRAY);
	resize(gray, gray, Size(20, 20));
	GaussianBlur(gray, gray, Size(3, 3), 0, 0);
	adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -1);
	//threshold(gray, gray, 80, 255, CV_THRESH_BINARY);
	//imshow("xxx", gray);
	// copy the data to make the matrix continuous
	Mat temp;
	gray.copyTo(temp);
	Mat data = temp.reshape(1, 1);

	data.convertTo(data, CV_32FC1);

	Ptr<ml::SVM> svm = ml::SVM::load("cxy_svm_5_1.xml");

	int result = (int)svm->predict(data);
	//cout << "预测结果:" << result << endl;

	return result;
}

//指针处理图像
Mat ArmorDetector::pointProcess(Mat srcImg, int enemyColor, int color_threshold, int gry_threshold) {
	Mat tempBinary;
	Mat gryBinary;

	tempBinary = Mat::zeros(srcImg.size(), CV_8UC1);
	cvtColor(srcImg, gryBinary, CV_BGR2GRAY);

	uchar* pdata = (uchar*)srcImg.data;
	uchar* qdata = (uchar*)tempBinary.data;
	int srcData = srcImg.rows * srcImg.cols;

	if (enemyColor == 0) {//RED
		for (int i = 0; i < srcData; i++)
		{
			if ((*(pdata + 2) - *pdata - *(pdata + 1)) > color_threshold) //减去绿色和蓝色
				*qdata = 255;
			pdata += 3;
			qdata++;
		}
	}
	else if (enemyColor == 1) { //BLUE
		for (int i = 0; i < srcData; i++)
		{
			if (*pdata - *(pdata + 2) > color_threshold)
				*qdata = 255;
			pdata += 3;
			qdata++;
		}
	}
	imgProcess(tempBinary);
	GaussianBlur(gryBinary, gryBinary, Size(3, 3), 0, 0);
	adaptiveThreshold(gryBinary, gryBinary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, gry_threshold);//自适应阈值化
	//threshold(gryBinary, gryBinary, gry_threshold, 255, THRESH_BINARY);

	return tempBinary & gryBinary;
}

//膨胀腐蚀操作
Mat ArmorDetector::imgProcess(Mat tempBinary) {
	Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3));
	erode(tempBinary, tempBinary, kernel);
	dilate(tempBinary, tempBinary, kernel);
	dilate(tempBinary, tempBinary, kernel);
	erode(tempBinary, tempBinary, kernel);

	return tempBinary;
}

int ArmorDetector::a(RotatedRect box, int high, int low) {
	if ((box.size.width > box.size.height && box.angle > low) || (box.size.width < box.size.height && box.angle < high)) return -100000;
}