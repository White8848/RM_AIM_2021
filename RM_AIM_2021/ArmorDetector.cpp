#define _CRT_SECURE_NO_WARNINGS
#include"ArmorDetector.h"
#include<string.h>
#include<algorithm>
#include<iostream>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include"omp.h"
#define PI 3.14159265
#define RED 0
#define BLUE 1

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
void ArmorDetector::getResult(Mat src0)
{
	getSrcImage(src0);
	//if (!roiimg.empty())
	//    imshow("roi",roiimg);
	//if (!roinimg.empty())
	//    imshow("number",roinimg);
	getBinaryImage(RED);
	imshow("bin",binary);
	getContours();
	getTarget();
	imshow("out",outline);
	imshow("last",src);
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
#pragma omp parallel for
	for (int row = 0; row < src.rows; row++)
	{
#pragma omp parallel for
		for (int col = 0; col < src.cols; col++)
		{
			if (row <= roi.lefttop.y || row >= roi.lefttop.y + roi.rheight || col <= roi.lefttop.x || col >= roi.lefttop.x + roi.rwidth)
			{
				gry.at<uchar>(row, col) = 0;
				continue;
			}
		}
	}
	imshow("gry", gry);
	if (color == 0)binary = pointProcess(gry, color, 10,10);//RED
	else binary = pointProcess(gry, color, 80,10);//BLUE
	//imgProcess(binary);
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
#pragma omp parallel for
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
#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
#pragma omp parallel for
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

			if ((longi / shorti) >= 0.7 && (longi / shorti) <= 1.5 && (longj / shortj) >= 0.7 && (longj / shortj) <= 1.5) matchrank[i][j] -= 10000;
			if ((longi / shorti) >= 1.5 && (longi / shorti) <= 2.5 && (longj / shortj) >= 1.5 && (longj / shortj) <= 2.5) //两个轮廓的长宽比
				matchrank[i][j] += 100;
			//相对位置筛选
			if ((box[i].center.y - box[j].center.y) > 0.5 * longi || (box[i].center.y - box[j].center.y) > 0.5 * longj) matchrank[i][j] -= 10000;
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
#pragma omp parallel for
	for (int i = 0; i < num; i++)
#pragma omp parallel for
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

	//cout<<"i "<<besti<<" :x="<<boxi.center.x<<" y="<<boxi.center.y<<endl;
	//cout<<"j "<<bestj<<" :x="<<boxj.center.x<<" y="<<boxj.center.y<<endl;
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
#pragma omp parallel for
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
#pragma omp parallel for
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
	/*
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
	*/

	//roi get
	float x, y, xn, yn;
	if (target.rect[0].x - 80 < 0) x = 0; else x = target.rect[0].x - 80;
	if (target.rect[0].y - 80 < 0) y = 0; else y = target.rect[0].y - 80;
	//xn=target.rect[0].x;
	//if (target.rect[0].y-10<0) yn=0; else yn=target.rect[0].y-10;
	roi.lefttop = Point2f(x, y);
	int h, w, hn, wn;
	w = target.rect[2].x - target.rect[0].x + 160;
	h = target.rect[3].y - target.rect[0].y + 160;
	//wn=target.rect[2].x-target.rect[0].x;
	//hn=target.rect[3].y-target.rect[0].y+20;
	if (roi.lefttop.x + w > src.cols) roi.rwidth = src.cols - roi.lefttop.x; else roi.rwidth = w;
	if (roi.lefttop.y + h > src.rows) roi.rheight = src.rows - roi.lefttop.y; else roi.rheight = h;
	//if (xn+wn>src.cols) wn=src.cols-xn; else wn = wn;
	//if (yn+hn>src.rows) hn=src.rows-yn; else hn = hn;
	roiimg = src(Rect(roi.lefttop.x, roi.lefttop.y, roi.rwidth, roi.rheight));
	//roinimg=src(Rect(xn,yn,wn,hn));
}

/////////////////////////////////////PRIVATE//////////////////////////////////////////
//指针处理图像
Mat ArmorDetector::pointProcess(Mat srcImg, int enemyColor, int color_threshold, int gry_threshold) {

	Mat tempBinary;
	Mat gryBinary;

	tempBinary = Mat::zeros(srcImg.size(), CV_8UC1);

	uchar* pdata = (uchar*)srcImg.data;
	uchar* qdata = (uchar*)tempBinary.data;
	int srcData = srcImg.rows * srcImg.cols;

	if (enemyColor == 0) {//RED
#pragma omp parallel for
		for (int i = 0; i < srcData; i++)
		{
			if ((*(pdata + 2) - *pdata - *(pdata + 1)) > color_threshold) //减去绿色和蓝色
				*qdata = 255;
			pdata += 3;
			qdata++;
		}
	}
	else if (enemyColor == 1) { //BLUE
#pragma omp parallel for
		for (int i = 0; i < srcData; i++)
		{
			if (*pdata - *(pdata + 2) > color_threshold)
				*qdata = 255;
			pdata += 3;
			qdata++;
		}
	}
	imgProcess(tempBinary);
	threshold(tempBinary, gryBinary, gry_threshold, 255, THRESH_BINARY);

	return tempBinary&gryBinary;
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

int ArmorDetector::a(RotatedRect box,int high,int low) {
	if ((box.size.width > box.size.height && box.angle > low) || (box.size.width < box.size.height && box.angle < high)) return -100000;
}