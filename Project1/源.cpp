#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include<opencv2/imgproc/types_c.h> 
	
using namespace cv;
using namespace std;


//bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)//最小二乘法函数
//{
//	//Number of key points
//	int N = key_point.size();
//
//	//构造矩阵X
//	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
//	for (int i = 0; i < n + 1; i++)
//	{
//		for (int j = 0; j < n + 1; j++)
//		{
//			for (int k = 0; k < N; k++)
//			{
//				X.at<double>(i, j) = X.at<double>(i, j) +
//					std::pow(key_point[k].x, i + j);
//			}
//		}
//	}
//
//	//构造矩阵Y
//	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
//	for (int i = 0; i < n + 1; i++)
//	{
//		for (int k = 0; k < N; k++)
//		{
//			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
//				std::pow(key_point[k].x, i) * key_point[k].y;
//		}
//	}
//
//	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
//	//求解矩阵A
//	cv::solve(X, Y, A, cv::DECOMP_LU);
//	return true;
//}
//
//
//void find_fire(Mat& srcMat)
//{
//	Mat gray_srcMat;
//	Mat dstMat, binMat;
//	cvtColor(srcMat, gray_srcMat, COLOR_BGR2GRAY);
//	Mat fire_Mat = Mat::zeros(gray_srcMat.size(), gray_srcMat.type());
//	cvtColor(srcMat, dstMat, COLOR_BGR2HSV);
//	vector<Mat> channels;
//	split(dstMat, channels);
//	//将S通道的图像复制，然后处理
//	Mat S_Mat;
//	channels.at(1).copyTo(S_Mat);
//	int row_num = srcMat.rows;			//行数
//	int col_num = srcMat.cols;			//列数
//	//双重循环，遍历右下角像素值
//	for (int i = row_num * 0.75; i < row_num; i++)	//行循环
//	{
//		for (int j = col_num * 0.75; j < col_num; j++)	//列循环
//		{
//			//-------【开始处理每个像素】---------------
//			if ((gray_srcMat.at<uchar>(i, j) >= 150 && S_Mat.at<uchar>(i, j) >= 120))
//			{
//				fire_Mat.at<uchar>(i, j) = 255;
//			} 
//		}
//	}
//	
//	Mat element = getStructuringElement(MORPH_RECT, Size(30, 30));	//膨胀内核（越大离散的点就越少）
//	Mat firedstImage;
//	//通过findContours函数寻找连通域
//	morphologyEx(fire_Mat, fire_Mat, MORPH_DILATE, element);
//	vector<vector<Point>> contours;
//	findContours(fire_Mat, contours, RETR_LIST, CHAIN_APPROX_NONE);
//	//绘制轮廓
//	for (int i = 0; i < contours.size(); i++) {
//		RotatedRect rbox = minAreaRect(contours[i]);
//		drawContours(srcMat, contours, i, Scalar(0, 0, 255), 1, 8);
//		Point2f vtx[4];
//		rbox.points(vtx);
//		for (int j = 0; j < 4; ++j) {
//			cv::line(srcMat, vtx[j], vtx[j < 3 ? j + 1 : 0], Scalar(255, 0, 0), 2, LINE_AA);
//		}
//	}
//}
//
//
//
//int main()
//{
//	//实例化的同时初始化
//	VideoCapture capture("E:\\File_\\大二下\\数字图像处理\\IMG_0589.TRIM.mp4");		
//	//计数器
//	int cnt = 0;//总帧数266
//	Mat frame;
//	Mat bgMat_0;//作为背景差分的比较帧
//	std::vector<cv::Point> effectPoint;
//	Mat srcMat;
//	int starPoint = 0;
//	uint8_t  flag_startPoint = 0;
//	while (1)
//	{
//		capture >> frame;
//		flag_startPoint = 0;
//		srcMat = frame;
//		cvtColor(frame, frame, COLOR_BGR2GRAY);
//		if (cnt == 0) {
//			//第一帧，获得背景图像
//			bgMat_0 = frame;
//		}
//		cnt++;
//		if (cnt == 255)
//			break;
//		Mat result = bgMat_0.clone();
//		//背景差分
//		absdiff(bgMat_0, frame, result);
//
//		imshow("result", result);
//
//		/*****************去噪声******************/
//		for (int i = 0; i < 90; i++)	//通过观察、发现0-180列为干扰，320-480列且0-90行为干扰
//		{
//			for (int j = 320; j < result.cols; j++)
//			{
//				result.at<uchar>(i, j) = 0;
//			}
//		}
//		for (int i = 0; i < result.rows; i++)
//		{
//			for (int j = 0; j < 180; j++)
//			{
//				result.at<uchar>(i, j) = 0;
//			}
//		}
//		
//		/*****************二值化******************/
//		threshold(result, result, 90, 255, THRESH_BINARY);
//		imshow("tu", result);
//		for (int i = 0; i < result.cols; i++)
//		{
//			for (int j = 0; j < result.rows; j++)
//			{
//				if (result.at<uchar>(j, i) == 255 && flag_startPoint == 0)
//				{
//					starPoint = i;                           //寻找起点
//					flag_startPoint = 1;
//				}
//				if (result.at<uchar>(j, i) == 255)
//				{
//					effectPoint.push_back(cv::Point(i, j));
//					break;
//				}
//			}
//		}
//		cv::Mat A;
//		/*****************最小二乘法进行拟合******************/
//		polynomial_curve_fit(effectPoint, 2, A);//利用二次曲线进行拟合
//		std::cout << "A = " << starPoint << std::endl;
//
//		std::vector<cv::Point> points_fitted;
//
//		for (int x = starPoint; x < 430; x++)
//		{
//			double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +            
//				A.at<double>(2, 0) * std::pow(x, 2) ;
//
//			points_fitted.push_back(cv::Point(x, y));
//		}
//		cv::polylines(srcMat, points_fitted, false, cv::Scalar(0, 0, 255), 1, 8, 0);
//		find_fire(srcMat);
//		imshow("result", srcMat);
//		
//		waitKey(30);
//		
//	}
//	
//	return 0;
//}
int main()
{

}

