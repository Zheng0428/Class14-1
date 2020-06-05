#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include<opencv2/imgproc/types_c.h> 
	
using namespace cv;
using namespace std;


//bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)//��С���˷�����
//{
//	//Number of key points
//	int N = key_point.size();
//
//	//�������X
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
//	//�������Y
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
//	//������A
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
//	//��Sͨ����ͼ���ƣ�Ȼ����
//	Mat S_Mat;
//	channels.at(1).copyTo(S_Mat);
//	int row_num = srcMat.rows;			//����
//	int col_num = srcMat.cols;			//����
//	//˫��ѭ�����������½�����ֵ
//	for (int i = row_num * 0.75; i < row_num; i++)	//��ѭ��
//	{
//		for (int j = col_num * 0.75; j < col_num; j++)	//��ѭ��
//		{
//			//-------����ʼ����ÿ�����ء�---------------
//			if ((gray_srcMat.at<uchar>(i, j) >= 150 && S_Mat.at<uchar>(i, j) >= 120))
//			{
//				fire_Mat.at<uchar>(i, j) = 255;
//			} 
//		}
//	}
//	
//	Mat element = getStructuringElement(MORPH_RECT, Size(30, 30));	//�����ںˣ�Խ����ɢ�ĵ��Խ�٣�
//	Mat firedstImage;
//	//ͨ��findContours����Ѱ����ͨ��
//	morphologyEx(fire_Mat, fire_Mat, MORPH_DILATE, element);
//	vector<vector<Point>> contours;
//	findContours(fire_Mat, contours, RETR_LIST, CHAIN_APPROX_NONE);
//	//��������
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
//	//ʵ������ͬʱ��ʼ��
//	VideoCapture capture("E:\\File_\\�����\\����ͼ����\\IMG_0589.TRIM.mp4");		
//	//������
//	int cnt = 0;//��֡��266
//	Mat frame;
//	Mat bgMat_0;//��Ϊ������ֵıȽ�֡
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
//			//��һ֡����ñ���ͼ��
//			bgMat_0 = frame;
//		}
//		cnt++;
//		if (cnt == 255)
//			break;
//		Mat result = bgMat_0.clone();
//		//�������
//		absdiff(bgMat_0, frame, result);
//
//		imshow("result", result);
//
//		/*****************ȥ����******************/
//		for (int i = 0; i < 90; i++)	//ͨ���۲졢����0-180��Ϊ���ţ�320-480����0-90��Ϊ����
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
//		/*****************��ֵ��******************/
//		threshold(result, result, 90, 255, THRESH_BINARY);
//		imshow("tu", result);
//		for (int i = 0; i < result.cols; i++)
//		{
//			for (int j = 0; j < result.rows; j++)
//			{
//				if (result.at<uchar>(j, i) == 255 && flag_startPoint == 0)
//				{
//					starPoint = i;                           //Ѱ�����
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
//		/*****************��С���˷��������******************/
//		polynomial_curve_fit(effectPoint, 2, A);//���ö������߽������
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

