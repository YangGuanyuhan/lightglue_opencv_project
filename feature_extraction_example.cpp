#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

// 辅助打印函数：用于打印 KeyPoint 类型的特征点
void printKeypoints(const string& algoName, const vector<KeyPoint>& keypoints) {
    cout << "========== " << algoName << " 算法 ==========" << endl;
    cout << "总计提取特征点数量: " << keypoints.size() << endl;
    
    // 只打印前 3 个点的坐标，避免控制台刷屏
    int printCount = min((int)keypoints.size(), 3);
    if (printCount > 0) {
        cout << "前 " << printCount << " 个特征点坐标 (x, y): ";
        for (int i = 0; i < printCount; i++) {
            cout << "(" << keypoints[i].pt.x << ", " << keypoints[i].pt.y << ")  ";
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    // 0. 模拟您的输入：彩色图片 image1 (这里我们读取一张本地图片演示)
    // 实际使用中，image1 可能是您通过摄像头捕获或上游传入的 cv::Mat
    Mat image1 = imread("test.jpg", IMREAD_COLOR);
    if (image1.empty()) {
        cout << "无法读取图像，请确保当前目录下有 test.jpg 供测试使用。" << endl;
        // 为了确保代码能跑，如果没有图片，我们生成一张彩色测试图
        image1 = Mat::zeros(300, 300, CV_8UC3);
        rectangle(image1, Point(50, 50), Point(150, 150), Scalar(255, 0, 0), -1); // 蓝色矩形
        circle(image1, Point(200, 200), 40, Scalar(0, 0, 255), -1);               // 红色圆形
    }

    // ==========================================
    // 预处理阶段
    // ==========================================
    Mat grayImage;
    // 【强制要求】将 CV_8UC3 (彩色) 转化为 CV_8UC1 (灰度)
    cvtColor(image1, grayImage, COLOR_BGR2GRAY);

    // ==========================================
    // 1. 现代基于 KeyPoint 对象的特征检测器
    // ==========================================
    vector<KeyPoint> keypoints;

    // 1.1 SIFT (尺度不变特征变换)
    // 特点：精度极高，对旋转、尺度缩放、亮度变化鲁棒，但计算耗时。
    Ptr<SIFT> sift = SIFT::create();
    sift->detect(grayImage, keypoints);
    printKeypoints("SIFT", keypoints);

    // 1.2 ORB (Oriented FAST and Rotated BRIEF)
    // 特点：速度极快，是 SIFT 的极佳免费替代品，主要用于实时特征提取。
    Ptr<ORB> orb = ORB::create();
    keypoints.clear();
    orb->detect(grayImage, keypoints);
    printKeypoints("ORB", keypoints);

    // 1.3 FAST (加速段测试特征)
    // 特点：专门针对速度优化的角点检测算法，但不具备尺度和旋转不变性。
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();
    keypoints.clear();
    fast->detect(grayImage, keypoints);
    printKeypoints("FAST", keypoints);

    // 1.4 BRISK (二进制鲁棒尺度不变关键点)
    // 特点：构建了尺度空间，具备尺度和旋转不变性，性能介于 SIFT 和 ORB 之间。
    Ptr<BRISK> brisk = BRISK::create();
    keypoints.clear();
    brisk->detect(grayImage, keypoints);
    printKeypoints("BRISK", keypoints);

    // 1.5 AKAZE (加速非线性尺度空间极值)
    // 特点：在非线性尺度空间寻找特征，能更好地保留图像边缘细节。
    Ptr<AKAZE> akaze = AKAZE::create();
    keypoints.clear();
    akaze->detect(grayImage, keypoints);
    printKeypoints("AKAZE", keypoints);


    // ==========================================
    // 2. 经典的角点检测方法 (输出形式非 KeyPoint)
    // ==========================================

    // 2.1 Shi-Tomasi 角点检测 (goodFeaturesToTrack)
    // 输出：并非 KeyPoint 数组，而是直接输出 vector<Point2f> 坐标点数组。
    vector<Point2f> corners;
    // 参数含义：输入图像, 输出角点, 最大角点数, 质量因子(0.01表示最大特征值的1%), 角点间的最小像素距离
    goodFeaturesToTrack(grayImage, corners, 100, 0.01, 10);
    
    cout << "========== Shi-Tomasi 算法 ==========" << endl;
    cout << "总计提取特征点数量: " << corners.size() << endl;
    if (!corners.empty()) {
        cout << "前 3 个特征点坐标 (x, y): ";
        for (int i = 0; i < min((int)corners.size(), 3); i++) {
            cout << "(" << corners[i].x << ", " << corners[i].y << ")  ";
        }
        cout << endl;
    }
    cout << endl;

    // 2.2 Harris 角点检测
    // 输出：一张 32 位浮点型的“响应图” (CV_32FC1)，需要手动设置阈值提取坐标。
    Mat harrisResponse;
    // 参数含义：输入灰度图, 输出响应图, 邻域大小(blockSize), Sobel算子孔径大小(ksize), 自由参数 k(通常取0.04~0.06)
    cornerHarris(grayImage, harrisResponse, 2, 3, 0.04);

    // 为了方便提取阈值，通常会对响应图进行归一化处理
    Mat harrisNormalized;
    normalize(harrisResponse, harrisNormalized, 0, 255, NORM_MINMAX, CV_32FC1);

    vector<Point2f> harrisPoints;
    float threshold = 100.0f; // 设定阈值：0~255之间，值越大筛选出的角点越尖锐
    
    // 遍历响应图，将大于阈值的点保存为坐标
    for (int j = 0; j < harrisNormalized.rows; j++) {
        for (int i = 0; i < harrisNormalized.cols; i++) {
            if (harrisNormalized.at<float>(j, i) > threshold) {
                harrisPoints.push_back(Point2f((float)i, (float)j));
            }
        }
    }

    cout << "========== Harris 算法 ==========" << endl;
    cout << "总计提取特征点数量: " << harrisPoints.size() << endl;
    if (!harrisPoints.empty()) {
        cout << "前 3 个特征点坐标 (x, y): ";
        for (int i = 0; i < min((int)harrisPoints.size(), 3); i++) {
            cout << "(" << harrisPoints[i].x << ", " << harrisPoints[i].y << ")  ";
        }
        cout << endl;
    }
    cout << endl;

    return 0;
}