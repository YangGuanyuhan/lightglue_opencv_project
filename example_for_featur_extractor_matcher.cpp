// OpenCV 常用特征点提取与匹配示范
// 说明：
// 1) 本文件演示从“检测 -> 描述 -> 匹配 -> 过滤 -> 可视化”的完整流程。
// 2) 同时包含经典角点（Shi-Tomasi、Harris）和现代局部特征（SIFT/ORB/BRISK/AKAZE）。
// 3) 输出图片会保存到当前目录，方便在无 GUI 环境下查看。

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;
using namespace cv;

struct FeatureResult {
	vector<KeyPoint> keypoints;
	Mat descriptors;
};

// 打印前若干个 KeyPoint，避免控制台刷屏
void printKeypointsSummary(const string& algoName, const vector<KeyPoint>& keypoints, int topK = 5) {
	cout << "========== " << algoName << " ==========" << endl;
	cout << "特征点数量: " << keypoints.size() << endl;

	int printCount = min(static_cast<int>(keypoints.size()), topK);
	if (printCount > 0) {
		cout << "前 " << printCount << " 个点: " << endl;
		for (int i = 0; i < printCount; ++i) {
			const KeyPoint& kp = keypoints[i];
			cout << "  #" << i
				 << " pt=(" << kp.pt.x << ", " << kp.pt.y << ")"
				 << " size=" << kp.size
				 << " angle=" << kp.angle
				 << " response=" << kp.response
				 << endl;
		}
	}
	cout << endl;
}

void printDescriptorSummary(const string& algoName, const Mat& descriptors) {
	cout << "[" << algoName << "] 描述子矩阵: rows=" << descriptors.rows
		 << ", cols=" << descriptors.cols
		 << ", type=" << descriptors.type() << endl;
}

// 统一封装：检测 + 计算描述子
FeatureResult detectAndCompute(const string& name, Ptr<Feature2D> detector, const Mat& gray) {
	FeatureResult result;
	detector->detectAndCompute(gray, noArray(), result.keypoints, result.descriptors);
	printKeypointsSummary(name, result.keypoints);
	printDescriptorSummary(name, result.descriptors);
	return result;
}

// 绘制并保存特征点可视化图
void drawAndSaveKeypoints(const Mat& image,
						  const vector<KeyPoint>& keypoints,
						  const string& outName,
						  const Scalar& color = Scalar(0, 255, 0)) {
	Mat vis;
	drawKeypoints(image, keypoints, vis, color, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imwrite(outName, vis);
	cout << "已保存特征点可视化: " << outName << endl;
}

// BFMatcher + KNN + Lowe ratio test
vector<DMatch> knnRatioMatch(const Mat& desc1, const Mat& desc2, int normType, float ratio = 0.75f) {
	vector<vector<DMatch>> knnMatches;
	BFMatcher matcher(normType, false);
	matcher.knnMatch(desc1, desc2, knnMatches, 2);

	vector<DMatch> goodMatches;
	goodMatches.reserve(knnMatches.size());
	for (const auto& m : knnMatches) {
		if (m.size() < 2) {
			continue;
		}
		if (m[0].distance < ratio * m[1].distance) {
			goodMatches.push_back(m[0]);
		}
	}
	return goodMatches;
}

// BFMatcher 交叉验证（crossCheck=true），简单且稳定，但通常匹配数会更少
vector<DMatch> bfCrossCheckMatch(const Mat& desc1, const Mat& desc2, int normType) {
	BFMatcher matcher(normType, true);
	vector<DMatch> matches;
	matcher.match(desc1, desc2, matches);
	sort(matches.begin(), matches.end(), [](const DMatch& a, const DMatch& b) {
		return a.distance < b.distance;
	});
	return matches;
}

// FLANN + KNN + ratio（常见于 SIFT/SURF 等浮点描述子）
vector<DMatch> flannKnnRatioMatch(const Mat& desc1, const Mat& desc2, float ratio = 0.75f) {
	Mat d1 = desc1, d2 = desc2;

	// FLANN 对 SIFT 描述子通常要求 CV_32F
	if (d1.type() != CV_32F) d1.convertTo(d1, CV_32F);
	if (d2.type() != CV_32F) d2.convertTo(d2, CV_32F);

	FlannBasedMatcher matcher;
	vector<vector<DMatch>> knnMatches;
	matcher.knnMatch(d1, d2, knnMatches, 2);

	vector<DMatch> goodMatches;
	goodMatches.reserve(knnMatches.size());
	for (const auto& m : knnMatches) {
		if (m.size() < 2) {
			continue;
		}
		if (m[0].distance < ratio * m[1].distance) {
			goodMatches.push_back(m[0]);
		}
	}
	return goodMatches;
}

// 使用 RANSAC 在“好匹配”中进一步筛掉几何不一致的离群点
vector<DMatch> geometricFilterByHomography(const vector<KeyPoint>& kpts1,
										   const vector<KeyPoint>& kpts2,
										   const vector<DMatch>& matches,
										   double reprojThreshold = 3.0) {
	vector<DMatch> inlierMatches;
	if (matches.size() < 4) {
		return inlierMatches;
	}

	vector<Point2f> pts1;
	vector<Point2f> pts2;
	pts1.reserve(matches.size());
	pts2.reserve(matches.size());

	for (const auto& m : matches) {
		pts1.push_back(kpts1[m.queryIdx].pt);
		pts2.push_back(kpts2[m.trainIdx].pt);
	}

	vector<uchar> inlierMask;
	findHomography(pts1, pts2, RANSAC, reprojThreshold, inlierMask);

	for (size_t i = 0; i < matches.size(); ++i) {
		if (i < inlierMask.size() && inlierMask[i]) {
			inlierMatches.push_back(matches[i]);
		}
	}
	return inlierMatches;
}

// 把匹配结果可视化保存
void drawAndSaveMatches(const Mat& img1,
						const vector<KeyPoint>& kpts1,
						const Mat& img2,
						const vector<KeyPoint>& kpts2,
						const vector<DMatch>& matches,
						const string& outName,
						int maxDraw = 80) {
	Mat vis;
	vector<DMatch> drawMatchesVec = matches;
	if (static_cast<int>(drawMatchesVec.size()) > maxDraw) {
		drawMatchesVec.resize(maxDraw);
	}

	drawMatches(img1, kpts1, img2, kpts2, drawMatchesVec, vis,
				Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite(outName, vis);
	cout << "已保存匹配可视化: " << outName
		 << " (绘制数量=" << drawMatchesVec.size() << ")" << endl;
}

// 经典角点: Shi-Tomasi
vector<Point2f> runShiTomasi(const Mat& gray, int maxCorners = 200, double quality = 0.01, double minDist = 8.0) {
	vector<Point2f> corners;
	goodFeaturesToTrack(gray, corners, maxCorners, quality, minDist);

	cout << "========== Shi-Tomasi ==========" << endl;
	cout << "角点数量: " << corners.size() << endl;
	for (int i = 0; i < min(static_cast<int>(corners.size()), 5); ++i) {
		cout << "  #" << i << " (" << corners[i].x << ", " << corners[i].y << ")" << endl;
	}
	cout << endl;
	return corners;
}

// 经典角点: Harris
vector<Point2f> runHarris(const Mat& gray,
						  int blockSize = 2,
						  int ksize = 3,
						  double k = 0.04,
						  float threshold = 120.0f) {
	Mat harrisResp;
	cornerHarris(gray, harrisResp, blockSize, ksize, k);

	Mat normResp;
	normalize(harrisResp, normResp, 0, 255, NORM_MINMAX, CV_32FC1);

	vector<Point2f> points;
	for (int y = 0; y < normResp.rows; ++y) {
		for (int x = 0; x < normResp.cols; ++x) {
			if (normResp.at<float>(y, x) > threshold) {
				points.emplace_back(static_cast<float>(x), static_cast<float>(y));
			}
		}
	}

	cout << "========== Harris ==========" << endl;
	cout << "角点数量: " << points.size() << endl;
	for (int i = 0; i < min(static_cast<int>(points.size()), 5); ++i) {
		cout << "  #" << i << " (" << points[i].x << ", " << points[i].y << ")" << endl;
	}
	cout << endl;
	return points;
}

// 将 Point2f 角点绘制到图像（用于 Shi-Tomasi/Harris）
void drawAndSaveCorners(const Mat& image,
						const vector<Point2f>& points,
						const string& outName,
						const Scalar& color = Scalar(0, 255, 255),
						int radius = 3) {
	Mat vis = image.clone();
	for (const auto& p : points) {
		circle(vis, p, radius, color, -1, LINE_AA);
	}
	imwrite(outName, vis);
	cout << "已保存角点可视化: " << outName << endl;
}

int main() {
	cout << "===== OpenCV 特征提取与匹配示范开始 =====" << endl;
	cout << CV_VERSION << endl;

	// 0) 输入图像
	// 建议准备两张有重叠区域的图：test1.jpg 和 test2.jpg
	Mat image1 = imread("test1.jpg", IMREAD_COLOR);
	Mat image2 = imread("test2.jpg", IMREAD_COLOR);

	// 如果读取失败，自动生成一对“可匹配”的测试图，确保示例能直接运行
	if (image1.empty() || image2.empty()) {
		cout << "未读取到 test1.jpg / test2.jpg，自动生成演示图像。" << endl;

		image1 = Mat::zeros(420, 640, CV_8UC3);
		rectangle(image1, Point(70, 90), Point(230, 260), Scalar(255, 180, 30), -1);
		circle(image1, Point(380, 220), 70, Scalar(40, 70, 255), -1);
		line(image1, Point(40, 380), Point(610, 330), Scalar(0, 255, 0), 4);
		putText(image1, "Feature Demo", Point(170, 70), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

		// 通过仿射变换 + 亮度扰动构造第二张图
		Mat M = getRotationMatrix2D(Point2f(320.0f, 210.0f), 8.0, 1.05);
		M.at<double>(0, 2) += 22.0;
		M.at<double>(1, 2) += -14.0;
		warpAffine(image1, image2, M, image1.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(20, 20, 20));
		image2 += Scalar(8, 12, 10);

		imwrite("synthetic_test1.jpg", image1);
		imwrite("synthetic_test2.jpg", image2);
		cout << "已保存自动生成输入图: synthetic_test1.jpg, synthetic_test2.jpg" << endl;
	}

	// 1) 预处理：彩色转灰度
	Mat gray1, gray2;
	cvtColor(image1, gray1, COLOR_BGR2GRAY);
	cvtColor(image2, gray2, COLOR_BGR2GRAY);

	// 2) 现代特征算法：检测 + 描述
	// SIFT: 浮点描述子，鲁棒但较慢
	Ptr<SIFT> sift = SIFT::create();
	FeatureResult sift1 = detectAndCompute("SIFT @ image1", sift, gray1);
	FeatureResult sift2 = detectAndCompute("SIFT @ image2", sift, gray2);

	// ORB: 二进制描述子，速度快，实时场景常用
	Ptr<ORB> orb = ORB::create(1500);
	FeatureResult orb1 = detectAndCompute("ORB @ image1", orb, gray1);
	FeatureResult orb2 = detectAndCompute("ORB @ image2", orb, gray2);


	// 仅检测（不计算描述子）示例：FAST
	Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(20, true);
	vector<KeyPoint> fastKpts;
	fast->detect(gray1, fastKpts);
	printKeypointsSummary("FAST @ image1 (detect only)", fastKpts);

	// 3) 经典角点方法（输出 Point2f）
	vector<Point2f> shiCorners = runShiTomasi(gray1);
	vector<Point2f> harrisCorners = runHarris(gray1);

	// 4) 可视化各类特征点
	drawAndSaveKeypoints(image1, sift1.keypoints, "out_kpts_sift_img1.jpg");
	drawAndSaveKeypoints(image1, orb1.keypoints, "out_kpts_orb_img1.jpg", Scalar(255, 80, 0));
	drawAndSaveKeypoints(image1, fastKpts, "out_kpts_fast_img1.jpg", Scalar(0, 220, 255));
	drawAndSaveCorners(image1, shiCorners, "out_corners_shitomasi_img1.jpg");
	drawAndSaveCorners(image1, harrisCorners, "out_corners_harris_img1.jpg", Scalar(0, 120, 255));

	// 5) 匹配示例
	// 5.1 SIFT + BF(L2) + KNN ratio
	vector<DMatch> siftGood = knnRatioMatch(sift1.descriptors, sift2.descriptors, NORM_L2, 0.75f);
	cout << "SIFT BF+KNN+Ratio 好匹配数量: " << siftGood.size() << endl;

	// 5.2 SIFT + FLANN + KNN ratio
	vector<DMatch> siftFlannGood = flannKnnRatioMatch(sift1.descriptors, sift2.descriptors, 0.75f);
	cout << "SIFT FLANN+KNN+Ratio 好匹配数量: " << siftFlannGood.size() << endl;

	// 5.3 ORB + BF(Hamming) + KNN ratio
	vector<DMatch> orbGood = knnRatioMatch(orb1.descriptors, orb2.descriptors, NORM_HAMMING, 0.78f);
	cout << "ORB BF(Hamming)+KNN+Ratio 好匹配数量: " << orbGood.size() << endl;

	// 5.4 ORB + BF(Hamming) + crossCheck
	vector<DMatch> orbCross = bfCrossCheckMatch(orb1.descriptors, orb2.descriptors, NORM_HAMMING);
	cout << "ORB BF(Hamming)+CrossCheck 匹配数量: " << orbCross.size() << endl;

	// 6) 几何一致性过滤（RANSAC）
	vector<DMatch> siftInliers = geometricFilterByHomography(sift1.keypoints, sift2.keypoints, siftGood, 3.0);
	vector<DMatch> orbInliers = geometricFilterByHomography(orb1.keypoints, orb2.keypoints, orbGood, 3.0);

	cout << "SIFT RANSAC 内点数量: " << siftInliers.size() << " / " << siftGood.size() << endl;
	cout << "ORB  RANSAC 内点数量: " << orbInliers.size() << " / " << orbGood.size() << endl;

	// 7) 匹配可视化
	drawAndSaveMatches(image1, sift1.keypoints, image2, sift2.keypoints,
					   siftGood, "out_match_sift_bf_ratio.jpg", 100);
	drawAndSaveMatches(image1, sift1.keypoints, image2, sift2.keypoints,
					   siftFlannGood, "out_match_sift_flann_ratio.jpg", 100);
	drawAndSaveMatches(image1, sift1.keypoints, image2, sift2.keypoints,
					   siftInliers, "out_match_sift_ransac_inliers.jpg", 100);

	drawAndSaveMatches(image1, orb1.keypoints, image2, orb2.keypoints,
					   orbGood, "out_match_orb_bf_ratio.jpg", 120);
	drawAndSaveMatches(image1, orb1.keypoints, image2, orb2.keypoints,
					   orbCross, "out_match_orb_bf_crosscheck.jpg", 120);
	drawAndSaveMatches(image1, orb1.keypoints, image2, orb2.keypoints,
					   orbInliers, "out_match_orb_ransac_inliers.jpg", 120);

	cout << "\n===== 示例结束 =====" << endl;
	cout << "你可以打开当前目录下 out_*.jpg 查看效果。" << endl;

	return 0;
}
