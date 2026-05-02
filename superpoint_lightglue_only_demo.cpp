int main(){
    return 0;
}

// #include <opencv2/opencv.hpp>
// #include <opencv2/features.hpp>

// #ifdef HAVE_OPENCV_DNN
// #include <opencv2/dnn.hpp>
// #endif

// #include <algorithm>
// #include <iostream>
// #include <vector>

// using namespace cv;
// using namespace std;

// int main()
// {
// #ifndef HAVE_OPENCV_DNN
//     cerr << "This demo requires OpenCV built with DNN support." << endl;
//     return 1;
// #else
    
//     const string imgPath0 = "../images/image1.jpg";
//     const string imgPath1 = "../images/image2.jpg";
//     const string superpointModel = "../model/superpoint.onnx";
//     const string lightglueModel = "../model/superpoint_lightglue.onnx";
//     const string outPath = "./sp_lg_matches.jpg";

//     Mat img0 = imread(imgPath0, IMREAD_COLOR);
//     Mat img1 = imread(imgPath1, IMREAD_COLOR);
//     if (img0.empty() || img1.empty())
//     {
//         cerr << "Failed to read input images." << endl;
//         return 2;
//     }

//     try
//     {
//         // 1) SuperPoint
//         cv::features::SuperPoint::Params spParams;
//         spParams.modelPath = superpointModel;
//         spParams.dnnEngine = dnn::ENGINE_ORT;
//         spParams.inputSize = Size(640, 480);
//         spParams.preferGrayInput = true;

//         Ptr<cv::features::FeatureExtractor> extractor = cv::features::SuperPoint::create(spParams);

//         vector<KeyPoint> kpts0, kpts1;
//         Mat desc0, desc1;
//         extractor->extract(img0, kpts0, desc0);
//         extractor->extract(img1, kpts1, desc1);

//         // Get original image size
//         Size sz0 = img0.size();
//         Size sz1 = img1.size();

//         // Calculate scaling ratio
//         float scale_x0 = (float)sz0.width / spParams.inputSize.width;
//         float scale_y0 = (float)sz0.height / spParams.inputSize.height;

//         float scale_x1 = (float)sz1.width / spParams.inputSize.width;
//         float scale_y1 = (float)sz1.height / spParams.inputSize.height;

        
//         for (auto &kp : kpts0)
//         {
//             kp.pt.x *= scale_x0;
//             kp.pt.y *= scale_y0;
//         }

//         for (auto &kp : kpts1)
//         {
//             kp.pt.x *= scale_x1;
//             kp.pt.y *= scale_y1;
//         }

//         // 2) LightGlue
//         vector<Point2f> pts0, pts1;
//         KeyPoint::convert(kpts0, pts0);
//         KeyPoint::convert(kpts1, pts1);

//         cv::features::LightGlue::Params lgParams;
//         lgParams.modelPath = lightglueModel;
//         lgParams.dnnEngine = dnn::ENGINE_ORT;
//         lgParams.disableWinograd = true;

//         Ptr<cv::features::FeatureMatcher> matcher = cv::features::LightGlue::create(lgParams);

//         vector<DMatch> matches;
//         matcher->match(Mat(pts0), desc0, Mat(pts1), desc1, matches, noArray(), img0.size(), img1.size());

//         sort(matches.begin(), matches.end(),
//              [](const DMatch &a, const DMatch &b)
//              { return a.distance < b.distance; });

//         vector<DMatch> visMatches(matches.begin(),
//                                   matches.begin() + min<size_t>(matches.size(), 300));

//         Mat vis;
//         drawMatches(img0, kpts0, img1, kpts1, visMatches, vis);

//         imwrite(outPath, vis);

//         cout << "Saved: " << outPath << endl;
//         // SuperPoint output check
//         cout << "img0 keypoints: " << kpts0.size() << endl;
//         cout << "img1 keypoints: " << kpts1.size() << endl;
//         cout << "desc0 size: " << desc0.rows << " x " << desc0.cols << endl;
//         cout << "desc1 size: " << desc1.rows << " x " << desc1.cols << endl;

//         if (kpts0.empty() || kpts1.empty() || desc0.empty() || desc1.empty())
//         {
//             cerr << "SuperPoint failed: empty output!" << endl;
//             return 4;
//         }

    
//         matcher->match(Mat(pts0), desc0, Mat(pts1), desc1, matches, noArray(), img0.size(), img1.size());

//         cout << "Raw matches: " << matches.size() << endl;

//         if (matches.empty())
//         {
//             cerr << "No matches found!" << endl;
//             return 5;
//         }

//         // Sort
//         sort(matches.begin(), matches.end(),
//              [](const DMatch &a, const DMatch &b)
//              { return a.distance < b.distance; });

//         // Print top 10 matches
//         cout << "Top matches (distance): ";
//         for (int i = 0; i < min(10, (int)matches.size()); i++)
//         {
//             cout << matches[i].distance << " ";
//         }
//         cout << endl;

//         // Safe truncation
//         size_t keep = min<size_t>(matches.size(), 300);
      

//         cout << "Keep matches: " << keep << endl;
//     }
//     catch (const cv::Exception &e)
//     {
//         cerr << "OpenCV exception: " << e.what() << endl;
//         return 8;
//     }

//     return 0;
// #endif
// }