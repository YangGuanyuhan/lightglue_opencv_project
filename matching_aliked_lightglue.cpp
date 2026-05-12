#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <vector>

using namespace cv;
using namespace std;

const string ALIKED_MODEL_PATH = "../model/aliked-n16rot-top1k-640.onnx";
const string ALIKED_LG_MODEL_PATH = "../model/aliked_lightglue.onnx";
const string IMG_0_PATH = "../images/image1.jpg";
const string IMG_1_PATH = "../images/image2.jpg";

const int TARGET_SIZE = 640;
const int DESCRIPTOR_DIM = 128; // ALIKED uses 128-dim descriptors
const float MATCH_CONFIDENCE_THRESHOLD = 0.2f;

const int ENGINE = dnn::ENGINE_ORT;

static const char *engine_to_string(int engine)
{
    if (engine == dnn::ENGINE_ORT)
        return "ENGINE_ORT";
    if (engine == dnn::ENGINE_NEW)
        return "ENGINE_NEW";
    return "UNKNOWN_ENGINE";
}

static void print_runtime_info()
{
    cout << "================ Runtime Info ================" << endl;
    cout << "OpenCV version string: " << CV_VERSION << endl;
    cout << "OpenCV major/minor/revision: "
         << CV_VERSION_MAJOR << "."
         << CV_VERSION_MINOR << "."
         << CV_VERSION_REVISION << endl;
    cout << "DNN inference engine: " << engine_to_string(ENGINE)
         << " (enum=" << static_cast<int>(ENGINE) << ")" << endl;
    cout << "==============================================" << endl;
}

// ------------------------------
// Extract ALIKED features from an image
// ------------------------------
bool extract_aliked(
    dnn::Net &net,
    const Mat &img,
    Mat &out_kpts,    // N x 2 (normalized coords)
    Mat &out_desc,    // N x D
    Mat &out_scores)  // N x 1
{
    // blobFromImage: resize, /255, BGR->RGB, HWC->NCHW
    Mat blob = dnn::blobFromImage(img, 1.0 / 255.0, Size(TARGET_SIZE, TARGET_SIZE), Scalar(), true, false);

    net.setInput(blob, "image");

    vector<string> outNames = {"keypoints", "descriptors", "scores"};
    vector<Mat> outs;
    net.forward(outs, outNames);

    out_kpts = outs[0].clone();
    out_desc = outs[1].clone();
    out_scores = outs[2].clone();

    // Ensure continuous memory
    if (!out_kpts.isContinuous()) out_kpts = out_kpts.clone();
    if (!out_desc.isContinuous()) out_desc = out_desc.clone();
    if (!out_scores.isContinuous()) out_scores = out_scores.clone();

    return true;
}

// ------------------------------
// Run LightGlue matching
// Output: matches0 is Nx2 int64 (idx_img0, idx_img1 per row)
//         mscores0 is N float (confidence per match)
// ------------------------------
void run_lightglue(
    dnn::Net &net,
    const Mat &kpts0_norm, const Mat &desc0,
    const Mat &kpts1_norm, const Mat &desc1,
    int num_kpts0, int num_kpts1,
    Mat &out_matches0, Mat &out_mscores0)
{
    // Construct 3D blobs: 1 x N x 2 for keypoints, 1 x N x D for descriptors
    int shape_kpts0[] = {1, num_kpts0, 2};
    Mat blob_kpts0(3, shape_kpts0, CV_32FC1, (void *)kpts0_norm.ptr<float>());

    int shape_kpts1[] = {1, num_kpts1, 2};
    Mat blob_kpts1(3, shape_kpts1, CV_32FC1, (void *)kpts1_norm.ptr<float>());

    int shape_desc0[] = {1, num_kpts0, desc0.cols};
    Mat blob_desc0(3, shape_desc0, CV_32FC1, (void *)desc0.ptr<float>());

    int shape_desc1[] = {1, num_kpts1, desc1.cols};
    Mat blob_desc1(3, shape_desc1, CV_32FC1, (void *)desc1.ptr<float>());

    net.setInput(blob_kpts0, "kpts0");
    net.setInput(blob_desc0, "desc0");
    net.setInput(blob_kpts1, "kpts1");
    net.setInput(blob_desc1, "desc1");

    vector<string> outNames = {"matches0", "mscores0"};
    vector<Mat> outs;
    net.forward(outs, outNames);

    // matches0: (N, 2) int64, each row is (idx0, idx1)
    out_matches0 = outs[0].clone();
    if (!out_matches0.isContinuous()) out_matches0 = out_matches0.clone();
    // Convert to CV_32F for easier access
    out_matches0.convertTo(out_matches0, CV_32F);

    // mscores0: (N,) float
    out_mscores0 = outs[1].clone();
    if (!out_mscores0.isContinuous()) out_mscores0 = out_mscores0.clone();
    out_mscores0 = out_mscores0.reshape(1, out_mscores0.total());
}

// ------------------------------
// Print match results and draw visualization
// ------------------------------
void print_and_draw_matches(
    const Mat &img0, const Mat &img1,
    const Mat &kpts0_norm, const Mat &kpts1_norm,
    int num_kpts0, int num_kpts1,
    const Mat &matches0, const Mat &mscores0,
    float threshold)
{
    // Convert normalized keypoints [-1,1] to pixel coordinates
    vector<Point2f> pts0(num_kpts0), pts1(num_kpts1);
    for (int i = 0; i < num_kpts0; i++)
    {
        pts0[i].x = (kpts0_norm.at<float>(i, 0) + 1.0f) / 2.0f * (img0.cols - 1);
        pts0[i].y = (kpts0_norm.at<float>(i, 1) + 1.0f) / 2.0f * (img0.rows - 1);
    }
    for (int i = 0; i < num_kpts1; i++)
    {
        pts1[i].x = (kpts1_norm.at<float>(i, 0) + 1.0f) / 2.0f * (img1.cols - 1);
        pts1[i].y = (kpts1_norm.at<float>(i, 1) + 1.0f) / 2.0f * (img1.rows - 1);
    }

    // Build DMatch list
    vector<KeyPoint> kp0, kp1;
    for (int i = 0; i < num_kpts0; i++)
        kp0.emplace_back(pts0[i].x, pts0[i].y, 1.0f);
    for (int i = 0; i < num_kpts1; i++)
        kp1.emplace_back(pts1[i].x, pts1[i].y, 1.0f);

    // matches0 is Nx2: each row is (idx_img0, idx_img1)
    int num_matches = matches0.rows;
    vector<DMatch> dmatches;
    int total_valid = 0;
    for (int i = 0; i < num_matches; i++)
    {
        int idx0 = (int)matches0.at<float>(i, 0);
        int idx1 = (int)matches0.at<float>(i, 1);
        float score = mscores0.at<float>(i);

        if (idx0 >= 0 && idx1 >= 0 && score > threshold)
        {
            dmatches.emplace_back(idx0, idx1, 1.0f - score);
            total_valid++;
        }
    }

    cout << "\n========== Match Results ==========" << endl;
    cout << "Image 1 keypoints: " << num_kpts0 << endl;
    cout << "Image 2 keypoints: " << num_kpts1 << endl;
    cout << "Valid matches (confidence > " << threshold << "): " << total_valid << endl;

    // Print all valid matches
    cout << "\n--- All valid matches ---" << endl;
    cout << "IdxImg0 -> IdxImg1  Score" << endl;
    for (size_t i = 0; i < dmatches.size(); i++)
    {
        int qIdx = dmatches[i].queryIdx;
        int tIdx = dmatches[i].trainIdx;
        float score = 1.0f - dmatches[i].distance;
        cout << "  " << qIdx << " -> " << tIdx << "  score=" << score;

        // Print pixel coordinates
        cout << "  (" << pts0[qIdx].x << ", " << pts0[qIdx].y << ") -> ("
             << pts1[tIdx].x << ", " << pts1[tIdx].y << ")" << endl;
    }

    // Draw matches
    Mat vis_img;
    drawMatches(
        img0, kp0, img1, kp1, dmatches, vis_img,
        Scalar(0, 255, 0), Scalar(0, 0, 255),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    string out_filename = "aliked_lightglue_matches.jpg";
    imwrite(out_filename, vis_img);
    cout << "\nResult saved as " << out_filename << endl;

    imshow("ALIKED + LightGlue Matches", vis_img);
    waitKey(0);
    destroyAllWindows();
}

// ------------------------------
// Main
// ------------------------------
int main()
{
    print_runtime_info();

    // Check model files
    if (!filesystem::exists(ALIKED_MODEL_PATH))
    {
        cerr << "ALIKED model not found: " << ALIKED_MODEL_PATH << endl;
        return -1;
    }
    if (!filesystem::exists(ALIKED_LG_MODEL_PATH))
    {
        cerr << "ALIKED LightGlue model not found: " << ALIKED_LG_MODEL_PATH << endl;
        return -1;
    }

    // Load models with ENGINE_ORT
    cout << "Loading ALIKED model: " << ALIKED_MODEL_PATH << endl;
    dnn::Net net_aliked = dnn::readNetFromONNX(ALIKED_MODEL_PATH, ENGINE);
    cout << "  -> Loaded with " << engine_to_string(ENGINE) << endl;

    cout << "Loading ALIKED LightGlue model: " << ALIKED_LG_MODEL_PATH << endl;
    dnn::Net net_lg = dnn::readNetFromONNX(ALIKED_LG_MODEL_PATH, ENGINE);
    net_lg.enableWinograd(false);
    cout << "  -> Loaded with " << engine_to_string(ENGINE) << endl;

    // Read images
    Mat img0 = imread(IMG_0_PATH);
    Mat img1 = imread(IMG_1_PATH);
    if (img0.empty())
    {
        cerr << "Cannot read image: " << IMG_0_PATH << endl;
        return -1;
    }
    if (img1.empty())
    {
        cerr << "Cannot read image: " << IMG_1_PATH << endl;
        return -1;
    }

    // Resize images to target size
    resize(img0, img0, Size(TARGET_SIZE, TARGET_SIZE));
    resize(img1, img1, Size(TARGET_SIZE, TARGET_SIZE));
    cout << "\nImages resized to " << TARGET_SIZE << "x" << TARGET_SIZE << endl;

    // Extract features with ALIKED
    Mat kpts0_norm, desc0, scores0;
    Mat kpts1_norm, desc1, scores1;

    cout << "\n[1/2] Extracting features from image 1..." << endl;
    extract_aliked(net_aliked, img0, kpts0_norm, desc0, scores0);
    int num_kpts0 = kpts0_norm.rows;
    cout << "  -> " << num_kpts0 << " keypoints extracted" << endl;

    cout << "[2/2] Extracting features from image 2..." << endl;
    extract_aliked(net_aliked, img1, kpts1_norm, desc1, scores1);
    int num_kpts1 = kpts1_norm.rows;
    cout << "  -> " << num_kpts1 << " keypoints extracted" << endl;

    // Print sample keypoints (first 5)
    cout << "\n--- Sample keypoints from image 1 (first 5, normalized) ---" << endl;
    for (int i = 0; i < min(5, num_kpts0); i++)
    {
        cout << "  kpt[" << i << "]: ("
             << kpts0_norm.at<float>(i, 0) << ", "
             << kpts0_norm.at<float>(i, 1) << ")  score="
             << scores0.at<float>(i) << endl;
    }

    // Run LightGlue matching
    Mat matches0, mscores0;
    cout << "\n[LightGlue] Running matching..." << endl;
    run_lightglue(net_lg, kpts0_norm, desc0, kpts1_norm, desc1,
                  num_kpts0, num_kpts1, matches0, mscores0);
    cout << "  -> Matching complete" << endl;

    // Print raw LightGlue output (Fused format: matches0 is Nx2)
    cout << "\n--- Raw LightGlue output (first 10 matches) ---" << endl;
    int num_raw_matches = matches0.rows;
    for (int i = 0; i < min(10, num_raw_matches); i++)
    {
        cout << "  match[" << i << "]: idx0=" << (int)matches0.at<float>(i, 0)
             << "  idx1=" << (int)matches0.at<float>(i, 1)
             << "  score=" << mscores0.at<float>(i) << endl;
    }

    // Print and draw results
    print_and_draw_matches(img0, img1, kpts0_norm, kpts1_norm,
                           num_kpts0, num_kpts1,
                           matches0, mscores0,
                           MATCH_CONFIDENCE_THRESHOLD);

    return 0;
}
