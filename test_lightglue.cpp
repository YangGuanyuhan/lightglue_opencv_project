#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <random>
#include <filesystem>

using namespace cv;
using namespace std;


const string LG_MODEL_PATH = "../model/superpoint_lightglue.onnx";
const int DESCRIPTOR_DIM = 256;

const int NUM_KPTS_0 = 500;
const int NUM_KPTS_1 = 500;

const int IMG_H0 = 480, IMG_W0 = 640;
const int IMG_H1 = 480, IMG_W1 = 640;

// ------------------------------
// Normalize keypoints N 2
// ------------------------------
Mat normalize_keypoints(const Mat &kpts, int H, int W)
{
    Mat kpts_norm;
    kpts.convertTo(kpts_norm, CV_32F); // Generates single channel CV_32FC1

    for (int i = 0; i < kpts_norm.rows; i++)
    {
        kpts_norm.at<float>(i, 0) /= (W - 1); // x
        kpts_norm.at<float>(i, 1) /= (H - 1); // y
    }

    // Return N 2 CV_32FC1 matrix directly do not use reshape
    return kpts_norm;
}

// ------------------------------
// Generate random keypoints descriptors
// ------------------------------
void generate_dummy_features(
    int num_kpts,
    int descriptor_dim,
    int H, int W,
    Mat &out_kpts, // N 2
    Mat &out_desc  // N D
)
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dx(0, W - 1);
    std::uniform_real_distribution<float> dy(0, H - 1);

    out_kpts = Mat(num_kpts, 2, CV_32F);
    out_desc = Mat(num_kpts, descriptor_dim, CV_32F);

    for (int i = 0; i < num_kpts; i++)
    {
        out_kpts.at<float>(i, 0) = dx(rng);
        out_kpts.at<float>(i, 1) = dy(rng);

        // random descriptor
        float norm = 0.0f;
        for (int d = 0; d < descriptor_dim; d++)
        {
            float v = rng() / float(rng.max());
            out_desc.at<float>(i, d) = v;
            norm += v * v;
        }

        // L2 normalize
        norm = std::sqrt(norm) + 1e-6f;
        for (int d = 0; d < descriptor_dim; d++)
        {
            out_desc.at<float>(i, d) /= norm;
        }
    }
}
// ------------------------------
// Run LightGlue
// ------------------------------
void run_lightglue(
    dnn::Net &net,
    const Mat &kpts0, const Mat &desc0,
    const Mat &kpts1, const Mat &desc1,
    int H0, int W0,
    int H1, int W1)
{
    cout << "Running LightGlue inference" << endl;

    // 1 Normalize keypoints return N 2 matrix
    Mat kpts0_norm = normalize_keypoints(kpts0, H0, W0);
    Mat kpts1_norm = normalize_keypoints(kpts1, H1, W1);

    // 2 Construct 3D Tensors 1 N 2
    int shape_kpts0[] = {1, kpts0_norm.rows, 2};
    Mat blob_kpts0(3, shape_kpts0, CV_32FC1, kpts0_norm.data);

    int shape_kpts1[] = {1, kpts1_norm.rows, 2};
    Mat blob_kpts1(3, shape_kpts1, CV_32FC1, kpts1_norm.data);

    // 3 Construct descriptor 3D Tensors 1 N 256
    int shape_desc0[] = {1, desc0.rows, desc0.cols};
    Mat blob_desc0(3, shape_desc0, CV_32FC1, desc0.data);

    int shape_desc1[] = {1, desc1.rows, desc1.cols};
    Mat blob_desc1(3, shape_desc1, CV_32FC1, desc1.data);

    // 4 Set inputs
    net.setInput(blob_kpts0, "kpts0");
    net.setInput(blob_desc0, "desc0");
    net.setInput(blob_kpts1, "kpts1");
    net.setInput(blob_desc1, "desc1");

    // Output nodes
    vector<string> outNames = {"matches0", "mscores0", "matches1", "mscores1"};
    vector<Mat> outs;
    
    // Execute inference
    net.forward(outs, outNames);

    // Parse output usually outs 0 is 1 N Tensor
    Mat matches0 = outs[0].reshape(1, outs[0].total());
    Mat mscores0 = outs[1].reshape(1, outs[1].total());

    cout << "Inference complete" << endl;

    cout << "Output matches0 shape " << matches0.total() << endl;
    cout << "Output mscores0 shape " << mscores0.total() << endl;

    // Count valid matches
    int valid = 0;
    for (int i = 0; i < matches0.total(); i++)
    {
        // LightGlue outputs -1 if no match found
        if (matches0.at<float>(i) > -0.5f) 
            valid++;
    }
    cout << "Found " << valid << " valid matches" << endl;

    // Print first 10 results
    cout << "\n--- First 10 match results ---" << endl;
    for (int i = 0; i < 10 && i < matches0.total(); i++)
    {
        cout << "i=" << i
             << " -> match=" << matches0.at<float>(i)
             << ", score=" << mscores0.at<float>(i) << endl;
    }
}

// ------------------------------
// Main
// ------------------------------
int main()
{

    cout << "OpenCV version " << CV_VERSION << endl;

    if (!std::filesystem::exists(LG_MODEL_PATH))
    {
        cerr << "Model does not exist " << LG_MODEL_PATH << endl;
        return -1;
    }

    // Load ONNX
    cout << "Loading LightGlue model " << LG_MODEL_PATH << endl;
    dnn::Net net = dnn::readNetFromONNX(LG_MODEL_PATH,dnn::ENGINE_NEW);
    net.enableWinograd(false);// Disable Winograd optimization to avoid potential issues
    cout << "Model loaded successfully" << endl;

    // Generate dummy data
    Mat kpts0, desc0, kpts1, desc1;
    generate_dummy_features(NUM_KPTS_0, DESCRIPTOR_DIM, IMG_H0, IMG_W0, kpts0, desc0);
    generate_dummy_features(NUM_KPTS_1, DESCRIPTOR_DIM, IMG_H1, IMG_W1, kpts1, desc1);

    cout << "Dummy data generation complete" << endl;

    // Run inference
    run_lightglue(net, kpts0, desc0, kpts1, desc1, IMG_H0, IMG_W0, IMG_H1, IMG_W1);

    return 0;
}
