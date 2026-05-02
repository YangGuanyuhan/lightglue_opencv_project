#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdlib>
#include <iostream>
#include <random>
#include <filesystem>

using namespace cv;
using namespace std;

const int DESCRIPTOR_DIM = 128;  // DISK uses 128-D, NOT 256-D like SuperPoint
const int NUM_KPTS_0 = 500;
const int NUM_KPTS_1 = 500;
const int IMG_H0 = 480, IMG_W0 = 640;
const int IMG_H1 = 480, IMG_W1 = 640;
const int ENGINE = dnn::ENGINE_NEW;

static const char* engine_to_string(int engine)
{
    if (engine == dnn::ENGINE_NEW)
        return "ENGINE_NEW";
    if (engine == dnn::ENGINE_ORT)
        return "ENGINE_ORT";
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
    cout << "OpenCV detailed build information:" << endl;
    cout << cv::getBuildInformation() << endl;
    cout << "DNN inference engine: " << engine_to_string(ENGINE)
         << " (enum=" << static_cast<int>(ENGINE) << ")" << endl;
    cout << "DESCRIPTOR_DIM: " << DESCRIPTOR_DIM << endl;
    const char* traceEnv = std::getenv("DNN_TRACE_ALL");
    cout << "DNN_TRACE_ALL=" << (traceEnv ? traceEnv : "<unset>") << endl;
    cout << "==============================================" << endl;
}

Mat normalize_keypoints(const Mat &kpts, int H, int W)
{
    Mat kpts_norm;
    kpts.convertTo(kpts_norm, CV_32F);
    for (int i = 0; i < kpts_norm.rows; i++)
    {
        kpts_norm.at<float>(i, 0) /= (W - 1);
        kpts_norm.at<float>(i, 1) /= (H - 1);
    }
    return kpts_norm;
}

void generate_dummy_features(
    int num_kpts,
    int H, int W,
    Mat &out_kpts,
    Mat &out_desc
)
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dx(0, W - 1);
    std::uniform_real_distribution<float> dy(0, H - 1);

    out_kpts = Mat(num_kpts, 2, CV_32F);
    out_desc = Mat(num_kpts, DESCRIPTOR_DIM, CV_32F);

    for (int i = 0; i < num_kpts; i++)
    {
        out_kpts.at<float>(i, 0) = dx(rng);
        out_kpts.at<float>(i, 1) = dy(rng);

        float norm = 0.0f;
        for (int d = 0; d < DESCRIPTOR_DIM; d++)
        {
            float v = rng() / float(rng.max());
            out_desc.at<float>(i, d) = v;
            norm += v * v;
        }
        norm = std::sqrt(norm) + 1e-6f;
        for (int d = 0; d < DESCRIPTOR_DIM; d++)
        {
            out_desc.at<float>(i, d) /= norm;
        }
    }
}

void print_mat_shape(const string &name, const Mat &m)
{
    cout << "  " << name << " shape=[";
    for (int i = 0; i < m.dims; i++)
    {
        cout << m.size[i];
        if (i + 1 < m.dims) cout << " x ";
    }
    cout << "] type=" << m.type() << " total=" << m.total() << endl;
}

void run_matcher_standard(
    dnn::Net &net,
    const Mat &kpts0, const Mat &desc0,
    const Mat &kpts1, const Mat &desc1,
    int H0, int W0, int H1, int W1)
{
    cout << "Running LightGlue inference (standard format)" << endl;

    Mat kpts0_norm = normalize_keypoints(kpts0, H0, W0);
    Mat kpts1_norm = normalize_keypoints(kpts1, H1, W1);

    int shape_kpts0[] = {1, kpts0_norm.rows, 2};
    Mat blob_kpts0(3, shape_kpts0, CV_32FC1, kpts0_norm.data);
    int shape_kpts1[] = {1, kpts1_norm.rows, 2};
    Mat blob_kpts1(3, shape_kpts1, CV_32FC1, kpts1_norm.data);

    int shape_desc0[] = {1, desc0.rows, desc0.cols};
    Mat blob_desc0(3, shape_desc0, CV_32FC1, desc0.data);
    int shape_desc1[] = {1, desc1.rows, desc1.cols};
    Mat blob_desc1(3, shape_desc1, CV_32FC1, desc1.data);

    net.setInput(blob_kpts0, "kpts0");
    net.setInput(blob_desc0, "desc0");
    net.setInput(blob_kpts1, "kpts1");
    net.setInput(blob_desc1, "desc1");

    vector<string> outNames = {"matches0", "mscores0", "matches1", "mscores1"};
    vector<Mat> outs;
    net.forward(outs, outNames);

    for (size_t i = 0; i < outs.size(); i++)
        print_mat_shape(outNames[i], outs[i]);

    // Parse matches0: shape [1, N] with -1 sentinel for unmatched
    Mat matches0 = outs[0].reshape(1, outs[0].total());
    Mat mscores0 = outs[1].reshape(1, outs[1].total());

    int valid = 0;
    for (int i = 0; i < matches0.total(); i++)
    {
        float m = matches0.at<float>(i);
        if (m > -0.5f) valid++;
    }
    cout << "Found " << valid << " valid matches (out of " << matches0.total() << ")" << endl;

    cout << "\n--- First 10 match results ---" << endl;
    for (int i = 0; i < 10 && i < matches0.total(); i++)
    {
        cout << "i=" << i
             << " -> match=" << matches0.at<float>(i)
             << ", score=" << mscores0.at<float>(i) << endl;
    }
}

void run_matcher_fused(
    dnn::Net &net,
    const Mat &kpts0, const Mat &desc0,
    const Mat &kpts1, const Mat &desc1,
    int H0, int W0, int H1, int W1)
{
    cout << "Running LightGlue inference (fused format)" << endl;

    Mat kpts0_norm = normalize_keypoints(kpts0, H0, W0);
    Mat kpts1_norm = normalize_keypoints(kpts1, H1, W1);

    int shape_kpts0[] = {1, kpts0_norm.rows, 2};
    Mat blob_kpts0(3, shape_kpts0, CV_32FC1, kpts0_norm.data);
    int shape_kpts1[] = {1, kpts1_norm.rows, 2};
    Mat blob_kpts1(3, shape_kpts1, CV_32FC1, kpts1_norm.data);

    int shape_desc0[] = {1, desc0.rows, desc0.cols};
    Mat blob_desc0(3, shape_desc0, CV_32FC1, desc0.data);
    int shape_desc1[] = {1, desc1.rows, desc1.cols};
    Mat blob_desc1(3, shape_desc1, CV_32FC1, desc1.data);

    net.setInput(blob_kpts0, "kpts0");
    net.setInput(blob_desc0, "desc0");
    net.setInput(blob_kpts1, "kpts1");
    net.setInput(blob_desc1, "desc1");

    // Fused model only has matches0 and mscores0
    vector<string> outNames = {"matches0", "mscores0"};
    vector<Mat> outs;
    net.forward(outs, outNames);

    for (size_t i = 0; i < outs.size(); i++)
        print_mat_shape(outNames[i], outs[i]);

    // Fused format: matches0 is [1, M, 2] or [M, 2] (pairs of indices)
    Mat matches0 = outs[0];
    Mat mscores0 = outs[1];

    // Handle both [1, M, 2] and [M, 2] shapes
    int M, pair_stride;
    const float* match_data;
    const float* score_data;

    if (matches0.dims == 3)
    {
        // Shape [1, M, 2]
        M = matches0.size[1];
        pair_stride = 2;
        match_data = matches0.ptr<float>(0);
        score_data = mscores0.ptr<float>(0);
    }
    else
    {
        // Shape [M, 2]
        M = matches0.size[0];
        pair_stride = matches0.size[1];  // should be 2
        match_data = matches0.ptr<float>(0);
        score_data = mscores0.ptr<float>(0);
    }

    cout << "Found " << M << " match pairs" << endl;
    cout << "\n--- First 10 match pairs ---" << endl;
    for (int i = 0; i < 10 && i < M; i++)
    {
        cout << "i=" << i
             << " -> (kpt0=" << match_data[i * pair_stride]
             << ", kpt1=" << match_data[i * pair_stride + 1]
             << "), score=" << score_data[i] << endl;
    }
}

void test_model(const string &model_path, bool is_fused)
{
    cout << "\n========== Testing: " << model_path << " ==========" << endl;

    if (!std::filesystem::exists(model_path))
    {
        cerr << "Model does not exist: " << model_path << endl;
        return;
    }

    cout << "Loading model..." << endl;
    dnn::Net net = dnn::readNetFromONNX(model_path, ENGINE);
    net.enableWinograd(false);
    cout << "Model loaded successfully" << endl;

    Mat kpts0, desc0, kpts1, desc1;
    generate_dummy_features(NUM_KPTS_0, IMG_H0, IMG_W0, kpts0, desc0);
    generate_dummy_features(NUM_KPTS_1, IMG_H1, IMG_W1, kpts1, desc1);
    cout << "Dummy data generation complete" << endl;

    if (is_fused)
        run_matcher_fused(net, kpts0, desc0, kpts1, desc1, IMG_H0, IMG_W0, IMG_H1, IMG_W1);
    else
        run_matcher_standard(net, kpts0, desc0, kpts1, desc1, IMG_H0, IMG_W0, IMG_H1, IMG_W1);
}

int main()
{
    print_runtime_info();

    test_model("../model/disk_lightglue.onnx", false);
    test_model("../model/disk_lightglue_flash.onnx", false);
    test_model("../model/disk_lightglue_fused.onnx", true);

    return 0;
}
