#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdlib>
#include <iostream>
#include <random>
#include <filesystem>

using namespace cv;
using namespace std;

const int IMG_H = 480, IMG_W = 640;
const int ENGINE = dnn::ENGINE_ORT;

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
    const char* traceEnv = std::getenv("DNN_TRACE_ALL");
    cout << "DNN_TRACE_ALL=" << (traceEnv ? traceEnv : "<unset>") << endl;
    cout << "Image input size: " << IMG_W << "x" << IMG_H << endl;
    cout << "==============================================" << endl;
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

void generate_random_image_blob(Mat &blob)
{
    int shape[] = {1, 3, IMG_H, IMG_W};
    blob = Mat(4, shape, CV_32F);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float* data = blob.ptr<float>(0);
    size_t total = blob.total();
    for (size_t i = 0; i < total; i++)
        data[i] = dist(rng);

    cout << "Generated random image blob: [1 x 3 x " << IMG_H << " x " << IMG_W << "]" << endl;
}

void print_output_samples(const string &name, const Mat &out_mat, int max_samples = 10)
{
    int type = out_mat.type();
    int dims = out_mat.dims;

    // Flatten for sampling: create a 1D view of the data
    if (dims == 0)
    {
        cout << "  " << name << ": empty Mat" << endl;
        return;
    }

    cout << "  " << name << " samples: ";
    int count = min((int)out_mat.total(), max_samples);

    // Handle different types
    if (type == CV_32F || type == CV_32FC1)
    {
        const float* data = out_mat.ptr<float>(0);
        for (int i = 0; i < count; i++)
            cout << data[i] << " ";
    }
    else if (type == CV_64F)
    {
        const double* data = out_mat.ptr<double>(0);
        for (int i = 0; i < count; i++)
            cout << data[i] << " ";
    }
    else if (type == CV_32S)
    {
        const int32_t* data = out_mat.ptr<int32_t>(0);
        for (int i = 0; i < count; i++)
            cout << data[i] << " ";
    }
    else if (type == CV_64S)
    {
        const int64_t* data = out_mat.ptr<int64_t>(0);
        for (int i = 0; i < count; i++)
            cout << data[i] << " ";
    }
    else if (type == CV_8U)
    {
        const uint8_t* data = out_mat.ptr<uint8_t>(0);
        for (int i = 0; i < count; i++)
            cout << (int)data[i] << " ";
    }
    else
    {
        // Fallback: try to interpret as float
        cout << "(unsupported type " << type << ")";
    }
    cout << endl;
}

void print_keypoint_samples(const Mat &kpts_mat, int max_samples = 5)
{
    int N = kpts_mat.total() / 2;
    cout << "  Keypoint count (from shape): " << N << endl;

    int count = min(N, max_samples);
    cout << "  Keypoint samples (x, y):" << endl;

    int type = kpts_mat.type();
    if (type == CV_32F || type == CV_32FC1)
    {
        const float* data = kpts_mat.ptr<float>(0);
        for (int i = 0; i < count; i++)
            cout << "    #" << i << ": (" << data[i * 2] << ", " << data[i * 2 + 1] << ")" << endl;
    }
    else if (type == CV_64F)
    {
        const double* data = kpts_mat.ptr<double>(0);
        for (int i = 0; i < count; i++)
            cout << "    #" << i << ": (" << data[i * 2] << ", " << data[i * 2 + 1] << ")" << endl;
    }
    else if (type == CV_32S)
    {
        const int32_t* data = kpts_mat.ptr<int32_t>(0);
        for (int i = 0; i < count; i++)
            cout << "    #" << i << ": (" << data[i * 2] << ", " << data[i * 2 + 1] << ")" << endl;
    }
    else if (type == CV_64S)
    {
        const int64_t* data = kpts_mat.ptr<int64_t>(0);
        for (int i = 0; i < count; i++)
            cout << "    #" << i << ": (" << data[i * 2] << ", " << data[i * 2 + 1] << ")" << endl;
    }
    else
    {
        cout << "  (unsupported keypoint type " << type << ")" << endl;
        print_output_samples("keypoints_raw", kpts_mat, 10);
    }
}

void test_extractor(const string &model_path)
{
    cout << "\n========== Testing: " << model_path << " ==========" << endl;

    if (!std::filesystem::exists(model_path))
    {
        cerr << "Model does not exist: " << model_path << endl;
        return;
    }

    cout << "Loading model..." << endl;
    dnn::Net net = dnn::readNetFromONNX(model_path, ENGINE);
    cout << "Model loaded successfully" << endl;

    // Generate random RGB image and run inference
    Mat blob;
    generate_random_image_blob(blob);
    net.setInput(blob, "image");

    vector<string> outNames = net.getUnconnectedOutLayersNames();
    cout << "Output layer names (" << outNames.size() << "): ";
    for (const auto &n : outNames) cout << n << " ";
    cout << endl;

    vector<Mat> outs;
    cout << "Running inference..." << endl;
    net.forward(outs, outNames);
    cout << "Inference complete" << endl;

    // Print all outputs
    for (size_t i = 0; i < outs.size(); i++)
    {
        cout << "\n--- Output[" << i << "]: " << outNames[i] << " ---" << endl;
        print_mat_shape(outNames[i], outs[i]);

        // Detect keypoint tensor: last dim == 2
        if (outs[i].dims >= 2 && outs[i].size[outs[i].dims - 1] == 2)
        {
            print_keypoint_samples(outs[i]);
        }
        else
        {
            print_output_samples(outNames[i], outs[i]);
        }
    }
}

int main()
{
    print_runtime_info();

    test_extractor("../model/disk.onnx");
    test_extractor("../model/disk_1024.onnx");

    return 0;
}
