#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <random>
#include <filesystem>

using namespace cv;
using namespace std;


const string LG_MODEL_PATH = "../model/superpoint_lightglue_fixed.onnx";
const int DESCRIPTOR_DIM = 256;

const int NUM_KPTS_0 = 500;
const int NUM_KPTS_1 = 500;

const int IMG_H0 = 480, IMG_W0 = 640;
const int IMG_H1 = 480, IMG_W1 = 640;

// ------------------------------
// 归一化关键点 (N,2) -> (1,N,2)
// ------------------------------
Mat normalize_keypoints(const Mat &kpts, int H, int W)
{
    Mat kpts_norm;
    kpts.convertTo(kpts_norm, CV_32F);

    for (int i = 0; i < kpts_norm.rows; i++)
    {
        kpts_norm.at<float>(i, 0) /= (W - 1); // x
        kpts_norm.at<float>(i, 1) /= (H - 1); // y
    }

    // reshape -> (1, N, 2)
    return kpts_norm.reshape(2, 1);
}

// ------------------------------
// 生成随机 keypoints + descriptors
// ------------------------------
void generate_dummy_features(
    int num_kpts,
    int descriptor_dim,
    int H, int W,
    Mat &out_kpts, // (N,2)
    Mat &out_desc  // (N,D)
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
// 运行 LightGlue
// ------------------------------
void run_lightglue(
    dnn::Net &net,
    const Mat &kpts0, const Mat &desc0,
    const Mat &kpts1, const Mat &desc1,
    int H0, int W0,
    int H1, int W1)
{
    cout << "正在运行 LightGlue 推理..." << endl;

    // 归一化关键点
    Mat kpts0_norm = normalize_keypoints(kpts0, H0, W0); // (1,N,2)
    Mat kpts1_norm = normalize_keypoints(kpts1, H1, W1);

    Mat desc0_batch = desc0.reshape(1, 1); // (1,N,D)
    Mat desc1_batch = desc1.reshape(1, 1);

    // 设置输入
    net.setInput(kpts0_norm, "kpts0");
    net.setInput(desc0_batch, "desc0");
    net.setInput(kpts1_norm, "kpts1");
    net.setInput(desc1_batch, "desc1");

    // 输出节点
    vector<string> outNames = {"matches0", "mscores0", "matches1", "mscores1"};
    vector<Mat> outs;
    net.forward(outs, outNames);

    Mat matches0 = outs[0].reshape(1, outs[0].total());
    Mat mscores0 = outs[1].reshape(1, outs[1].total());

    cout << "推理完成。" << endl;

    cout << "输出 matches0 形状: " << matches0.total() << endl;
    cout << "输出 mscores0 形状: " << mscores0.total() << endl;

    // 统计有效匹配
    int valid = 0;
    for (int i = 0; i < matches0.total(); i++)
    {
        if (matches0.at<float>(i) > -1)
            valid++;
    }
    cout << "找到 " << valid << " 个有效匹配。" << endl;

    // 打印前 10 个结果
    cout << "\n--- 前 10 个匹配结果 ---" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << "i=" << i
             << " -> match=" << matches0.at<float>(i)
             << ", score=" << mscores0.at<float>(i) << endl;
    }
}

// ------------------------------
// 主程序
// ------------------------------
int main()
{

    cout << "OpenCV version: " << CV_VERSION << endl;

    if (!std::filesystem::exists(LG_MODEL_PATH))
    {
        cerr << "模型不存在: " << LG_MODEL_PATH << endl;
        return -1;
    }

    // 载入 ONNX
    cout << "加载 LightGlue 模型: " << LG_MODEL_PATH << endl;
    dnn::Net net = dnn::readNetFromONNX(LG_MODEL_PATH, dnn::ENGINE_CLASSIC);
    net.enableWinograd(false);// 关闭 Winograd 优化以避免潜在问题
    cout << "模型加载成功！" << endl;

    // 生成假数据
    Mat kpts0, desc0, kpts1, desc1;
    generate_dummy_features(NUM_KPTS_0, DESCRIPTOR_DIM, IMG_H0, IMG_W0, kpts0, desc0);
    generate_dummy_features(NUM_KPTS_1, DESCRIPTOR_DIM, IMG_H1, IMG_W1, kpts1, desc1);

    cout << "假数据生成完成。" << endl;

    // 运行推理
    run_lightglue(net, kpts0, desc0, kpts1, desc1, IMG_H0, IMG_W0, IMG_H1, IMG_W1);

    return 0;
}
