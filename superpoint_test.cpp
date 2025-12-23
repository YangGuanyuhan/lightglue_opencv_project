#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <random>
#include <filesystem>

// 打印多维 Mat 的 shape
void printShape(const cv::Mat &mat)
{
    std::cout << "  Output shape: [";
    for (int i = 0; i < mat.dims; i++)
    {
        std::cout << mat.size[i];
        if (i + 1 < mat.dims)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

int main()
{
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    // 固定 SuperPoint 输入尺寸
    const int H = 480;
    const int W = 640;

    std::string model_path = "../model/superpoint_fixed_480x640.onnx";

    if (!std::filesystem::exists(model_path))
    {
        std::cerr << "模型不存在: " << model_path << std::endl;
        return -1;
    }

    std::cout << "Loading SuperPoint model: " << model_path << " ...\n";

    cv::dnn::Net net;
    try
    {
        net = cv::dnn::readNetFromONNX(model_path);
    }
    catch (cv::Exception &e)
    {
        std::cerr << "ERROR loading model: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "SuperPoint 模型加载成功\n";

    // ==================================
    // 生成随机输入图像 (1×1×H×W)
    // ==================================
    cv::Mat rand_img(H, W, CV_32FC1);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);

    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            rand_img.at<float>(i, j) = dist(gen);

    std::cout << "已生成随机图像\n";

    // create input blob
    cv::Mat inputBlob = cv::dnn::blobFromImage(
        rand_img, 1.0 / 255.0, cv::Size(W, H),
        cv::Scalar(), false, false, CV_32F);

    std::cout << "Input blob shape = [";
    for (int i = 0; i < inputBlob.dims; i++)
        std::cout << inputBlob.size[i] << (i + 1 < inputBlob.dims ? "," : "");
    std::cout << "]\n";

    net.setInput(inputBlob, "image");

    // ===========================
    // 按层执行 forward 调试
    // ===========================
    std::vector<std::string> layer_names = net.getLayerNames();
    std::cout << "模型层数: " << layer_names.size() << "\n";

    for (size_t i = 0; i < layer_names.size(); i++)
    {
        const std::string &name = layer_names[i];
        int layer_id = net.getLayerId(name);
        cv::Ptr<cv::dnn::Layer> layer = net.getLayer(layer_id);

        std::cout << "\n====================================\n";
        std::cout << "[" << i + 1 << "/" << layer_names.size() << "] Layer\n";
        std::cout << "Name : " << name << "\n";
        std::cout << "Type : " << layer->type << "\n";

        try
        {
            cv::Mat out = net.forward(name);
            printShape(out);
        }
        catch (cv::Exception &e)
        {
            std::cout << "ERROR during forward!\n";
            std::cout << "Message: " << e.what() << "\n";
            break;
        }
    }

    std::cout << "\n逐层调试结束！\n";

    // ===========================
    // 输出最终结果（正常推理）
    // ===========================
    std::vector<std::string> output_names = {
        "keypoints",
        "scores",
        "descriptors"};

    std::vector<cv::Mat> outputs;
    net.forward(outputs, output_names);

    cv::Mat keypoints = outputs[0];
    cv::Mat scores = outputs[1];
    cv::Mat descriptors = outputs[2];

    std::cout << "\n===== SuperPoint 最终推理成功 =====\n";
    printShape(keypoints);
    printShape(scores);
    printShape(descriptors);

    return 0;
}
