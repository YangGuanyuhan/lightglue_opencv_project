#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
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
    std::string model_path = "../model/mnist-8.onnx";

    if (!std::filesystem::exists(model_path))
    {
        std::cout << "Error: Cannot find file " << model_path << std::endl;
        return 0;
    }

    std::cout << "Loading model: " << model_path << " ..." << std::endl;

    cv::dnn::Net net;
    try
    {
        net = cv::dnn::readNetFromONNX(model_path, cv::dnn::ENGINE_CLASSIC);
    }
    catch (cv::Exception &e)
    {
        std::cout << "Failed to load model. Error: " << e.what() << std::endl;
        return 0;
    }

    // Prepare MNIST input: (1,1,28,28)
    cv::Mat img(28, 28, CV_32F);
    cv::randu(img, 0.0f, 1.0f);

    cv::Mat blob = cv::dnn::blobFromImage(
        img, 1.0f, cv::Size(28, 28),
        cv::Scalar(), false, false, CV_32F);

    net.setInput(blob, "Input3");

    // ---- Layer-by-layer forward ----
    std::vector<std::string> layer_names = net.getLayerNames();
    std::cout << "Model has " << layer_names.size() << " layers.\n";

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

    std::cout << "\nDebug finished." << std::endl;
    return 0;
}
