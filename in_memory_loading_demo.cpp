#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
//#include <android/asset_manager.h> 

int main()
{
    const std::string model_path = "../model/superpoint_lightglue.onnx";
    std::vector<char> model_bytes;

    // =====================================================================
    // Option 1:  ifstream::read 放在heap
    // =====================================================================
    
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (file) {
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        model_bytes.resize(size);
        if (!file.read(model_bytes.data(), size)) {
            std::cerr << "Option 1: Failed to read data" << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Option 1: Failed to open file" << std::endl;
    }
    

    // =====================================================================
    // Option 2: 使用 std::istreambuf_iterator (代码最简洁，C++ 风格拉满)
    // =====================================================================
    /*
    std::ifstream file(model_path, std::ios::binary);
    if (file) {
        // 直接利用迭代器将文件流分配给 vector，不需要手动求大小
        model_bytes.assign(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
        file.close();
    } else {
        std::cerr << "Option 2: Failed to open file" << std::endl;
    }
    */

    // =====================================================================
    // Option 3: 使用 std::stringstream (适合较小文件，大文件会有额外的内存拷贝开销)
    // =====================================================================
    /*
    std::ifstream file(model_path, std::ios::binary);
    if (file) {
        std::stringstream buffer;
        buffer << file.rdbuf(); // 将文件 buffer 读入 stringstream
        std::string str_data = buffer.str();
        model_bytes.assign(str_data.begin(), str_data.end());
        file.close();
    } else {
        std::cerr << "Option 3: Failed to open file" << std::endl;
    }
    */

   

    // =====================================================================
    // Option 5: 针对 Android NDK 的 AAssetManager 真实操作 (如果是做安卓端部署)
    // 假设你已经通过 JNI 传进来了 AAssetManager* mgr 对象
    // =====================================================================
    /*
    // AAssetManager* mgr = ... ; // 从 JNI 环境获取
    // AAsset* asset = AAssetManager_open(mgr, model_path.c_str(), AASSET_MODE_BUFFER);
    // if (asset) {
    //     off_t size = AAsset_getLength(asset);
    //     model_bytes.resize(size);
    //     AAsset_read(asset, model_bytes.data(), size);
    //     AAsset_close(asset);
    // } else {
    //     std::cerr << "Option 5: Failed to open asset" << std::endl;
    // }
    */

    
    if (!model_bytes.empty())
    {
        std::cout << "Successfully retrieved model memory data stream! Total bytes: " << model_bytes.size() << std::endl;

       
         cv::dnn::Net net = cv::dnn::readNetFromONNX(model_bytes.data(), model_bytes.size());
    }
    else
    {
        std::cerr << "Failed to retrieve model memory data stream!" << std::endl;
    }
    std::cout << "Done." << std::endl;

    return 0;
}