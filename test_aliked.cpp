#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <vector>

using namespace cv;
using namespace std;

const string ALIKED_MODEL_PATH = "../model/aliked-n16rot-top1k-640.onnx";
const string IMAGE_PATH = "../images/image1.jpg";
const int TARGET_SIZE = 640;
const int ENGINE = dnn::ENGINE_NEW;

static const char *engine_to_string(int engine)
{
    if (engine == dnn::ENGINE_NEW)
        return "ENGINE_NEW";
    if (engine == dnn::ENGINE_ORT)
        return "ENGINE_ORT";
    return "UNKNOWN_ENGINE";
}

void test_aliked()
{
    cout << "OpenCV version " << CV_VERSION << endl;

    // 1 Check if files exist
    if (!std::filesystem::exists(ALIKED_MODEL_PATH)) {
        cerr << "Model does not exist " << ALIKED_MODEL_PATH << endl;
        return;
    }
    if (!std::filesystem::exists(IMAGE_PATH)) {
        cerr << "Image does not exist " << IMAGE_PATH << endl;
        return;
    }

    // 2 Load model
    cout << "Loading ALIKED model from " << ALIKED_MODEL_PATH << " " << endl;
    dnn::Net net = dnn::readNetFromONNX(ALIKED_MODEL_PATH, ENGINE);
    // net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    cout << "Model loaded successfully in " << engine_to_string(ENGINE)  << endl;

    // 3 Read and preprocess image
    Mat img = imread(IMAGE_PATH);
    if (img.empty()) {
        cerr << "Unable to read image" << endl;
        return;
    }

    // Save resized image for visualization
    Mat vis_img;
    resize(img, vis_img, Size(TARGET_SIZE, TARGET_SIZE));

    // blobFromImage automatically completes Resize 1/255.0 normalization BGR to RGB HWC to NCHW
    Mat blob = dnn::blobFromImage(img, 1.0 / 255.0, Size(TARGET_SIZE, TARGET_SIZE), Scalar(), true, false);
    cout << "Blob shape " << blob.size[0] << "x" << blob.size[1] << "x" << blob.size[2] << "x" << blob.size[3] << endl;

    // 4 Run inference
    net.setInput(blob, "image"); // Set input input name is image from Python
    
    // Output nodes are keypoints descriptors scores
    vector<string> outNames = {"keypoints", "descriptors", "scores"};
    vector<Mat> outs;
    
    cout << "Running inference" << endl;
    net.forward(outs, outNames);
    cout << "Inference successful" << endl;

    // 5 Parse output
    Mat kpts_mat = outs[0]; // keypoints
    Mat desc_mat = outs[1]; // descriptors
    Mat scores_mat = outs[2]; // scores

    // Ensure continuous memory for safe pointer access
    if (!kpts_mat.isContinuous()) kpts_mat = kpts_mat.clone();

    // total represents total number of elements
    int num_kpts = kpts_mat.total() / 2; 
    float* kpts_data = (float*)kpts_mat.data;

    cout << "\nExtracted " << num_kpts << " keypoints" << endl;
    cout << "Sample keypoints first 1 " << kpts_data[0] << " " << kpts_data[1] << " " << endl;

    // 6 Handle normalized coordinates and draw
    // Check if conversion from -1 1 and scaling to actual pixels is needed
    double minVal, maxVal;
    minMaxIdx(kpts_mat, &minVal, &maxVal);
    
    bool is_normalized = (maxVal <= 1.0);
    bool needs_shift = (minVal < 0.0);

    if (is_normalized) {
        cout << "Detected normalized coordinates Scaling to image size " << TARGET_SIZE << "x" << TARGET_SIZE << " " << endl;
    }

    int drawn_count = 0;
    for (int i = 0; i < num_kpts; i++) {
        float x = kpts_data[i * 2];
        float y = kpts_data[i * 2 + 1];

        // Restore real coordinates
        if (is_normalized) {
            if (needs_shift) {
                x = (x + 1.0f) / 2.0f;
                y = (y + 1.0f) / 2.0f;
            }
            x *= TARGET_SIZE;
            y *= TARGET_SIZE;
        }

        int ix = cvRound(x);
        int iy = cvRound(y);

        // Draw image prevent out of bounds
        if (ix >= 0 && ix < vis_img.cols && iy >= 0 && iy < vis_img.rows) {
            circle(vis_img, Point(ix, iy), 3, Scalar(0, 255, 0), -1);
            drawn_count++;
        }
    }

    cout << "Scaled keypoints first 1 " << cvRound(kpts_data[0] * (is_normalized ? (needs_shift ? 0.5 : 1) * TARGET_SIZE : 1) + (needs_shift ? TARGET_SIZE/2.0 : 0)) << " " << " " << " " << endl;
    cout << "Actually drawn " << drawn_count << " points within boundaries" << endl;

    // 7 Save results
    string out_filename = "cpp_output_features.jpg";
    imwrite(out_filename, vis_img);
    cout << "Image saved as " << out_filename << " " << endl;
}

int main()
{
    test_aliked();
    return 0;
}