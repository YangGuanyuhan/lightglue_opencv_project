#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

const string XFEAT_MODEL_PATH = "../model/xfeat.onnx";
const string IMAGE_PATH = "../image.png";
const int TARGET_SIZE = 640;
const float THRESHOLD = 0.5f;

void test_xfeat()
{
    cout << "OpenCV version " << CV_VERSION << endl;

    // 1 Check if files exist
    if (!std::filesystem::exists(XFEAT_MODEL_PATH)) {
        cerr << "Model does not exist " << XFEAT_MODEL_PATH << endl;
        return;
    }
    if (!std::filesystem::exists(IMAGE_PATH)) {
        cerr << "Image does not exist " << IMAGE_PATH << endl;
        return;
    }

    // 2 Load model
    cout << "Loading XFeat model from " << XFEAT_MODEL_PATH << " " << endl;
    dnn::Net net = dnn::readNetFromONNX(XFEAT_MODEL_PATH);
    // net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    cout << "Model loaded successfully" << endl;

    // 3 Read and preprocess image
    Mat img = imread(IMAGE_PATH);
    if (img.empty()) {
        cerr << "Unable to read image" << endl;
        return;
    }

    int h = img.rows;
    int w = img.cols;

    // Grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Calculate Scale and Resize
    float scale = (float)TARGET_SIZE / std::max(h, w);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    Mat img_resized;
    resize(gray, img_resized, Size(new_w, new_h));

    // Padding to 640x640 aligned top left fill 0
    Mat padded = Mat::zeros(TARGET_SIZE, TARGET_SIZE, CV_8UC1);
    img_resized.copyTo(padded(Rect(0, 0, new_w, new_h)));

    // blobFromImage automatically completes 1/255.0 normalization HWC to NCHW
    // Single channel gray image input becomes 1 1 640 640
    Mat blob = dnn::blobFromImage(padded, 1.0 / 255.0, Size(TARGET_SIZE, TARGET_SIZE), Scalar(), false, false);
    cout << "Blob shape " << blob.size[0] << "x" << blob.size[1] << "x" << blob.size[2] << "x" << blob.size[3] << endl;

    // 4 Run inference
    net.setInput(blob); 
    
    // Get all output node names automatically
    vector<string> outNames = net.getUnconnectedOutLayersNames();
    vector<Mat> outs;
    
    cout << "Running inference" << endl;
    net.forward(outs, outNames);
    cout << "Inference successful" << endl;

    // 5 Parse output
    // XFeat score_map should be 1 1 80 80 other feature layers have more channels
    // Find Mat with channel count size 1 1 as score_map
    Mat score_map;
    for (const auto& out : outs) {
        if (out.dims == 4 && out.size[1] == 1) {
            score_map = out;
            break;
        }
    }
    
    // Fallback to 3rd output according to Python logic
    if (score_map.empty() && outs.size() >= 3) {
        score_map = outs[2]; 
    }

    // Ensure continuous memory
    if (!score_map.isContinuous()) score_map = score_map.clone();

    int score_h = score_map.size[2]; // Usually 80
    int score_w = score_map.size[3]; // Usually 80
    float stride = (float)TARGET_SIZE / score_h;
    
    const float* score_data = (const float*)score_map.data;

    // 6 Extract keypoints and map back to original image coordinates
    vector<Point> keypoints;
    for (int y = 0; y < score_h; y++) {
        for (int x = 0; x < score_w; x++) {
            // Locate value in 1D array y x
            float score = score_data[y * score_w + x];
            
            if (score > THRESHOLD) {
                // Coordinate restoration logic consistent with Python code int x * stride / scale
                int px = static_cast<int>(x * stride / scale);
                int py = static_cast<int>(y * stride / scale);

                // Boundary protection retain points in valid image area
                if (px >= 0 && px < w && py >= 0 && py < h) {
                    keypoints.push_back(Point(px, py));
                }
            }
        }
    }

    cout << "\nExtracted keypoints " << keypoints.size() << endl;
    if (!keypoints.empty()) {
        cout << "Sample keypoints first 1 " << keypoints[0].x << " " << keypoints[0].y << " " << endl;
    }

    // 7 Draw keypoints and save results
    Mat vis = img.clone();
    for (const auto& pt : keypoints) {
        circle(vis, pt, 2, Scalar(0, 255, 0), -1);
    }

    string out_filename = "cpp_xfeat_output.jpg";
    imwrite(out_filename, vis);
    cout << "Image saved as " << out_filename << " " << endl;
}

int main()
{
    test_xfeat();
    return 0;
}