#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <numeric>

using namespace cv;
using namespace std;
using namespace std::chrono;

// ======================== Config ========================
const string DISK_MODEL       = "../model/disk.onnx";
const string DISK_LG_MODEL    = "../model/disk_lightglue.onnx";
const string ALIKED_MODEL     = "../model/aliked-n16rot-top1k-640.onnx";
const string ALIKED_LG_MODEL  = "../model/aliked_lightglue.onnx";
const string IMG_0_PATH       = "../images/image1.jpg";
const string IMG_1_PATH       = "../images/image2.jpg";

const int    ALIKED_TARGET    = 640;
const int    DESCRIPTOR_DIM   = 128;   // both DISK and ALIKED use 128-D
const float  MATCH_THRESHOLD  = 0.2f;
const int    ENGINE           = dnn::ENGINE_NEW;

// ======================== Utilities ========================
static const char* engine_str(int e) {
    if (e == dnn::ENGINE_NEW) return "ENGINE_NEW";
    if (e == dnn::ENGINE_ORT) return "ENGINE_ORT";
    return "UNKNOWN";
}

static void print_header(const string &title) {
    cout << "\n============================================================" << endl;
    cout << "  " << title << endl;
    cout << "============================================================" << endl;
}

// ======================== DISK Extraction ========================
// DISK extractor outputs:
//   outs[0] = keypoints [1,N,2] int16 (type=11 due to stride padding)
//   outs[1] = scores     [1,N]   float32
//   outs[2] = descriptors[1,N,128] float32
bool extract_disk(dnn::Net &net, const Mat &img,
                  Mat &out_kpts, Mat &out_desc, Mat &out_scores,
                  int &out_img_h, int &out_img_w)
{
    out_img_h = img.rows;
    out_img_w = img.cols;

    // DISK takes RGB [1,3,H,W] float32, no normalization needed (blobFromImage does /255)
    Mat blob = dnn::blobFromImage(img, 1.0, Size(), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outs;
    net.forward(outs);

    if (outs.size() < 3) {
        cerr << "DISK extractor: expected >=3 outputs, got " << outs.size() << endl;
        return false;
    }

    // --- Parse keypoints (int16 with stride padding, type=11) ---
    Mat kpts_raw = outs[0];
    int N = kpts_raw.size[1];  // [1, N, 2]

    out_kpts = Mat(N, 2, CV_32F);
    int kpts_type = kpts_raw.type();
    cout << "  DISK keypoints raw type=" << kpts_type << " dims=" << kpts_raw.dims
         << " size=[1," << N << ",2]" << endl;

    if (kpts_type == 11) {
        // int16 data padded to 8-byte stride; read via int64 pointer
        const int64_t* data = kpts_raw.ptr<int64_t>(0);
        for (int i = 0; i < N; i++) {
            out_kpts.at<float>(i, 0) = (float)(int16_t)(data[i * 2] & 0xFFFF);
            out_kpts.at<float>(i, 1) = (float)(int16_t)(data[i * 2 + 1] & 0xFFFF);
        }
    } else if (kpts_type == CV_32F || kpts_type == CV_32FC1) {
        kpts_raw.reshape(1, N).copyTo(out_kpts);
    } else {
        cerr << "DISK: unsupported keypoint type " << kpts_type << endl;
        return false;
    }

    // --- Scores ---
    out_scores = outs[1].reshape(1, N).clone();

    // --- Descriptors ---
    out_desc = outs[2].reshape(1, N).clone();  // [1,N,128] -> [N,128]

    cout << "  DISK extracted " << N << " keypoints, desc shape=["
         << out_desc.rows << " x " << out_desc.cols << "]" << endl;

    return true;
}

// Subsample top-K keypoints by score (to avoid OOM in LightGlue)
void subsample_top_k(Mat &kpts, Mat &desc, Mat &scores, int max_kpts)
{
    int N = kpts.rows;
    if (N <= max_kpts) return;

    // Build index array sorted by score descending
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return scores.at<float>(a) > scores.at<float>(b);
    });

    Mat new_kpts(max_kpts, 2, CV_32F);
    Mat new_desc(max_kpts, desc.cols, CV_32F);
    Mat new_scores(max_kpts, 1, CV_32F);

    for (int i = 0; i < max_kpts; i++) {
        int src = idx[i];
        new_kpts.at<float>(i, 0) = kpts.at<float>(src, 0);
        new_kpts.at<float>(i, 1) = kpts.at<float>(src, 1);
        for (int d = 0; d < desc.cols; d++)
            new_desc.at<float>(i, d) = desc.at<float>(src, d);
        new_scores.at<float>(i) = scores.at<float>(src);
    }

    kpts   = new_kpts;
    desc   = new_desc;
    scores = new_scores;
    cout << "  Subsampled to top " << max_kpts << " keypoints (by score)" << endl;
}

// ======================== DISK LightGlue Matching ========================
// Standard format output:
//   matches0 [1,N0] int64  ( -1 = unmatched )
//   mscores0 [1,N0] float32
//   matches1 [1,N1] int64
//   mscores1 [1,N1] float32
bool match_disk_lightglue(dnn::Net &net,
                          const Mat &kpts0, const Mat &desc0,
                          const Mat &kpts1, const Mat &desc1,
                          int H0, int W0, int H1, int W1,
                          vector<Point2f> &out_pts0, vector<Point2f> &out_pts1,
                          vector<DMatch> &out_matches, vector<float> &out_scores)
{
    // DISK outputs pixel coordinates, normalize to [0,1]
    Mat kpts0_norm = kpts0.clone();
    Mat kpts1_norm = kpts1.clone();
    for (int i = 0; i < kpts0_norm.rows; i++) {
        kpts0_norm.at<float>(i, 0) /= (W0 - 1);
        kpts0_norm.at<float>(i, 1) /= (H0 - 1);
    }
    for (int i = 0; i < kpts1_norm.rows; i++) {
        kpts1_norm.at<float>(i, 0) /= (W1 - 1);
        kpts1_norm.at<float>(i, 1) /= (H1 - 1);
    }

    int N0 = kpts0.rows, N1 = kpts1.rows;

    // Build 3D blobs
    int sh_k0[] = {1, N0, 2};
    Mat bk0(3, sh_k0, CV_32FC1, kpts0_norm.data);
    int sh_d0[] = {1, N0, DESCRIPTOR_DIM};
    Mat bd0(3, sh_d0, CV_32FC1, desc0.data);
    int sh_k1[] = {1, N1, 2};
    Mat bk1(3, sh_k1, CV_32FC1, kpts1_norm.data);
    int sh_d1[] = {1, N1, DESCRIPTOR_DIM};
    Mat bd1(3, sh_d1, CV_32FC1, desc1.data);

    net.setInput(bk0, "kpts0");
    net.setInput(bd0, "desc0");
    net.setInput(bk1, "kpts1");
    net.setInput(bd1, "desc1");

    // Standard format: request all 4 outputs (ENGINE_NEW requires exact match)
    vector<string> outNames = {"matches0", "mscores0", "matches1", "mscores1"};
    vector<Mat> outs;
    net.forward(outs, outNames);

    // Parse matches0: [1, N0] with -1 sentinel
    Mat matches0 = outs[0].reshape(1, outs[0].total());
    Mat mscores0 = outs[1].reshape(1, outs[1].total());

    int match_type = matches0.type();
    cout << "  DISK LightGlue matches0 type=" << match_type << " total=" << matches0.total() << endl;

    // Store pixel coordinates for visualization
    out_pts0.resize(N0);
    out_pts1.resize(N1);
    for (int i = 0; i < N0; i++) {
        out_pts0[i].x = kpts0.at<float>(i, 0);
        out_pts0[i].y = kpts0.at<float>(i, 1);
    }
    for (int i = 0; i < N1; i++) {
        out_pts1[i].x = kpts1.at<float>(i, 0);
        out_pts1[i].y = kpts1.at<float>(i, 1);
    }

    // Parse matches
    int valid = 0;
    for (int i = 0; i < matches0.total(); i++) {
        double m;
        if (match_type == 11 || match_type == CV_64S || match_type == CV_64F)
            m = (double)matches0.ptr<int64_t>(0)[i];
        else if (match_type == CV_32S)
            m = (double)matches0.ptr<int32_t>(0)[i];
        else
            m = (double)matches0.ptr<float>(0)[i];

        float score = mscores0.at<float>(i);
        if (m > -0.5 && score > MATCH_THRESHOLD) {
            out_matches.emplace_back(i, (int)m, 1.0f - score);
            out_scores.push_back(score);
            valid++;
        }
    }

    cout << "  DISK LightGlue: " << valid << " valid matches (threshold=" << MATCH_THRESHOLD << ")" << endl;
    return true;
}

// ======================== ALIKED Extraction ========================
// ALIKED outputs normalized [-1,1] float32 coordinates
bool extract_aliked(dnn::Net &net, const Mat &img,
                    Mat &out_kpts, Mat &out_desc, Mat &out_scores)
{
    Mat blob = dnn::blobFromImage(img, 1.0 / 255.0, Size(ALIKED_TARGET, ALIKED_TARGET),
                                  Scalar(), true, false);
    net.setInput(blob, "image");

    vector<string> outNames = {"keypoints", "descriptors", "scores"};
    vector<Mat> outs;
    net.forward(outs, outNames);

    out_kpts   = outs[0].clone();
    out_desc   = outs[1].clone();
    out_scores = outs[2].clone();

    if (!out_kpts.isContinuous())   out_kpts   = out_kpts.clone();
    if (!out_desc.isContinuous())   out_desc   = out_desc.clone();
    if (!out_scores.isContinuous()) out_scores = out_scores.clone();

    int N = out_kpts.rows;
    cout << "  ALIKED extracted " << N << " keypoints, kpts type=" << out_kpts.type()
         << " desc shape=[" << out_desc.rows << " x " << out_desc.cols << "]" << endl;

    return true;
}

// ======================== ALIKED LightGlue Matching ========================
// Fused format output: matches0 [M,2] int64, mscores0 [M] float32
bool match_aliked_lightglue(dnn::Net &net,
                            const Mat &kpts0, const Mat &desc0,
                            const Mat &kpts1, const Mat &desc1,
                            int num0, int num1,
                            vector<Point2f> &out_pts0, vector<Point2f> &out_pts1,
                            vector<DMatch> &out_matches, vector<float> &out_scores,
                            int imgW0, int imgH0, int imgW1, int imgH1)
{
    int sh_k0[] = {1, num0, 2};
    Mat bk0(3, sh_k0, CV_32FC1, (void*)kpts0.ptr<float>());
    int sh_d0[] = {1, num0, kpts0.cols > 2 ? kpts0.cols : DESCRIPTOR_DIM};
    // Actually desc0 is 128-D
    int sh_d0_fixed[] = {1, num0, desc0.cols};
    Mat bd0(3, sh_d0_fixed, CV_32FC1, (void*)desc0.ptr<float>());

    int sh_k1[] = {1, num1, 2};
    Mat bk1(3, sh_k1, CV_32FC1, (void*)kpts1.ptr<float>());
    int sh_d1_fixed[] = {1, num1, desc1.cols};
    Mat bd1(3, sh_d1_fixed, CV_32FC1, (void*)desc1.ptr<float>());

    net.setInput(bk0, "kpts0");
    net.setInput(bd0, "desc0");
    net.setInput(bk1, "kpts1");
    net.setInput(bd1, "desc1");

    vector<string> outNames = {"matches0", "mscores0"};
    vector<Mat> outs;
    net.forward(outs, outNames);

    // Fused format: matches0 = [M, 2]
    Mat matches0 = outs[0].clone();
    if (!matches0.isContinuous()) matches0 = matches0.clone();
    matches0.convertTo(matches0, CV_32F);

    Mat mscores0 = outs[1].clone();
    if (!mscores0.isContinuous()) mscores0 = mscores0.clone();
    mscores0 = mscores0.reshape(1, mscores0.total());

    // Convert normalized [-1,1] to pixel coords
    out_pts0.resize(num0);
    out_pts1.resize(num1);
    for (int i = 0; i < num0; i++) {
        out_pts0[i].x = (kpts0.at<float>(i, 0) + 1.0f) / 2.0f * (imgW0 - 1);
        out_pts0[i].y = (kpts0.at<float>(i, 1) + 1.0f) / 2.0f * (imgH0 - 1);
    }
    for (int i = 0; i < num1; i++) {
        out_pts1[i].x = (kpts1.at<float>(i, 0) + 1.0f) / 2.0f * (imgW1 - 1);
        out_pts1[i].y = (kpts1.at<float>(i, 1) + 1.0f) / 2.0f * (imgH1 - 1);
    }

    int total = 0;
    int M = matches0.rows;
    for (int i = 0; i < M; i++) {
        int idx0 = (int)matches0.at<float>(i, 0);
        int idx1 = (int)matches0.at<float>(i, 1);
        float score = mscores0.at<float>(i);
        if (idx0 >= 0 && idx1 >= 0 && score > MATCH_THRESHOLD) {
            out_matches.emplace_back(idx0, idx1, 1.0f - score);
            out_scores.push_back(score);
            total++;
        }
    }

    cout << "  ALIKED LightGlue: " << total << " valid matches (threshold=" << MATCH_THRESHOLD << ")" << endl;
    return true;
}

// ======================== Draw & Save ========================
void draw_and_save(const Mat &img0, const Mat &img1,
                   const vector<Point2f> &pts0, const vector<Point2f> &pts1,
                   const vector<DMatch> &matches,
                   const string &filename, const string &title)
{
    vector<KeyPoint> kp0, kp1;
    for (const auto &p : pts0) kp0.emplace_back(p.x, p.y, 1.0f);
    for (const auto &p : pts1) kp1.emplace_back(p.x, p.y, 1.0f);

    // Resize img1 to match img0 height for drawMatches
    Mat img1_resized;
    if (img1.rows != img0.rows) {
        float scale = (float)img0.rows / img1.rows;
        resize(img1, img1_resized, Size((int)(img1.cols * scale), img0.rows));
    } else {
        img1_resized = img1;
    }

    // Recompute pts1 for resized image
    vector<KeyPoint> kp1_resized;
    if (img1.rows != img0.rows) {
        float scale = (float)img0.rows / img1.rows;
        for (const auto &p : pts1)
            kp1_resized.emplace_back(p.x * scale, p.y * scale, 1.0f);
    } else {
        kp1_resized = kp1;
    }

    Mat vis;
    drawMatches(img0, kp0, img1_resized, kp1_resized, matches, vis,
                Scalar(0, 255, 0), Scalar(0, 0, 255),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imwrite(filename, vis);
    cout << "  Saved: " << filename << " (" << vis.cols << "x" << vis.rows << ")" << endl;

    // Skip GUI on WSL — just save to disk
}

// ======================== Main ========================
int main()
{
    cout << "========================================================" << endl;
    cout << "  DISK vs ALIKED — Feature Extraction & Matching" << endl;
    cout << "  Engine: " << engine_str(ENGINE) << endl;
    cout << "  OpenCV:  " << CV_VERSION << endl;
    cout << "========================================================" << endl;

    // ---- Check files ----
    for (const auto &f : {DISK_MODEL, DISK_LG_MODEL, ALIKED_MODEL, ALIKED_LG_MODEL, IMG_0_PATH, IMG_1_PATH}) {
        if (!filesystem::exists(f)) {
            cerr << "ERROR: file not found: " << f << endl;
            return -1;
        }
    }

    // ---- Load images ----
    Mat img0 = imread(IMG_0_PATH, IMREAD_COLOR);
    Mat img1 = imread(IMG_1_PATH, IMREAD_COLOR);
    if (img0.empty() || img1.empty()) {
        cerr << "ERROR: cannot read images" << endl;
        return -1;
    }
    cout << "\nImages: " << img0.cols << "x" << img0.rows
         << "  /  " << img1.cols << "x" << img1.rows << endl;

    // ================================================================
    //                        DISK PIPELINE
    // ================================================================
    print_header("DISK Pipeline: disk.onnx + disk_lightglue.onnx");

    dnn::Net disk_net, disk_lg_net;
    Mat disk_k0, disk_d0, disk_s0, disk_k1, disk_d1, disk_s1;
    int disk_h0, disk_w0, disk_h1, disk_w1;
    vector<Point2f> disk_pts0, disk_pts1;
    vector<DMatch> disk_matches;
    vector<float> disk_match_scores;

    auto t0_disk = high_resolution_clock::now();

    try {
        // Load DISK extractor
        cout << "\n[1] Loading DISK extractor: " << DISK_MODEL << endl;
        disk_net = dnn::readNetFromONNX(DISK_MODEL, ENGINE);
        cout << "    -> Loaded OK" << endl;

        // Extract features
        cout << "\n[2] Extracting DISK features from image 1..." << endl;
        if (!extract_disk(disk_net, img0, disk_k0, disk_d0, disk_s0, disk_h0, disk_w0))
            return -1;

        cout << "\n[3] Extracting DISK features from image 2..." << endl;
        if (!extract_disk(disk_net, img1, disk_k1, disk_d1, disk_s1, disk_h1, disk_w1))
            return -1;

        // Subsample to avoid OOM (LightGlue attention is quadratic)
        const int DISK_MAX_KPTS = 2048;
        subsample_top_k(disk_k0, disk_d0, disk_s0, DISK_MAX_KPTS);
        subsample_top_k(disk_k1, disk_d1, disk_s1, DISK_MAX_KPTS);

        // Load DISK LightGlue
        cout << "\n[4] Loading DISK LightGlue: " << DISK_LG_MODEL << endl;
        disk_lg_net = dnn::readNetFromONNX(DISK_LG_MODEL, ENGINE);
        disk_lg_net.enableWinograd(false);
        cout << "    -> Loaded OK" << endl;

        // Match
        cout << "\n[5] Running DISK LightGlue matching..." << endl;
        match_disk_lightglue(disk_lg_net,
                             disk_k0, disk_d0, disk_k1, disk_d1,
                             disk_h0, disk_w0, disk_h1, disk_w1,
                             disk_pts0, disk_pts1, disk_matches, disk_match_scores);
    } catch (const cv::Exception &e) {
        cerr << "\n!!! DISK pipeline OpenCV error: " << e.what() << endl;
    } catch (const std::exception &e) {
        cerr << "\n!!! DISK pipeline runtime error: " << e.what() << endl;
    }

    auto t1_disk = high_resolution_clock::now();
    auto disk_time = duration_cast<milliseconds>(t1_disk - t0_disk).count();

    // ================================================================
    //                       ALIKED PIPELINE
    // ================================================================
    print_header("ALIKED Pipeline: aliked-n16rot-top1k-640.onnx + aliked_lightglue.onnx");

    // Resize for ALIKED (fixed 640x640)
    Mat aliked_img0, aliked_img1;
    resize(img0, aliked_img0, Size(ALIKED_TARGET, ALIKED_TARGET));
    resize(img1, aliked_img1, Size(ALIKED_TARGET, ALIKED_TARGET));

    dnn::Net aliked_net, aliked_lg_net;
    Mat aliked_k0, aliked_d0, aliked_s0, aliked_k1, aliked_d1, aliked_s1;
    vector<Point2f> aliked_pts0, aliked_pts1;
    vector<DMatch> aliked_matches;
    vector<float> aliked_match_scores;

    auto t0_aliked = high_resolution_clock::now();

    try {
        // Load ALIKED extractor
        cout << "\n[1] Loading ALIKED extractor: " << ALIKED_MODEL << endl;
        aliked_net = dnn::readNetFromONNX(ALIKED_MODEL, ENGINE);
        cout << "    -> Loaded OK" << endl;

        // Extract features
        cout << "\n[2] Extracting ALIKED features from image 1..." << endl;
        extract_aliked(aliked_net, aliked_img0, aliked_k0, aliked_d0, aliked_s0);

        cout << "\n[3] Extracting ALIKED features from image 2..." << endl;
        extract_aliked(aliked_net, aliked_img1, aliked_k1, aliked_d1, aliked_s1);

        // Load ALIKED LightGlue
        cout << "\n[4] Loading ALIKED LightGlue: " << ALIKED_LG_MODEL << endl;
        aliked_lg_net = dnn::readNetFromONNX(ALIKED_LG_MODEL, ENGINE);
        aliked_lg_net.enableWinograd(false);
        cout << "    -> Loaded OK" << endl;

        // Match
        cout << "\n[5] Running ALIKED LightGlue matching..." << endl;
        match_aliked_lightglue(aliked_lg_net,
                               aliked_k0, aliked_d0, aliked_k1, aliked_d1,
                               aliked_k0.rows, aliked_k1.rows,
                               aliked_pts0, aliked_pts1, aliked_matches, aliked_match_scores,
                               ALIKED_TARGET, ALIKED_TARGET, ALIKED_TARGET, ALIKED_TARGET);
    } catch (const cv::Exception &e) {
        cerr << "\n!!! ALIKED pipeline OpenCV error: " << e.what() << endl;
    } catch (const std::exception &e) {
        cerr << "\n!!! ALIKED pipeline runtime error: " << e.what() << endl;
    }

    auto t1_aliked = high_resolution_clock::now();
    auto aliked_time = duration_cast<milliseconds>(t1_aliked - t0_aliked).count();

    // ================================================================
    //                        COMPARISON SUMMARY
    // ================================================================
    print_header("COMPARISON SUMMARY");

    cout << "\n              |  DISK          |  ALIKED" << endl;
    cout << "--------------+----------------+----------------" << endl;
    cout << "  Extractor   | disk.onnx      | aliked-n16rot-top1k-640.onnx" << endl;
    cout << "  LightGlue   | disk_lightglue | aliked_lightglue" << endl;
    cout << "  Engine      | ENGINE_NEW     | ENGINE_NEW" << endl;
    cout << "  Descriptor  | 128-D          | 128-D" << endl;
    cout << "  Image size  | " << img0.cols << "x" << img0.rows << " (original)  | " << ALIKED_TARGET << "x" << ALIKED_TARGET << " (resized)" << endl;
    cout << "  Kpts img1   | " << disk_k0.rows << "            | " << aliked_k0.rows << endl;
    cout << "  Kpts img2   | " << disk_k1.rows << "            | " << aliked_k1.rows << endl;
    cout << "  Matches     | " << disk_matches.size() << "             | " << aliked_matches.size() << endl;

    // Average confidence
    float disk_avg_conf = 0, aliked_avg_conf = 0;
    for (auto s : disk_match_scores)   disk_avg_conf   += s;
    for (auto s : aliked_match_scores) aliked_avg_conf += s;
    if (!disk_match_scores.empty())   disk_avg_conf   /= disk_match_scores.size();
    if (!aliked_match_scores.empty()) aliked_avg_conf /= aliked_match_scores.size();

    cout << "  Avg conf    | " << disk_avg_conf << "       | " << aliked_avg_conf << endl;
    cout << "  Total time  | " << disk_time << " ms         | " << aliked_time << " ms" << endl;

    // ================================================================
    //                        VISUALIZATION
    // ================================================================
    print_header("VISUALIZATION");

    if (!disk_matches.empty()) {
        cout << "\n[DISK] Drawing " << disk_matches.size() << " matches..." << endl;
        draw_and_save(img0, img1, disk_pts0, disk_pts1, disk_matches,
                      "disk_lightglue_matches.jpg", "DISK + LightGlue (ENGINE_NEW)");
    } else {
        cout << "\n[DISK] No matches to draw" << endl;
    }

    if (!aliked_matches.empty()) {
        cout << "\n[ALIKED] Drawing " << aliked_matches.size() << " matches..." << endl;
        draw_and_save(aliked_img0, aliked_img1, aliked_pts0, aliked_pts1, aliked_matches,
                      "aliked_lightglue_matches.jpg", "ALIKED + LightGlue (ENGINE_NEW)");
    } else {
        cout << "\n[ALIKED] No matches to draw" << endl;
    }

    cout << "\nDone! Check output images in build/ directory." << endl;
    return 0;
}
