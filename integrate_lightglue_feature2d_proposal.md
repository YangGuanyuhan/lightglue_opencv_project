用户体验应接近传统的 feature2d 流程。

## 1. 目标

- 将公共 API 保留在 `opencv2/features.hpp` 中现有的 feature2d 抽象之下
- 使默认用法与类似 SIFT 的工作流对齐（先 `detectAndCompute` 然后 `match`）
- 保持 DNN 依赖为可选项 (`HAVE_OPENCV_DNN`)

## 2. 提议的公共 API

### 2.1 作为 Feature2D 的 ALIKED

C++

```
class CV_EXPORTS_W ALIKED : public Feature2D
{
public:
    struct Params
    {
        Params();
        Size inputSize;
        bool normalizeDescriptors;
        int dnnEngine;
        int dnnBackend;
        int dnnTarget;
    };

    // 保持与 TrackerNano 一致：直接接受已加载的 dnn::Net，实例创建后网络不可变
    CV_WRAP static Ptr<ALIKED> create(const dnn::Net& net, const Params& params = Params());

    // 兼容传统基于文件/内存的加载方式（内部会转换为 dnn::Net）
    CV_WRAP static Ptr<ALIKED> create(const String& modelPath, const Params& params = Params());
    static Ptr<ALIKED> create(const std::vector<uchar>& modelData, const Params& params = Params());

    CV_WRAP virtual String getModel() const = 0;
    CV_WRAP virtual String getDefaultName() const CV_OVERRIDE = 0;
};
```

**ALIKED 与 SuperPoint 的关键差异：**

| 特性 | SuperPoint | ALIKED |
|------|-----------|--------|
| 描述子维度 | 256-D | 128-D |
| 输入图像 | 灰度 [1,1,H,W] | RGB [1,3,H,W] |
| 关键点坐标 | 像素坐标 (int64) | 归一化坐标 [-1,1] (float32) |
| 输入尺寸 | 可变 (如 640x480) | 固定 640x640 |
| LightGlue 输出格式 | Standard [1,N] | Fused [M,2] |

### 2.2 作为 DescriptorMatcher 的 LightGlue

C++

```
class CV_EXPORTS_W LightGlueMatcher : public DescriptorMatcher
{
public:
    struct Params
    {
        Params();
        float scoreThreshold;
        bool disableWinograd;
        int dnnEngine;
        int dnnBackend;
        int dnnTarget;
    };

    // 保持与 TrackerNano 一致
    CV_WRAP static Ptr<LightGlueMatcher> create(const dnn::Net& net, const Params& params = Params());


    CV_WRAP static Ptr<LightGlueMatcher> create(const String& modelPath, const Params& params = Params());
    static Ptr<LightGlueMatcher> create(const std::vector<uchar>& modelData, const Params& params = Params());

    // 为非 ALIKED 或跨进程使用提供可选的手动几何信息输入。
    CV_WRAP virtual void setPairInfo(InputArray queryKpts,
                                     InputArray trainKpts,
                                     Size queryImageSize = Size(),
                                     Size trainImageSize = Size()) = 0;
    CV_WRAP virtual void clearPairInfo() = 0;
};
```

**行为：** 主要的匹配器调用保持为继承的标准 API：

- `match(queryDesc, trainDesc, matches)`
- `knnMatch(...)`
- `radiusMatch(...)`

LightGlue 的几何信息需求由上下文解析来处理（第 3 节）。

Params 构造函数的实现 in cpp file

~~~
// ALIKED::Params 构造函数实现
ALIKED::Params::Params()
{
    // 默认参数初始化
    inputSize = Size(640, 640);
    normalizeDescriptors = true;

#ifdef HAVE_OPENCV_DNN
    dnnEngine = dnn::ENGINE_AUTO(自己本地测试使用ort,但是代码中使用auto);
    dnnBackend = dnn::DNN_BACKEND_DEFAULT;
    dnnTarget = dnn::DNN_TARGET_CPU;
#else
    dnnEngine = -1;  // invalid value
    dnnBackend = -1; // invalid value
    dnnTarget = -1;  // invalid value
#endif
}

// LightGlueMatcher::Params 构造函数实现
LightGlueMatcher::Params::Params()
{
    scoreThreshold = 0.0f;
    disableWinograd = false;

#ifdef HAVE_OPENCV_DNN
    dnnEngine = dnn::ENGINE_AUTO(自己本地测试使用ort,但是代码中使用auto);
    dnnBackend = dnn::DNN_BACKEND_DEFAULT;
    dnnTarget = dnn::DNN_TARGET_CPU;
#else
    dnnEngine = -1;  // invalid value
    dnnBackend = -1; // invalid value
    dnnTarget = -1;  // invalid value
#endif
}
~~~



## 3. 几何信息处理策略

### 3.1 自动上下文路径（默认）

当描述符由进程内的 OpenCV ALIKED 生成时，实现会存储内部匹配上下文：

- 关键点坐标（Nx2，浮点型，归一化坐标 [-1,1]）
- 源图像大小

然后正常的 `DescriptorMatcher::match(desc0, desc1, ...)` 调用就能以类似 SIFT 的风格工作。

### 3.2 手动上下文路径（调整）

对于来自外部来源、文件或不同进程边界的描述符，用户可以在匹配之前显式提供几何信息：

- `setPairInfo(queryKpts, trainKpts, queryImageSize, trainImageSize)` 然后调用标准的 `match(queryDesc, trainDesc, matches)`

### 3.3 歧义消除规则

- 如果自动和手动上下文同时存在，则下一次 match 调用时手动上下文优先。
- 一次成功匹配后，手动上下文会被自动清除。
- `clearPairInfo()` 显式清除手动上下文。
- 如果没有可用的有效上下文，匹配器将抛出 `StsBadArg` 并给出准确的提示。

### 3.4 生命周期和线程规则

- 上下文管理是针对每个匹配器实例的；不存在进程范围的可变单例行为。
- 与描述符输出关联的自动上下文被视为短期的，并尽可能为即时匹配工作流提供支持 (best-effort)。
- 跨线程或延迟匹配应显式使用 `setPairInfo(...)` 以避免生命周期歧义。

## 4. 用户用法

### 4.1 默认用法（传统类似 SIFT 的模式）

C++

```
cv::ALIKED::Params sp;
sp.inputSize = cv::Size(640, 640);
sp.dnnEngine = cv::dnn::ENGINE_ORT;

cv::LightGlueMatcher::Params lg;
lg.modelPath = "aliked_lightglue.onnx";
lg.scoreThreshold = 0.0f;
lg.dnnEngine = cv::dnn::ENGINE_ORT;

cv::Ptr<cv::Feature2D> extractor = cv::ALIKED::create("aliked-n16rot-top1k-640.onnx", sp);
cv::Ptr<cv::DescriptorMatcher> matcher = cv::LightGlueMatcher::create(lg);

std::vector<cv::KeyPoint> kpts0, kpts1;
cv::Mat desc0, desc1;
extractor->detectAndCompute(img0, cv::noArray(), kpts0, desc0);
extractor->detectAndCompute(img1, cv::noArray(), kpts1, desc1);

std::vector<cv::DMatch> matches;
matcher->match(desc0, desc1, matches);
```

这使顶层调用形式与现有的 feature2d 代码保持一致。

**注意事项：**
- ALIKED 输入为 RGB 图像，内部会自动完成 BGR→RGB 转换和归一化
- ALIKED 输出的关键点坐标为归一化坐标 [-1,1]，`detectAndCompute` 内部会自动转换为像素坐标存储到 `KeyPoint` 中
- 匹配时内部会将像素坐标归一化回 [-1,1] 传给 LightGlue
- ALIKED LightGlue 输出为 Fused 格式 [M,2]，即直接给出 M 对匹配索引

### 4.2 外部描述符用法（手动几何上下文）

C++

```
cv::LightGlueMatcher::Params lg;
lg.modelPath = "aliked_lightglue.onnx";
lg.dnnEngine = cv::dnn::ENGINE_ORT;

cv::Ptr<cv::LightGlueMatcher> matcher = cv::LightGlueMatcher::create(lg);

matcher->setPairInfo(cv::Mat(pts0), cv::Mat(pts1), img0.size(), img1.size());

std::vector<cv::DMatch> matches;
matcher->match(desc0, desc1, matches);
```

这里的 `pts0` 和 `pts1` 是 `vector<Point2f>`（或等效的 Nx2 浮点矩阵），坐标应为归一化坐标 [-1,1]。

## 5. 构建和模块集成

### 5.1 CMake

在 `modules/features/CMakeLists.txt` 中，添加可选的 DNN 依赖：

- 从 `OPTIONAL opencv_flann`
- 变更为 `OPTIONAL opencv_flann opencv_dnn`

### 5.2 头文件依赖准则

- 保持公共头文件中没有具体的 DNN 网络对象
- 将大量使用 DNN 的包含文件/逻辑放在带有 `#ifdef HAVE_OPENCV_DNN` 保护的 `.cpp` 文件中

### 5.3 非 DNN 构建

- 公共类依然存在
- 依赖 DNN 的操作会返回明确的 `StsNotImplemented` 错误
- 特定于 DNN 的参数保留在 `#ifdef HAVE_OPENCV_DNN` 下

## 6. 双重加载与底层架构对齐 (Dual Loading & Architecture)

为了确保内部实现的高度一致性，并支持跨平台（如 Android assets, 文件系统）的灵活加载，`ALIKED` 和 `LightGlue` 在内部设计上严格对齐 `TrackerNano` 的模式。

基于文件路径或内存数据创建实例时，内部都会先统一调用 `cv::dnn::readNet` 解析为 `dnn::Net`，随后透传给具体的 `Impl` 类：

C++

```
// 内部实现高度对齐 TrackerNano
class ALIKEDImpl : public ALIKED
{
    cv::dnn::Net net;
    Params params;
public:
    ALIKEDImpl(const dnn::Net& _net, const Params& _params)
    {
        CV_Assert(!_net.empty());
        net = _net;
        params = _params;
    }
    // ... detectAndCompute 实现 ...
    // 内部处理：
    //   1. blobFromImage(img, 1.0/255.0, Size(640,640), Scalar(), true, false)  // RGB, NCHW
    //   2. net.setInput(blob, "image")
    //   3. net.forward(outs, {"keypoints", "descriptors", "scores"})
    //   4. 关键点坐标从归一化 [-1,1] 转换为像素坐标存入 KeyPoint
    //   5. 描述子 128-D 直接输出
};

// 接口实现
Ptr<ALIKED> ALIKED::create(const dnn::Net& net, const Params& params)
{
    return makePtr<ALIKEDImpl>(net, params);
}

Ptr<ALIKED> ALIKED::create(const String& modelPath, const Params& params)
{
    dnn::Net net = dnn::readNet(modelPath);
    return makePtr<ALIKEDImpl>(net, params);
}

```
参考这个

```cpp
TrackerNanoImpl(const dnn::Net& backbone, const dnn::Net& neckhead);
 TrackerNanoImpl(const dnn::Net& _backbone, const dnn::Net& _neckhead)
    {
        CV_Assert(!_backbone.empty());
        CV_Assert(!_neckhead.empty());

        backbone = _backbone;
        neckhead = _neckhead;
    }

```

## 7. 从当前 WIP 迁移

当前的 WIP 引入了：

- `cv::features::FeatureExtractor`
- `cv::features::FeatureMatcher`
- `cv::features::SuperPoint` (弃用，由 ALIKED 替代)
- `cv::features::LightGlue`

迁移建议：

- 将 ALIKED 和 LightGlue API 移动到 `features.hpp` 的 `cv` 命名空间中
- 直接继承 `Feature2D` 和 `DescriptorMatcher`
- 移除平行的提取器/匹配器抽象
- 移除 SuperPoint 相关代码，ALIKED 完全替代其特征提取功能
- 在 `modules/features/src` 中保持实现的分离：
  - `feature2d_aliked.cpp` (替代原 `feature2d_superpoint.cpp`)
  - `matchers_lightglue.cpp`
  - 在编译保护下的可选 DNN 专用文件

## 8. 测试计划

### 8.1 单元测试

- ALIKED `detectAndCompute()` 输出类型/形状一致性（128-D 描述子，关键点像素坐标）
- 经过 ALIKED 提取后，LightGlue 匹配器使用默认的仅描述符 `match(...)` 调用
- 带有手动 `setPairInfo(...)` 路径的 LightGlue 匹配器
- 手动与自动上下文的优先级和生命周期规则
- 上下文缺失时的错误消息清晰度
- 非 DNN 构建行为 (`StsNotImplemented`)

### 8.2 集成测试

- 在 `box.png` 和 `box_in_scene.png` 上进行端到端匹配
- 基本的索引边界和分数阈值检查
- ALIKED 归一化坐标到像素坐标的转换正确性验证
- ALIKED LightGlue Fused 输出格式 [M,2] 的解析正确性验证
