# Wide-field-of-view-holography

这是一个大视场灰度全息影像的超表面设计与优化项目，当前实现的是基于 FFT 的 9 通道全息优化原型。

项目当前做的事情不是直接求真实纳米结构几何，而是先用两组代理参数 `phdx` 和 `phdy` 表示片上结构的可调自由度，再通过 FFT 模拟各衍射级次的远场强度，优化得到更接近目标图像的多通道结果。

## 项目目标

- 优化两组相位代理参数 `phdx` 和 `phdy`
- 让 9 个衍射通道在远场重建出目标灰度图像
- 让失败原因可诊断，而不是只看 loss 是否下降

当前重点是：

- 提高重建质量
- 提高灰度表现
- 减少通道能量塌缩
- 让诊断过程更清晰

当前不做：

- RCWA / FDTD 全波验证
- 超表面单元库映射
- GDS 导出
- 材料建模

## 推荐阅读顺序

第一次接触这个仓库，建议按下面顺序读：

1. `AGENT.MD`
2. `holo_opt/runner.py`
3. `holo_opt/targets.py`
4. `holo_opt/line_targets.py`
5. `holo_opt/field.py`
6. `holo_opt/export.py`

## 代码结构

建议按 3 个部分理解这套代码。

### 1. 目标图像生成与输入准备

- `holo_opt/config.py`
  - 定义实验配置、级次、损失权重、输出路径
- `holo_opt/targets.py`
  - 生成标准灰度阶梯 target
  - 加载 `.mat` target
- `holo_opt/line_targets.py`
  - 从普通 RGB 图片生成黑底线稿灰度 target
- `holo_opt/lineart_preview.py`
  - 只做线稿预览，不启动优化
- `holo_opt/cli.py`
  - 解析命令行参数，组装实验配置

### 2. 片上结构代理优化与远场模拟

- `holo_opt/field.py`
  - 用 `phdx`、`phdy` 和各级次 `(m, n)` 做 FFT 远场模拟
  - 计算各损失项
- `holo_opt/weights.py`
  - 根据通道效率和误差动态更新通道权重
- `holo_opt/runner.py`
  - 串起 target 加载、参数初始化、优化循环、诊断记录和最优状态选择

### 3. 评估、导出与结果管理

- `holo_opt/metrics.py`
  - 计算图像误差、灰度误差、效率和 score
- `holo_opt/export.py`
  - 导出图、表、配置和相位结果到输出目录

## 环境与安装

主要运行平台是 Windows，命令默认使用 PowerShell 和 `py`。

安装依赖：

```powershell
py -m pip install -r requirements.txt
```

当前依赖见 [requirements.txt](/D:/Hankstudy/github/Wide-field-of-view-holography/requirements.txt)：

- `numpy`
- `torch`
- `scipy`
- `matplotlib`
- `pillow`

完整测试：

```powershell
py -m unittest discover -s tests -q
```

## 目录约定

### 输入目录

- `inputs/lineart_sources/`
  - 放普通 RGB 图片
  - 给 `lineart_preview` 和 `lineart` 模式使用

### 输出目录

- `outputs/holo_experiments/`
  - 正式实验结果
- `outputs/lineart_preview/`
  - 线稿预览图
- `outputs/test_export/`
  - 导出测试临时结果
- `outputs/test_runner/`
  - runner 测试临时结果

如果你只是做实验分析，主要看：

- `outputs/holo_experiments/`
- `outputs/lineart_preview/`

## Target 模式

当前支持 3 种 target 模式。

### 1. `standard`

内置标准灰度阶梯 target。

适合：

- 检查流程是否跑通
- 做灰度能力诊断
- 比较不同损失权重和训练设置

### 2. `mat`

从 `.mat` 文件读取 target 栈。

适合：

- 使用你自己的多通道目标图
- 对接外部目标生成流程

### 3. `lineart`

从一张普通 RGB 图片生成黑底线稿灰度 target 图，再复制为 9 个通道的 target 栈。

当前线稿逻辑是：

- 保持宽高比缩放到正方形
- 自动提取边缘
- 只保留线条，背景为黑色
- 线中心更亮，边缘更暗

适合：

- 先从轮廓图开始做远场结构优化
- 快速验证一个图案是否适合进入优化阶段

## 最常用的 4 条命令

### 1. 跑完整测试

```powershell
py -m unittest discover -s tests -q
```

### 2. 预览线稿效果

先把图片放到：

```text
inputs/lineart_sources/
```

然后运行：

```powershell
py -m holo_opt.lineart_preview --input demo.png --size 256
```

这条命令会：

- 读取 `inputs/lineart_sources/demo.png`
- 输出原图副本到 `outputs/lineart_preview/demo_original.png`
- 输出线稿图到 `outputs/lineart_preview/demo_lineart.png`

如果你想直接传绝对路径，也可以：

```powershell
py -m holo_opt.lineart_preview --input D:/path/to/demo.png --size 256
```

### 3. 跑标准诊断优化

```powershell
py -m holo_opt.cli --target-mode standard --size 64 --epochs-per-chunk 300 --outer-loops 5 --device cpu --output-root outputs/holo_experiments --label diagnostic --eta-balance-weight 0.05 --gray-monotonic-weight 0.1 --phase-smoothness-weight 0.0001 --background-weight 0.0
```

### 4. 跑线稿优化

把原图放到：

```text
inputs/lineart_sources/
```

然后运行：

```powershell
py -m holo_opt.cli --target-mode lineart --target-path inputs/lineart_sources/demo.png --size 128 --device cpu --output-root outputs/holo_experiments --label lineart
```

也可以直接传绝对路径：

```powershell
py -m holo_opt.cli --target-mode lineart --target-path D:/path/to/demo.png --size 128 --device cpu --output-root outputs/holo_experiments --label lineart
```

## VS Code 运行方式

仓库已经提供了 `.vscode/launch.json` 预设。

推荐使用：

- `Holo smoke: standard 16`
  - 只检查流程能否跑通
- `Holo diagnostic: standard 64`
  - 推荐的调试配置
- `Holo quick9: standard 128`
  - 较快的标准 9 通道运行
- `Holo quality: standard 128`
  - 更重的质量优先配置
- `Holo MAT template`
  - 读取用户自己的 `.mat` 目标图

## 一次优化到底在做什么

核心流程如下：

1. 根据 `target-mode` 生成或加载 target 图像
2. 初始化 `phdx` 和 `phdy`
3. 对每个 channel 的 `(m, n)` 级次构造相位
4. 对每个通道做 FFT，得到远场强度图
5. 把远场图和 target 图做比较，计算 loss
6. 反向传播更新 `phdx` 和 `phdy`
7. 每个 outer loop 做一次评估、诊断和通道权重更新
8. 导出最优结果和全部诊断信息

如果你只想看入口，直接读：

- `holo_opt/runner.py`

## 一个实验结果文件夹里有什么

`outputs/holo_experiments/` 下每个子文件夹都代表一次独立实验。

文件夹名字通常像这样：

```text
diagnostic_9ch_64_20260510_000000_123456
```

含义是：

- `diagnostic`：这次实验的 label
- `9ch`：9 个通道
- `64`：目标尺寸是 64
- 后面是时间戳

### 最值得先看的文件

- `summary.png`
  - 第一行是目标图
  - 第二行是重建图
- `outer_###_summary.png`
  - 每个 outer loop 的中间结果
- `loss_terms.png`
  - 各损失项的变化
- `diagnostics.csv`
  - 每轮诊断值

### 其他常见文件

- `config.json`
  - 本次运行的完整配置
- `metrics.json`
  - 最终指标
- `metrics.csv`
  - 适合表格查看的通道指标
- `optimized_results.npz`
  - 原始数值结果打包
- `phdx.csv`
  - 最终优化出的 x 方向相位代理参数
- `phdy.csv`
  - 最终优化出的 y 方向相位代理参数
- `loss_curve.png`
  - 总 loss 曲线
- `eta_curve.png`
  - 各通道效率曲线
- `gray_levels.png`
  - 灰度级响应或误差图
- `loss_terms.csv`
  - 每一步的损失分项

## 建议怎么看结果

推荐按下面顺序看：

1. `summary.png`
2. 最新的 `outer_###_summary.png`
3. `loss_terms.png`
4. `diagnostics.csv`
5. `metrics.json` / `metrics.csv`
6. `phdx.csv` / `phdy.csv`

不要一开始就盯着 `phdx.csv` 和 `phdy.csv`。

先看图像像不像，再判断是哪个损失项或哪个通道拖后腿。

## 常见判断方法

- `summary.png` 像噪声
  - 看 `loss_terms.png`
  - 看 `phase_smoothness` 和 `image_mse`

- 某些通道明显更暗
  - 看 `eta_curve.png`
  - 看 `metrics.csv`

- 灰度不对
  - 看 `gray_levels.png`
  - 看 `gray_level_error`

- loss 在降，但图像没改善
  - 不要只加步数
  - 重点看 `image_mse` 是否也在降

## 路径和命令建议

- 运行和测试优先使用 `py`
- 正式实验结果默认写到 `outputs/holo_experiments/`
- 线稿预览结果默认写到 `outputs/lineart_preview/`
- 线稿原图建议统一放到 `inputs/lineart_sources/`

## 文档说明

- `README.md`
  - 负责快速上手和常用命令
- `AGENT.MD`
  - 负责完整维护上下文、设计约定和工作流说明

如果你修改了：

- target 生成逻辑
- 优化逻辑
- 导出结构
- 命令行入口
- 测试方式

请同步更新 `AGENT.MD`。
