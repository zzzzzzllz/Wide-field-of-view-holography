# 项目背景

本项目是一个大视场灰度全息成像的超表面设计与优化工程。核心目标是在 9 个衍射通道上生成灰度阶梯或用户提供的目标图像，通过 PyTorch 优化两组相位参数 `phdx` 和 `phdy`，并导出重建强度、指标、诊断数据和可视化结果。

默认级次为：

```text
(-2, 2), (-2, 1), (-2, 0),
(-3, 2), (-3, 1), (-3, 0),
(-4, 2), (-4, 1), (-4, 0)
```

默认物理参数继承当前设计设定：

- `lambda_nm = 532`
- `px_nm = 830`
- `py_nm = 830`
- guided mode 默认开启
- `neff = 2.05`
- `alpha_deg = -16.7`

## 代码结构

- `holo_opt/cli.py`：命令行入口，负责解析实验参数并启动优化。
- `holo_opt/config.py`：实验配置、物理参数、损失权重、评分权重、默认级次和配置校验。
- `holo_opt/field.py`：光场强度计算、训练损失和各损失分项计算。
- `holo_opt/runner.py`：实验主流程，包括目标加载、优化循环、权重更新、诊断记录、进度打印和结果导出。
- `holo_opt/batch.py`：多 seed 批处理入口，复用单次运行配置，逐个 seed 运行并导出 `seed_summary.csv`。
- `holo_opt/export.py`：导出运行目录、CSV/JSON/NPZ 数据以及 PNG 可视化图。
- `holo_opt/metrics.py`：重建质量、效率、灰度级误差等评估指标。
- `holo_opt/targets.py`：标准灰度阶梯目标和 `.mat` 目标加载。
- `holo_opt/line_targets.py`：从普通 RGB 图片生成线稿或灰度 target；灰度模式按 3x3 切块分配到 9 个通道。
- `holo_opt/lineart_preview.py`：只做线稿预览，不启动优化；输入一张 RGB 图片，导出原图副本和处理后的线稿图。
- `holo_opt/weights.py`：基于效率和通道误差的通道权重更新。
- `tests/`：使用 `unittest` 编写的单元测试和导出/运行烟雾测试。
- `.vscode/launch.json`：VS Code 调试入口，适合不想手写命令时使用。

建议按下面 3 个部分理解和维护这套代码：

1. 目标图像生成与输入准备
   - `config.py`：定义这次实验的 target 模式、尺寸、通道数和输出路径。
   - `targets.py`：生成标准灰度阶梯 target，或从 `.mat` 文件读取 target 栈。
   - `line_targets.py`：从一张 RGB 图片自动提线或生成压暗灰度图；灰度 target 按 3x3 切块分配到 9 个通道。
   - `lineart_preview.py`：给第一部分单独预览用，先验证“取线条效果”再进入优化。
   - `cli.py`：把命令行参数装配成统一配置对象。
2. 片上结构代理优化与远场模拟
   - `field.py`：使用 `phdx`、`phdy` 和级次对每个通道做 FFT 远场模拟，并计算损失项。
   - `weights.py`：根据各通道误差和效率做动态重加权。
  - `runner.py`：串起 target 加载、参数初始化、优化循环、诊断记录和最优状态选择。
3. 评估、导出与结果管理
   - `metrics.py`：计算每个通道和整体的图像误差、灰度误差、效率和 score。
   - `export.py`：把图、表、配置和相位结果写到 `outputs/holo_experiments/`。

## 协作与平台约定

- 本仓库以 Windows 为主运行平台，以 PowerShell 命令和 `py` 启动方式为标准工作流。
- 所有 GitHub 相关操作一律走 CLI 路线，例如 `git` 和 `gh`；不要使用 GitHub connector / app tools 处理 branch、push、PR、review、comment 或 issue。
- macOS 主要用于轻量阅读、文档补充、少量代码调整和 review；不要把仅在 macOS 验证过的行为表述成“项目已完全验证”。
- 写命令示例时，优先给出 Windows 可直接执行的版本；如果当前只在 macOS 做了静态修改或局部检查，要明确说明“Windows 侧需要按下列命令复核”。
- 新增逻辑时避免引入只适配单一平台的路径处理、shell 写法或缓存目录假设；路径拼接优先走 Python 标准库，不写死 macOS 风格路径。
- 若发现某个验证在当前机器上无法完成，要把原因写明白，例如 `torch` 缺失、CUDA 不可用、matplotlib/font cache 权限告警等，不要模糊带过。
- 默认采用两步交付：先完成修改并停下来让用户检查；只有用户明确确认后，才 stage、commit、push 或创建 PR。除非用户明确要求“一次性提交”，否则不要在同一轮里自动提交。

## 文档同步规则

- 以后每次修改核心代码，只要涉及算法行为、训练流程、导出格式、调参方式、运行入口或测试策略，就要同步更新本文件。
- 同步到 `AGENT.MD` 时，至少补齐下面 5 类信息：
  - 改动动机：为什么改。
  - 影响范围：改了哪些模块，例如 `field.py`、`runner.py`、`export.py`。
  - 用户可见变化：新增了什么参数、输出文件、图像、CSV 或诊断方式。
  - 验证方式：跑了哪些测试，哪些没有跑。
  - 平台说明：Windows 怎么跑，macOS 当前只验证到了什么程度。
- `README.md` 不做大面积删减，优先渐进补充。`README.md` 负责快速入口，`AGENT.MD` 负责完整上下文、维护约定和设计细节。
- 如果改动只动文档，也要重新通读相关段落，确保 `README.md`、`AGENT.MD`、`docs/superpowers/` 三处描述不互相冲突。

## 输出说明约定

- 每次任务结束后的说明默认使用中文。
- 说明要尽量直白，避免只说“优化了”“调整了”这种泛化表述，要说清楚“改了哪里、为什么改、会影响什么”。
- 如果改动涉及算法，要单独说明“算法修改精进情况”，至少回答下面几个问题：
  - 当前优化是朝哪个失败模式在收敛，例如雪花噪声、灰度不单调、通道能量塌缩。
  - 这次是新增能力、修正错误，还是增强诊断可见性。
  - 用户下一步应该看哪些输出文件来判断是否真的变好。
- 每次任务结束时，不要等用户再次提醒，默认同时提供一份可直接发出的中文 comments/PR 说明草稿。
- 这份草稿至少包含两部分：
  - `comments 标题`：一句话概括这次改动。
  - `comments 细节`：按“改动点、原因、风险或未验证点、Windows 侧建议复核命令”的顺序写清楚。
- 如果这次改动较大，还要额外给出一份 `PR 摘要`，方便用户直接复制到 pull request 描述里。
- 除非用户明确说不要，否则这份 comments/PR 草稿属于默认交付内容，不需要用户再提第二遍。

## 当前算法演进状态

当前主线不是空白原型，而是已经完成了一轮“质量优化”落地，后续 AI 需要在这个基础上继续推进，而不是重复造轮子。

当前后续推进方向的唯一正式入口是 `docs/ROADMAP.md`。`docs/superpowers/plans/2026-05-07-holography-quality-optimization.md` 是历史已实施计划，只用于追溯上一轮质量优化如何落地，不再作为新任务清单。

- 已落地能力：
  - `LossConfig` 和 `diagnostic_interval` 已进入配置层。
  - target 侧已支持 4 种输入模式：`standard`、`mat`、`lineart`、`grayscale`。
  - `field.py` 已支持结构化损失项：`image_mse`、`eta_balance`、`gray_monotonic`、`phase_smoothness`、`background`；`image_mse` 现在按 target 总能量匹配重建强度后再计算，避免最大亮噪点归一化导致 loss 下降但图像仍是雪花噪声。
  - `runner.py` 已记录 `loss_terms_history`、`diagnostics`、`outer_summaries`，并按 score 选最优状态。
  - best state 选择支持 `--selection-metric score|image_error|gray_level_error|efficiency_balance_penalty`，默认 `score`；做 MSE 优先实验时使用 `image_error`。
  - `batch.py` 已支持多 seed 批处理，输出 `seed_summary.csv` 并标出最低 score 的 best seed。
  - `export.py` 已导出 `diagnostics.csv`、`loss_terms.csv`、`loss_terms.png`、`outer_###_summary.png`。
  - `cli.py` 和 `.vscode/launch.json` 已暴露 diagnostic/quality 工作流。
  - 相关测试已覆盖配置、损失项、导出和 runner 烟雾流程。
- 这轮设计的核心目标：
  - 不只是让 loss 下降，还要让失败原因可诊断。
  - 优先解决噪声重建、灰度不单调、通道能量塌缩、训练不稳定这几类问题。
  - 保持 FFT 原型架构，不扩展到 RCWA/FDTD、单元库映射、GDS 导出等更重的物理设计链条。
- 下一阶段更值得继续推进的方向：
  - 球形视场与平面 target 的几何映射。
  - 灰度质量与单通道能量预算优化。
  - 输入图片适配与 image2 辅助 target 设计。
  - 收敛稳定性、多 seed 与实验批处理。
  - 工作流与实验报告自动化。

## 推荐运行方式

优先使用 VS Code 的“运行和调试”面板：

- `Holo smoke: standard 16`：只检查流程能否跑通，结果质量通常没有参考价值。
- `Holo diagnostic: standard 64`：当前推荐的调试配置，会导出逐轮 summary、诊断 CSV 和损失分项。
- `Holo quick9: standard 128`：较快的标准 9 通道运行。
- `Holo quality: standard 128`：更长步数的质量优先配置，当前使用 `cuda`，没有 CUDA 时需要改成 `cpu` 或改用其他配置。
- `Holo MAT template`：读取用户自己的 `.mat` 目标图，需要先把 `--target-path` 改成真实路径。
- `lineart` 模式：读取一张普通 RGB 图片，自动生成黑底线稿灰度 target 图，适合从轮廓图开始做远场结构优化。

常用命令行命令：

```powershell
py -m holo_opt.cli --target-mode standard --device cpu --size 128 --epochs-per-chunk 300 --outer-loops 3
```

标准诊断运行示例：

```powershell
py -m holo_opt.cli --target-mode standard --size 64 --epochs-per-chunk 300 --outer-loops 5 --device cpu --output-root outputs/holo_experiments --label diagnostic --eta-balance-weight 0.05 --gray-monotonic-weight 0.1 --phase-smoothness-weight 0.0001 --background-weight 0.0
```

MAT 目标运行示例：

```powershell
py -m holo_opt.cli --target-mode mat --target-path D:/path/to/targets.mat --mat-variable bw_all --size 128 --device cpu --output-root outputs/holo_experiments --label mat9
```

RGB 线稿目标运行示例：

```powershell
py -m holo_opt.cli --target-mode lineart --target-path D:/path/to/input.png --size 128 --device cpu --output-root outputs/holo_experiments --label lineart
```

线稿预览运行示例：

```powershell
py -m holo_opt.lineart_preview --input demo.png --size 256
```

多 seed 对比运行示例：

```powershell
py -m holo_opt.batch --target-mode standard --size 64 --epochs-per-chunk 300 --outer-loops 3 --device cuda --output-root outputs/holo_experiments --label seed_probe --seeds 42 43 44
```

MSE 优先选择 best outer 示例：

```powershell
py -m holo_opt.cli --target-mode grayscale --target-path D:/path/to/input.png --size 128 --epochs-per-chunk 10000 --outer-loops 10 --device cuda --output-root outputs/holo_experiments --label mse_focused --selection-metric image_error
```

线稿预览与正式优化的路径约定：

- 建议把原始图片放到 `inputs/lineart_sources/`
- `py -m holo_opt.lineart_preview --input demo.png` 会优先在 `inputs/lineart_sources/demo.png` 查找输入图
- 预览模式默认把两张图写到 `outputs/lineart_preview/`
- 正式 `lineart` 优化模式把实验结果写到 `outputs/holo_experiments/`

当前本机 `py` 可导入 `torch`。普通 `python` 可能是 Microsoft Store 占位入口，运行项目和测试时优先使用 `py`。

## 测试方式

运行完整测试：

```powershell
py -m unittest discover -s tests -q
```

如果当前是在 macOS 上临时调整代码，但不准备在本机完整跑通，结束说明里要明确写出建议在 Windows 上执行的复核命令，例如：

```powershell
py -m unittest discover -s tests -q
py -m holo_opt.cli --target-mode standard --size 64 --epochs-per-chunk 300 --outer-loops 5 --device cpu --output-root outputs/holo_experiments --label diagnostic
```

修改训练逻辑、导出逻辑或指标后，至少运行完整测试。只改文档时可以不跑测试，但需要重新读取文档确认内容无误。

## 输出结果

实验结果默认写入 `outputs/holo_experiments/` 下的时间戳运行目录。典型输出包括：

- `config.json`：本次运行的完整配置。
- `metrics.json`：最终指标，包含每个通道和汇总指标。
- `metrics.csv`：适合表格查看的通道指标。
- `optimized_results.npz`：目标、重建强度、相位、loss、效率和权重历史。
- `phdx.csv`：最终优化出的 x 方向相位参数。
- `phdy.csv`：最终优化出的 y 方向相位参数。
- `summary.png`：最重要的总览图，第一行是目标，第二行是重建结果。
- `outer_###_summary.png`：每个诊断轮次的中间重建结果。
- `loss_curve.png`：总 loss 曲线。
- `eta_curve.png`：各通道效率曲线。
- `gray_levels.png`：灰度级响应曲线或灰度误差图。
- `diagnostics.csv`：每个 outer loop 的 `loss`、`score`、`mean_eta`、`image_error`、`gray_level_error` 等诊断值。
- `loss_terms.csv`：每一步的损失分项。
- `loss_terms.png`：`image_mse`、`eta_balance`、`gray_monotonic`、`phase_smoothness` 和 `background` 损失分项的曲线。
- `seed_summary.csv`：多 seed 批处理目录下的汇总表，包含每个 seed 的 `run_dir`、`score`、`image_error`、`gray_level_error` 和 `best` 标记。

结果判断优先级：

1. 先看 `summary.png` 和最新的 `outer_###_summary.png`，确认是否能看出目标结构。
2. 再看 `loss_curve.png`，判断训练是否仍在下降、是否震荡。
3. 再看 `loss_terms.csv` 或 `loss_terms.png`，判断到底是图像误差、灰度单调性、相位平滑性还是效率均衡在主导问题。
4. 最后看 `phdx.csv` 和 `phdy.csv`，它们是后续做单元结构映射或版图设计时才优先关注的结果。

## 调参线索

- `image_mse` 高：目标图像本身没重建好，优先增加步数、检查学习率和目标图。
- `gray_monotonic` 高：灰度台阶关系差，考虑提高 `--gray-monotonic-weight`。
- `phase_smoothness` 高且图像像雪花：考虑提高 `--phase-smoothness-weight`，或先用更小尺寸诊断。
- `eta_balance` 高：各通道效率差距大，考虑提高 `--eta-balance-weight` 或检查权重更新。
- `background` 高：暗背景太亮，启用并提高 `--background-weight`。
- loss 持续下降但图像不改善：不要只加步数，重点检查 `image_mse` 是否也在下降。
- 如果 `loss` 和 `image_mse` 下降但 `summary.png` 仍像雪花，优先检查当前代码是否仍在使用旧的“最大值归一化强度 MSE”；当前基线应使用按 target 总能量匹配后的强度 MSE。

## 维护注意事项

- 不要改变已有导出文件名，测试、VS Code 配置和历史结果依赖这些约定。
- 修改训练损失时，同步检查 `tests/test_field.py` 和 `tests/test_runner.py`。
- 修改导出文件、图像或 CSV 时，同步检查 `tests/test_export.py` 和 `tests/test_runner.py`。
- 新增导出数据时，优先在 `export_results()` 中集中处理，并补充非空文件或内容校验测试。
- 训练进度目前每 500 步在终端打印一次当前 step 和 total loss。
- 输出目录下可能存在大量历史实验结果，除非明确要求，不要清理或覆盖。
- 仓库里可能有未提交实验输出、VS Code 本地配置或资料文件；提交前只 stage 本次任务相关文件。
- 如果修改了算法设计思路但没有同步 `AGENT.MD`，这次任务就算没有完成。

## 2026-05-10 灰度图 target 输入更新

- 改动动机：在已有 `lineart` 取线条能力之外，新增直接从 RGB 图片生成灰度 target 的能力；灰度图会先转为亮度图，再压低整体亮度，避免大面积色块在单个 diffraction channel 中过白、占满固定亮度预算。正式 `grayscale` target 会按 3x3 切图分给 9 个 channel，不复制整图。
- 影响范围：`holo_opt/line_targets.py` 新增 `generate_grayscale_image_targets()`、亮度压缩逻辑和 `split_grayscale_image_into_channel_tiles()`；`holo_opt/runner.py`、`holo_opt/config.py`、`holo_opt/cli.py` 接入新的 `target_mode="grayscale"`；`holo_opt/grayscale_preview.py` 提供与线稿预览一致的原图/处理图输出；`.vscode/launch.json` 新增灰度图模板。
- 用户可见变化：正式运行可用 `py -m holo_opt.cli --target-mode grayscale --target-path inputs/lineart_sources/demo_preview.png --size 128 --device cpu`；预览可用 `py -m holo_opt.grayscale_preview --input demo_preview.png --size 64`，输出文件名为 `*_original.png` 和 `*_grayscale.png`。
- 算法细节：灰度图不使用线稿路径里的 autocontrast 拉满动态范围，而是使用原始 RGB luminance，经过 gamma 压缩、最大亮度上限和低梯度区域压暗；默认最大亮度约为 `0.65`，大块平坦区域进一步乘以 `0.55` 附近的暗化系数。切块顺序为 row-major：左上、上中、右上、中左、中间、中右、左下、下中、右下；每个 tile resize 到 channel target 尺寸后参与优化。
- 验证方式：已运行 `py -m unittest discover -s tests -q`，并执行灰度预览命令确认输出 `outputs/lineart_preview/demo_preview_grayscale.png`。

## 2026-05-18 灰度预处理增强与输入适配诊断

- 改动动机：仅靠全局 gamma 压缩和大块平坦区压暗，容易把局部细节一起压没，导致输入图在进入全息优化前已经丢掉轮廓层次；同时 3x3 切块后各通道亮度预算分配不透明，难以判断某张图是不是天然不适合当前固定能量设置。
- 影响范围：`holo_opt/line_targets.py` 新增更强的灰度预处理链路、tile 预算均衡和 `GrayscaleTargetArtifacts` 诊断产物；`holo_opt/config.py`、`holo_opt/cli.py` 新增灰度预处理配置；`holo_opt/runner.py`、`holo_opt/export.py` 现在会在 `grayscale` 运行时额外导出 `preprocess_comparison.png` 与 `target_energy_report.csv`；`holo_opt/grayscale_preview.py` 改为复用正式 `grayscale` 预处理链路；相关测试已同步更新。
- 用户可见变化：`grayscale` 模式现在支持 `--grayscale-max-intensity`、`--grayscale-gamma`、`--grayscale-flat-darkening`、`--grayscale-detail-boost`、`--grayscale-tile-balance-strength`、`--grayscale-tile-balance-clip` 六个调参入口；正式运行后可直接查看 `preprocess_comparison.png` 判断原图灰度、处理后灰度和 stitched target 的差异，再查看 `target_energy_report.csv` 判断 9 个 tile 的亮度预算是否过于失衡。
- 算法细节：预处理先对非黑背景前景做稳健亮度归一化，再做 gamma 压缩；随后用 DoG 型局部细节信号把亮边缘和细节重新抬回来，避免“整体压暗后细节一起消失”；对低梯度大块区域仍保留压暗；最后在切成 3x3 tile 后，对非空 tile 做温和亮度预算均衡，减少个别 tile 过亮或过暗导致的单通道压力失衡。
- 验证方式：已运行 `py -m unittest discover -s tests -q`；已执行 `py -m holo_opt.grayscale_preview --input outputs/tmp_preprocess_probe/probe.png --size 64 --output-dir outputs/tmp_preprocess_probe/preview`；已执行 `py -m holo_opt.cli --target-mode grayscale --target-path outputs/tmp_preprocess_probe/probe.png --size 16 --epochs-per-chunk 2 --outer-loops 1 --device cpu --output-root outputs/tmp_preprocess_probe/runs --label grayscale_probe`，确认导出 `preprocess_comparison.png` 和 `target_energy_report.csv`。

## 2026-05-18 grayscale preview 多 preset 对比

- 改动动机：用户当前希望先逐步看“预处理前后效果”，不着急看 target 或最终全息优化结果，所以需要把 `grayscale_preview` 从单张处理图扩成可直接比较多种预处理风格的轻量预览工具。
- 影响范围：`holo_opt/grayscale_preview.py` 新增 `balanced`、`detail`、`budget` 三个预设风格，支持 `--preset` 一次输出多组处理结果，并新增 `*_comparison.png` 和 `*_preview_report.json`；`tests/test_grayscale_preview.py` 改为覆盖单 preset、多 preset、报告和参数解析；`README.md` 补充了 preview-only 的使用方式。
- 用户可见变化：现在可直接运行 `py -m holo_opt.grayscale_preview --input demo_preview.png --size 128 --preset balanced detail budget`，只看 `source RGB`、`source grayscale` 和三种处理结果的并排对比，不需要先进入 target 切块或正式优化。
- 算法细节：`balanced` 使用当前默认参数；`detail` 提高局部细节回提和亮区保留；`budget` 更强调压低峰值亮度和收紧 tile 预算波动。`*_preview_report.json` 会汇总每个 preset 的整体均值亮度、边缘密度、平坦区比例，以及 tile 预算缩放范围，方便在不跑优化的情况下先做输入图筛选。
- 验证方式：已运行 `py -m unittest tests.test_grayscale_preview -q` 和 `py -m unittest discover -s tests -q`；已执行 `py -m holo_opt.grayscale_preview --input outputs/tmp_preprocess_probe/probe.png --size 64 --preset balanced detail budget --output-dir outputs/tmp_preprocess_probe/preview_multi`，确认输出多张 preset 处理图、对比图和 JSON 报告。

## 2026-05-19 训练设备与耗时输出、速度优化

- 改动动机：正式优化运行时需要直接在终端看到实际使用的是 `cuda` 还是 `cpu`，并在结束时看到总耗时；同时当前训练循环在小尺寸上比参考脚本慢很多，主要瓶颈来自逐通道 FFT 循环和灰阶单调损失里的多层 Python/mask 循环。
- 影响范围：`holo_opt/runner.py` 增加实际设备打印、运行耗时统计和 CUDA 同步计时，并把每步损失分项的 CPU 同步合并；`holo_opt/field.py` 将 `compute_intensities()` 改为批量 FFT，将 `gray_monotonic_loss()` 改为 `scatter_add_` 聚合；`tests/test_field.py` 和 `tests/test_runner.py` 增加对应覆盖。
- 用户可见变化：每次运行 `py -m holo_opt.cli ...` 时，开始阶段会打印 `Device: cuda` 或 `Device: cpu`；导出完成后会打印 `Run finished in X.X s`。原有导出文件名、CSV/PNG/NPZ 内容约定不变。
- 算法细节：远场强度计算现在一次构造 `(channels, H, W)` phase stack 并批量执行 FFT，避免每个通道单独调用 FFT；灰阶单调损失按通道和灰阶 bin 聚合均值，只比较相邻且实际存在像素的灰阶对，不再跨过缺失灰阶比较。训练循环继续完整记录 `loss_terms.csv`，但减少了每步反复 `.cpu().item()` 带来的同步开销。
- 验证方式：需要运行 `py -m unittest discover -s tests -q`；建议再运行 `py -m holo_opt.cli --target-mode standard --size 64 --epochs-per-chunk 3 --outer-loops 1 --device cpu --output-root outputs/holo_experiments --label timing_smoke`。如果本机 CUDA 可用，再运行同等 `--device cuda` smoke，确认终端显示 `Device: cuda` 且流程完成。

## 2026-05-25 image_mse 能量匹配修正

- 改动动机：排查“结果图像质量太差、全部都是噪点”时发现，旧版 `image_mse` 使用每通道最大像素归一化后的强度图计算 MSE。远场亮噪点会主导最大值，导致 loss 可以下降，但 `summary.png` 仍然是雪花噪声。
- 影响范围：`holo_opt/field.py` 新增 `match_target_energy()`，`compute_loss_terms()` 的 `image_mse` 改为先把每个通道的重建强度按 target 总能量缩放，再与原始 target 计算 MSE；`tests/test_field.py` 新增回归测试覆盖该计算方式；`docs/ROADMAP.md` 同步更新当前基线。
- 用户可见变化：不新增命令行参数，不改变导出文件名。正式优化时，`image_mse` 会更直接惩罚散斑噪声和错误能量分布；判断效果仍优先看 `summary.png`、`outer_###_summary.png`、`loss_terms.csv` 和 `diagnostics.csv`。
- 算法细节：辅助损失仍使用原有接口；`gray_monotonic` 和 `background` 继续基于最大值归一化图像做灰阶/背景诊断，`image_mse` 则改为固定能量预算下的强度拟合，避免单个亮噪点把整个通道缩放到看似合理。
- 验证方式：本次需要运行 `py -m unittest tests.test_field -q`、`py -m unittest discover -s tests -q`，并建议运行 `py -m holo_opt.cli --target-mode standard --size 16 --epochs-per-chunk 2 --outer-loops 2 --device cpu --output-root outputs/holo_experiments --label smoke` 做流程 smoke。若要看质量变化，再用 CUDA 跑 `--size 64 --epochs-per-chunk 300 --outer-loops 5` 级别的诊断实验。

## 2026-05-25 多 seed 批处理入口

- 改动动机：固定单 seed 运行容易把随机初相的偶然失败误判为算法失败；多 seed 对比可以更稳地选择当前配置下最好的初始化。
- 影响范围：新增 `holo_opt/batch.py`，复用 `holo_opt.cli` 的参数解析和 `run_experiment()`；每个 seed 在批处理目录的 `runs/` 下生成独立实验目录；`tests/test_batch.py` 覆盖参数解析、默认 seed 行为和 summary 写出。
- 用户可见变化：新增命令 `py -m holo_opt.batch ... --seeds 42 43 44`。运行后会在 `outputs/holo_experiments/<label>_seeds_<timestamp>/seed_summary.csv` 写出各 seed 指标，并用 `best=1` 标出最低 score。
- 算法细节：不改变单次优化逻辑，不改变 score 公式；batch 只负责复制基础配置、替换 seed 和 label，然后汇总每次 `final_metrics["summary"]`。
- 验证方式：需要运行 `py -m unittest discover -s tests -q`；建议运行 `py -m holo_opt.batch --target-mode standard --size 8 --epochs-per-chunk 1 --outer-loops 1 --device cpu --output-root outputs/holo_experiments --label batch_smoke --seeds 1 2`，确认生成 `seed_summary.csv`。

## 2026-05-25 best state 选择指标配置

- 改动动机：综合 `score` 会混入灰阶单调性、效率均衡和平均效率，不等价于图像 MSE；当实验目标是直接压低 `image_error` 时，按 `score` 选最终 outer 可能选不到 MSE 最低的状态。
- 影响范围：`holo_opt/config.py` 新增 `selection_metric`；`holo_opt/cli.py` 新增 `--selection-metric`；`holo_opt/runner.py` 在保留诊断 `score` 的同时，按配置指标选择 best state；测试覆盖配置、CLI 和 runner 选择行为。
- 用户可见变化：默认仍是 `--selection-metric score`，历史命令行为不变。MSE 优先实验使用 `--selection-metric image_error`；可选值还有 `gray_level_error` 和 `efficiency_balance_penalty`。
- 算法细节：该参数只改变“保存哪一轮 outer 的结果”，不改变训练 loss，也不改变 `diagnostics.csv` 的记录。所有可选指标当前都是越低越好。
- 验证方式：需要运行 `py -m unittest discover -s tests -q`；MSE 优先实测可配合 `--gray-monotonic-weight 0.0` 做对照，判断灰阶单调项是否牵制了 `image_error`。

## 2026-05-25 实验性冗余入口移除

- 改动动机：完整图多通道布局和额外平坦区抑制损失在实测中没有成为 MSE 主线，继续保留会增加调参入口和判断成本。
- 影响范围：移除灰度完整图多通道 CLI/配置/target 分支，灰度图输入固定回 3x3 tile 分配；移除额外平坦区抑制损失的配置、CLI、训练项、导出项和测试。
- 用户可见变化：不再支持 `--grayscale-channel-layout` 和 `--speckle-weight`。当前推荐主线是默认 `grayscale` tiled target，加 `--selection-metric image_error` 做 MSE 优先选择。
- 算法细节：保留 `image_mse` 能量匹配修正、灰度预处理、tile 预算均衡、多 seed 批处理和 best state 指标选择；删除的两项不再出现在 `loss_terms.csv` 或 `loss_terms.png`。
- 验证方式：需要运行 `py -m unittest discover -s tests -q`，并至少跑一次小尺寸 CLI smoke，确认已删除参数不会影响标准训练流程。

## 2026-05-26 target 区域分区与 signal-window 损失

- 改动动机：全图 MSE 会让 phase-only 或近似 phase-only 模型在大面积平坦灰度区域硬拟合高频误差，导致主体结构出现但平坦区仍有明显雪花噪声；需要把边缘、主体、平坦区、暗区和 relaxed 区分开优化。
- 影响范围：新增 `holo_opt/region_masks.py` 和 `holo_opt/mask_preview.py`；`config.py`、`cli.py`、`runner.py`、`field.py`、`export.py` 接入区域 mask 和 opt-in 的 `signal_window` 图像损失；相关测试覆盖配置、CLI、mask 生成、导出、runner 和 field loss。
- 用户可见变化：新增 `py -m holo_opt.mask_preview ...` 预览命令；正式运行可用 `--region-mask-enabled --image-loss-mode signal_window`；导出新增 `mask_summary.png` 和 `region_mask_report.csv`，启用 signal-window 时 `loss_terms.csv` 增加 `signal_window`、`edge_mse`、`signal_mse`、`flat_lowpass_mse`、`relaxed_lowpass_mse`、`dark_leakage`。
- 算法细节：分区在最终 channel target 栈上逐通道生成，优先级为 `edge > signal > flat > dark > relaxed`；边缘和主体区强拟合，平坦和 relaxed 区只做低频拟合，暗区只惩罚超过阈值的亮泄漏。默认 `--image-loss-mode global` 保留历史行为，`signal_window` 和 `hybrid` 需要显式开启。
- 验证方式：需要运行 `py -m unittest discover -s tests -q`；建议运行 `py -m holo_opt.mask_preview --target-mode grayscale --target-path inputs/lineart_sources/test_input.png --size 64 --output-dir outputs/mask_preview` 和小尺寸 CPU smoke；质量实验建议在 CUDA 上对比 `global`、`signal_window` 和 `hybrid`。
