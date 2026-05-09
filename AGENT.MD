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
- `holo_opt/export.py`：导出运行目录、CSV/JSON/NPZ 数据以及 PNG 可视化图。
- `holo_opt/metrics.py`：重建质量、效率、灰度级误差等评估指标。
- `holo_opt/targets.py`：标准灰度阶梯目标和 `.mat` 目标加载。
- `holo_opt/weights.py`：基于效率和通道误差的通道权重更新。
- `tests/`：使用 `unittest` 编写的单元测试和导出/运行烟雾测试。
- `.vscode/launch.json`：VS Code 调试入口，适合不想手写命令时使用。

## 协作与平台约定

- 本仓库以 Windows 为主运行平台，以 PowerShell 命令和 `py` 启动方式为标准工作流。
- macOS 主要用于轻量阅读、文档补充、少量代码调整和 review；不要把仅在 macOS 验证过的行为表述成“项目已完全验证”。
- 写命令示例时，优先给出 Windows 可直接执行的版本；如果当前只在 macOS 做了静态修改或局部检查，要明确说明“Windows 侧需要按下列命令复核”。
- 新增逻辑时避免引入只适配单一平台的路径处理、shell 写法或缓存目录假设；路径拼接优先走 Python 标准库，不写死 macOS 风格路径。
- 若发现某个验证在当前机器上无法完成，要把原因写明白，例如 `torch` 缺失、CUDA 不可用、matplotlib/font cache 权限告警等，不要模糊带过。

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

- 已落地能力：
  - `LossConfig` 和 `diagnostic_interval` 已进入配置层。
  - `field.py` 已支持结构化损失项：`image_mse`、`eta_balance`、`gray_monotonic`、`phase_smoothness`、`background`。
  - `runner.py` 已记录 `loss_terms_history`、`diagnostics`、`outer_summaries`，并按 score 选最优状态。
  - `export.py` 已导出 `diagnostics.csv`、`loss_terms.csv`、`loss_terms.png`、`outer_###_summary.png`。
  - `cli.py` 和 `.vscode/launch.json` 已暴露 diagnostic/quality 工作流。
  - 相关测试已覆盖配置、损失项、导出和 runner 烟雾流程。
- 这轮设计的核心目标：
  - 不只是让 loss 下降，还要让失败原因可诊断。
  - 优先解决噪声重建、灰度不单调、通道能量塌缩、训练不稳定这几类问题。
  - 保持 FFT 原型架构，不扩展到 RCWA/FDTD、单元库映射、GDS 导出等更重的物理设计链条。
- 下一阶段更值得继续推进的方向：
  - 改进初始化策略，降低随机初相导致的不稳定性。
  - 基于真实实验结果继续调权重，而不是只停留在默认超参数。
  - 增加更明确的收敛诊断，例如 early-stop 依据、best-outer 对比、目标类型分支诊断。
  - 强化 Windows 环境下的一键运行体验和依赖说明，减少解释器不一致带来的误判。

## 推荐运行方式

优先使用 VS Code 的“运行和调试”面板：

- `Holo smoke: standard 16`：只检查流程能否跑通，结果质量通常没有参考价值。
- `Holo diagnostic: standard 64`：当前推荐的调试配置，会导出逐轮 summary、诊断 CSV 和损失分项。
- `Holo quick9: standard 128`：较快的标准 9 通道运行。
- `Holo quality: standard 128`：更长步数的质量优先配置，当前使用 `cuda`，没有 CUDA 时需要改成 `cpu` 或改用其他配置。
- `Holo MAT template`：读取用户自己的 `.mat` 目标图，需要先把 `--target-path` 改成真实路径。

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
- `loss_terms.png`：`image_mse`、`eta_balance`、`gray_monotonic`、`phase_smoothness` 和 `background` 五个损失分项的曲线。

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

## 维护注意事项

- 不要改变已有导出文件名，测试、VS Code 配置和历史结果依赖这些约定。
- 修改训练损失时，同步检查 `tests/test_field.py` 和 `tests/test_runner.py`。
- 修改导出文件、图像或 CSV 时，同步检查 `tests/test_export.py` 和 `tests/test_runner.py`。
- 新增导出数据时，优先在 `export_results()` 中集中处理，并补充非空文件或内容校验测试。
- 训练进度目前每 500 步在终端打印一次当前 step 和 total loss。
- 输出目录下可能存在大量历史实验结果，除非明确要求，不要清理或覆盖。
- 仓库里可能有未提交实验输出、VS Code 本地配置或资料文件；提交前只 stage 本次任务相关文件。
- 如果修改了算法设计思路但没有同步 `AGENT.MD`，这次任务就算没有完成。
