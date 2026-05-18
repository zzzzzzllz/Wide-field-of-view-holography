# 项目路线图与推进方向

本文件是当前项目后续推进的正式计划入口。旧的 `docs/superpowers/plans/2026-05-07-holography-quality-optimization.md` 是已经落地的历史实施计划，不再作为下一阶段任务清单使用。

## 当前基线

项目当前已经具备：

- 基于 FFT 的 9 通道全息优化主流程。
- `standard`、`mat`、`lineart`、`grayscale` 四类 target 输入。
- 结构化损失项：`image_mse`、`eta_balance`、`gray_monotonic`、`phase_smoothness`、`background`。
- 诊断导出：`diagnostics.csv`、`loss_terms.csv`、`loss_terms.png`、`outer_###_summary.png`、`stitched_comparison.png`。
- 仓库协作规范 skill：`skills/holography-workflow/SKILL.md`。
- 输入输出目录规则：`inputs/` 和 `outputs/` 只保留 `.gitkeep`，不提交实验图片或结果。

## 文件架构

后续维护按 5 个层次理解：

1. **项目说明与计划**
   - `README.md`：快速上手、常用命令、结果查看顺序。
   - `AGENT.MD`：完整维护上下文、协作规则、算法演进状态。
   - `docs/ROADMAP.md`：当前后续推进方向。
   - `skills/holography-workflow/SKILL.md`：Codex 和协作者执行任务时的流程规范。

2. **目标图像生成与输入准备**
   - `holo_opt/targets.py`：标准灰度阶梯和 `.mat` target。
   - `holo_opt/line_targets.py`：线稿 target、灰度 target、3x3 通道切分。
   - `holo_opt/lineart_preview.py`、`holo_opt/grayscale_preview.py`：正式优化前的输入预览。

3. **片上结构代理优化与远场模拟**
   - `holo_opt/field.py`：FFT 远场模拟和损失项。
   - `holo_opt/weights.py`：通道权重动态更新。
   - `holo_opt/runner.py`：训练主循环、诊断记录、best state 选择。

4. **评估、导出与结果管理**
   - `holo_opt/metrics.py`：图像误差、灰度误差、效率和 score。
   - `holo_opt/export.py`：图像、CSV、JSON、NPZ、相位结果导出。

5. **验证**
   - `tests/`：所有单元测试和 smoke 测试。
   - `.vscode/launch.json`：常用运行配置。

## 后续推进方向

### 1. 球形视场与平面 target 的几何映射

目标：解决当前正方形平面 target 与真实球形视场之间的畸变问题。

建议先做：

- 新增独立几何层，例如 `holo_opt/fov.py`。
- 保留 `planar` 模式作为对照。
- 新增 `spherical_fov` 模式，把平面图像映射到角度坐标，再投影回优化网格。
- 导出 `fov_mapping.png`，显示原平面 target、球面重采样 target、反投影误差。

适合先推进的原因：这是项目物理目标和当前代码模型之间最大的结构性缺口。

### 2. 灰度质量与能量预算优化

目标：让灰度不只是“像不像”，还要在固定通道能量下稳定、可分、不过曝。

建议先做：

- 增加灰度动态范围指标、暗部噪声指标、亮部饱和指标。
- 对 `standard`、`lineart`、`grayscale` 使用不同灰度诊断。
- 在 `grayscale` 预处理里加入更明确的能量预算报告。
- 导出 `preprocess_comparison.png` 和 `target_energy_report.csv`。

适合先推进的原因：你当前明确提到灰度优化差、单通道能量有限，这条线最直接改善图片质量。

### 3. 输入图片适配与 image2 辅助 target 设计

目标：在进入全息优化前，把普通图片处理成更适合固定能量、多通道重建的 target。

当前已落地的第一步：

- `grayscale` 路径已经补上局部细节保留、低梯度区域压暗、3x3 tile 温和预算均衡，以及 `preprocess_comparison.png` / `target_energy_report.csv` 两类输入适配诊断。
- 后续这条线继续推进时，不要重复实现基础灰度压缩，重点往 `preprocess_mode` 抽象、输入图可读报告增强和 `image2_assisted` 候选图生成上走。

建议先做：

- 形成 `preprocess_mode`：`none`、`grayscale_budget`、`lineart`、`image2_assisted`。
- 对输入图输出可读报告：亮度分布、边缘密度、平坦区域比例、预计能量压力。
- image2 只用于生成更适合全息显示的候选图；最终好坏仍由本项目 metrics 和重建图判断。

适合先推进的原因：它能最快让你看到“同一张图处理前后重建效果是否变好”。

### 4. 收敛稳定性、多 seed 与实验批处理

目标：减少随机初相造成的偶然失败，让实验结果更可复现。

建议先做：

- 新增 batch runner，例如 `holo_opt/batch.py`。
- 支持多个 seed 自动运行，生成 `seed_summary.csv`。
- 增加 best-seed、best-outer、early-stop 诊断。
- 保持单次 runner 行为不破坏。

适合先推进的原因：当前优化对初始相位敏感，多 seed 是低风险、高收益的稳定性补强。

### 5. 工作流与实验报告自动化

目标：让每次改动后“怎么验证、看什么文件、结论是什么”变成固定输出。

建议先做：

- 每次实验自动导出 `experiment_report.md`。
- 报告包含命令、配置、核心指标、关键图片列表、失败模式判断。
- PR 模板引用报告路径，减少人工解释成本。
- 后续可加 GitHub Actions 只跑测试，不上传本地实验输出。

适合先推进的原因：这条线不直接改善算法，但能显著改善协作和复现实验。

## 推荐选择顺序

如果目标是先解决物理模型正确性，选 **1 球形视场与平面 target 的几何映射**。

如果目标是先让图片效果更好，选 **2 灰度质量与能量预算优化**。

如果目标是先让输入图更适合项目，选 **3 输入图片适配与 image2 辅助 target 设计**。

如果目标是先让实验结果更稳定，选 **4 收敛稳定性、多 seed 与实验批处理**。

如果目标是先让你和同学协作更顺，选 **5 工作流与实验报告自动化**。

我的建议：优先选 **1** 或 **2**。如果你最关心“球形视场真实物理问题”，从 1 开始；如果你最关心“当前图像灰度效果差”，从 2 开始。

## 每条方向的共同验收标准

- 必须补测试或说明为什么不需要测试。
- 必须运行：

```powershell
py -m unittest discover -s tests -q
```

- 如果改动影响实验输出，必须给出一个小尺寸 smoke 命令。
- 如果改动影响算法行为，必须同步更新 `AGENT.MD` 和本文件。
- 不提交 `inputs/`、`outputs/` 中的真实实验图片或结果。
- 默认分两步交付：实现后先给用户检查，不直接提交；用户检查确认后，再单独执行 stage、commit、push 或 PR。
