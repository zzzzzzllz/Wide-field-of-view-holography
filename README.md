# Wide-field-of-view-holography
这是一个大视场灰度全息影像的超表面设计与优化项目，当前实现的是基于 FFT 的 9 通道全息优化原型。

## 项目定位

- 目标：优化两组相位参数 `phdx` 和 `phdy`，让 9 个衍射通道重建出灰度阶梯图或用户提供的目标图。
- 当前重点：提高重建质量，并把失败原因变得可诊断，而不只是让 loss 数值下降。
- 当前不做：RCWA/FDTD、超表面单元库映射、GDS 导出、材料建模。

## 仓库入口

- 快速使用和维护上下文：`AGENT.MD`
- 核心训练主流程：`holo_opt/runner.py`
- 光场与损失项：`holo_opt/field.py`
- 导出与可视化：`holo_opt/export.py`
- VS Code 运行预设：`.vscode/launch.json`

## 运行平台约定

- 主要运行平台是 Windows，默认命令以 PowerShell 和 `py` 为准。
- macOS 主要用于阅读、review 和轻量修改，不作为主要实验运行平台。
- 如果你在 macOS 上只做了文档或少量代码调整，建议回到 Windows 再做完整测试和实验验证。

## Windows 常用命令

完整测试：

```powershell
py -m unittest discover -s tests -q
```

标准诊断运行：

```powershell
py -m holo_opt.cli --target-mode standard --size 64 --epochs-per-chunk 300 --outer-loops 5 --device cpu --output-root outputs/holo_experiments --label diagnostic --eta-balance-weight 0.05 --gray-monotonic-weight 0.1 --phase-smoothness-weight 0.0001 --background-weight 0.0
```

## 输出结果

一次运行通常会导出：

- `summary.png`：目标图和重建图总览
- `loss_curve.png`：总 loss 曲线
- `eta_curve.png`：各通道效率曲线
- `gray_levels.png`：灰度级响应或误差图
- `diagnostics.csv`：每轮 outer loop 的关键诊断值
- `loss_terms.csv` / `loss_terms.png`：各损失分项的历史

## 文档约定

- `README.md` 负责快速入口，不做大面积删减，优先持续补充。
- `AGENT.MD` 是完整维护手册。以后只要修改了算法、训练流程、导出结构、运行参数或测试方式，都应该同步更新 `AGENT.MD`。
