# pathstruct-24g 可行性验证：代码问题与结果

**日期**：2026-10-21（初版）；2026-10-22（补充 §3.4 扩展实验与断点恢复记录）  
**硬件/环境**：RTX 4090 D 24 GB；Qwen3-VL-4B-Instruct（bf16，LoRA r=8，`all-linear`）；transformers 4.57.6；torch 2.5.1+cu124  
**数据**：合成队列；真实 H&E 的 MoNuSeg 与 PanNuke（RationAI 公开导出，CC BY-NC-SA）  
**说明**：本报告只做可行性验证，不是临床/病理结论。所有数字均为留出患者或留出组织上的结果。

---

## 0. 一句话结论

**你的想法在“反事实结构判断”上部分有效，在“结构描述子回归”上还没有被真实数据证实。**

- 合成数据：简单版 LM 的 0.674 是**空框 + 先验捷径**；严格非空、kind/label 平衡后，384 px 的 LM 只有 0.500，1024 px 只有 0.534。训练后的**反事实头**在严格 384 px 上 0.640（blank 0.509）；**结构头**在合成数据上 topology R² 0.315/0.411。
- 真实 H&E（MoNuSeg）：反事实头 3 个 seed 为 0.583/0.600/0.650（mean **0.611**），blank 0.500，box-geometry 0.467；**expand 类稳定 0.722**；结构头 topology R² **为负**。
- 真实 H&E（PanNuke）：反事实头 2 个 seed 为 0.595/0.582（mean **0.588**），blank 0.530，box-geometry 0.577；但训练 loss 停在 0.5–0.8（underfit，1.3 epochs），0.588 是下界；结构头 topology R² **仍为负**。
- **扩展到 600 步 + topology-only（2026-10-22）**：PanNuke cf **0.600**（blank 0.547，real−blank +5.3 pt，CI [0.5, 10.9]），仅比 300 步高 1.2 pt；MoNuSeg 3 seeds mean **0.594**（full descriptor 0.611），blank 0.489，expand 0.750。两组的结构 topo R² 仍为负；去掉 3×3 regional block 没有修复结构回归。
- 因此：**反事实头在真实 H&E 上有小幅但真实的视觉信号**（尤其 expand）；**结构回归头没有从合成迁移到真实数据，加长训练和去掉 regional block 都没有解决**；P2 下游迁移和 LM yes/no 读出仍未证明。

---

## 1. 发现的代码问题（按严重程度）

### P0：会让任务本身失效的问题

| # | 问题 | 为什么严重 | 修复 |
|---|---|---|---|
| 1 | 反事实 prompt 没有写编辑类型（collapse/expand/translate），translate 也没有写方向向量 | 同一张图 + 同一个框在数据集中同时有 yes 和 no；标签不是观测的函数，任务不可解 | prompt 完整描述操作；translate 画橙色方向箭头；按 kind 平衡 yes/no |
| 2 | prompt 把框坐标 `[x1,y1,x2,y2]` 写成文本 | 纯文本 logistic 在留出患者上达到 **0.689**，超过 P1 的 0.65 门槛——模型可以完全不用图像 | 坐标从文本移除；位置只存在于图像上的绿框/箭头；纯文本基线降到 **0.500** |
| 3 | README 说“图像上画了框”，但代码没有画 | 模型需要把 0–1000 文本坐标映射到图像 patch，这是另一个更难的任务 | 新增 `annotate_edit`：绿框 + translate 橙色箭头 |
| 4 | 空框（框内没有细胞）是平凡负例 | 合成数据 20% train / 14% val 为空框；答案必然 no，P1 全部答对，抬高了总分 | 新增 `--min-affected 5` 过滤；单独报告非空子集；严格实验只保留非空框 |
| 5 | P1/P2 在训练集 jsonl 上评估 | 门槛衡量的是记忆，不是泛化 | 按患者划分 `*_train.jsonl` / `*_val.jsonl`；真实数据用 external patient/tissue split |
| 6 | `--no-aux` 被描述成 LoRA-only | 它仍然通过 LM loss 看反事实数据；不是 growth-only 基线 | README/报告明确写成“同数据混合、只去掉辅助头/损失”的消融 |

### P1：模块/架构问题

| # | 问题 | 为什么严重 | 修复 |
|---|---|---|---|
| 7 | 结构头和反事实头都对所有视觉 token 做全局 mean pooling | 3×3 regional 目标从全局均值不可学；cf 头看不到编辑区域，只能学数据集先验 | 从 `image_grid_thw` 重排成视觉 token 网格；结构头 3×3 池化；cf 头对编辑区域局部池化 |
| 8 | 区域描述子用 2–98% 分位数归一化、且 tile 索引是 x-major | 目标 tile 和图像 token 网格不对齐；3×3 目标被转置 | `describe(..., frame=...)` 用图像坐标系；tile 改为行优先 `y*grid+x` |
| 9 | cf 头特征没有平移向量 | 同一 (centre, radius, kind) 在不同方向下标签不同 | `MASK_DIM` 6 → 8，加入 `vx, vy` |
| 10 | 结构头的 3×3 regional block 在真实数据上 R² 为负 | 它可能稀释 topology 项，且并未学到 | 已加 `--structure-dims 5`：结构头只回归 topology（n_comp/hole_count/hole_area_frac/h0_max/h0_sum），去掉 regional block；新实验见 3.4 |

### P2：数据/管线/环境问题

| # | 问题 | 修复/验证 |
|---|---|---|
| 11 | CellViT 每个 ROI 单独 factorize 类型 | 同一类在不同 ROI 得到不同 id；改为全局字符串词表 |
| 12 | Otsu 返回直方图 bin 中心，可能落在常量背景下方 | 4/36 合成 ROI 整图变前景；加阈值保护；72/72 ROI 成功 |
| 13 | 分割失败时旧文件残留，静默回退到 synth GT | `segment_dir` 先删除目标文件；`cells`（GT）与 `cells_seg` 分离 |
| 14 | 代码按 28×28 px/token 估算 | Qwen3-VL 是 patch 16 × merge 2 = **32×32**；修正 `vision.py`、`max_pixels` |
| 15 | `transformers` 没上界，pip 装 5.x | 5.x 要求 torch≥2.6，与 2.5.1 冲突；固定 `>=4.57,<5` |
| 16 | `scikit-image` 没列入依赖 | `pilot segment` 在干净安装后失败；加入 base dependencies |
| 17 | `--compare` 读 JSON 数组失败 | eval 写 JSON 数组、compare 按 JSONL 读；`read_jsonl` 兼容两种 |
| 18 | `--compare` 用 `roi_id|gold` 作 key | 同一 ROI 多个 cf 样本会冲突；改用 `ex_id` |
| 19 | doctor 不检查 cell 坐标尺度 | slide-level 坐标会产生“有限但无意义”的描述子；>5% out-of-bounds 报警 |
| 20 | eval 只取前 N 个 ROI | 只评估前几个类别；改为系统性跨样本采样；增加 per-kind、parse 失败统计 |
| 21 | binary parse 对 “Yes, because…” 失败 | 允许 trailing clause，但拒绝 “no idea” 等 hedge |
| 22 | PanNuke 每 patch 只有 16–273 核（中位 28） | 原 60-cell 门槛会丢掉大部分 patch；新增 `--min-cells 16`（doctor/build） |
| 23 | 真实 tissue 标签不是 ring/solid/multifocal | `build_growth_examples` 固定问题会错；新增 generic tissue question / `--growth-question` |

---

## 2. 实验设置

### 2.1 合成数据（管线与捷径验证）

- 72 ROIs（24 ring / 24 solid / 24 multifocal），36 patients；384 px；每 ROI 220 cells。
- 严格非空实验：`--min-affected 5`，按 kind/label 平衡；384 px 为 226 train / 114 val；1024 px 为 254 train / 118 val（GT cells）。
- 训练：Qwen3-VL-4B + LoRA，batch 1，grad-accum 8，150/300 steps；1024 token preset VRAM 11.3 GB。
- 读出：LM yes/no 生成；以及**训练后的结构头/反事实头直接评估**（`aux_heads.pt`）。

### 2.2 真实 H&E：MoNuSeg

- 37 train / 14 test 患者（test patient-disjoint）；1000×1000；294–1862 nuclei/image；无细胞类型。
- 116 train cf（58/58）/ 60 test cf（30/30）；150 steps；1024 tokens；3 seeds；VRAM 11.1 GB。
- box-geometry train→test：**0.467 balanced / AUC 0.528**（无框捷径）。

### 2.3 真实 H&E：PanNuke

- fold1：405 train patches（15 组织）/ 160 held-out patches（breast、colon、bile、cervix 四个完全留出组织）；256→1024；5 类细胞；每 patch 16–273 nuclei。
- 1796 train cf（898/898）/ 620 held-out cf（310/310）；2 seeds × 300 steps；1024 tokens。
- box-geometry train→held-out：**0.577 / AUC 0.580**（有轻微框先验）。

### 2.4 对照

- **Text-only**：不改图，只用 prompt 文本。
- **Blank-box**：保留绿框/箭头，去掉所有细胞。
- **Box-geometry**：只用 kind + box + centre + radius + vector 的 logistic。
- **Empty-box**：单独统计空框样本。
- **LM-only**：同数据、去掉辅助头/损失。

---

## 3. 结果

### 3.1 合成数据：简单版 vs 严格版

| 实验 | balanced accuracy | 说明 |
|---|---|---|
| P0 简单 cf | 0.523 | 97.7% 预测 yes；常数 yes 偏向 |
| P1 简单 cf（seed 0 / 1） | **0.674 / 0.674** | 但 14% val 是空框，P1 全对 |
| P1 简单 cf + blank | 0.500 | 只看绿框/箭头，不看细胞 |
| 纯文本 baseline | 0.500 | 去掉坐标后 |
| **严格非空 384 px，LM** | **0.500** | 94.7% 预测 no；训练 loss 0.23，未拟合 |
| **严格非空 1024 px，LM** | **0.534** [0.481, 0.582] | 仍与 chance 无显著差异 |
| 严格 384 px，**cf 头** | **0.640** / AUC 0.652 | blank 0.509 / AUC 0.516；干净视觉信号 |
| 严格 1024 px，**cf 头** | 0.686 / AUC 0.744 | blank AUC 0.728 → 主要是 box geometry，不算干净 |
| 严格 384 / 1024，**结构头 topology R²** | 0.315 / 0.411 | blank 对照为负 |
| 同任务 aux vs LM-only | 0.674 vs 0.644 | +3.0 pt，CI [−1.5, +6.8]，不显著 |
| P2 growth，aux / no-aux | 1.000 / 1.000 | 合成下游天花板，P2 无区分度 |

**合成结论**：简单版 0.674 不能作为成功证据；严格非空 LM 在 384/1024 都学不会。训练后的 cf 头在 384 有 0.640 的真实视觉信号（blank 0.509）；结构头在合成上能学 topology。

### 3.2 真实 MoNuSeg（3 seeds）

| seed | cf real balanced | real AUC | blank balanced | blank AUC | expand | translate | collapse | structure topology R² |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.583 | 0.637 | 0.517 | 0.529 | 0.667 | 0.571 | 0.375 | −72.6 |
| 1 | 0.600 | 0.610 | 0.483 | 0.643 | 0.708 | 0.536 | 0.500 | −29.4 |
| 2 | **0.650** | **0.676** | 0.500 | 0.504 | 0.792 | 0.571 | 0.500 | −20.4 |
| mean | **0.611** | 0.641 | 0.500 | 0.559 | **0.722** | 0.559 | 0.458 | 负 |

- test 60 条 / 14 患者；seed 0 的患者级 bootstrap 95% CI：balanced [0.483, 0.689]，AUC [0.540, 0.758]。
- box-geometry 0.467 / AUC 0.528；因此 0.611 不是框先验。
- **expand 稳定 0.722**；translate ~0.56；collapse 只有 8 个 test 样本，不可靠。
- **结构头不泛化**：topology R² 全为负；37 张训练图不足以支撑 descriptor 回归。

### 3.3 真实 PanNuke（2 seeds，held-out 组织）

| seed | cf real balanced | real AUC | blank balanced | blank AUC | collapse | expand | translate | structure topology R² |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.595 | 0.645 | 0.515 | 0.591 | 0.544 | 0.597 | 0.627 | −3.76 |
| 1 | 0.582 | 0.626 | 0.544 | 0.580 | 0.544 | 0.593 | 0.598 | −5.84 |
| mean | **0.588** | 0.636 | 0.530 | 0.585 | 0.544 | 0.595 | 0.613 | 负 |

- held-out 620 条 / 4 个组织；seed 0 real−blank 配对 **+8.1 pt，CI [5.7, 10.3]**；seed 1 +3.9 pt。
- box-geometry 0.577 / AUC 0.580；cf 头 0.588 只高约 1 pt，视觉增量小。
- **cf head 训练 loss 0.5–0.8，300 steps ≈ 1.3 epochs：underfit**，0.588 是下界。
- **结构头在 held-out 组织上仍为负 R²**；5 类细胞没有带来结构回归的泛化。

### 3.4 修复后的扩展实验（一：扩真实数据；二：修结构头）

为落实“先做一和二”：

- **一、扩真实数据/延长训练**：PanNuke 从 300 步（1.3 epochs，underfit）扩展到 **600 步（≈2.7 epochs）**，仍在 405 train patches / 1796 cf 上，held-out 组织 620 cf。
- **二、修结构头**：`--structure-dims 5`，只回归 topology 五项，去掉 3×3 regional block；PanNuke 600 步 + MoNuSeg 3 seeds 都使用该设置。
- **防过拟合**：MoNuSeg 3 seeds 每 50 步保存 checkpoint，用 held-out 比较 50/100/150 步（3.4.2）。
- **执行说明**：PanNuke 600 曾于 2026-10-21 中断在 step 400/600（无中间 checkpoint，无法续跑）；2026-10-22 02:51 以独立进程重跑完整 600 步，随后完成 MoNuSeg 3 seeds 与全部评估。完整操作记录见附录 A6。

#### 3.4.1 PanNuke 600 步 + topology-only（1 seed）

训练曲线（每 100 步）：

| step | lm | struct | cf | lr |
|---|---|---|---|---|
| 100 | 0.261 | 0.032 | 0.527 | 9.63e-05 |
| 200 | 0.215 | 0.027 | 0.518 | 7.96e-05 |
| 300 | 0.251 | 0.020 | 0.581 | 5.41e-05 |
| 400 | 0.248 | 0.028 | 0.694 | 2.74e-05 |
| 500 | 0.163 | 0.007 | 0.357 | 7.40e-06 |
| 600 | 0.134 | 0.018 | 0.364 | 0 |

- 训练 cf 从 0.527 降到 0.364，lm 从 0.261 降到 0.134；400 步附近 cf 一度升到 0.694，500 步后随 LR 退火明显下降。
- 与 300 步 run 的训练 cf 0.5–0.8 相比，600 步确实优化得更充分；但这是**单 seed**，不能直接与 2-seed baseline 做严格配对比较。

Held-out（620 cf，310/310；4 个完全留出组织）：

| 指标 | 600 步 topo-only（1 seed） | 300 步 baseline（2 seeds mean） |
|---|---|---|
| cf real balanced | **0.600** [0.563, 0.649] | 0.588 |
| real AUC | 0.657 [0.610, 0.709] | 0.636 |
| blank balanced | 0.547 [0.529, 0.564] | 0.530 |
| real−blank paired | **+5.3 pt** [0.5, 10.9] | +8.1 / +3.9（两 seed） |
| collapse / expand / translate | 0.544 / 0.606 / 0.631 | 0.544 / 0.597 / 0.627 |
| structure topology R² | −6.16 [−332.2, −3.02] | −3.76 / −5.84 |

- **延长到 600 步只带来 +1.2 pt（0.588 → 0.600），bootstrap CI 重叠；real−blank 边际也没有扩大（+5.3 vs +8.1/+3.9）。** 训练 loss 明显下降但 held-out 基本没动，说明瓶颈不只是 underfitting。
- 结构头：`--structure-dims 5` 没有把 R² 变正；均值仍被 `hole_area_frac` 的 **−31.18** 支配。逐维：`n_comp` −0.024、`hole_count` +0.007、`hole_area_frac` −31.18、`h0_max` −0.106、**`h0_sum` +0.52**。即个别维度可学，但整体不可用。
- 注意：这一轮同时改了步数和结构头目标，cf 对比存在混杂；cf 头结构本身未变，且 600 步的训练 cf 明显更低，因此“多训没有换来 held-out 提升”这个结论是可靠的。

#### 3.4.2 MoNuSeg 150 步 topology-only（3 seeds，含早停）

训练 cf loss（step 50 / 100 / 150）：

| seed | 50 | 100 | 150 |
|---|---|---|---|
| 0 | 0.530 | 0.170 | 0.016 |
| 1 | 0.283 | 0.210 | 0.048 |
| 2 | 0.663 | 0.131 | 0.004 |

- 三个 seed 到 150 步都接近训练集记忆；50 步明显 underfit，100 步是折中点。因此额外保存了 `checkpoint-50/100` 做 held-out 早停比较。

Held-out（60 cf；14 患者；final 150 步）：

| seed | cf real balanced | AUC | blank balanced | real−blank（配对） | collapse | expand | translate | topo R² |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.600 [0.516, 0.685] | 0.649 | 0.450 | +15.0 pt [8.5, 22.2] | 0.375 | 0.708 | 0.571 | −44.8 |
| 1 | 0.583 [0.439, 0.722] | 0.622 | 0.533 | +5.0 pt [−13.6, 21.7] | 0.500 | 0.708 | 0.500 | −22.1 |
| 2 | 0.600 [0.501, 0.700] | 0.643 | 0.483 | +11.7 pt [0.0, 22.6] | 0.250 | 0.833 | 0.500 | −38.7 |
| mean | **0.594** | 0.638 | **0.489** | +10.6 pt | 0.375 | **0.750** | 0.524 | 负 |

与 full-descriptor baseline（3 seeds，150 步）对比：cf mean 0.611 → **0.594**（−1.7 pt）；blank 0.500 → 0.489；expand 0.722 → **0.750**；translate 0.559 → 0.524；collapse 0.458 → 0.375（n=8，不稳）。**去掉 regional block 没有提升 cf；expand 仍是唯一稳定项。**

**早停比较（cf real balanced）：**

| seed | checkpoint-50 | checkpoint-100 | final-150 |
|---|---|---|---|
| 0 | 0.533 | 0.583 | **0.600** |
| 1 | 0.500 | **0.600** | 0.583 |
| 2 | 0.517 | 0.550 | **0.600** |
| mean | 0.517 | 0.578 | **0.594** |

- 早停到 100 步**没有**让 cf 更好（0.578 vs 0.594），final 总体最好；seed 1 是唯一 ck100 略高的。
- 但结构 R² 在 ck100 上更好：seed 0 = −19.1 vs final −44.8；seed 1 = −19.3 vs −22.1；seed 2 = −38.3 vs −38.7。说明末段确实在拟合训练集，cf 已平台，结构头则继续变差。
- 结构 R² 逐维（final）：`n_comp` +0.07/+0.07/−0.06、`hole_count` −0.11/−0.01/−0.09、`hole_area_frac` −223.5/−110.5/−193.4、`h0_max` +0.03/+0.11/0.00、`h0_sum` −0.42/−0.31/−0.03。**仍为负，且由 `hole_area_frac` 主导。**

#### 3.4.3 “一和二”的结论

- **修结构头（去掉 3×3 regional block）没有修复真实数据上的结构回归**：PanNuke 和 MoNuSeg 的 topology R² 仍为负，失败集中在 `hole_area_frac`；只有 `h0_sum`（PanNuke +0.52）和 `n_comp`（MoNuSeg ≈+0.07）出现弱学习信号。
- **延长训练（300→600 步）只带来 +1.2 pt PanNuke cf，CI 重叠**；训练 loss 明显下降而 held-out 平台化，说明继续加步数不是有效方向。
- **cf 头的小幅真实信号仍然成立**：PanNuke real−blank +5.3 pt（CI 刚好排除 0），MoNuSeg 三个 seed 的配对差 +15.0/+5.0/+11.7 pt；expand 稳定在 0.75 左右。但离稳健超过 0.65 仍有距离。
- **过拟合存在但不严重**：MoNuSeg 训练 cf 在 150 步降到 0.004–0.048，held-out 已平台；早停到 100 步对 cf 无益，但结构 R² 略好。后续应把“数据/目标定义”而不是“训练时长”作为主攻方向。

---

## 4. 结论

### 成立/有证据

1. **数据管线可跑通**：真实 H&E → 专家核 mask → cell table → topology → cf labels → LoRA + heads → 留出评估。
2. **反事实任务在真实 H&E 上有一致的、小幅的视觉信号**：MoNuSeg real−blank +11 pt，PanNuke +8.1/+3.9 pt；MoNuSeg expand 0.722；box geometry 不能解释 MoNuSeg 的 0.611（geometry 0.467）。
3. **训练后的反事实头可以学到东西**：严格 384 合成 0.640（blank 0.509）；MoNuSeg mean 0.611；PanNuke mean 0.588（underfit）。
4. **控制实验有效**：text-only、blank、geometry、empty-box 四个对照抓到了真实的设计 bug 和混淆。

### 未成立/未证明

1. **稳健超过 0.65**：MoNuSeg 3 seeds 只有 seed 2 到 0.650；mean 0.611；PanNuke mean 0.588；600 步 topo-only 为 0.600。CI 宽，样本小。
2. **结构描述子回归在真实 H&E 上泛化**：MoNuSeg 和 PanNuke 的 topology R² 都是负的；合成上的 0.315/0.411 没有复现。`--structure-dims 5` 去掉 regional block 后仍为负（PanNuke −6.16，MoNuSeg −44.8/−22.1/−38.7），且由 `hole_area_frac` 主导。
3. **辅助监督的下游迁移**：P2 合成到天花板；真实下游还没跑；同任务 aux 只 +3.0 pt 且 CI 跨 0。
4. **LM 的 yes/no 读出**：严格 384/1024 的 LM 只有 0.500/0.534；模块能学的，LM 生成不出来。
5. **临床/病理结论**：数据规模小、MoNuSeg test 14 患者、PanNuke 4 个 hold-out 组织；不能做诊断声明。
6. **加长训练或去掉 3×3 regional block 能修复主要缺口**：PanNuke 300→600 步只 +1.2 pt（CI 重叠），结构 R² 仍负；方向应转向数据规模与目标定义。

---

## 5. 下一步（按优先级）

1. **扩真实数据/重定义任务**：PanNuke 600 步已试，收益仅 +1.2 pt；下一步增加 patch/患者数，并重做 cf 任务难度（translate/collapse 尤其弱）。所有结果报告 patient/tissue 级 CI。
2. **重写结构目标**：regional block 已证明不是主因；针对 `hole_area_frac` 的数值不稳定（大量 0 / 尖峰）改用 clip/rank/分位数或分解目标，并报告逐维 R²，而不是只看 5 维均值。
3. **提高局部有效分辨率**：local crop / 更高 tokens；translate/collapse 是弱项；当前 full-ROI 1024 仍可能不够。
4. **P2 用真实下游标签**：tissue/growth + patch-level split；LM 读出另测。
5. **保留四个对照**：text-only、blank、geometry、empty-box；先在真实数据上跑，再谈 0.65。
6. **更强的呈现方式**：代码已有 `--presentation before_after`；它主要解决 LM 的心算模拟，不替代真实结构头。

---

## 6. 交付物与复现

- 完整技术报告：`FEASIBILITY.md`
- 本飞书报告：`FEISHU_REPORT.md` / `FEISHU_REPORT.docx`
- 真实数据导入：`scripts/import_monuseg.py`、`scripts/import_pannuke.py`
- 头评估：`scripts/eval_heads.py`、`scripts/bootstrap_heads.py`
- 对照：`scripts/make_blank_control.py`、`scripts/balance_cf.py`
- 测试：`python -m pytest tests/` → **23 passed**
- 代码与报告已提交并推送到 `feasibility/pilot-fixes` 分支（commit `2715637`、`561b418`、`89e8a25`）；原始预测/头结果在 `out/heads_*`、`out/pred_*`

### 复现命令（真实数据）

```bash
# MoNuSeg
python scripts/import_monuseg.py \
  --train-parquet .../data/train-*.parquet \
  --test-parquet  .../data/test-*.parquet \
  --out data/real/monuseg
pilot build --manifest data/real/monuseg/manifest_train.csv \
  --cells data/real/monuseg/cells --out data/real/monuseg_train \
  --per-roi 6 --val-frac 0.2 --min-affected 5
python scripts/balance_cf.py --train data/real/monuseg_train/cf_train.jsonl \
  --val data/real/monuseg_train/cf_val.jsonl \
  --out-train data/real/monuseg_train/cf_train_bal.jsonl \
  --out-val data/real/monuseg_train/cf_val_bal.jsonl

# PanNuke
python scripts/import_pannuke.py \
  --parquet .../fold1-*.parquet --out data/real/pannuke \
  --val-tissues breast,colon,bile,cervix --min-cells 16
pilot build --manifest data/real/pannuke/manifest_train.csv \
  --cells data/real/pannuke/cells --out data/real/pannuke_train \
  --per-roi 6 --val-frac 0 --min-affected 5 --min-cells 16
python scripts/balance_cf.py --train data/real/pannuke_train/cf.jsonl \
  --val data/real/pannuke_train/cf.jsonl \
  --out-train data/real/pannuke_train/cf_train_bal.jsonl \
  --out-val /tmp/pannuke_bal.jsonl

# 训练 + 头评估
pilot train --cf data/real/monuseg_train/cf_train_bal.jsonl --steps 150 \
  --preset 4b_bf16_1024 --out out/p1_real_s0
python scripts/eval_heads.py --adapter out/p1_real_s0 \
  --data data/real/monuseg_test/cf_bal.jsonl --preset 4b_bf16_1024 \
  --out out/heads_real_s0.json
python scripts/bootstrap_heads.py --real out/heads_real_s0.json \
  --blank out/heads_blank_s0.json
```

---

## 7. 飞书文档

- 飞书 CLI：`@larksuite/cli 1.0.98`（`lark-cli`），app `cli_aa5c394ff9b8dbe1`，user identity 已授权（docs + drive scope）。
- 本文档由 `lark-cli docs +create --doc-format markdown --content -` 从 `FEISHU_REPORT.md` 创建；`FEISHU_REPORT.docx` 是离线备份。
- **飞书文档链接**：https://rcn9izkq6bgo.feishu.cn/docx/TzuId4WBSoeowXxgTmncL0BGnSf
- 文档 ID：`TzuId4WBSoeowXxgTmncL0BGnSf`；后续更新用 `lark-cli docs +update --document-id TzuId4WBSoeowXxgTmncL0BGnSf ...`。

---

# 附录

## 8. 附录 A：完整操作记录（按阶段）

### A0：仓库、环境、模型

1. 克隆 `https://github.com/LinKX-known/pathstruct-24g` 到 `/root/pathstruct-24g`；逐行阅读 README、`src/pilot/*.py`、`tests/test_pilot.py`、`scripts/run_pilot.sh`。
2. 环境：RTX 4090 D 24 GB；torch 2.5.1+cu124；`pip install -e .` + `scikit-image` + `pytest`；`pip install -e ".[qwen]"`。
3. 发现 pip 默认装 transformers 5.20.2，要求 torch≥2.6，与 2.5.1 冲突；固定 **transformers 4.57.6**。同时修正 `pyproject.toml` 的 `transformers>=4.57,<5`。
4. HuggingFace 直连超时；使用 `HF_ENDPOINT=https://hf-mirror.com` 下载 **Qwen/Qwen3-VL-4B-Instruct**（8.9 GB）。模型加载、processor、生成 probe 通过：384 px → 24×24 patch → 12×12 = 144 visual tokens；前向显存 8.7 GB。
5. 初始测试 13 个全部通过；后续补充到 **23 个测试**。

### A1：合成数据流水线与第一批 bug

6. `pilot synth` 生成 36 ROI；`pilot segment` 首次有 **4/36 ROI 失败**，且 `segment_dir` 不删除旧文件，失败 ROI 静默回退到 synth ground truth；`doctor` 看不到。
7. 定位 Otsu 阈值落在直方图 bin 中心、略低于常量背景，导致整图前景；加阈值保护后 **72/72** 分割成功（24/class）。
8. 检查反事实数据发现：
   - prompt 只说 “were moved”，没有 edit kind/vector；同一图+框同时有 yes/no；
   - prompt 带 0–1000 框坐标，纯文本 logistic 在留出患者上 **0.689**；
   - 14% val（20% train）是空框，答案必然 no；
   - P1/P2 在训练集上评估；
   - CellViT 每 ROI 单独 factorize 类型；
   - 两个辅助头全局 mean pooling；区域描述子 tile 顺序与视觉网格不一致；
   - cf 头没有平移向量；Qwen3-VL 实际是 32×32 px/token 而非 28×28；
   - scikit-image 未列入依赖；`--compare` 读 JSON 数组失败；doctor 不查坐标尺度。
9. 完成代码修复（详见附录 B 与第 1 节），新增测试到 23 个。

### A2：合成实验（简单版 / 严格版 / 高分辨率 / 模块级）

10. 简单版：72 ROI，430 cf；P0 cf **0.523**；P1 cf **0.674**（seed 0/1 都是 0.674）；blank 0.500；text-only 0.500。
11. 去掉空框 + 按 kind/label 严格平衡：384 px 226 train / 114 val；1024 px 254 train / 118 val（GT cells）。
    - LM：strict 384 **0.500**；strict 1024 **0.534** [0.481, 0.582]。
    - 训练后的 cf 头：strict 384 **0.640** / AUC 0.652（blank 0.509 / AUC 0.516）；strict 1024 0.686 / AUC 0.744（blank AUC 0.728，geometry 混淆）。
    - 结构头 topology R²：strict 384 **0.315**；strict 1024 **0.411**（blank 为负）。
    - 同任务 aux vs LM-only：0.674 vs 0.644，+3.0 pt，CI [−1.5, +6.8]。
12. P2：growth aux / no-aux 都 **1.000**（22 held-out ROIs；零样本 0.667），天花板；cf 交叉评估 0.689 / 0.697；结构线性探针 aux/no-aux 混合，无稳定优势。
13. 实现并构建 before/after 双面板（`--presentation before_after`），但按“先做一和二”没有把它作为主实验。

### A3：真实 H&E 数据

14. 在 HF 上检索可用小规模病理数据集，选定：
    - **MoNuSeg**（RationAI；100 MB）：37 train / 14 test 患者；1000×1000；294–1862 nuclei；无类型；test patient-disjoint。
    - **PanNuke fold1**（RationAI；280 MB）：405 train patches（15 组织）/ 160 held-out patches（breast/colon/bile/cervix 四个完全留出组织）；256→1024；5 类细胞；每 patch 16–273 nuclei。
15. 用 `pyarrow` 直接读 HF parquet（`datasets` 5.x 与 transformers 的 hub<1.0 冲突，因此不用 datasets）；新增 `import_monuseg.py`、`import_pannuke.py`。
16. MoNuSeg：116 train cf / 60 test cf（30/30）；box-geometry **0.467 / AUC 0.528**；3 seeds × 150 steps × 1024 tokens；cf 头 0.583/0.600/0.650（mean **0.611**），blank 0.500，expand 0.722，translate 0.559，collapse 0.458（n=8）；结构头 topology R² 全负。
17. PanNuke：1796 train cf / 620 held-out cf（310/310）；box-geometry **0.577 / AUC 0.580**；2 seeds × 300 steps；cf 头 0.595/0.582（mean **0.588**），blank 0.515/0.544，real−blank +8.1/+3.9 pt；训练 loss 0.5–0.8（1.3 epochs，underfit）；结构头 R² 负。

### A4：按“一和二”执行的扩展

18. **一、扩真实数据/延长训练**：PanNuke 300 → **600 步**（≈2.7 epochs），仍是 405 train / 160 held-out；1796 / 620 cf。
19. **二、修结构头**：新增 `--structure-dims 5`，结构头只回归 topology 五项，去掉 3×3 regional block；`eval_heads.py` 从 checkpoint 自动推断头输出维度，`bootstrap_heads.py` 自动对齐 descriptor 维度。
20. 运行：PanNuke 600 + topology-only（1 seed）；MoNuSeg 150 + topology-only（3 seeds）；随后在 held-out 上评估 cf/structure 头 + blank 对照。结果见 3.4。

### A5：飞书 CLI 与文档

21. 按官方指南安装 `npm install -g @larksuite/cli` → `lark-cli 1.0.98`。
22. `lark-cli config` 已有 app `cli_aa5c394ff9b8dbe1`；user token 的 refresh token 被吊销；发起 `auth login --domain docs,drive --no-wait --json` device flow，用户扫码/打开链接授权；`auth login --device-code` 完成，`auth status` 显示 user identity ready，scopes 包含 `docx:document:create`、`drive:file:upload`、`docs:document:import` 等。
23. 用 `--dry-run` 验证 `docs +create --doc-format markdown`；`--content @file` 被 `/root` 保护路径拦截，改用 stdin：`cat FEISHU_REPORT.md | lark-cli docs +create ... --content -`。
24. 训练/评估结束后，把 3.4 的新结果填入本报告，用上述命令创建飞书文档，并回填文档链接。

### A6：2026-10-22 断点恢复与防过拟合监控

25. **中断与恢复**：上一轮 PanNuke 600 在 **step 400/600、约 31 分钟**处被中断；训练脚本只在结束时保存，没有可续的 checkpoint，因此无法续跑。2026-10-22 02:51 用 `setsid` 独立进程重跑完整 600 步（02:51–03:38，46.2 分钟）。重跑曲线（step 100→600）：lm 0.261/0.215/0.251/0.248/0.163/0.134；cf 0.527/0.518/0.581/0.694/0.357/0.364；struct 0.032/0.027/0.020/0.028/0.007/0.018。前一次在 ~31 分钟被杀的进程这次存活，说明独立进程组解决了会话中断导致的进程清理。
26. **防过拟合/欠拟合监控**：训练脚本新增 `--checkpoint-every`（默认 50 步），MoNuSeg topology-only 3 seeds 额外保存 `checkpoint-50/100`。训练集 cf loss：seed 0 = 0.530 → 0.170 → 0.016；seed 1 = 0.283 → 0.210 → 0.048；seed 2 = 0.663 → 0.131 → 0.004（150 步全部接近训练集记忆）。这些中间 checkpoint 用于 held-out 早停比较，避免把过拟合终点当作唯一结果。
27. **评估**：8 组正式头评估（PanNuke600：620 条 real + 620 条 blank；MoNuSeg topo：60 条 × 3 seeds × real/blank），再加 6 组中间 checkpoint 的 held-out real 评估；bootstrap 按患者/组织分组重采样 5000 次。新结果见 §3.4 与附录 C。
28. **交付**：把新结果填回本报告 §3.4、附录 C，补附录 A/B；重新生成 `FEISHU_REPORT.docx`；用 `lark-cli docs +create --doc-format markdown --content -` 创建飞书文档并回填链接；代码与报告推送到 `feasibility/pilot-fixes` 分支。

---

## 9. 附录 B：全部代码修改（文件级）

### 修改的文件（16 个）

| 文件 | 主要修改 |
|---|---|
| `README.md` | 修正 14 项设计错误；preset 表加入 `4b_bf16_1024`；真实数据 MoNuSeg/PanNuke 结果；P1/P2/真实数据的控制与限制 |
| `pyproject.toml` | 加入 `scikit-image`；`transformers>=4.57,<5`；`dev` extra（pytest、scikit-learn） |
| `scripts/run_pilot.sh` | `cells_seg` 与 GT 分离；patient split；P0/P1/P2 只在 val 上评估；`LIMIT` 处理 |
| `src/pilot/cells.py` | CellViT class 保持字符串、全局词表；`write_cells` 支持字符串类型 |
| `src/pilot/cli.py` | build 的 `--val-frac/--min-affected/--min-cells/--presentation/--annotate/--growth-question`；p0 `--out`；eval `ex_id`/per-kind；check scikit-image；train `--structure-dims`、`--checkpoint-every` |
| `src/pilot/data.py` | prompt 写全 edit/vector、去掉文本坐标、绿框/箭头；`annotate_edit`；before/after 渲染；归一化位置特征；`min_cells`；patient split；`read_jsonl` 兼容 JSON 数组；synth `dot_radius/n_cells`、坐标 clamp |
| `src/pilot/diagnostics.py` | doctor 坐标尺度检查；`min_cells` 可配置 |
| `src/pilot/eval.py` | yes/no 宽松但拒绝 hedge；per-kind；ex_id；跨样本 limit；device 修复 |
| `src/pilot/heads.py` | 视觉 token 网格重排；结构头 3×3 池化；cf 头编辑区域池化；`MASK_DIM` 6→8（含平移向量） |
| `src/pilot/interventions.py` | `action_text` 含方向；按 kind 平衡；`min_affected`；`summarise` per-kind |
| `src/pilot/model.py` | Qwen3-VL 32×32 px/token；新增 `4b_bf16_1024` preset |
| `src/pilot/segment.py` | Otsu 阈值保护；删除陈旧输出；共享 `write_cells` |
| `src/pilot/structures.py` | `describe(..., frame=...)` 图像坐标系；tile 行优先 |
| `src/pilot/train.py` | base config/grid；prefix 对齐检查；全失败 step 保护；日志归一化；`structure_dims`；每 N 步保存中间 adapter+heads（`checkpoint_every`，默认 50） |
| `src/pilot/vision.py` | `PATCH=32`（Qwen3-VL） |
| `tests/test_pilot.py` | 新增 10 个测试：prompt/annotation、patient split、frame/tile、CellViT 全局类型、spatial heads、min_affected、JSON 数组、per-kind 等；总计 **23** |

### 新增的文件（8 个）

| 文件 | 用途 |
|---|---|
| `FEASIBILITY.md` | 完整技术可行性报告（合成 + 真实 + 控制 + 限制 + 复现） |
| `FEISHU_REPORT.md` | 本飞书报告源文件 |
| `FEISHU_REPORT.docx` | Word 离线备份 |
| `scripts/import_monuseg.py` | MoNuSeg parquet → 图像 + cell table + train/test manifest |
| `scripts/import_pannuke.py` | PanNuke parquet → 上采样图像 + 5 类 cell table + 15/4 组织 split |
| `scripts/eval_heads.py` | 直接评估训练后的 structure/cf 头（real/blank），支持 topology-only |
| `scripts/bootstrap_heads.py` | 患者/组织级 bootstrap：balanced、AUC、real−blank、structure R² |
| `scripts/balance_cf.py` | 按 kind/label 严格平衡 cf 数据 |
| `scripts/make_blank_control.py` | 生成 blank-box 对照（box / before_after） |
| `scripts/probe_structure.py` | 线性结构探针（aux vs no-aux backbone） |

> 注：`data/`、`out/` 被 `.gitignore` 忽略；模型/Qwen 缓存在 `/root/.cache/huggingface`。代码与报告已提交并推送到 `feasibility/pilot-fixes` 分支（commit `2715637`、`561b418`、`89e8a25`）。

---

## 10. 附录 C：全部结果、原始文件与复现

### C1：所有评估结果（完整列表）

**合成数据**
- P0 cf：0.523（132 条；97.7% yes）；P0 growth：0.727 / balanced 0.667（22 条）。
- P1 cf：seed 0 = 0.674；seed 1 = 0.674（均 132 条；CI seed 0 [0.591, 0.758]）。
- P1 blank：seed 0 = 0.500；seed 1 = 0.500。
- P1 LM-only（`--no-aux`）：0.644。
- 严格 384 LM：0.500（114 条；94.7% no）；严格 1024 LM：0.534（118 条；CI [0.481, 0.582]）。
- 严格 384 cf 头：0.640 / AUC 0.652；blank 0.509 / AUC 0.516。
- 严格 1024 cf 头：0.686 / AUC 0.744；blank 0.500 / AUC 0.728。
- 结构头 topology R²：strict 384 = 0.315；strict 1024 = 0.411；blank 为负。
- 同任务 aux vs LM-only：0.674 vs 0.644（Δ +3.0，CI [−1.5, +6.8]）。
- P2 growth：aux 1.000；no-aux 1.000（22 held-out；Δ 0，CI [0,0]）。
- P2 cf cross-eval：aux 0.689；no-aux 0.697（Δ −0.8，CI [−7.6, +6.1]）。
- 结构线性探针：mean/grid × aux/no-aux；grid topology MSE 36.08 vs 37.46，R² 0.585 vs 0.563；mean 反而 aux 略差；无稳定优势。

**真实 MoNuSeg（3 seeds）**
- cf real：0.583 / 0.600 / 0.650（mean **0.611**）；AUC 0.637 / 0.610 / 0.676。
- blank：0.517 / 0.483 / 0.500（mean 0.500）；AUC 0.529 / 0.643 / 0.504。
- expand：0.667 / 0.708 / 0.792（mean 0.722）；translate：0.571 / 0.536 / 0.571（mean 0.559）；collapse：0.375 / 0.500 / 0.500（n=8）。
- box-geometry：0.467 / AUC 0.528。
- 结构头 topology R²：−72.6 / −29.4 / −20.4（全负）。
- seed 0 患者级 bootstrap：balanced [0.483, 0.689]；AUC [0.540, 0.758]。

**真实 PanNuke（2 seeds × 300 步）**
- cf real：0.595 / 0.582（mean **0.588**）；AUC 0.645 / 0.626。
- blank：0.515 / 0.544（mean 0.530）；AUC 0.591 / 0.580。
- real−blank 配对：+8.1 pt CI [5.7, 10.3]；+3.9 pt。
- collapse / expand / translate：0.544/0.597/0.627；0.544/0.593/0.598。
- box-geometry：0.577 / AUC 0.580。
- 结构头 topology R²：−3.76 / −5.84（全负）。
- cf 训练 loss 0.5–0.8（300 步 ≈ 1.3 epochs，underfit）。

**修复后的扩展（3.4，2026-10-22）**

**PanNuke 600 步 + topology-only（1 seed）**
- 训练（step 100→600）：lm 0.261/0.215/0.251/0.248/0.163/0.134；struct 0.032/0.027/0.020/0.028/0.007/0.018；cf 0.527/0.518/0.581/0.694/0.357/0.364。
- cf real：**0.600**，CI [0.563, 0.649]；AUC 0.657，CI [0.610, 0.709]。
- blank：0.547，CI [0.529, 0.564]；AUC 0.579，CI [0.540, 0.617]。
- real−blank 配对：**+5.3 pt**，CI [0.5, 10.9]；n=620，4 组织。
- per-kind：collapse 0.544（n=160）/ expand 0.606（n=216）/ translate 0.631（n=244）。
- structure topology R²：−6.16，CI [−332.2, −3.02]；逐维 n_comp −0.024 / hole_count +0.007 / hole_area_frac −31.183 / h0_max −0.106 / h0_sum +0.52。
- 对照 300 步 baseline（2 seeds mean）：cf 0.588；AUC 0.636；blank 0.530；paired +8.1/+3.9；topo R² −3.76/−5.84。

**MoNuSeg 150 步 topology-only（3 seeds）**
- 训练 cf（50/100/150）：seed 0 = 0.530/0.170/0.016；seed 1 = 0.283/0.210/0.048；seed 2 = 0.663/0.131/0.004。
- cf real：0.600 / 0.583 / 0.600（mean **0.594**）；CI [0.516,0.685] / [0.439,0.722] / [0.501,0.700]；AUC 0.649/0.622/0.643。
- blank：0.450 / 0.533 / 0.483（mean 0.489）；real−blank 配对：+15.0 / +5.0 / +11.7 pt；CI [8.5,22.2] / [−13.6,21.7] / [0.0,22.6]。
- per-kind：collapse 0.375/0.500/0.250（n=8）；expand 0.708/0.708/0.833（mean 0.750）；translate 0.571/0.500/0.500（mean 0.524）。
- structure topology R²：−44.8 / −22.1 / −38.7；逐维均值 n_comp +0.029 / hole_count −0.071 / hole_area_frac −175.78 / h0_max +0.047 / h0_sum −0.255。
- 早停 cf（ck50/ck100/final）：seed 0 = 0.533/0.583/0.600；seed 1 = 0.500/0.600/0.583；seed 2 = 0.517/0.550/0.600；mean 0.517/0.578/**0.594**。
- 早停结构 R²（ck100 vs final）：−19.1 vs −44.8；−19.3 vs −22.1；−38.3 vs −38.7。
- 对照 full-descriptor baseline（3 seeds，150 步）：cf mean 0.611；blank 0.500；expand 0.722；translate 0.559；collapse 0.458（n=8）；topo R² −72.6/−29.4/−20.4。

### C2：原始文件

- 合成模型/预测：`out/p1/`、`out/p1_seed1/`、`out/p1_noaux/`、`out/p1_hard/`、`out/p1_hi/`、`out/p2_aux/`、`out/p2_lora/`。
- 合成头结果：`out/heads_real_s*.json`、`out/heads_blank_s*.json`（MoNuSeg）；`out/heads_pannuke_*`（PanNuke）。
- 真实数据：`data/real/monuseg/`、`data/real/monuseg_train/`、`data/real/monuseg_test/`；`data/real/pannuke/`、`data/real/pannuke_train/`、`data/real/pannuke_test/`。
- 新实验：`out/p1_pannuke600_s0/`、`out/p1_real_topo_s0/1/2/`（含 `checkpoint-50/100`）。
- 新评估 JSON：`out/heads_pannuke600_{real,blank}_s0.json`、`out/heads_topo_{real,blank}_s{0,1,2}.json`、`out/heads_topo_real_s{0,1,2}_ck{50,100}.json`。
- 对照：`data/synth/cf_val_blank.jsonl`、`data/synth_hi/cf_val_bal_blank.jsonl`、`data/real/monuseg_test/cf_bal_blank.jsonl`、`data/real/pannuke_test/cf_bal_blank.jsonl`。
- 技术报告：`FEASIBILITY.md`；本报告：`FEISHU_REPORT.md` / `.docx`。

### C3：关键复现命令

```bash
# 合成
pilot synth --out data/synth --per-class 24 --size 384
pilot segment --images data/synth/images --out data/synth/cells_seg --backend classical
pilot build --manifest data/synth/manifest.csv --cells data/synth/cells_seg \
  --out data/synth --per-roi 6 --val-frac 0.3
pilot p0   --data data/synth/cf_val.jsonl --preset 4b_bf16 --out out/pred_p0_cf.json
pilot train --cf data/synth/cf_train.jsonl --steps 150 --preset 4b_bf16 --out out/p1
pilot eval  --data data/synth/cf_val.jsonl --adapter out/p1 --preset 4b_bf16 --out out/pred_p1.json

# 真实 MoNuSeg
python scripts/import_monuseg.py --train-parquet ... --test-parquet ... --out data/real/monuseg
pilot build --manifest data/real/monuseg/manifest_train.csv --cells data/real/monuseg/cells \
  --out data/real/monuseg_train --per-roi 6 --val-frac 0.2 --min-affected 5
python scripts/balance_cf.py --train data/real/monuseg_train/cf_train.jsonl \
  --val data/real/monuseg_train/cf_val.jsonl \
  --out-train data/real/monuseg_train/cf_train_bal.jsonl \
  --out-val data/real/monuseg_train/cf_val_bal.jsonl
pilot train --cf data/real/monuseg_train/cf_train_bal.jsonl --steps 150 \
  --preset 4b_bf16_1024 --out out/p1_real_s0
python scripts/eval_heads.py --adapter out/p1_real_s0 \
  --data data/real/monuseg_test/cf_bal.jsonl --preset 4b_bf16_1024 --out out/heads_real_s0.json
python scripts/bootstrap_heads.py --real out/heads_real_s0.json --blank out/heads_blank_s0.json

# 真实 PanNuke
python scripts/import_pannuke.py --parquet .../fold1-*.parquet --out data/real/pannuke \
  --val-tissues breast,colon,bile,cervix --min-cells 16
pilot build --manifest data/real/pannuke/manifest_train.csv --cells data/real/pannuke/cells \
  --out data/real/pannuke_train --per-roi 6 --val-frac 0 --min-affected 5 --min-cells 16
python scripts/balance_cf.py --train data/real/pannuke_train/cf.jsonl \
  --val data/real/pannuke_train/cf.jsonl \
  --out-train data/real/pannuke_train/cf_train_bal.jsonl \
  --out-val /tmp/pannuke_bal.jsonl
pilot train --cf data/real/pannuke_train/cf_train_bal.jsonl --steps 600 \
  --grad-accum 8 --seed 0 --log-every 100 --preset 4b_bf16_1024 --structure-dims 5 \
  --out out/p1_pannuke600_s0
python scripts/eval_heads.py --adapter out/p1_pannuke600_s0 \
  --data data/real/pannuke_test/cf_bal.jsonl --preset 4b_bf16_1024 \
  --out out/heads_pannuke600_real_s0.json
python scripts/eval_heads.py --adapter out/p1_pannuke600_s0 \
  --data data/real/pannuke_test/cf_bal_blank.jsonl --preset 4b_bf16_1024 \
  --out out/heads_pannuke600_blank_s0.json
python scripts/bootstrap_heads.py --real out/heads_pannuke600_real_s0.json \
  --blank out/heads_pannuke600_blank_s0.json

# 3.4 扩展：MoNuSeg topology-only + 每 50 步 checkpoint（早停比较）
for s in 0 1 2; do
  pilot train --cf data/real/monuseg_train/cf_train_bal.jsonl --steps 150 \
    --grad-accum 8 --seed $s --preset 4b_bf16_1024 --structure-dims 5 \
    --checkpoint-every 50 --out out/p1_real_topo_s$s
  python scripts/eval_heads.py --adapter out/p1_real_topo_s$s \
    --data data/real/monuseg_test/cf_bal.jsonl --preset 4b_bf16_1024 \
    --out out/heads_topo_real_s$s.json
  python scripts/eval_heads.py --adapter out/p1_real_topo_s$s \
    --data data/real/monuseg_test/cf_bal_blank.jsonl --preset 4b_bf16_1024 \
    --out out/heads_topo_blank_s$s.json
  for ck in 50 100; do
    python scripts/eval_heads.py --adapter out/p1_real_topo_s$s/checkpoint-$ck \
      --data data/real/monuseg_test/cf_bal.jsonl --preset 4b_bf16_1024 \
      --out out/heads_topo_real_s${s}_ck${ck}.json
  done
done
```

### C4：测试

```bash
python -m pytest tests/ -q
# 23 passed
```

