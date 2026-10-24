# 从“结构头失败”到 P2：拓扑监督的诊断、修正与下一步

**日期**：2026-10-22  
**目标**：用拓扑结构把 VLM 的视觉证据和文本证据接起来。P2 是决定这条路成不成立的关键：如果加入结构监督后，DHMC 生长模式分类（或组织分类）不动，前面的 cf 数字再漂亮也只是测量工具。

---

## 0. 先修正三个判断

### 0.1 “结构头失败”主要是指标假象，但不能说头没问题

用现有 `out/heads_*.json` 和训练集描述子在 CPU 上重算（`scripts/eval/diagnose_structure.py`）：

| 队列 | 模型 | mean R²（旧口径） | 非退化维 mean R² | Spearman（全维 / 非退化） | `hole_area_frac` R² | 训练均值基线 R² |
|---|---|---|---|---|---|---|
| MoNuSeg topo s0 | 5 dims | −44.8 | −0.107 | 0.152 / 0.328 | −223.5 | −0.027 |
| MoNuSeg topo s1 | 5 dims | −22.1 | −0.043 | 0.196 / 0.468 | −110.5 | −0.027 |
| MoNuSeg topo s2 | 5 dims | −38.7 | −0.030 | 0.225 / 0.386 | −193.4 | −0.027 |
| PanNuke 600 topo s0 | 5 dims | −6.16 | **+0.130** | **0.299 / 0.467** | −31.2 | −0.019 |

- **训练均值基线并没有 −200**，而是 −0.02 到 −0.03。所以 `hole_area_frac` 的 R² 不是“指标完全不能用”；它确实说明头在这一维上远差于常数预测（MAE 是均值预测的 13–23 倍）。
- 但把这一维和另外四维**做无权重平均**得到 −44.8，是没有意义的：这一维的方差只有 `3.9e-6`（MoNuSeg）到 `1.3e-5`（PanNuke），R² 的分母极小。正确的结论是：
  - **`h0_sum` 在 PanNuke 上有真实信号**：R² +0.52、skill +0.59、Spearman 0.71、MAE 只有均值预测的 0.60。
  - **`n_comp` / `h0_max` 有弱信号**：PanNuke skill +0.12/+0.11，Spearman 0.35；MoNuSeg 也接近 0 到弱正。
  - **`hole_area_frac` 确实没学会**，而且是零膨胀、尖峰、方差极小的目标。它需要单独的 presence + positive-value 损失，而不是一个连续回归。
- 因此报告里的“结构头失败”应改成：**“结构头在零膨胀的 lumen 面积维上失败；在连通性/持久性维上有弱但可测的信号；旧的五维平均 R² 不能作为结论。”**
- 另一处需要保留的警告：**PanNuke 的 cf 视觉增量仍未证实**。300 步 cf 0.588 vs box-geometry 0.577（只高 1 pt）；600 步 run 只有 blank 0.547，没有 geometry 对照。P2 必须继续保留 geometry 对照，否则 real−blank 会被误读成“看了局部结构”。

### 0.2 LM 读出不是“架构读不出表征”，是数据量

训练集 226 条时，训练 cf loss 可以降到 0.004–0.048，说明 **LM 能记住**；但 60 条留出样本只有 0.500/0.534。与此同时，同一个 backbone 的表征喂给头能到 0.61。正确的表述是：

> 226 条训练样本足以让 LM 记住，但不足以让它泛化；留出读出约 0.5。这不是“LM 读不出表征”，而是数据量/泛化问题——同样的表征，头能到 0.61。

这会改变下一步：不是先改 LM 架构，而是先加数据/加约束（P2 的真实下游标签、topology-text 对齐、rationale 监督）。

### 0.3 错框对照必须补

blank 去掉所有细胞，只证明“有没有细胞证据”；它不能证明“有没有看对区域”。同一张图、框挪到随机位置、保持编辑类型/半径/箭头不变、**用细胞表重新计算精确标签**，才能区分：

- 模型仍答 yes → 全局纹理/框位置先验；
- 模型正确翻成 no → 真的在看框内局部结构。

---

## 1. 这次代码已经改了什么

### 1.1 指标与诊断

- `src/pilot/structure_eval.py`：逐维输出 R²、训练均值基线 R²、skill、log 空间 MAE、MAE/均值基线、Spearman、校准差、零膨胀标志、presence AUC、正样本 MAE/R²/Spearman。聚合时同时给 `r2_mean_all`（兼容旧报告）、`r2_median_all`、`r2_mean_nondegenerate`、`skill_mean`、`spearman_mean`。
- `scripts/eval/diagnose_structure.py`：**不需要 GPU**，直接读 `heads_*.json` + 训练 JSONL，复现上面的表。
- `scripts/eval/eval_heads.py`：加 `--train-data`、`--metrics-out`；逐维诊断；统计并打印 `skipped batch / no_image_tokens / grid fallback`。
- `scripts/eval/bootstrap_heads.py`：加入结构指标的 patient/tissue 分组 bootstrap；不再只报一个被 `hole_area_frac` 支配的“R² mean”。

### 1.2 错框对照

- `scripts/controls/make_wrongbox_control.py`：同图、挪框、保持 kind/radius/vector，用 `measure_effect` 重新计算精确 `changed`；默认 `--select-label no`，只保留“挪框后真值为 no”的样本；也可 `flip`/`any`。
- `scripts/eval/analyze_wrongbox.py`：配对分析 `wrongbox_yes_rate`（捷径率）、`flip_no_rate`（原 yes→挪框 no 的正确翻转率）、`paired_logit_drop`，按患者分组 bootstrap。
- 已在 MoNuSeg test 上生成 50 条全 no 的错框样本：`data/real/monuseg_test/cf_bal_wrongbox.jsonl`（22 条原 yes→挪框 no，28 条原 no→挪框 no；collapse 只有 2 条，expand 20，translate 28——collapse 的 no-op 位置很难找，报告时要按 kind 分层）。

### 1.3 代码层修复

- `CounterfactualHead.forward`：中心必须在 `[0,1]`、半径必须在 `(0,2]`，否则抛出明确错误（像素坐标不会再被静默当成归一化坐标）。
- `grid_or_global_mean`：统一 train/eval 的 fallback，返回 `grid|fallback|failed`，两处都计数并报告。
- 重排后的 `scripts/{data,eval,controls,build}/*.py` 的 `sys.path` 修正为 `parents[2]`。
- `train.py`：结构头现在可以**同时在下游 growth 样本上训练**（`--structure-on-growth` 默认开；`--lam-structure-growth`）；`collate_aux` 允许 growth 行没有 `changed`；辅助 loss 的日志按实际生效批次数归一化。
- `build_growth_examples(..., cells_dir=..., with_descriptor=True)`：下游样本可携带与 cf 样本同定义的 topology descriptor（`pilot build` 默认开启，`--no-growth-descriptors` 关闭）。
- `wsi_id` 贯穿 growth/cf/eval 记录；`scripts/eval/aggregate_slide_predictions.py` 把 patch 预测按 slide 多数投票并计算 slide-level 指标。
- `scripts/data/import_dhmc.py`：解析 DHMC 的多种 metadata（`label` 列或 `LUAD_*` 列，2=predominant），tile WSI，组织比例过滤，写 manifest；`openslide` 优先，PIL 回退用于小测试图。

### 1.4 文档修正

- 本报告和 `docs/RESULTS.md` 中“LM 读不出表征”的表述已按 §0.2 修正。
- 新的结构指标口径已写入 `docs/RESULTS.md` 的 §3.4 / §4（见对应 commit）。

---

## 2. 相关工作和可以借的东西

> 目标不是“再做一个拓扑特征”，而是让拓扑成为视觉证据和文本证据之间的**可验证中间表示**。

### 2.1 TopoSlide（CVPR 2026）— 最直接的目标/损失模板

- 论文：<https://openaccess.thecvf.com/content/CVPR2026/html/Abousamra_TopoSlide_Topologically-Informed_Histopathology_Whole_Slide_Image_Representation_Learning_CVPR_2026_paper.html>
- 代码：<https://github.com/plevritis-lab/TopoSlide>
- 做法：把 patch embedding 聚类成组织模式，对每个 cluster 的空间点云做 DTM-Rips 的 H0/H1 持久同调；用**持久性直方图 + 累积直方图**作为目标，Huber loss；对零/非零条目分别算 loss；再加 binary presence loss；token 级也预测 topology；条件于 slide embedding 和 cluster prototype。
- 可借：
  1. 用 H0/H1 持久性直方图替代/补充 `hole_area_frac`；
  2. Huber on histogram + CDF（CDF 对 bin 边界的微小移动更稳）；
  3. zero/nonzero 分开算 loss（直接解决零膨胀）；
  4. 多尺度/多 cluster 的 topology token（如果做 WSI 级 P2）。

### 2.2 PMIL（2025）— MIL 融合的现成蓝图

- 论文/代码：<https://github.com/ahmadobeid/PMIL>
- 做法：从 nuclei point cloud 生成 persistence images（PI）或 persistence curves（PC），用 CNN 提取 PI 特征，再与 RGB patch 特征在 CLAM/MIL 里做特征级融合；在 PANDA 前列腺分级上报告 ACC/AUC 提升。
- 可借：DHMC 是 slide-level label，最自然的 P2 就是 patch bag + MIL。PMIL 给出“RGB 分支 + topology 分支 + 融合”的具体工程路径。

### 2.3 TG-HEM（MedIA 2026）— 拓扑引导的难例挖掘

- 论文：<https://www.sciencedirect.com/science/article/abs/pii/S1361841526002240>
- 代码：<https://github.com/caki35/TGHEM>
- 做法：用 GT 与预测细胞分布的持久同调差异（Vietoris–Rips）给分割 loss 加权，做 region/pixel 两级 hard example mining。
- 可借：如果 DHMC 的 nuclei 分割质量是瓶颈，用拓扑差异给 patch/像素加权，优先修正“拓扑错”的区域；也可以用它做**分割质量筛选**，只保留 topology 稳定的 patch。

### 2.4 GeoTop（2025）— 几何 + 拓扑融合

- 代码：<https://github.com/MorillaLab/GeoTop>
- 做法：TDA（H0/H1、persistence entropy、diagram amplitude）+ Lipschitz-Killing curvatures（area/perimeter/Euler characteristic 随阈值变化）融合，随机森林分类。
- 可借：腺体结构不只靠连通性，还靠面积/周长/欧拉数。若 DHMC 上 topology-only 不够，可加 LKC/几何描述子作为第二分支；GeoTop 的 184 维特征工程可直接移植到 patch 级。

### 2.5 GraPHFormer（CVPR 2026）— 拓扑-形态对比对齐

- 论文：<https://openaccess.thecvf.com/content/CVPR2026/html/Shah_GraPHFormer_A_Multimodal_Graph_Persistent_Homology_Transformer_for_the_Analysis_of_Neuroscience_Morphologies_CVPR_2026_paper.html>
- 做法：3 通道 persistence images（DINOv2-ViT-S）+ 骨架图 TreeLSTM，CLIP 式对比对齐。
- 可借：把“predicted topology embedding”和“class description text embedding”做 InfoNCE，这是最直接的 visual-text 桥；不依赖 LM 生成 yes/no。

### 2.6 Persistent Homology as a Diagnostic Lens in Vision-Language Medical Systems（2026）

- 章节：<https://www.igi-global.com/chapter/persistent-homology-as-a-diagnostic-lens-in-vision-language-medical-systems/409644>
- 做法：用 cubical persistence / Betti curve / bottleneck-Wasserstein drift 审计 VLM 的输出，检测“看起来合理但拓扑错了”的 silent structural failure。
- 可借：把它作为**评价指标**而不是训练损失：模型说“有腺腔/实性片状”时，用分割出的 topology 去核对；rationale 与测量不一致就记一次 visual-text inconsistency。

### 2.7 其他

- **Noise-aware topological consistency（MICCAI 2024）**：可微 topology loss 做半监督分割；如果以后要把分割和分类端到端，参考它的 topo-consistency/topo-removal loss。
- **SlideBank（2026，arXiv 2609.00342）**：WSI 证据库/一致性推理；可作为“检索相似拓扑模式的证据”方向，不是当前第一优先级。
- **CellViT++ / HoVer-Net / Cellpose**：H&E nuclei 分割。仓库已有 CellViT JSON 桥；DHMC 上建议先用 CellViT++，classical Otsu 只做 fallback，并用 `segbench` 在 PanNuke 上先量 descriptor 稳定性。
- **DeepSlide / BMIRDS**：DHMC 官方 baseline 和 tiling 参考：<https://github.com/BMIRDS/deepslide>。

---

## 3. P2 设计：DHMC 生长模式分类

### 3.1 数据事实

- 公开 DHMC：143 张 H&E WSI，20x，Aperio AT2，pyramidal TIFF；**只有 slide-level predominant pattern**，没有区域级标注。
- 5 类及数量（来自 dependency-mil 的公开 label CSV）：acinar 59、solid 51、lepidic 19、micropapillary 9、papillary 5。
- 获取：<https://bmirds.github.io/LungCancer/> 填表申请，下载链接邮件发送、4 小时过期；无公开 HF/Kaggle 镜像。
- 含义：
  - 5 类严重不平衡，papillary/micropapillary 只有 5/9 张；5-fold 分层后每折验证集里可能只有 1–2 张罕见类。**balanced accuracy / macro F1 会比 accuracy 更合适，但 CI 会很宽**。
  - slide-level label 对 patch 是弱标签；patch 分类 + 多数投票会有标签噪声。MIL/CLAM 更符合标签粒度，但工程更重。

### 3.2 两种可执行协议

**协议 A（先跑，便宜）：patch 分类 + slide 多数投票**

1. `import_dhmc.py` 每张 slide 采 32–64 个 512×512 patch（20x，组织比例 ≥0.5）。
2. `pilot segment` 或 CellViT 提取 nuclei → `pilot build --growth-question "What is the predominant growth pattern? Choose one of: lepidic, acinar, papillary, micropapillary, solid."` → growth 样本带 descriptor；cf 样本照常生成。
3. `pilot train` 两个 arm：`--no-aux` vs 默认（structure on growth + cf）。
4. `pilot eval` 在留出 slide 的 patch 上生成预测；`aggregate_slide_predictions.py` 多数投票到 slide。
5. 5-fold slide-disjoint、3 seeds；报告 slide-level balanced accuracy / macro F1 / per-class。

**协议 B（更贴合标签，稍重）：patch bag + MIL**

1. 同一批 patch；用冻结 VLM 视觉塔提取 patch embedding（或 LoRA 微调）。
2. `src/pilot/mil.py` 的 ABMIL/CLAM 头聚合 patch；可选 topology 分支（descriptor 或 persistence image）做特征级融合。
3. slide-level label 直接监督 bag；5-fold。
4. 这是 PMIL 的路线；如果协议 A 的 patch 噪声太大，直接切 B。

> 建议：先用协议 A 跑通端到端和统计管线；如果 aux vs no-aux 的差异被 patch 噪声淹没，再上协议 B。

### 3.3 实验 arm

| arm | LM/分类损失 | 结构监督 | 拓扑融合/对齐 | 目的 |
|---|---|---|---|---|
| A0 baseline | ✓ | — | — | 下界 |
| A1 struct-aux | ✓ | growth + cf descriptor | — | 当前方法 |
| A2 struct-fuse | ✓ | growth descriptor | descriptor → classifier | topology 作为瓶颈 |
| A3 topo-text | ✓ | growth descriptor | InfoNCE(descriptor, class description) | 直接对齐视觉/文本 |
| A4 rationale | ✓ | growth descriptor | LM 先输出 topology rationale | 强制文本证据可核验 |

- **A1 vs A0** 是用户要的“加结构监督有没有用”。
- **A3** 是最贴合“视觉证据和文本证据脱节”的 arm；class description 要写成拓扑可核验的句子，例如：
  - lepidic: “cells line alveolar septa; air spaces are preserved; low architectural complexity.”
  - acinar: “round glands with central lumina; one or more enclosed spaces.”
  - papillary: “finger-like projections; fibrovascular cores.”
  - micropapillary: “small tufts without fibrovascular cores; high local complexity.”
  - solid: “sheets of cells; few or no lumina; high cell density.”
- **A4**：用测量到的 descriptor 生成 rationale，例如 “0 lumina, 1 connected component, solid sheet”; 训练 LM 先输出 rationale 再输出答案；评价 rationale 的 topology 与测量值是否一致。

### 3.4 指标、统计与功效

- 主指标：slide-level **balanced accuracy** 和 **macro F1**；次指标：accuracy、per-class recall、AUROC（若概率可用）。
- 5-fold slide-disjoint、分层；3 seeds；报告 mean ± std，以及 pooled out-of-fold 的 slide-grouped bootstrap 95% CI。
- 配对比较：同一 slide 上 aux vs no-aux 的 correctness，做 McNemar / slide-grouped paired bootstrap。
- **功效警告**：143 张 slide、5 类，2 个点的差异在 slide 级基本不可检测（McNemar 在 discordance 10–20% 时，n=143 对 ~5 个点才接近 80% power）。因此：
  - 第一轮把 2 点作为**方向性目标**，不要当作硬 gate；
  - 同时报告 patch-level（cluster-robust）指标，但结论仍以 slide 为单位；
  - 若要硬 gate，需要合并罕见类（如 papillary+micropapillary）做 3–4 类，或加 TCGA-LUAD/更多 WSI 扩大 n。
- 建议先做 **3-class 敏感性分析**：lepidic / acinar / solid（n=129），把 papillary/micropapillary 作为“其他”或留出；这样统计功效更好，也能先验证 topology 是否对“有腔 vs 实性”这种拓扑差异敏感。

### 3.5 拓扑目标重设计（直接回应 `hole_area_frac`）

1. **Hurdle 分解**：对零膨胀维（`hole_count`, `hole_area_frac`, H1 count/sum）拆成：
   - presence head：BCE，预测 `x > 0`；
   - positive head：只在 `x > 0` 的样本上做 Huber(log1p(x))；
   - 评价：presence AUC + positive MAE/Spearman，不再用单一 R²。
2. **持久性直方图**（TopoSlide 风格）：
   - H0：MST 死亡时间（按最近邻间距归一化）→ 8–16 bin 直方图 + CDF；
   - H1：raster hole 面积/DTM-Rips 持久性 → 8–16 bin 直方图 + CDF；
   - loss：Huber(hist) + Huber(CDF) + BCE(presence)。
3. **稳健化**：训练集 1–99% winsorize；log1p；按训练均值/标准差标准化；对零膨胀维用 rank/quantile 目标作为敏感性分析。
4. **逐维报告**：永远同时给 `R²_model`、`R²_train_mean`、`skill`、`MAE/MAE_mean`、Spearman、presence AUC；不要只报五维平均。

---

## 4. 代码落地顺序（没卡也能做）

### 已完成（本次）

- [x] 结构指标逐维诊断 + 训练均值基线 + CPU 复现脚本。
- [x] MoNuSeg 错框对照生成（50 条全 no）+ 配对分析脚本。
- [x] CF head 坐标校验、grid fallback 计数、sys.path 修复。
- [x] 结构头在 growth 样本上的监督（`--structure-on-growth`）。
- [x] growth 样本携带 descriptor；`wsi_id` 贯穿；slide 多数投票脚本。
- [x] DHMC metadata 解析 + WSI tiling importer（openslide 优先，PIL 回退）。
- [x] 45 tests passed。

### 有卡后第一批（不需要新数据）

1. `eval_heads.py` 对现有 MoNuSeg topo s0/s1/s2 跑 `cf_bal_wrongbox.jsonl`，`analyze_wrongbox.py` 出捷径率/翻转率。
2. `diagnose_structure.py` 对所有现有 `heads_*.json` 出逐维表；更新报告 §3.4/§4。
3. 用 `bootstrap_heads.py --train-data` 重算 CI，确认“非退化维”结论。

### 有 DHMC 数据后（协议 A）

1. 申请/下载 DHMC；`import_dhmc.py` 每 slide 32–64 patch。
2. `pilot segment`（CellViT/Cellpose/classical）→ `pilot build`（growth descriptor on）。
3. 5-fold × {A0, A1} × 3 seeds；`pilot eval` + `aggregate_slide_predictions.py`。
4. 若 A1 有方向性收益：加 A2/A3；若 flat：先检查 patch 标签噪声、分割质量、class 不平衡，再决定是否上 MIL。
5. 用 `analyze_wrongbox` 和 rationale consistency 作为机制证据，而不只看准确率。

---

## 5. 需要你决定的事

1. **DHMC 数据**：你是否有 WSI，或需要我写申请说明/下载后的目录约定？申请说明、落盘约定和导入命令见 `docs/DHMC_REQUEST.md`（2026-10-25：确认暂无数据）。
2. **分割后端**：DHMC 上用 CellViT++（准但重）、Cellpose（易装）还是 classical（快但噪声大）？我建议先用 CellViT++/Cellpose，classical 只做 fallback。
3. **类别策略**：先跑 5 类，还是先跑 3 类（lepidic/acinar/solid）验证拓扑敏感性？我建议两个都跑，3 类作为主分析、5 类作为报告。
4. **是否加 TCGA-LUAD**：如果 2 点 gate 必须成立，143 张 slide 功效不够；加外部数据或合并类别是更诚实的做法。

---

## 6. 参考链接

- TopoSlide: <https://github.com/plevritis-lab/TopoSlide> / <https://openaccess.thecvf.com/content/CVPR2026/html/Abousamra_TopoSlide_Topologically-Informed_Histopathology_Whole_Slide_Image_Representation_Learning_CVPR_2026_paper.html>
- PMIL: <https://github.com/ahmadobeid/PMIL>
- TG-HEM: <https://github.com/caki35/TGHEM> / <https://www.sciencedirect.com/science/article/abs/pii/S1361841526002240>
- GeoTop: <https://github.com/MorillaLab/GeoTop>
- GraPHFormer: <https://openaccess.thecvf.com/content/CVPR2026/html/Shah_GraPHFormer_A_Multimodal_Graph_Persistent_Homology_Transformer_for_the_Analysis_of_Neuroscience_Morphologies_CVPR_2026_paper.html>
- PH as a diagnostic lens for medical VLMs: <https://www.igi-global.com/chapter/persistent-homology-as-a-diagnostic-lens-in-vision-language-medical-systems/409644>
- DHMC dataset: <https://bmirds.github.io/LungCancer/>
- DeepSlide: <https://github.com/BMIRDS/deepslide>
- dependency-mil DHMC labels (public derived CSV): <https://github.com/GeorgeBatch/dependency-mil>
