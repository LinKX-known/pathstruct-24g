# DHMC 申请与导入说明（2026-10-25）

公开 DHMC（Dartmouth Lung Adenocarcinoma Growth Pattern）是 P2 的关键下游数据。当前机器上没有 DHMC WSI（`/root/autodl-tmp/slides` 里的 42 张是 TCGA-COAD，不是 DHMC/LUAD），所以协议 A 还不能开跑。本文是申请、落盘、导入的约定，数据到手后按此执行。

## 1. 数据事实

- 143 张 H&E WSI，20x，Aperio AT2，pyramidal TIFF。
- 只有 **slide-level predominant growth pattern**：lepidic / acinar / papillary / micropapillary / solid。
- 类别数（dependency-mil 公开派生 CSV）：acinar 59、solid 51、lepidic 19、micropapillary 9、papillary 5。
- 官方页面：<https://bmirds.github.io/LungCancer/>；填表申请后邮件发送下载链接，**链接约 4 小时过期**。
- 无公开 HF/Kaggle 镜像。派生标签 CSV：<https://github.com/GeorgeBatch/dependency-mil>（`DETAILED_DHMC_LUAD_LUSC_BENIGN.csv`，`LUAD_*` 列，2 = predominant）。

## 2. 申请时需要填写/说明的内容

表单通常要求申请人信息与研究用途。建议用下面这段（按实际信息替换方括号）：

> **Project title:** Topology-aware vision-language modelling for lung adenocarcinoma growth-pattern classification
>
> **Applicant:** [姓名], [单位/实验室], [邮箱], [导师/PI]
>
> **Purpose:** We are developing a research prototype that uses cell/tissue topology (connected components, lumina, persistence features) as an intermediate representation between pathology images and textual evidence. The DHMC WSIs will be used solely for non-commercial research to evaluate whether topology supervision improves slide-level growth-pattern classification (lepidic / acinar / papillary / micropapillary / solid). We will not attempt re-identification, will not redistribute the data, and will cite the DHMC publication in any resulting work.
>
> **Data handling:** WSIs will be stored on a secured research server, accessible only to project members, and deleted or returned according to the data-use terms.

注意事项：

- 用机构邮箱；如果 PI 需要签字，先让 PI 确认。
- 说明只做 slide-level 分类，不做诊断声明。
- 链接邮件可能进垃圾箱；收到后**立即下载**（4 小时过期）。
- 下载后核对文件数（应为 143 张）和 `MetaData.csv`。

## 3. 落盘约定

WSI 很大，放 `/root/autodl-tmp`（约 107 GB 可用），不要放 `/root` overlay（约 9 GB 可用）：

```
/root/autodl-tmp/dhmc/
  WSI/                 # 143 个 .tif（保持原始文件名）
  MetaData.csv         # 官方 metadata
  DETAILED_DHMC_LUAD_LUSC_BENIGN.csv   # dependency-mil 派生标签（可选）
```

## 4. 导入与检查

```bash
# 1) 只解析 metadata，确认列名和类别映射（不读 WSI）
python scripts/data/import_dhmc.py \
  --metadata /root/autodl-tmp/dhmc/MetaData.csv \
  --wsi-dir /root/autodl-tmp/dhmc/WSI \
  --out data/real/dhmc --dry-run

# 2) 每 slide 采 32 个 512 px patch，组织比例 >= 0.5
python scripts/data/import_dhmc.py \
  --metadata /root/autodl-tmp/dhmc/MetaData.csv \
  --wsi-dir /root/autodl-tmp/dhmc/WSI \
  --out data/real/dhmc \
  --patch-size 512 --patches-per-slide 32 --min-tissue-frac 0.5

# 3) 核对：slide 数、类别计数、每 slide patch 数、组织比例分布
python - <<'PY'
import pandas as pd
m = pd.read_csv('data/real/dhmc/manifest.csv')
print(m.shape)
print(m['label'].value_counts())
print(m.groupby('wsi_id').size().describe())
PY
```

如果 `openslide` 未安装，先 `pip install 'pathstruct-24g[wsi]'`；PIL fallback 会把整张 WSI 读进内存，只适合小测试图。

## 5. 协议 A（数据到手后）

1. `pilot segment` 或 CellViT/Cellpose 提取 nuclei；先在少量 slide 上检查分割质量和 descriptor 稳定性。
2. `pilot build --growth-question "What is the predominant growth pattern? Choose one of: lepidic, acinar, papillary, micropapillary, solid."`，growth 样本会带 topology descriptor。
3. A0 = `--no-aux`；A1 = `--structure-on-growth`（默认开）。5-fold slide-disjoint × 3 seeds。
4. `pilot eval` 后 `scripts/eval/aggregate_slide_predictions.py` 做 slide 多数投票。
5. 主分析先做 3 类敏感性：lepidic / acinar / solid（n=129），再报 5 类；主指标 slide-level balanced accuracy / macro F1，附 slide-grouped bootstrap CI。
6. 功效警告：143 张 slide、5 类，2 pt 差异基本不可检测；把 2 pt 当方向性目标，不要当硬 gate。

## 6. 数据检查清单

- [ ] WSI 数量 = 143，文件可被 openslide 打开（`openslide.OpenSlide(path).level_count`）。
- [ ] 每张 slide 的 label 非空，且属于 5 类。
- [ ] 类别计数与 59/51/19/9/5 一致（或记录差异）。
- [ ] 每 slide 至少 32 个组织比例 ≥0.5 的 patch；不足时记录并降低阈值。
- [ ] 5-fold 分层后，每折验证集里罕见类（papillary/micropapillary）数量 ≥1；若不足，合并为 "other" 或先做 3 类。
- [ ] manifest 中保留 `wsi_id`，用于 slide 级聚合和 slide-disjoint 划分。
