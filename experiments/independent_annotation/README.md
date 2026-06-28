# 独立人工标注包（强化 judge 验证）

目的：现在论文的 judge 验证是**作者一人标注**（reviewer 攻击点），且 abstention judge
**完全没有人工验证**。引入一个独立的普通人标注者，可同时补上这两处：

- **主研究 100 题**：独立标注者重标同一批 → 报 inter-annotator κ + 独立人机一致性。
- **abstention 50 题**：abstention judge 的**首次人工验证**（论文 Limitations 现写"future work"）。

标注者**不需要专业知识、不需要署名**（进致谢即可，且保持"独立"这一卖点）。

## 流程

1. 生成标注工具（已生成 `annotate.html`，改了数据才需重跑）：
   ```
   python build_independent_kit.py
   ```
2. **只把 `annotate.html` 发给朋友**。他双击用浏览器打开（推荐 Chrome），
   一题一题点按钮，约 20-30 分钟。进度自动存本机，可中途关。
3. 做完点「导出结果」，他把下载的 `annotations_export.json` 发回。
4. 你判分：
   ```
   python score_independent.py annotations_export.json
   ```
   输出：独立人机一致性（overall + 分层 κ）、作者-独立 inter-annotator κ、
   abstention 人机一致性。直接抄进 judge-validation 附录。

## 文件

- `build_independent_kit.py` — 读现有盲表，生成自包含 HTML（删掉了 judge 判定/pipeline，保证盲标）。
- `annotate.html` — **发给朋友的唯一文件**，零安装零依赖。
- `score_independent.py` — 收回结果后算一致性。

## 数据来源（只读，不改）

- 主研究 100 题：`../results/judge_agreement/annotation_sheet.csv` + `key.json`
- abstention 50 题：`../results/cat5_blind_labels.csv` + `..._KEY.csv` + `..._annotator2.csv`
