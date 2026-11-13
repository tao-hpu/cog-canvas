# Independent annotation kit (judge validation)

This directory holds the materials for the independent human re-annotation used to validate the LLM judge (paper Appendix L.7, "Judge Validation: Human-Agreement Study").

An annotator who is not an author relabeled the same 100 items that the author had labeled. The sheet was blind: each item shows the question, the gold answer, and one system answer. The judge verdict and the identity of the pipeline are withheld.

## Files

| File | Purpose |
|---|---|
| `build_independent_kit.py` | Reads the blind annotation sheet and writes a self-contained HTML tool. Judge verdicts and pipeline labels are removed. |
| `annotate.html` | The tool given to the annotator. It runs in a browser with no installation. Progress is stored in the browser, and the result is exported as JSON. |
| `annotations_export.json` | The independent annotator's exported labels. |
| `score_independent.py` | Computes agreement and Cohen's kappa between the independent annotator, the LLM judge, and the author annotator. |
| `build_discordant_kit.py`, `annotate_discordant.html`, `score_discordant.py` | A second kit of the same design, restricted to questions on which the two representations disagree. |

## Procedure

1. Build the tool (only needed if the source sheet changes):
   ```
   python build_independent_kit.py
   ```
2. Give `annotate.html` to the annotator. The task takes about 20 to 30 minutes.
3. The annotator exports `annotations_export.json` and returns it.
4. Score:
   ```
   python score_independent.py annotations_export.json
   ```

## Instructions shown to the annotator

The on-screen instructions in `annotate.html` are in Chinese, the annotator's first language. An English translation follows.

> This is a manual check of answers produced by an AI question-answering system. No specialist knowledge is needed; use common sense. It takes about 20 to 30 minutes.
>
> Part 1 (compare with the reference answer). Each item shows a question, a reference answer, and the system's answer. Decide whether the system's answer is correct. It counts as correct if its meaning matches the reference answer, even when the wording differs. It counts as incorrect if it gives wrong information, does not answer the question, or omits key information.
>
> Part 2 (decide whether the system declined to answer). These items have no reference answer. Decide only whether the system declined or gave an answer. "Declined" means the system said it does not know, that the information is not in the context, or that it cannot answer. "Gave an answer" means the system produced specific content, whether or not it is right.
>
> Progress is saved in this browser. You can close the file and reopen it to continue. When you finish, click Export and send back the downloaded file.

## Results

Run `python score_independent.py annotations_export.json` to reproduce these figures.

Both parts are reported in Appendix L.7 of the paper. Part 1 covers 100 correctness items, stratified 25 per benchmark and representation cell. The independent annotator agrees with the LLM judge on 89% of items (Cohen's kappa 0.78) and with the author annotator on 90% (kappa 0.80).

Part 2 (50 LoCoMo category-5 answers, 25 per pipeline, labeled "declined" or "gave an answer") validates the abstention judge and is reported in the same appendix. The independent annotator agrees with the abstention judge on all 50 items (kappa 1.00; 5 declined, 45 answered). An independent LLM annotator (`cat5_blind_labels_annotator2.csv`) labeled the same sample and also agrees on all 50.

## Source data (read-only)

- Correctness items: `../results/judge_agreement/annotation_sheet.csv` and `key.json`
- Abstention items: `../results/cat5_blind_labels.csv`, `cat5_blind_labels_KEY.csv`, `cat5_blind_labels_annotator2.csv`

## Participation

The annotator was a volunteer, gave informed consent, received no payment, and is credited in the paper's Acknowledgments. The items are model outputs on public benchmarks and contain no personal data.
