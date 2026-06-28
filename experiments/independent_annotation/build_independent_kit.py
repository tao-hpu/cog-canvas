"""Build a zero-install, self-contained HTML annotation kit for an INDEPENDENT
human annotator (not an author), to strengthen the judge-validation study:

  1. Main correctness (100 items): re-annotate the same triples the author
     labeled, blind, so we can report inter-annotator agreement (author vs.
     independent) AND independent-human vs. LLM-judge agreement.
  2. Abstention (50 items): the FIRST human validation of the category-5
     abstention judge (currently checked only against a lexical detector and an
     LLM annotator -- Limitations flags "direct human validation remains future
     work").

Output: a single annotate.html that embeds all items (verdicts/pipeline stripped),
autosaves progress to localStorage, and exports the annotator's labels as JSON.
Send ONLY annotate.html to the annotator; they send back annotations_export.json.

Usage:
    python build_independent_kit.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
OUT = HERE / "annotate.html"

MAIN_SHEET = RESULTS / "judge_agreement" / "annotation_sheet.csv"
CAT5_SHEET = RESULTS / "cat5_blind_labels.csv"


def load_main():
    items = []
    with MAIN_SHEET.open() as f:
        for r in csv.DictReader(f):
            items.append({
                "id": r["id"],
                "question": r["question"],
                "gold": r["ground_truth"],
                "answer": r["answer"],
            })
    return items


def load_cat5():
    items = []
    with CAT5_SHEET.open() as f:
        reader = csv.DictReader(f)
        # tolerate the verbose header "human_label_abstained (1/0)"
        for r in reader:
            items.append({
                "id": str(r["id"]),
                "question": r["question"],
                "answer": r["answer"],
            })
    return items


HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>答案核对（约 20-30 分钟）</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         max-width: 820px; margin: 0 auto; padding: 24px 18px 120px; color: #1a1a1a;
         line-height: 1.6; background: #fafafa; }
  h1 { font-size: 20px; } h2 { font-size: 16px; color:#555; }
  .bar { position: sticky; top: 0; background: #fafafa; padding: 10px 0; border-bottom: 1px solid #ddd; z-index: 5; }
  .prog { height: 8px; background: #e5e5e5; border-radius: 4px; overflow: hidden; }
  .prog > i { display:block; height:100%; background:#2d7; width:0%; transition: width .2s; }
  .meta { font-size: 13px; color:#888; margin-top:6px; }
  .card { background:#fff; border:1px solid #e2e2e2; border-radius:10px; padding:18px; margin-top:16px;
          box-shadow:0 1px 3px rgba(0,0,0,.04); }
  .lbl { font-size:12px; font-weight:700; letter-spacing:.04em; color:#888; text-transform:uppercase; margin-bottom:4px; }
  .field { margin-bottom:14px; }
  .q { font-size:16px; font-weight:600; }
  .gold { background:#eef7ee; border-left:3px solid #2d7; padding:8px 12px; border-radius:4px; white-space:pre-wrap; }
  .ans { background:#f4f6fb; border-left:3px solid #58c; padding:8px 12px; border-radius:4px; white-space:pre-wrap; }
  .btns { display:flex; gap:12px; margin-top:18px; flex-wrap:wrap; }
  button.choice { flex:1; min-width:140px; padding:14px; font-size:15px; font-weight:600; border:2px solid #ccc;
          border-radius:8px; background:#fff; cursor:pointer; transition:all .12s; }
  button.choice:hover { border-color:#888; }
  button.choice.sel-good { background:#2d7; border-color:#2d7; color:#fff; }
  button.choice.sel-bad  { background:#e55; border-color:#e55; color:#fff; }
  .nav { display:flex; justify-content:space-between; margin-top:16px; }
  .nav button { padding:8px 18px; font-size:14px; border:1px solid #bbb; border-radius:6px; background:#fff; cursor:pointer; }
  .nav button:disabled { opacity:.4; cursor:default; }
  .intro, .done { background:#fff; border:1px solid #e2e2e2; border-radius:10px; padding:22px; margin-top:16px; }
  .intro ul { padding-left:20px; } .intro li { margin:6px 0; }
  #export { padding:14px 26px; font-size:16px; font-weight:700; background:#2d7; color:#fff; border:none; border-radius:8px; cursor:pointer; }
  input#name { padding:8px; font-size:14px; border:1px solid #bbb; border-radius:6px; width:220px; }
  .warn { color:#c33; font-size:13px; }
  kbd { background:#eee; border:1px solid #ccc; border-radius:4px; padding:1px 6px; font-size:12px; }
</style>
</head>
<body>
<h1>答案核对任务</h1>

<div id="intro" class="intro">
  <p>谢谢帮忙！这是一个给 AI 问答系统打分的人工核对任务，<b>不需要任何专业知识</b>，凭常识判断即可，约 20-30 分钟。</p>
  <p>你会看到两部分，共 <b id="total-pre">?</b> 道题：</p>
  <p><b>第一部分（对照标准答案，判断对错）：</b></p>
  <ul>
    <li>每题给你一个<b>问题</b>、一个<b>标准答案</b>、和系统给出的<b>系统答案</b>。</li>
    <li>判断<b>系统答案是否答对了</b>——只要意思和标准答案一致就算「正确」，措辞不同没关系。</li>
    <li>系统答案信息错误、答非所问、或漏掉关键信息，算「错误」。</li>
  </ul>
  <p><b>第二部分（判断是否「拒答」）：</b></p>
  <ul>
    <li>这部分<b>没有标准答案</b>。只判断系统答案是<b>「拒答」</b>还是<b>「给出了答案」</b>。</li>
    <li>「拒答」= 系统说不知道 / 上下文里没有这个信息 / 无法回答 这类。</li>
    <li>「给出了答案」= 系统实际给出了具体内容（不管对不对）。</li>
  </ul>
  <p class="warn">进度会自动保存在本机浏览器里，中途关掉再打开同一文件可以接着做。全部做完后点导出，把下载的文件发回即可。</p>
  <p>请先填你的名字（随便英文/拼音都行，用于区分标注者）：<br>
     <input id="name" placeholder="annotator name"></p>
  <p><button id="start" class="nav" style="padding:12px 28px;font-size:15px;font-weight:600;border-color:#2d7;color:#2d7;">开始</button></p>
</div>

<div id="app" style="display:none;">
  <div class="bar">
    <div class="prog"><i id="progfill"></i></div>
    <div class="meta"><span id="counter"></span> · <span id="phase"></span> · <span id="answered"></span></div>
  </div>
  <div id="card"></div>
  <div class="nav">
    <button id="prev">← 上一题</button>
    <button id="next">下一题 →</button>
  </div>
</div>

<div id="done" class="done" style="display:none;">
  <h2>全部完成 ✅</h2>
  <p>点下面按钮导出结果文件，然后把下载的 <code>annotations_export.json</code> 发回即可。</p>
  <p><button id="export">导出结果</button></p>
  <p class="meta">没看到下载？检查浏览器下载栏，或换 Chrome 打开本文件。</p>
</div>

<script>
const MAIN = __MAIN_JSON__;
const CAT5 = __CAT5_JSON__;
const ITEMS = [
  ...MAIN.map(x => ({...x, phase:"main"})),
  ...CAT5.map(x => ({...x, phase:"abstention"})),
];
const KEY = "indep_annot_v1";
document.getElementById("total-pre").textContent = ITEMS.length;

let state = JSON.parse(localStorage.getItem(KEY) || "{}");
state.labels = state.labels || {};   // id -> verdict string
state.name = state.name || "";
let idx = state.idx || 0;

const $ = id => document.getElementById(id);

function save(){ state.idx = idx; localStorage.setItem(KEY, JSON.stringify(state)); }

function render(){
  const it = ITEMS[idx];
  const isMain = it.phase === "main";
  $("counter").textContent = `第 ${idx+1} / ${ITEMS.length} 题`;
  $("phase").textContent = isMain ? "第一部分：判断对错" : "第二部分：是否拒答";
  const done = Object.keys(state.labels).length;
  $("answered").textContent = `已答 ${done}`;
  $("progfill").style.width = (100*(idx)/ITEMS.length) + "%";

  let html = `<div class="card">`;
  html += `<div class="field"><div class="lbl">问题</div><div class="q">${esc(it.question)}</div></div>`;
  if (isMain) html += `<div class="field"><div class="lbl">标准答案</div><div class="gold">${esc(it.gold)}</div></div>`;
  html += `<div class="field"><div class="lbl">系统答案</div><div class="ans">${esc(it.answer)}</div></div>`;
  const cur = state.labels[it.phase + ":" + it.id];
  if (isMain){
    html += `<div class="btns">
      <button class="choice ${cur==='CORRECT'?'sel-good':''}" data-v="CORRECT">✓ 正确</button>
      <button class="choice ${cur==='INCORRECT'?'sel-bad':''}" data-v="INCORRECT">✗ 错误</button></div>`;
  } else {
    html += `<div class="btns">
      <button class="choice ${cur==='ABSTAINED'?'sel-good':''}" data-v="ABSTAINED">🚫 拒答 / 没有信息</button>
      <button class="choice ${cur==='ANSWERED'?'sel-bad':''}" data-v="ANSWERED">💬 给出了答案</button></div>`;
  }
  html += `</div>`;
  $("card").innerHTML = html;

  document.querySelectorAll(".choice").forEach(b => b.onclick = () => {
    state.labels[it.phase + ":" + it.id] = b.dataset.v;
    save();
    if (idx < ITEMS.length - 1){ idx++; render(); } else { render(); maybeDone(); }
  });
  $("prev").disabled = idx === 0;
  $("next").textContent = idx === ITEMS.length-1 ? "完成 ✓" : "下一题 →";
}

function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function maybeDone(){
  if (Object.keys(state.labels).length === ITEMS.length){
    $("app").style.display = "none";
    $("done").style.display = "block";
  }
}

$("prev").onclick = () => { if (idx>0){ idx--; render(); } };
$("next").onclick = () => { if (idx<ITEMS.length-1){ idx++; render(); } else maybeDone(); };

$("start").onclick = () => {
  state.name = $("name").value.trim() || "anonymous";
  save();
  $("intro").style.display = "none";
  $("app").style.display = "block";
  render();
};

$("export").onclick = () => {
  const main = {}, abst = {};
  for (const it of MAIN){ const v = state.labels["main:"+it.id]; if(v) main[it.id]=v; }
  for (const it of CAT5){ const v = state.labels["abstention:"+it.id]; if(v) abst[it.id]=v; }
  const out = { annotator: state.name, main, abstention: abst };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotations_export.json";
  a.click();
};

// resume
if (state.name){ $("name").value = state.name; }
if (Object.keys(state.labels).length > 0 && Object.keys(state.labels).length < ITEMS.length){
  $("intro").style.display = "none"; $("app").style.display = "block"; render();
} else if (Object.keys(state.labels).length === ITEMS.length){
  $("done").style.display = "block";
}
</script>
</body>
</html>
"""


def main():
    main_items = load_main()
    cat5_items = load_cat5()
    html = (HTML
            .replace("__MAIN_JSON__", json.dumps(main_items, ensure_ascii=False))
            .replace("__CAT5_JSON__", json.dumps(cat5_items, ensure_ascii=False)))
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  main correctness items: {len(main_items)}")
    print(f"  abstention items:       {len(cat5_items)}")
    print(f"  total:                  {len(main_items)+len(cat5_items)}")
    print("\nSend ONLY annotate.html to the annotator.")
    print("They send back annotations_export.json -> run: python score_independent.py annotations_export.json")


if __name__ == "__main__":
    main()
