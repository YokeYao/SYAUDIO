# ALM-Sycophancy

[中文](#中文) | [English](#english)

## 中文
Qwen2-Audio 在 MMAU / MMAR / GSM8K / MMLU 上的多轮 MCQ 评测脚本，包含基线推理与多种 sycophancy case。

### 数据配置
- `mmau`: `benchmark/MMAU/mmau-test-mini.json`，音频根目录 `benchmark/MMAU`，字段 `audio_id`
- `mmar`: `benchmark/MMAR/MMAR-meta.json`，音频根目录 `benchmark/MMAR`，字段 `audio_path`
- `gsm8k`: `benchmark/GSM8K/test_mcq.jsonl`，音频根目录 `benchmark/GSM8K`，字段 `audio_path`
- `mmlu`: `benchmark/MMLU/mmlu_combined.jsonl`，音频根目录 `benchmark/MMLU`，字段 `audio_path`

### 第 1 轮：基线推理
示例：在 MMAU 上跑 10 条基线并记录日志：
```bash
python code/audio_eval.py \
  --dataset mmau \
  --prompt baseline \
  --limit 10 \
  --max-gen-len 1024
# 日志：result/mmau_baseline_limit10.log
```
参数：
- `--dataset`: 选择数据集（mmau/mmar/gsm8k/mmlu）
- `--prompt`: `baseline` 这里只填这个，其他的在 sychophancy.py 跑
- `--limit`: 处理样本数
- `--max-gen-len`: 生成长度

### 第 2 轮：Sycophancy 评测（基于基线）
在读取基线日志后，使用二轮 prompt 做追问/诱导，结果写入 `result/sychophancy/`。

基础命令格式：
```bash
python code/sychophancy.py \
  --prompt <prompt_key> \
  --variant <仅 bias_feedback 需填 strong/medium/low> \
  --baseline-log result/<dataset>_baseline_limit<k>.log \
  --max-gen-len 1024
# 输出：result/sychophancy/<dataset>_<prompt_key>_<variant|auto>.log
```

各 sycophancy case 与参数示例：
- `bias_feedback`（strong/medium/low）：拒绝第一轮答案，观察是否随从
  ```bash
  python code/sychophancy.py --prompt bias_feedback --variant strong --baseline-log result/mmau_baseline_limit50.log
  ```
- `are_you_sure`：追问“你确定吗？”无需 variant
  ```bash
  python code/sychophancy.py --prompt are_you_sure --baseline-log result/mmau_baseline_limit50.log
  ```
- `answer_sycophancy`：基于基线正确性自动切换 variant（基线对则诱导错，基线错则提示正确）
  ```bash
  python code/sychophancy.py --prompt answer_sycophancy --baseline-log result/mmau_baseline_limit50.log
  ```
- `mimicry_sycophancy`：提供一段与基线答案一致/相反的提示文本（variant 自动根据基线正确性选择 correct/incorrect）
  ```bash
  python code/sychophancy.py --prompt mimicry_sycophancy --baseline-log result/mmau_baseline_limit50.log
  ```

其他可选参数：
- `--limit`: 仅取基线日志前 k 条样本做 sycophancy（便于快速 sanity check）
- `--max-gen-len`: 二轮生成长度

---

## English
Qwen2-Audio multi-turn MCQ evaluation on MMAU / MMAR / GSM8K / MMLU, covering baseline inference and multiple sycophancy cases.

### Data config
- `mmau`: `benchmark/MMAU/mmau-test-mini.json`, audio root `benchmark/MMAU`, field `audio_id`
- `mmar`: `benchmark/MMAR/MMAR-meta.json`, audio root `benchmark/MMAR`, field `audio_path`
- `gsm8k`: `benchmark/GSM8K/test_mcq.jsonl`, audio root `benchmark/GSM8K`, field `audio_path`
- `mmlu`: `benchmark/MMLU/mmlu_combined.jsonl`, audio root `benchmark/MMLU`, field `audio_path`

### Round 1: Baseline inference
Example: run 10 baseline samples on MMAU and log:
```bash
python code/audio_eval.py \
  --dataset mmau \
  --prompt baseline \
  --limit 10 \
  --max-gen-len 1024
# Log: result/mmau_baseline_limit10.log
```
Flags:
- `--dataset`: choose dataset (mmau/mmar/gsm8k/mmlu)
- `--prompt`: use `baseline` here; other prompts run via sychophancy.py
- `--limit`: number of samples
- `--max-gen-len`: generation length

### Round 2: Sycophancy evaluation (uses baseline history)
Reads a baseline log, applies a second-round prompt, writes to `result/sychophancy/`.

General command:
```bash
python code/sychophancy.py \
  --prompt <prompt_key> \
  --variant <strong/medium/low only for bias_feedback> \
  --baseline-log result/<dataset>_baseline_limit<k>.log \
  --max-gen-len 1024
# Output: result/sychophancy/<dataset>_<prompt_key>_<variant|auto>.log
```

Cases and examples:
- `bias_feedback` (strong/medium/low): reject round-1 answer to test compliance
  ```bash
  python code/sychophancy.py --prompt bias_feedback --variant strong --baseline-log result/mmau_baseline_limit50.log
  ```
- `are_you_sure`: ask “are you sure?”; no variant needed
  ```bash
  python code/sychophancy.py --prompt are_you_sure --baseline-log result/mmau_baseline_limit50.log
  ```
- `answer_sycophancy`: auto-switch variant by baseline correctness (if baseline correct, suggest wrong; if wrong, hint correct)
  ```bash
  python code/sychophancy.py --prompt answer_sycophancy --baseline-log result/mmau_baseline_limit50.log
  ```
- `mimicry_sycophancy`: injects a hint aligned/opposed to baseline answer (variant auto-selects correct/incorrect)
  ```bash
  python code/sychophancy.py --prompt mimicry_sycophancy --baseline-log result/mmau_baseline_limit50.log
  ```

Other flags:
- `--limit`: truncate baseline records for quick runs
- `--max-gen-len`: generation length in round 2
