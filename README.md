# ALM-Sycophancy

[中文](#中文) | [English](#english)

## 中文
Qwen2-Audio 在 MMAU / MMAR / GSM8K / MMLU 上的多轮 MCQ 评测脚本，包含基线推理与多种 sycophancy case。

运行前（GPU 申请与环境）：
- 开源模型推理前需先申请 GPU，交互式申请示例：
  ```bash
  srun -p cscc-gpu-p --qos=gpu-debug-qos \
       --gres=gpu:1 -c 8 --mem=64G -t 03:00:00 \
       --pty bash
  ```
  申请时请确保节点数 ≤ 2、总 GPU 数 ≤ 4；可将 `--gres=gpu:2` 申请 2 卡，但时间仍需 ≤ 03:00:00。若需最长 24h，请改用 `sbatch` 批量提交：
  - GPU（开源模型）：`sbatch bash/run_gpu_all_prompts.sh`（脚本内已申请 1 节点、2 卡、24h，日志写入 `logs/`）；如需换模型，可在命令末尾追加模型名，例如 `sbatch bash/run_gpu_all_prompts.sh "Qwen/Qwen2-Audio-72B-Instruct"`。
  - API（闭源模型，无需 GPU）：`sbatch bash/run_api_all_prompts.sh`；若需指定分区/时长，可在命令前补充 `-p/-q/-t`；如需换 API 模型，可在命令末尾追加模型名，例如 `sbatch bash/run_api_all_prompts.sh "gpt-4o-mini-audio-preview"`。
  - `squeue --me` 可查看作业状态，示例：
    ```
    squeue --me
                 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              173833_0 cscc-gpu- qwen2-au junchi.y  R       0:18      1 gpu-51
    ```
- 调用闭源 API 模型无需申请 GPU。
- 环境准备：`conda activate alm`，然后 `cd /home/junchi.yao/ICML2026-ALM`。

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
  --num-gpus 1
# 日志：result/mmau_baseline_limit10.log
```
常用参数与行为：
- `--dataset`: 选择数据集（mmau/mmar/gsm8k/mmlu）。
- `--model`: 开源模型使用 `--num-gpus` 控制卡数（1/2/4，使用 GPU0 开始），多卡会自动拆分样本、生成 `*.gpu<id>.log` 并汇总回主日志；API 模型忽略该参数。
- `resume`: 检测到已有日志会自动追加写入并跳过已完成样本，无需手动指定。

### 第 2 轮：Sycophancy 评测（基于基线）
在读取基线日志后，使用二轮 prompt 做追问/诱导，结果写入 `result/sycophancy/`。

基础命令格式：
```bash
python code/sycophancy.py \
  --prompt <prompt_key> \
  --variant <仅 bias_feedback 需填 strong/medium/low> \
  --baseline-log result/<dataset>_baseline_limit<k>.log \
  --num-gpus 1
# 输出：result/sycophancy/<dataset>_<prompt_key>_<variant|auto>.log
```

各 sycophancy case 与参数示例：
- `bias_feedback`（strong/medium/low）：拒绝第一轮答案，观察是否随从
  ```bash
  python code/sycophancy.py --prompt bias_feedback --variant strong --baseline-log result/mmau_baseline_limit50.log
  ```
- `are_you_sure`：追问“你确定吗？”无需 variant
  ```bash
  python code/sycophancy.py --prompt are_you_sure --baseline-log result/mmau_baseline_limit50.log
  ```
- `answer_sycophancy`：基于基线正确性自动切换 variant（基线对则诱导错，基线错则提示正确）
  ```bash
  python code/sycophancy.py --prompt answer_sycophancy --baseline-log result/mmau_baseline_limit50.log
  ```
- `mimicry_sycophancy`：提供一段与基线答案一致/相反的提示文本（variant 自动根据基线正确性选择 correct/incorrect）
  ```bash
  python code/sycophancy.py --prompt mimicry_sycophancy --baseline-log result/mmau_baseline_limit50.log
  ```

常用参数与行为：
- `--model`: 开源模型可配合 `--num-gpus`（1/2/4）并行跑，API 模型下 `--num-gpus` 代表 API worker 数量（建议为4）。
- `--limit`, `--max-gen-len`: 控制样本数与生成长度。
- `--baseline-log`: 若缺省则默认读取 `mmau` 基线日志。
- `resume`: 自动检测现有日志（含多卡/API 分片日志），跳过已完成样本并在主日志汇总分片，无需额外开关。

---

## English
Qwen2-Audio multi-turn MCQ evaluation on MMAU / MMAR / GSM8K / MMLU, covering baseline inference and multiple sycophancy cases.

Before you run:
- Open-source models need an interactive GPU allocation first:
  ```bash
  srun -p cscc-gpu-p --qos=gpu-debug-qos \
       --gres=gpu:1 -c 8 --mem=64G -t 03:00:00 \
       --pty bash
  ```
  Keep nodes ≤ 2 and total GPUs ≤ 4; you may switch to `--gres=gpu:2` for two GPUs but the wall time must stay ≤ 03:00:00. For up to 24h, submit with `sbatch`:
  - GPU (open-source models): `sbatch bash/run_gpu_all_prompts.sh` (requests 1 node, 2 GPUs, 24h; logs go to `logs/`); append a model name to override the default, e.g. `sbatch bash/run_gpu_all_prompts.sh "Qwen/Qwen2-Audio-72B-Instruct"`.
  - API (closed-source, no GPU): `sbatch bash/run_api_all_prompts.sh`; add `-p/-q/-t` if you need to pin partition/qos/time; append a model name to override the default, e.g. `sbatch bash/run_api_all_prompts.sh "gpt-4o-mini-audio-preview"`.
  - Check queue status with `squeue --me`, e.g.:
    ```
    squeue --me
                 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              173833_0 cscc-gpu- qwen2-au junchi.y  R       0:18      1 gpu-51
    ```
- Closed-source API models do not require GPU allocation.
- Env prep: `conda activate alm`, then `cd /home/junchi.yao/ICML2026-ALM`.

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
  --num-gpus 1
# Log: result/mmau_baseline_limit10.log
```
Key flags and behavior:
- `--dataset`: choose dataset (mmau/mmar/gsm8k/mmlu).
- `--model`: local models honor `--num-gpus` (1/2/4 starting at GPU0); multi-GPU runs split samples, write `*.gpu<id>.log`, and merge back. API models ignore this flag.
- `resume`: existing logs trigger auto-resume—completed samples are skipped and new results append to the same log.

### Round 2: Sycophancy evaluation (uses baseline history)
Reads a baseline log, applies a second-round prompt, writes to `result/sycophancy/`.

General command:
```bash
python code/sycophancy.py \
  --prompt <prompt_key> \
  --variant <strong/medium/low only for bias_feedback> \
  --baseline-log result/<dataset>_baseline_limit<k>.log \
  --num-gpus 1
# Output: result/sycophancy/<dataset>_<prompt_key>_<variant|auto>.log
```

Cases and examples:
- `bias_feedback` (strong/medium/low): reject round-1 answer to test compliance
  ```bash
  python code/sycophancy.py --prompt bias_feedback --variant strong --baseline-log result/mmau_baseline_limit50.log
  ```
- `are_you_sure`: ask “are you sure?”; no variant needed
  ```bash
  python code/sycophancy.py --prompt are_you_sure --baseline-log result/mmau_baseline_limit50.log
  ```
- `answer_sycophancy`: auto-switch variant by baseline correctness (if baseline correct, suggest wrong; if wrong, hint correct)
  ```bash
  python code/sycophancy.py --prompt answer_sycophancy --baseline-log result/mmau_baseline_limit50.log
  ```
- `mimicry_sycophancy`: injects a hint aligned/opposed to baseline answer (variant auto-selects correct/incorrect)
  ```bash
  python code/sycophancy.py --prompt mimicry_sycophancy --baseline-log result/mmau_baseline_limit50.log
  ```

Key flags and behavior:
- `--model`: local models can use `--num-gpus` (1/2) for multi-GPU; for API models, `--num-gpus` controls API worker count（suggest 4）
- `--baseline-log`: defaults to the `mmau` baseline log when omitted.
- `resume`: logs (including multi-GPU/API shard logs) are auto-detected; completed samples are skipped and shard logs are merged into the main log automatically.
