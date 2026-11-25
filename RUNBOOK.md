# RUNBOOK: executing the study on real hardware

Commands Marcel runs on his own machines. Nothing here is run on the corporate Cloud Desktop. Every number in
Chapter 4 must come from `slmbench report` over the databases these steps produce (DESIGN.md section 0).

Conventions: `$PROTO` is the `prototype/` directory, `$DISS` is `../dissertation`. All servers bind `127.0.0.1`.
Write down anything you do differently; those notes become `[[CONFIRM]]` answers in Chapter 3.

Every device uses llama.cpp release tag `b12532` (commit `08c252e221759e0f69e80f0450b295c92d2344e1`, verified on GitHub
2026-11-19; record it in Chapter 3): `export LLAMA_CPP_TAG=b12532 LLAMA_CPP_COMMIT=08c252e221759e0f69e80f0450b295c92d2344e1`.
This tag replaced `--mlock` with `--load-mode`; the device profiles start llama-server with `--load-mode mmap+mlock
--cache-ram 0 --fit off` (weights locked in memory, no server prompt cache, no automatic argument fitting).

## Container-emulated edge pilot

Base and RAG can run before the hardware is ready, in Docker containers capped to 4 CPUs / 8 GB on a large x86
host (`edge-container-4c8g`, accuracy of record from `cpu-reference`). No GPU, so no FFT/LoRA/QLoRA, and no
energy counter. Results go to `results/pilot.db` and `dissertation/generated-pilot/` and are never presented as
Pi results. Steps, durations and the labelling rule: [../PILOT.md](../PILOT.md).

## Before each campaign (every device)

Run a read-only readiness check on the machine itself before any long `slmbench run` or training batch. It prints
a PASS/WARN/FAIL table and exits 1 on any FAIL; fix every FAIL first and read every WARN.

```bash
slmbench preflight --device gpu-workstation --db results/gpu-workstation.db
slmbench preflight --device edge-pi5 --db results/edge-pi5.db            # meter CSV: telemetry.meter_csv / --meter-csv, --tz
slmbench preflight --device edge-thinclient --db results/edge-thinclient.db
```

Checks: Python and the pinned `[gpu]`/`[edge]`, `[rag]`, `[analysis]` versions; llama-server `/health` on the
profile's 127.0.0.1 endpoint and `--jinja` (WARN if the server is not running but the runner manages it); every
GGUF the device's plans reference exists and matches the sha256 in `artifacts` (copy the workstation DB first;
`--no-sha` skips hashing); `performance` governor (edge); `vcgencmd get_throttled` = 0x0, SoC < 50 C and Wi-Fi
down (Pi; Wi-Fi is WARN only); RAPL `energy_uj` readable (thin client); NVML device and energy counter (GPU);
the meter CSV's latest row is under 60 s old with a time-zone-aware timestamp (when `block_energy: usb_meter_csv`);
free disk for results and models; results DB writable and free of `is_mock` rows (WARN). Checks that do not apply
to the profile show WARN "not applicable". Preflight writes nothing, not even an empty DB.

## 0. Data files (once, on the workstation)

| Dataset | Files | Source (verify against the librarian facts) |
|---|---|---|
| PubMedQA | `ori_pqal.json`, `ori_pqaa.json`, `test_ground_truth.json` | github.com/pubmedqa/pubmedqa (`data/`) |
| FinQA | `train.json`, `dev.json`, `test.json` | github.com/czyssrs/FinQA (`dataset/`) |
| ARC-Easy | Hugging Face `allenai/ai2_arc`, config `ARC-Easy`, split `test` | `--hf` |

Put the raw files in `~/slm/raw/`.

## 1. GPU workstation

### 1.1 Install

```bash
sudo nvidia-smi -pm 1                                  # persistence mode (DESIGN.md 3.4)
git clone --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp ~/llama.cpp
test "$(git -C ~/llama.cpp rev-parse HEAD)" = "$LLAMA_CPP_COMMIT"
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build -j --target llama-server llama-quantize
cd $PROTO
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[gpu,rag,analysis,edge,dev]"   # pins: torch>=2.3, transformers>=4.45, peft>=0.12,
                                                 # bitsandbytes>=0.43, accelerate>=0.34, datasets>=2.20
pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
pytest                                                  # offline suite must pass before any real run
```

Alternatively `docker build -f docker/Dockerfile.gpu -t slmbench-gpu .` (the image pins the same tag).

### 1.2 Accept the Llama licence

Open https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct, accept the Llama 3.2 Community License, then
`huggingface-cli login`. Qwen2.5 and SmolLM2 are Apache-2.0 and need no gate.

### 1.3 Environment record and data

```bash
slmbench env --db results/gpu-workstation.db --device gpu-workstation --ambient-c <measured C>
slmbench prepare-data --dataset pubmedqa --out data --labeled ~/slm/raw/ori_pqal.json \
  --artificial ~/slm/raw/ori_pqaa.json --test-ids ~/slm/raw/test_ground_truth.json
slmbench prepare-data --dataset finqa --out data --train ~/slm/raw/train.json --dev ~/slm/raw/dev.json \
  --test ~/slm/raw/test.json
slmbench prepare-data --dataset arc_easy --out data --hf
slmbench build-index --dataset pubmedqa --data-dir data
slmbench build-index --dataset finqa --data-dir data
slmbench estimate --counts-only            # expect accuracy 126, retention 63, edge 252 configs
```

Check `data/pubmedqa/manifest.json`: `official_test_split: true`, counts test 500 / val 500, and the
`artificial_removed_overlap` count. Copy the whole `data/` directory to each edge device; every device must use the
same files (the manifests carry sha256 hashes).

### 1.4 Training (3 models x {fft, lora, qlora} x 2 datasets)

```bash
slmbench train --list-jobs selection | head          # 36 jobs: 18 adapted models x 2 learning rates
slmbench train --dry-run --model qwen2.5-0.5b --strategy qlora --dataset pubmedqa --lr 1e-4   # sanity check
scripts/run_gpu_training.sh selection                 # seed 42, full LR grid, validation metric recorded
scripts/run_gpu_training.sh select                    # marks the best-validation LR per adapted model
scripts/run_gpu_training.sh variability               # seeds 43, 44 at the selected LR (cost only, not exported)
```

FFT loads fp32 master weights and trains under bf16 autocast. On GPUs without native bf16 (compute capability
< 8; emulated bf16 does not count) autocast and the QLoRA bnb compute dtype both fall back to fp16. Training
prompts use the same `context_tokens` and `prompt_overhead_tokens` as evaluation, so truncation is identical; the recipe is
stored in `training_runs.optimizer`, e.g. `adamw_torch fp32-master bf16-autocast`. QLoRA uses
`prepare_model_for_kbit_training`, and its fp32 upcast of non-quantised parameters is part of the measured QLoRA
memory (label `qlora-kbit-prep(...)`). `train --select` ignores `is_mock` rows.

If FFT runs out of memory, set `training.optimizer: adamw_bnb_8bit` in `configs/matrix.yaml`, rerun **all**
strategies (the optimiser must be identical across strategies) and state it in Chapter 3.

### 1.5 Export GGUF

```bash
scripts/run_gpu_training.sh export     # merges seed-42 selected adapters, F16 (explicit --outtype f16), Q8_0, Q4_K_M
sqlite3 results/gpu-workstation.db "SELECT model,strategy,dataset,precision,size_bytes FROM artifacts"
```

Expect 18 adapted x 3 + 3 base x 3 = 63 GGUF files. Copy `~/slm/models/` to each edge device (same paths).

### 1.6 Chat-template check (before every campaign, on every device)

`llama-server` runs with `--jinja` so it uses the GGUF's own chat template; the fixed
`chat_template_kwargs.date_string` in `configs/matrix.yaml` keeps the Llama 3 date header identical between
training and evaluation. Check one rendered prompt per model family against the server (needs `transformers`
for the tokenizer; exit 1 and a diff on mismatch):

```bash
for m in llama3.2-1b qwen2.5-0.5b smollm2-360m; do
  slmbench check-template --device gpu-workstation --plan accuracy --filter model=$m --filter strategy=base \
    --filter dataset=pubmedqa --data-dir data
done
```

Repeat with `--filter strategy=lora --filter dataset=pubmedqa` once the merged models exist.

### 1.7 Accuracy of record, retention, reference arms

```bash
slmbench run --db results/gpu-workstation.db --device gpu-workstation --plan accuracy  --data-dir data
slmbench run --db results/gpu-workstation.db --device gpu-workstation --plan retention --data-dir data
slmbench run --db results/gpu-workstation.db --device gpu-workstation --plan ref-gold  --data-dir data
pip install vllm   # or the docker `vllm` stage; vLLM pins its own torch, so use a separate venv if needed
slmbench run --db results/gpu-workstation.db --device gpu-workstation --plan ref-vllm  --data-dir data
```

Each command is resumable: rerun it after any interruption. `--retry-errors` re-attempts failed queries.

## 2. Raspberry Pi 5 (8 GB)

### 2.1 Install and prepare

```bash
sudo apt install -y git cmake build-essential python3-venv libcurl4-openssl-dev
git clone --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp ~/llama.cpp
test "$(git -C ~/llama.cpp rev-parse HEAD)" = "$LLAMA_CPP_COMMIT"
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build -j4 --target llama-server
cd $PROTO && python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[edge,rag,analysis,dev]" && pytest
sudo scripts/pin_cpu_governor.sh performance
sudo rfkill block wifi                                  # use wired Ethernet or local console; no other workload
slmbench env --db results/edge-pi5.db --device edge-pi5 --ambient-c <measured C>
```

Active cooler fitted, headless. Insert the inline USB-C power meter between the PSU and the Pi.

### 2.2 Pilot and estimate

```bash
slmbench run --db results/edge-pi5-pilot.db --device edge-pi5 --plan edge --data-dir data --pilot 5 --repeats 1
slmbench estimate --db results/edge-pi5-pilot.db --device edge-pi5 --plan edge
```

The Pi default is `block_energy: usb_meter_csv` (`configs/devices/edge-pi5.yaml`): the runner records the idle-start,
block-start and block-end timestamps, and the values come from the meter CSV (2.4). With `--block-energy
usb_meter_manual` the runner instead asks for the meter's
cumulative mWh three times per block: at the start of the 60 s idle window, at the end of idle (= block start),
and at block end. Each reading is timestamped when entered. Idle power and gross energy both come from the meter,
and idle is subtracted over exactly the interval between the two block readings. A reading below the previous one
(idle start <= block start <= block end) or a non-number is rejected and re-prompted. CSV logging is preferred
(3 repeats x 126 configs is about 1 100 manual readings per device). Use the estimate to confirm the 100-item subset fits;
if not, reduce `plans.edge.subset` and record the change as a refinement.

### 2.3 Full edge plan

```bash
scripts/run_edge_matrix.sh edge-pi5                     # or the two `slmbench run` lines it prints
```

Per block the runner: waits until the SoC is below 50 C, (re)starts `llama-server` with `--load-mode mmap+mlock`, discards
3 warm-up queries, measures a 60 s idle baseline with the model loaded, then runs the block. Blocks run in a
seeded random order. Temperature and `vcgencmd get_throttled` are logged per query; a block with a throttled
query is re-run once. Energy repeats default to 3 (`runner.energy_repeats`) and are interleaved: repeat r of
every config runs before repeat r + 1, so the per-config bootstrap CI over 3 blocks is approximate. Each block id
carries a per-invocation suffix, so a resumed run adds a new block row instead of overwriting the old one. Blocks
with fewer queries than the planned item count (pilot or resumed blocks) are excluded from the H1b energy decision
and bootstrap and counted as `partial_blocks`; run a full repeat if a config has fewer than 3 full blocks.
Prompts (including `/tokenize` calls) are built before the idle window and block start.

### 2.4 Meter CSV import (only if the meter exported a CSV)

```bash
slmbench energy-block import-csv --db results/edge-pi5.db --device edge-pi5 --csv ~/meter-export.csv --tz +02:00
slmbench energy-block list --db results/edge-pi5.db | head
```

`--device` is required. Only that device's blocks opened in meter mode (`source` starting `usb_meter`) are
updated; NVML, RAPL, tegrastats and PMIC blocks are never touched. Gross energy and the idle power are both
recomputed from the CSV over each block's own idle window and block interval.
The CSV needs a `timestamp` column and `mWh` or `Wh`. Timestamps without an offset are rejected unless `--tz`
gives the meter clock's zone (`+02:00` or `Africa/Johannesburg` for SAST); epoch seconds need no `--tz`. Keep the
meter clock and the Pi clock synchronised (NTP on the Pi; set the meter clock before the session).

Copy `results/edge-pi5.db` back to the workstation.

## 3. x86 thin client (8 GB)

Same as section 2 with `edge-thinclient`, except: no `vcgencmd`; energy comes from RAPL CPU-package counters
(labelled package energy, not wall power), which need root to read:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj   # resets at reboot; repeat after each boot
sudo scripts/pin_cpu_governor.sh performance
slmbench env --db results/edge-thinclient.db --device edge-thinclient --ambient-c <measured C>
scripts/run_edge_matrix.sh edge-thinclient
```

If a wall meter is available, add `--block-energy usb_meter_manual` (or `usb_meter_csv`) as on the Pi.
Run the Pi and the thin client in parallel. Copy `results/edge-thinclient.db` back.

## 4. Merge, analyse, report (workstation)

```bash
slmbench merge-db --into results/results.db results/gpu-workstation.db results/edge-pi5.db results/edge-thinclient.db
slmbench analyze --db results/results.db --out results/analysis       # refuses (exit 3) if any is_mock row
slmbench report  --db results/results.db \
  --out /local/home/marcelkp/.kiro/crew/workspace/slm-adaptation-research/dissertation/generated
```

`analyze`, `report` and `advise` refuse (exit 3) if any mock row is present: never merge a test DB into
`results.db`. `--allow-mock` exists only for testing and watermarks every output, including the CSVs. The tariff and grid
factor come from `configs/tariff.yaml`; override with `--tariff <ZAR/kWh>` and `--grid-factor <kg/kWh>`.
Check `tables/accuracy_parity.md`: any config flagged `flag_rerun_500` must be re-run on all 500 items on that
device (`slmbench run --plan edge --device <dev> --filter model=... --filter strategy=...` after setting
`plans.edge.subset: null` for that run) before the report is final.

## 5. Advisor

```bash
slmbench advise --db results/results.db --device edge-pi5 --ram-mb 6000 --latency-budget-s 10 \
  --monthly-queries 20000 --min-accuracy 0.6 --dataset pubmedqa --out results/advisor-pi5-pubmedqa.html
```

Open the HTML file in a browser; it is self-contained (no server, no scripts). The advisor excludes configs whose
edge answers disagree with the accuracy of record (parity flag) until they are re-run on 500 items, counts both
LR-grid training runs in adaptation energy, and shows "no metered energy" for configs measured only by the Pi PMIC.
