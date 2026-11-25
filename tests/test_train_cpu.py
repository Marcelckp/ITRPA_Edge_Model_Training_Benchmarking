import sys
import types

import pytest

from slmbench import db
from slmbench import train as T

CPUINFO_SPR = "processor\t: 0\nflags\t\t: fpu sse avx512f avx512_bf16 avx512_vnni\n"
CPUINFO_OLD = "processor\t: 0\nflags\t\t: fpu sse avx2\n"


def test_cpu_bf16_detection_and_device():
    assert T.cpu_bf16(CPUINFO_SPR) and not T.cpu_bf16(CPUINFO_OLD)
    assert T.cpu_bf16("flags : amx_bf16 amx_tile\n")
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    assert T.resolve_device("auto", fake) == "cpu" and T.resolve_device("cuda", fake) == "cuda"
    with pytest.raises(ValueError):
        T.resolve_device("tpu", fake)


def test_cpu_recipes_keep_fp32_master_and_cuda_unchanged(small_matrix):
    fft = T.make_plan(small_matrix, "toy-a", "fft", "pubmedqa", 42, 1e-5, "data")
    lora = T.make_plan(small_matrix, "toy-a", "lora", "pubmedqa", 42, 1e-4, "data")
    qlora = T.make_plan(small_matrix, "toy-a", "qlora", "pubmedqa", 42, 1e-4, "data")
    r = T.precision_recipe(fft, True, "cpu")
    assert r["load_dtype"] == "float32" and r["bf16"] and not r["fp16"] and r["label"].startswith("cpu ")
    assert "fp32-master bf16-autocast" in r["label"]
    r = T.precision_recipe(lora, True, "cpu")
    assert r["load_dtype"] == "bfloat16" and r["bnb_compute_dtype"] == "bfloat16"
    r = T.precision_recipe(lora, False, "cpu")
    assert r["load_dtype"] == "float32" and not r["bf16"] and not r["fp16"] and "none-autocast" in r["label"]
    assert "qlora-kbit-prep" in T.precision_recipe(qlora, True, "cpu")["label"]
    g = T.precision_recipe(fft, False)
    assert g["fp16"] and g["label"] == "adamw_torch fp32-master fp16-autocast"


def test_throughput_excludes_first_step_and_counter():
    assert T.throughput([(0.0, 100, 120)])["tokens_per_s"] is None
    t = T.throughput([(10.0, 100, 120), (12.0, 300, 360), (14.0, 500, 600)])
    assert t["tokens_per_s"] == pytest.approx(100.0) and t["padded_tokens_per_s"] == pytest.approx(120.0)
    assert t["s_per_step"] == pytest.approx(2.0) and t["timed_steps"] == 2

    class Mask(list):
        def sum(self):
            return sum(map(sum, self))

        def numel(self):
            return sum(map(len, self))

    c = T.TokenCounter(lambda feats: {"attention_mask": Mask([[1, 1, 0], [1, 1, 1]])})
    c([{}, {}])
    assert (c.tokens, c.padded) == (5, 6)


class _Tensor:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


class _Param:
    def __init__(self, n, grad):
        self.n, self.requires_grad = n, grad

    def numel(self):
        return self.n


def _fake_stack(monkeypatch, calls):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.float32, torch.bfloat16, torch.float16 = "float32", "bfloat16", "float16"
    torch.set_num_threads = lambda n: calls.setdefault("threads", n)

    tr = types.ModuleType("transformers")

    class Tok:
        pad_token, eos_token, chat_template, pad_token_id, eos_token_id = None, "</s>", None, 0, 0

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def __call__(self, text, add_special_tokens=False, **k):
            return {"input_ids": list(range(len(text.split())))}

        def save_pretrained(self, d):
            pass

    class Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, hf_id, **kw):
            calls["load"] = kw
            return cls()

        def parameters(self):
            return [_Param(10, True), _Param(90, False)] if calls.get("strategy") != "fft" else [_Param(100, True)]

        def gradient_checkpointing_enable(self):
            pass

        def save_pretrained(self, d):
            calls["saved"] = d

    class Args:
        def __init__(self, **kw):
            calls["args"] = kw

    class Trainer:
        def __init__(self, **kw):
            self.kw = kw

        def train(self):
            cb = self.kw["callbacks"][0]
            for i in range(3):
                self.kw["data_collator"].tokens += 100
                self.kw["data_collator"].padded += 120
                cb.on_step_end(None, types.SimpleNamespace(global_step=i + 1), types.SimpleNamespace())
            return types.SimpleNamespace(global_step=3, training_loss=1.25)

    class Callback:
        pass

    tr.AutoTokenizer, tr.AutoModelForCausalLM, tr.TrainingArguments, tr.Trainer = Tok, Model, Args, Trainer
    tr.DataCollatorForSeq2Seq = lambda tok, **k: (lambda feats: feats)
    tr.TrainerCallback = Callback
    tr.set_seed = lambda s: None
    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **kw: kw
    peft.get_peft_model = lambda model, cfg: model
    peft.prepare_model_for_kbit_training = lambda model, **kw: model
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tr)
    monkeypatch.setattr(T, "_validate", lambda *a, **k: 0.5)
    monkeypatch.setattr(T, "cpu_bf16", lambda cpuinfo=None: True)
    monkeypatch.setattr(T, "peak_rss_mb", lambda: 1234.0)


class _FakeCgroup:
    paths = None
    v = 100.0

    def __init__(self, *a, **k):
        pass

    def available(self):
        return True

    def cpu_s(self):
        _FakeCgroup.v += 40.0
        return _FakeCgroup.v


@pytest.fixture
def fake_cpu(monkeypatch, small_matrix, prepared_data, tmp_path):
    calls = {}
    _fake_stack(monkeypatch, calls)
    import slmbench.telemetry.cgroup as cgm

    monkeypatch.setattr(cgm, "CgroupSource", _FakeCgroup)
    monkeypatch.setattr(cgm, "read_cgroup", lambda p: types.SimpleNamespace(mem_peak_bytes=2 * 2**30))
    return calls


def test_run_training_cpu_path(fake_cpu, small_matrix, prepared_data, tmp_path):
    plan = T.make_plan(small_matrix, "toy-a", "fft", "pubmedqa", 42, 1e-5, prepared_data, str(tmp_path / "models"))
    fake_cpu["strategy"] = "fft"
    conn = db.connect(tmp_path / "r.db")
    tid = T.run_training(plan, conn, device="cpu", threads=24)
    args, load = fake_cpu["args"], fake_cpu["load"]
    assert args["use_cpu"] and args["bf16"] and not args["fp16"] and fake_cpu["threads"] == 24
    assert load["device_map"] == {"": "cpu"} and load["torch_dtype"] == "float32"
    row = conn.execute("SELECT * FROM training_runs WHERE train_id=?", (tid,)).fetchone()
    assert row["device"] == "cpu" and row["peak_vram_mb"] is None and row["energy_j"] is None
    assert row["peak_mem_mb"] == 1234.0 and row["mem_kind"] == "RSS-peak" and row["cpu_s"] == pytest.approx(40.0)
    assert row["optimizer"].startswith("cpu adamw_torch fp32-master bf16-autocast")
    env = db.get_env(conn, f"train/{tid}/")
    assert env[f"train/{tid}/device"].startswith("cpu: ") and env[f"train/{tid}/gpu"] is None
    assert env[f"train/{tid}/cgroup_peak_mb"] == "2048.0" and env[f"train/{tid}/energy_source"] == "none"
    u = conn.execute("SELECT * FROM cpu_usage WHERE scope='train'").fetchone()
    assert u["ref_id"] == str(tid) and u["cpu_s"] == pytest.approx(40.0)


def test_bench_run_writes_nothing(fake_cpu, small_matrix, prepared_data, tmp_path):
    plan = T.make_plan(small_matrix, "toy-a", "lora", "pubmedqa", 42, 1e-4, prepared_data, str(tmp_path / "models"))
    conn = db.connect(tmp_path / "r.db")
    res = T.run_training(plan, conn, device="cpu", threads=4, limits=T.TrainLimits(max_steps=3, skip_validation=True))
    assert res["tokens"] == 300 and res["timed_steps"] == 2 and res["peak_rss_mb"] == 1234.0
    assert fake_cpu["args"]["max_steps"] == 3 and fake_cpu["load"]["torch_dtype"] == "bfloat16"
    assert conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0] == 0
    assert "saved" not in fake_cpu


def test_cli_bench_flags_guarded(small_matrix, tmp_path, capsys):
    from slmbench.cli import main

    from .conftest import SMALL_MATRIX

    base = ["train", "--matrix", str(SMALL_MATRIX), "--db", str(tmp_path / "r.db"), "--model", "toy-a",
            "--strategy", "lora", "--dataset", "pubmedqa", "--lr", "1e-4"]
    assert main(base + ["--grad-accum", "1", "--dry-run"]) == 2
    assert main(base + ["--bench"]) == 2
    assert main(base + ["--bench", "--grad-accum", "1", "--per-device-batch-size", "2", "--dry-run"]) == 0
    assert '"effective_batch_size": 2' in capsys.readouterr().out


def test_subsample_is_seeded_and_shared(small_matrix):
    items = [{"i": i} for i in range(100)]
    a, b = T.subsample(items, 10), T.subsample(items, 10)
    assert a == b and len(a) == 10 and [x["i"] for x in a] == sorted(x["i"] for x in a)
    assert T.subsample(items, None) is items and T.subsample(items, 500) is items
    small_matrix["training"]["datasets"]["finqa"]["max_train_examples"] = 3
    assert T.make_plan(small_matrix, "toy-a", "lora", "finqa", 42, 1e-4, "data")["max_train_examples"] == 3


def test_tokenised_set_is_cached_across_strategies(small_matrix, prepared_data, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(T, "_tokenize_example", lambda tok, ds, it, plan: calls.append(1) or {"input_ids": [1, 2]})
    lora = T.make_plan(small_matrix, "toy-a", "lora", "pubmedqa", 42, 1e-4, prepared_data, str(tmp_path / "m"))
    fft = T.make_plan(small_matrix, "toy-a", "fft", "pubmedqa", 42, 2e-5, prepared_data, str(tmp_path / "m"))
    items = [{"q": i} for i in range(4)]
    assert T.tokenized_cache_path(lora) == T.tokenized_cache_path(fft)
    assert T.tokenized_train_set(None, lora, items) == T.tokenized_train_set(None, fft, items)
    assert len(calls) == 4
    small_matrix["training"]["max_seq_len"] = 128
    other = T.make_plan(small_matrix, "toy-a", "lora", "pubmedqa", 42, 1e-4, prepared_data, str(tmp_path / "m"))
    assert T.tokenized_cache_path(other) != T.tokenized_cache_path(lora)


def test_cpu_model_prefers_model_name(monkeypatch):
    from slmbench import env

    monkeypatch.setattr(env, "_read", lambda p: "processor\t: 0\ncpu family\t: 6\nmodel\t\t: 143\n"
                        "model name\t: Intel(R) Xeon(R) Platinum 8488C\n")
    assert env.cpu_model() == "Intel(R) Xeon(R) Platinum 8488C"
    monkeypatch.setattr(env, "_read", lambda p: "Hardware\t: BCM2835\nModel\t: Raspberry Pi 5 Model B Rev 1.0\n")
    assert env.cpu_model() == "BCM2835"
