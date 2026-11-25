import pytest

from slmbench import config as C


def test_planned_counts_match_design_3_6(full_matrix):
    acc = C.expand(full_matrix, ["accuracy"])
    ret = C.expand(full_matrix, ["retention"])
    edge = C.expand(full_matrix, ["edge"])
    assert len(acc) == 126
    assert len(ret) == 63
    assert len(edge) == 252
    assert C.count_runs(acc)["queries"] == 63000
    assert C.count_runs(ret)["queries"] == 18900
    assert C.count_runs(edge)["queries"] == 25200
    assert C.adapted_model_count(full_matrix) == 18
    assert len(C.training_plan(full_matrix, "selection")) == 36
    assert len(C.training_plan(full_matrix, "variability")) == 36


def test_per_model_per_dataset_is_21(full_matrix):
    acc = C.expand(full_matrix, ["accuracy"])
    sub = [c for c in acc if c.model == "qwen2.5-0.5b" and c.dataset == "finqa"]
    assert len(sub) == 21
    assert sorted({c.k for c in sub if c.strategy == "rag"}) == [3, 5, 10]
    assert all(c.k == 0 for c in sub if c.strategy != "rag")


def test_retention_has_base_and_adapted_models(full_matrix):
    ret = C.expand(full_matrix, ["retention"])
    assert sum(1 for c in ret if c.strategy == "base") == 9
    assert sum(1 for c in ret if c.strategy != "base") == 54
    assert not any(c.strategy == "rag" for c in ret)
    assert {c.context_mode for c in ret if c.strategy == "lora"} == {"retention:pubmedqa", "retention:finqa"}


def test_reference_arms_are_separate_and_tagged(full_matrix):
    gold = C.expand(full_matrix, ["ref-gold"])
    vllm = C.expand(full_matrix, ["ref-vllm"])
    assert gold and all(c.context_mode == "gold" and c.strategy == "base" for c in gold)
    assert len(vllm) == 3 * 4 * 2 and all(c.engine == "vllm" and c.precision == "bf16" for c in vllm)
    ids = {c.config_id for c in C.expand(full_matrix, ["accuracy"])}
    assert not ids & {c.config_id for c in gold + vllm}
    for c in gold + vllm:
        assert C.plan_of(c.db_row(), full_matrix) == c.plan


def test_config_id_stable_and_path_independent(full_matrix):
    a = C.expand(full_matrix, ["edge"])[0]
    b = C.ExpConfig(**{k: getattr(a, k) for k in C.ID_FIELDS}, model_path="/elsewhere.gguf")
    assert a.config_id == b.config_id
    assert len(a.config_id) == 16
    again = C.expand(full_matrix, ["edge"])[0]
    assert again.config_id == a.config_id
    mock = C.ExpConfig(**{k: getattr(a, k) for k in C.ID_FIELDS}, is_mock=1)
    assert mock.config_id != a.config_id


def test_known_config_id_regression():
    cfg = C.ExpConfig(model="m", strategy="base", precision="F16", engine="llamacpp", tier="edge", device="d",
                      dataset="pubmedqa", context_mode="closed", k=0, seed=42, temperature=0.0, max_new_tokens=16)
    assert cfg.config_id == C.make_config_id(cfg.as_dict())
    assert cfg.config_id == C.ExpConfig(**{k: getattr(cfg, k) for k in C.ID_FIELDS}).config_id


def test_exclude_and_filters(full_matrix):
    full_matrix["plans"]["edge"]["exclude"] = [{"device": "edge-pi5", "precision": "F16"}]
    edge = C.expand(full_matrix, ["edge"])
    assert len(edge) == 252 - 42
    only = C.expand(full_matrix, ["edge"], filters=C.parse_filters(["model=smollm2-360m", "k=3,5"]))
    assert {c.k for c in only} == {3, 5}


def test_invalid_matrix_rejected(full_matrix):
    full_matrix["plans"]["ref-vllm"]["precisions"] = ["Q4_K_M"]
    with pytest.raises(C.ConfigError):
        C.validate_matrix(full_matrix)


def test_devices_load_and_are_loopback():
    for name in ("gpu-workstation", "edge-pi5", "edge-thinclient", "edge-jetson"):
        dev = C.load_device(name)
        for b in dev["backends"].values():
            assert "127.0.0.1" in b["base_url"]


def test_model_registry_has_exact_param_counts(full_matrix):
    llama = next(m for m in full_matrix["models"] if m["id"] == "llama3.2-1b")
    assert llama["params"] == 1235814400
    exact = {m["id"]: m["params"] for m in full_matrix["models"]}
    assert exact == {"llama3.2-1b": 1235814400, "qwen2.5-0.5b": 494032768, "smollm2-360m": 361821120}


def test_tariff_values_and_override():
    t = C.load_tariff()
    assert t["tariff_zar_per_kwh"] == pytest.approx(4.14)
    assert t["grid_kgco2e_per_kwh"] == pytest.approx(0.960)
    assert "Cape Town" in t["tariff_source"] and "DFFE" in t["grid_source"]
    o = C.load_tariff(tariff=3.0, grid_factor=0.5)
    assert o["tariff_zar_per_kwh"] == 3.0 and o["tariff_source"] == "CLI override"
