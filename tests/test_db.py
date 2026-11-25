from slmbench import db

DESIGN_COLUMNS = {
    "configs": "config_id model strategy precision engine tier device dataset context_mode k model_path seed "
               "temperature max_new_tokens is_mock",
    "runs": "run_id config_id query_id block_id started_at prompt_tokens output_tokens ttft_s tpot_s e2e_latency_s "
            "tokens_per_s server_prompt_ms server_predicted_ms peak_mem_mb mem_kind peak_vram_mb energy_j avg_power_w "
            "temp_c throttled truncated retrieval_hit raw_output parsed_answer parse_ok gold_answer correct error is_mock",
    "energy_blocks": "block_id config_id device repeat_index started_at ended_at n_queries meter_start_mwh "
                     "meter_end_mwh energy_j idle_power_w idle_energy_j source is_mock idle_started_at idle_meter_start_mwh",
    "training_runs": "train_id model strategy dataset seed learning_rate selected val_metric n_train_examples epochs "
                     "peak_vram_mb duration_s energy_j avg_power_w trainable_params total_params final_loss optimizer "
                     "grad_checkpointing is_mock peak_mem_mb mem_kind device cpu_s",
    "artifacts": "artifact_id model strategy dataset precision path size_bytes sha256 train_id",
    "telemetry": "sample_id scope ref_id t power_w vram_mb mem_mb temp_c",
    "environment": "key value",
    "cpu_usage": "usage_id scope ref_id config_id device started_at ended_at wall_s cpu_s peak_mem_mb mem_kind "
                 "n_queries source coverage is_mock",
    "energy_estimates": "estimate_id usage_id scope ref_id config_id device wall_s cpu_s mem_gb n_queries "
                        "p_core_w_low p_core_w_central p_core_w_high p_mem_w_per_gb pue energy_j_low "
                        "energy_j_central energy_j_high energy_basis method_note is_mock",
}


def _cfg(cid="c1", mock=0):
    return {"config_id": cid, "model": "m", "strategy": "base", "precision": "F16", "engine": "llamacpp",
            "tier": "edge", "device": "d", "dataset": "pubmedqa", "context_mode": "closed", "k": 0,
            "model_path": "/x", "seed": 42, "temperature": 0.0, "max_new_tokens": 16, "is_mock": mock}


def test_schema_matches_design(tmp_path):
    conn = db.connect(tmp_path / "r.db")
    for table, cols in DESIGN_COLUMNS.items():
        have = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        assert have == cols.split(), table


def test_resumable_inserts_and_mock_propagation(tmp_path):
    conn = db.connect(tmp_path / "r.db")
    db.upsert_config(conn, _cfg(mock=1))
    assert db.insert_run(conn, {"config_id": "c1", "query_id": "q1", "correct": 1})
    assert not db.insert_run(conn, {"config_id": "c1", "query_id": "q1", "correct": 0})
    assert db.done_query_ids(conn, "c1") == {"q1"}
    row = conn.execute("SELECT is_mock, correct FROM runs").fetchone()
    assert tuple(row) == (1, 1)
    assert db.has_mock(conn)


def test_errors_can_be_retried(tmp_path):
    conn = db.connect(tmp_path / "r.db")
    db.upsert_config(conn, _cfg())
    db.insert_run(conn, {"config_id": "c1", "query_id": "q1", "error": "boom"})
    assert db.done_query_ids(conn, "c1", include_errors=False) == set()
    db.insert_run(conn, {"config_id": "c1", "query_id": "q1", "correct": 1}, replace=True)
    assert conn.execute("SELECT error FROM runs").fetchone()[0] is None


def test_energy_block_meter_conversion(tmp_path):
    conn = db.connect(tmp_path / "r.db")
    db.upsert_config(conn, _cfg())
    db.upsert_energy_block(conn, {"block_id": "b", "config_id": "c1", "device": "d", "meter_start_mwh": 100.0,
                                  "source": "usb_meter_manual"})
    db.upsert_energy_block(conn, {"block_id": "b", "meter_end_mwh": 110.0, "n_queries": 10})
    r = conn.execute("SELECT energy_j, n_queries, config_id FROM energy_blocks").fetchone()
    assert r[0] == 36.0 and r[1] == 10 and r[2] == "c1"


def test_merge_db_combines_devices_idempotently(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    for path, cid, dev in ((a, "c1", "pi"), (b, "c2", "thin")):
        conn = db.connect(path)
        db.upsert_config(conn, {**_cfg(cid), "device": dev})
        db.insert_run(conn, {"config_id": cid, "query_id": "q1", "correct": 1})
        db.set_env(conn, {f"{dev}/os": "x"})
        db.insert_training_run(conn, {"model": "m", "strategy": "lora", "dataset": "pubmedqa", "seed": 42,
                                      "learning_rate": 1e-4, "duration_s": 1.0})
        conn.close()
    dest = tmp_path / "all.db"
    db.merge_dbs(dest, [a, b])
    db.merge_dbs(dest, [a, b])
    conn = db.connect(dest)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM configs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0] == 1
    assert set(db.get_env(conn)) == {"pi/os", "thin/os"}
