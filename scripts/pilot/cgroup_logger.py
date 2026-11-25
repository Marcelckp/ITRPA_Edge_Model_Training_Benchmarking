#!/usr/bin/env python3
"""Host-side 1 Hz sampler of one Docker container's cgroup CPU time and memory (stdlib only, Python >= 3.6).

Waits for the named container, appends one CSV row per second until its cgroup disappears, then waits for the
next instance (a resumed campaign) until --while-tmux names a tmux session that no longer exists.

    taskset -c 39 python3 scripts/pilot/cgroup_logger.py --name slm-pilot-edge \
        --out ../pilot-logs/edge-cgroup.csv --while-tmux slm-pilot

The samples feed `slmbench energy-estimate` (estimated from CPU time, not metered).
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

FIELDS = ["utc", "unix_s", "container_id", "cgroup_version", "cpu_usage_ns", "mem_usage_bytes",
          "mem_peak_bytes", "mem_rss_bytes", "mem_cache_bytes", "nr_throttled", "throttled_ns"]


def _read(path):
    with open(path) as f:
        return f.read()


def _kv(text):
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


def cgroup_dirs(root, cid):
    """Return (version, {controller: dir}) for the container, or None if its cgroup does not exist."""
    v1 = [("docker/" + cid), ("system.slice/docker-%s.scope" % cid)]
    for rel in v1:
        cpu = os.path.join(root, "cpuacct", rel)
        if os.path.isfile(os.path.join(cpu, "cpuacct.usage")):
            return 1, {"cpuacct": cpu, "cpu": os.path.join(root, "cpu", rel),
                       "memory": os.path.join(root, "memory", rel)}
    for rel in ("system.slice/docker-%s.scope" % cid, "docker/" + cid):
        d = os.path.join(root, rel)
        if os.path.isfile(os.path.join(d, "cpu.stat")):
            return 2, {"unified": d}
    return None


def sample(version, dirs):
    """One reading as a dict of integers (missing counters are None)."""
    r = {k: None for k in FIELDS[4:]}
    if version == 1:
        r["cpu_usage_ns"] = int(_read(os.path.join(dirs["cpuacct"], "cpuacct.usage")))
        m = dirs["memory"]
        r["mem_usage_bytes"] = int(_read(os.path.join(m, "memory.usage_in_bytes")))
        r["mem_peak_bytes"] = int(_read(os.path.join(m, "memory.max_usage_in_bytes")))
        st = _kv(_read(os.path.join(m, "memory.stat")))
        r["mem_rss_bytes"] = st.get("total_rss", st.get("rss"))
        r["mem_cache_bytes"] = st.get("total_cache", st.get("cache"))
        try:
            cs = _kv(_read(os.path.join(dirs["cpu"], "cpu.stat")))
            r["nr_throttled"] = cs.get("nr_throttled")
            r["throttled_ns"] = cs.get("throttled_time")
        except OSError:
            pass
    else:
        d = dirs["unified"]
        cs = _kv(_read(os.path.join(d, "cpu.stat")))
        r["cpu_usage_ns"] = cs["usage_usec"] * 1000
        r["nr_throttled"] = cs.get("nr_throttled")
        if "throttled_usec" in cs:
            r["throttled_ns"] = cs["throttled_usec"] * 1000
        r["mem_usage_bytes"] = int(_read(os.path.join(d, "memory.current")))
        try:
            r["mem_peak_bytes"] = int(_read(os.path.join(d, "memory.peak")))
        except OSError:
            pass
        st = _kv(_read(os.path.join(d, "memory.stat")))
        r["mem_rss_bytes"] = st.get("anon")
        r["mem_cache_bytes"] = st.get("file")
    return r


def container_id(name):
    p = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}} {{.Id}}", name],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
    if p.returncode != 0:
        return None
    running, cid = p.stdout.split()
    return cid if running == "true" else None


def tmux_alive(session):
    return subprocess.run(["tmux", "has-session", "-t", session],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def log(msg):
    print("%s %s" % (datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), msg), flush=True)


def record(out, cid, root, interval):
    found = cgroup_dirs(root, cid)
    if found is None:
        log("container %s has no visible cgroup under %s" % (cid[:12], root))
        return 0
    version, dirs = found
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    n = 0
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        nxt = time.monotonic()
        while True:
            try:
                r = sample(version, dirs)
            except (OSError, ValueError, KeyError):
                break
            now = time.time()
            r.update(utc=datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                     unix_s="%.3f" % now, container_id=cid[:12], cgroup_version=version)
            w.writerow(r)
            f.flush()
            n += 1
            nxt += interval
            time.sleep(max(0.0, nxt - time.monotonic()))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="container name to watch")
    ap.add_argument("--out", required=True, help="CSV to append to")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between checks while waiting")
    ap.add_argument("--root", default="/sys/fs/cgroup")
    ap.add_argument("--while-tmux", help="stop once this tmux session is gone and the container is not running")
    ap.add_argument("--once", action="store_true", help="stop after the first container instance exits")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log("waiting for container %s -> %s" % (a.name, a.out))
    while True:
        cid = container_id(a.name)
        if cid:
            log("recording %s (%s)" % (a.name, cid[:12]))
            n = record(a.out, cid, a.root, a.interval)
            log("container %s gone after %d samples" % (cid[:12], n))
            if a.once:
                return 0
            time.sleep(a.poll)
            continue
        if a.while_tmux and not tmux_alive(a.while_tmux):
            log("tmux session %s is gone; stopping" % a.while_tmux)
            return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
