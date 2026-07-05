import json

SCRATCH = "/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/1820fb87-8dcc-46b9-a583-eecc7f146f28/scratchpad"

p = json.load(open(f"{SCRATCH}/trajectory_plan.json"))

assignment = {
    2: [("wave1", "openr1-mix_s0"), ("wave1", "openr1-mix_s1"), ("wave1", "openr1-mix_s2"),
        ("mixcontrol", "openr1-mix-ext_s0"), ("mixcontrol", "openr1-mix-ext_s1"), ("mixcontrol", "openr1-mix-ext_s2")],
    3: [("wave1", "wikitext-mix_s0"), ("wave1", "wikitext-mix_s1"), ("wave1", "wikitext-mix_s2"),
        ("mixcontrol", "wikitext-mix-ext_s0"), ("mixcontrol", "wikitext-mix-ext_s1"), ("mixcontrol", "wikitext-mix-ext_s2")],
    4: [("wave1ext", "openr1-mix-ext_s0"), ("wave1ext", "openr1-mix-ext_s1"), ("wave1ext", "openr1-mix-ext_s2")],
    5: [("wave1ext", "wikitext-mix-ext_s0"), ("wave1ext", "wikitext-mix-ext_s1"), ("wave1ext", "wikitext-mix-ext_s2")],
    6: [("wave2", "openr1-mix-ext_s0"), ("wave2", "openr1-mix-ext_s1"), ("wave2", "openr1-mix-ext_s2")],
    7: [("wave2", "wikitext-mix-ext_s0"), ("wave2", "wikitext-mix-ext_s1"), ("wave2", "wikitext-mix-ext_s2")],
}

OUT_BASE = "results/lm_rd_trackc/trajectory_probes"

for gpu, jobs in assignment.items():
    lines = [
        "#!/usr/bin/env bash",
        "set -e",
        "cd /home/nvidia/chapter2/deltanet_rd",
        f"export CUDA_VISIBLE_DEVICES={gpu}",
        f"mkdir -p {OUT_BASE}/wave1 {OUT_BASE}/wave1ext {OUT_BASE}/wave2 {OUT_BASE}/mixcontrol",
    ]
    for fam, run in jobs:
        paths = p[fam]["runs"][run]
        out = f"{OUT_BASE}/{fam}/{run}.json"
        lines.append(f'echo "=== GPU{gpu}: {fam} {run} ({len(paths)} ckpts) ==="')
        cmd_lines = ["/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \\", "  --checkpoints \\"]
        for path in paths:
            cmd_lines.append(f"    {path} \\")
        cmd_lines.append(f"  --out {out}")
        lines.extend(cmd_lines)
        lines.append(f'echo "=== GPU{gpu}: {fam} {run} DONE ==="')
    lines.append(f"echo ALL_DONE_GPU{gpu}")
    with open(f"{SCRATCH}/driver_gpu{gpu}.sh", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote driver_gpu{gpu}.sh, {len(jobs)} jobs, {sum(len(p[fam]['runs'][run]) for fam,run in jobs)} ckpts")
