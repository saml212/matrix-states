import json
d = json.load(open("/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/1820fb87-8dcc-46b9-a583-eecc7f146f28/scratchpad/diag_2p34_md0_xcheck_output.json"))
cells = d["focus_cells"]
print("teeth:", d["teeth"])
cur = None
for cid in cells:
    c = cells[cid]
    if "error" in c:
        print(cid, "ERROR:", c["error"]); continue
    cfg = "__".join(cid.split("__")[:3])
    if cfg != cur:
        print(f"\n== {cfg} ==")
        cur = cfg
    row = " ".join(f"D{r['D']}={r['crosscheck_rf90']:.2f}" if not r.get("excluded") else f"D{r['D']}=EXCL"
                   for r in c["profile"])
    nbit = sum(1 for r in c["profile"] if not r.get("excluded") and r["reproduction_bit_identical"])
    nev  = sum(1 for r in c["profile"] if not r.get("excluded"))
    print(f"{cid.split('__')[-1]} loss={c['final_loss']:.2e} {row}  [teeth {nbit}/{nev}]")
