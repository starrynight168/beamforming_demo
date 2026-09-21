import sys
from pathlib import Path

TRAIN_ROOT = Path("/home/shanjunjie/wangjiabao/paper1/deeplearn/train")
sys.path.insert(0, str(TRAIN_ROOT / "server_jobs"))
import si_master_pipeline as smp

specs = smp.build_all_specs()
total = 0
done_c = 0
pending_c = 0

print("=== MBAN SI Ablation 13 Modules Progress Audit ===")
for wave_name, wave_specs in specs.items():
    wave_done = 0
    wave_pending = []
    for name, out_dir, args in wave_specs:
        total += 1
        if smp.is_model_done(out_dir):
            done_c += 1
            wave_done += 1
        else:
            pending_c += 1
            wave_pending.append(name)
    pct = wave_done / len(wave_specs) * 100
    print(f"{wave_name:<10}: {wave_done:2d}/{len(wave_specs):2d} ({pct:5.1f}%) | Pending: {len(wave_pending):2d}")
    if wave_pending and len(wave_pending) <= 6:
        print(f"   -> {', '.join(wave_pending)}")
    elif wave_pending:
        print(f"   -> {', '.join(wave_pending[:5])} ... (+{len(wave_pending)-5})")

print("-" * 55)
print(f"TOTAL: {done_c}/{total} completed ({done_c/total*100:.1f}%), {pending_c} remaining.")
