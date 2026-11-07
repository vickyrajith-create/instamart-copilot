import subprocess
import sys
import time

STEPS = [
    "generate_data.py",
    "part_a_audit.py",
    "part_a_clean.py",
    "part_a_validate.py",
    "part_a_sql.py",
    "part_a_pandas.py",
    "part_b_rfm.py",
    "part_b_kmeans.py",
    "part_b_segments.py",
    "lapse_data.py",
    "part_b_models.py",
    "part_b_balance.py",
    "part_b_final.py",
    "part_c_redact.py",
    "part_c_triage.py",
    "part_d_rag.py",
    "part_d_run.py",
]

start = time.time()
for i, script in enumerate(STEPS, 1):
    print(f"\n===== [{i}/{len(STEPS)}] {script} =====", flush=True)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\nSTOPPED: {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

print(f"\nALL {len(STEPS)} STEPS FINISHED in {time.time() - start:.0f} seconds")