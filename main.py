"""
Command-line entry point.

  python main.py prep                      # clean data, write data_report.json (both datasets)
  python main.py repro                     # same seed twice -> identical output?  (needs GAN library)
  python main.py tune                      # choose the short epoch budget on validation data
  python main.py run                       # Chandigarh: 4 arms x 5 seeds  (resumable)
  python main.py run --dataset bilbao      # second dataset: Tier 1 and Tier 4 only
  python main.py summarize                 # results/<dataset>/summary.txt / .csv
  python main.py replicate --base-synth "<path to base repo>/3. AI models/1. Real vs Synth vs RealSynth/datasets/synth/60_Bilbao_synth.csv"
                                           # replication check vs the base paper (needs bilbao a1 runs)
  python main.py all                       # everything above, in order
  python main.py run --paper3 --epochs 2000 3000 5000 7500   # paper 3: new arms at several step budgets
Useful flags: --arms a1_base_long ...  --seeds 42 123  --force  --dataset bilbao
"""
import argparse, os
import config as C
from src import preprocess, experiments, replication_check


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prep", "repro", "tune", "run", "summarize", "replicate", "all"])
    ap.add_argument("--dataset", default="chandigarh", choices=list(C.DATASETS))
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--paper3", action="store_true", help="use the paper-3 arms (conditioned time, sliding windows)")
    ap.add_argument("--epochs", nargs="*", type=int, help="override the epoch budget (one run per value; files get an _ep<N> tag)")
    ap.add_argument("--force", action="store_true", help="recompute even if a result file exists")
    ap.add_argument("--base-synth", default=os.path.join(C.BASE_DIR, "data", "raw", "bilbao", "base_paper_synth_Bilbao.csv"),
                    help="base paper's released synthetic Bilbao CSV (for the `replicate` command)")
    a = ap.parse_args()

    def do_run(ds):
        arms = a.arms or (C.PAPER3_ARMS if a.paper3 else C.DATASETS[ds]["arms"])
        for seed in (a.seeds or C.FINAL_SEEDS):
            for arm in arms:
                for ep in (a.epochs or [None]):
                    experiments.run_single(ds, arm, seed, force=a.force, epochs=ep, tag=f"_ep{ep}" if ep else "")

    if a.command in ("prep", "all"):
        for ds in C.DATASETS:
            _, rep = preprocess.prepare_dataset(ds)
            print(f"[prep] {ds}: {rep['final_rows']} rows, {rep['segments']} segments, test rows {rep['test_rows']}, "
                  f"NOx handling: {rep['nox_handling']}")
    if a.command in ("repro", "all"):
        experiments.repro_check(a.dataset)
    if a.command in ("tune", "all"):
        experiments.tune("chandigarh", force=a.force)
    if a.command in ("run", "all"):
        for ds in ([a.dataset] if a.command == "run" else list(C.DATASETS)):
            do_run(ds)
    if a.command == "replicate":
        replication_check.run(a.base_synth)
    if a.command in ("summarize", "all"):
        for ds in ([a.dataset] if a.command == "summarize" else list(C.DATASETS)):
            experiments.summarize(ds)


if __name__ == "__main__":
    main()
