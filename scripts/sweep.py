"""Lanza los experimentos de scripts/experiments.py.

    python scripts/sweep.py q1                # una pregunta
    python scripts/sweep.py q1 q2 --jobs 3    # varias, con 3 ejecuciones a la vez
    python scripts/sweep.py all --part 1/3    # el primer tercio de todo (para repartir entre portátiles)
    python scripts/sweep.py all --dry-run     # enseña qué haría sin lanzar nada

Cada ejecución se guarda en results/<nombre automático>/ y su salida en runs/logs/. Si una ejecución
ya está completa (metrics.csv con todas sus rondas) se salta: se puede cortar con Ctrl+C y relanzar
el mismo comando sin perder nada. Las que se cortaron a medias se repiten enteras.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments import EXPERIMENTS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fl.utils import load_config, run_name  # noqa: E402

RESULTS_DIR = "results"
LOGS_DIR = ROOT / "runs" / "logs"


def collect_runs(names):
    """Lista sin repetir de (carpeta, overrides, rondas) de las preguntas pedidas, en orden fijo."""
    runs, seen = [], set()
    for name in names:
        for overrides in EXPERIMENTS[name]:
            overrides = overrides + [f"output.dir={RESULTS_DIR}"]
            cfg = load_config(overrides=overrides)
            folder = run_name(cfg)
            if folder not in seen:
                seen.add(folder)
                runs.append((folder, overrides, cfg.server.rounds))
    return runs


def is_done(folder, rounds):
    metrics = ROOT / RESULTS_DIR / folder / "metrics.csv"
    if not metrics.exists():
        return False
    with open(metrics) as f:
        return sum(1 for _ in f) == rounds + 1  # cabecera + una fila por ronda


def launch(folder, overrides, threads):
    # Con varias ejecuciones a la vez, cada una usa solo su parte de los núcleos.
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    start = time.perf_counter()
    with open(LOGS_DIR / f"{folder}.txt", "w") as log:
        result = subprocess.run(
            [sys.executable, "run.py", "--set", *overrides],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    return result.returncode, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--jobs", type=int, default=1, help="ejecuciones a la vez (por defecto 1)")
    parser.add_argument("--part", default="1/1", help="i/n: lanza solo la parte i de n (p. ej. 2/3)")
    parser.add_argument("--dry-run", action="store_true", help="enseña las ejecuciones sin lanzarlas")
    args = parser.parse_args()

    names = list(EXPERIMENTS) if "all" in args.experiments else args.experiments
    i, n = (int(x) for x in args.part.split("/"))
    if not 1 <= i <= n:
        parser.error(f"--part {args.part}: tiene que ser i/n con 1 <= i <= n")

    # El reparto se hace antes de mirar qué está hecho, para que sea el mismo en todos los portátiles.
    runs = collect_runs(names)[i - 1 :: n]
    pending = [(folder, overrides) for folder, overrides, rounds in runs if not is_done(folder, rounds)]
    print(f"{len(runs)} ejecuciones ({', '.join(names)}, parte {i}/{n}): "
          f"{len(runs) - len(pending)} ya hechas, {len(pending)} por lanzar")

    if args.dry_run:
        for folder, overrides in pending:
            print(f"  {folder}\n    python run.py --set {' '.join(overrides)}")
        return
    if not pending:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    threads = max(1, (os.cpu_count() or 1) // args.jobs)
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(launch, folder, overrides, threads): folder for folder, overrides in pending}
        for k, future in enumerate(as_completed(futures), start=1):
            folder = futures[future]
            code, seconds = future.result()
            status = "ok" if code == 0 else f"ERROR (ver runs/logs/{folder}.txt)"
            print(f"[{k}/{len(pending)}] {folder}: {status}, {seconds:.0f} s", flush=True)
            if code != 0:
                failed.append(folder)

    if failed:
        print(f"\n{len(failed)} ejecuciones han fallado: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
