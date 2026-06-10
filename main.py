"""
Ponto de entrada do trabalho: GA aplicado ao TSP nas instâncias da TSPLIB.

Modo interativo (padrão): basta rodar

    python main.py

que o programa lista os arquivos .tsp disponíveis na pasta data/ e pergunta
qual você quer rodar. Você digita o nome (ex.: "att48.tsp" ou só "att48").
Ele roda as variantes do GA, mostra o progresso e, no final, compara a melhor
solução encontrada com o ótimo global conhecido.

Também é possível pular a pergunta passando o nome por linha de comando:

    python main.py --instance att48
    python main.py --instance att48 --variants V2_OX_inv V4_memetic --runs 10

Coloque os arquivos .tsp baixados da TSPLIB na pasta data/.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# No Windows, o console costuma usar cp1252 e embaralha acentos e o símbolo ±.
# Forçamos a saída para UTF-8 para o print sair legível.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.experiment import (
    estimate_runtime_seconds,
    format_duration,
    print_summary_table,
    plot_convergence,
    plot_route,
    run_variant,
    save_summary_csv,
)
from src.ga import GAConfig
from src.tsplib import build_distance_matrix, read_tsp


# --------------------------------------------------------------------------- #
# Variantes do GA a comparar.
#
# A ideia é explorar variações de operadores, como o enunciado pede:
#   V1 -> crossover OX  + mutação swap        (base simples)
#   V2 -> crossover OX  + mutação inversion   (mutação 2-opt)
#   V3 -> crossover PMX + mutação inversion   (troca o crossover)
#   V4 -> OX + inversion + busca local 2-opt  (GA memético)
# Edite à vontade para testar outras combinações no relatório.
# --------------------------------------------------------------------------- #
def make_variants() -> list[GAConfig]:
    return [
        GAConfig(label="V1_OX_swap",      crossover="OX",  mutation="swap"),
        GAConfig(label="V2_OX_inv",       crossover="OX",  mutation="inversion"),
        GAConfig(label="V3_PMX_inv",      crossover="PMX", mutation="inversion"),
        GAConfig(label="V4_memetic",      crossover="OX",  mutation="inversion",
                 local_search=True, ls_max_passes=1, init="nearest_neighbor"),
    ]


# --------------------------------------------------------------------------- #
# Ajuste de esforço por tamanho da instância.
#
# Instâncias maiores precisam de mais população/gerações, mas a busca local
# (V4) fica MUITO cara em u1432 — por isso reduzimos o esforço lá. Ajuste
# conforme o tempo disponível na sua máquina.
# --------------------------------------------------------------------------- #
def scale_config(cfg: GAConfig, n: int) -> GAConfig:
    import copy
    c = copy.copy(cfg)
    if n <= 50:
        c.pop_size, c.generations = 100, 400
    elif n <= 150:
        c.pop_size, c.generations = 200, 800
    elif n <= 300:
        c.pop_size, c.generations = 300, 1200
    else:  # u1432
        c.pop_size, c.generations = 200, 800
        if c.local_search:           # 2-opt completo em 1432 cidades é pesado
            c.pop_size, c.generations = 60, 200
    return c


# --------------------------------------------------------------------------- #
# Seleção interativa da instância
# --------------------------------------------------------------------------- #
def list_available(data_dir: str) -> list[str]:
    """Nomes (sem .tsp) dos arquivos disponíveis na pasta de dados."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.tsp")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def ask_instance(data_dir: str) -> str | None:
    """
    Mostra os .tsp disponíveis e pergunta qual rodar. Aceita 'att48' ou
    'att48.tsp'. Devolve o nome sem extensão, ou None se o usuário cancelar.
    """
    available = list_available(data_dir)
    print(f"\nArquivos .tsp encontrados em '{data_dir}/':")
    if available:
        for name in available:
            print(f"   - {name}.tsp")
    else:
        print("   (nenhum arquivo .tsp encontrado — baixe-os da TSPLIB)")

    while True:
        raw = input("\nQual instância deseja rodar? (digite o nome, ou 'sair'): ").strip()
        if raw.lower() in ("sair", "exit", "quit", "q", ""):
            return None
        name = raw[:-4] if raw.lower().endswith(".tsp") else raw
        if os.path.exists(os.path.join(data_dir, f"{name}.tsp")):
            return name
        print(f"   [erro] '{name}.tsp' não existe em '{data_dir}/'. Tente de novo.")


def ask_yes_no(question: str, default_no: bool = True) -> bool:
    """Pergunta sim/não. Enter vazio assume o padrão (não, por segurança)."""
    suffix = "[s/N]" if default_no else "[S/n]"
    while True:
        ans = input(f"{question} {suffix}: ").strip().lower()
        if ans == "":
            return not default_no
        if ans in ("s", "sim", "y", "yes"):
            return True
        if ans in ("n", "nao", "não", "no"):
            return False
        print("   responda 's' (sim) ou 'n' (não).")


# --------------------------------------------------------------------------- #
# Comparação final da melhor solução encontrada com o ótimo
# --------------------------------------------------------------------------- #
def compare_against_optimum(all_stats) -> None:
    """Imprime, no final, a comparação da melhor rota achada com o ótimo global."""
    best_stats = min(all_stats, key=lambda s: s.best)
    inst = best_stats.instance
    opt = best_stats.optimum

    print("\n" + "#" * 70)
    print(f"#  COMPARAÇÃO FINAL — instância {inst}")
    print("#" * 70)
    print(f"  Melhor variante .............. {best_stats.variant}")
    print(f"  Melhor custo encontrado ...... {best_stats.best:.0f}")
    if opt is not None:
        diff = best_stats.best - opt
        gap = 100.0 * diff / opt
        print(f"  Ótimo global conhecido ....... {opt}")
        print(f"  Diferença (achado - ótimo) ... {diff:+.0f}")
        print(f"  Gap em relação ao ótimo ...... {gap:.2f}%")
        if diff <= 0:
            print("  >>> O GA ATINGIU (ou superou) o ótimo conhecido!")
        elif gap <= 1.0:
            print("  >>> Excelente: a menos de 1% do ótimo.")
        elif gap <= 5.0:
            print("  >>> Bom: a menos de 5% do ótimo.")
        else:
            print("  >>> Há espaço para melhorar (ajuste parâmetros/variantes).")
    else:
        print("  Ótimo global não cadastrado para esta instância.")

    # melhor rota encontrada (sequência de cidades, 1-indexada como na TSPLIB)
    route = [int(c) + 1 for c in best_stats.best_route]
    print(f"\n  Melhor rota (cidades 1..n): {route}")
    print("#" * 70 + "\n")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="pasta com os arquivos .tsp")
    ap.add_argument("--out", default="results", help="pasta de saída")
    ap.add_argument("--instance", default=None,
                    help="instância a rodar (ex.: att48). Se omitido, pergunta.")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="labels das variantes a rodar (padrão: todas)")
    ap.add_argument("--runs", type=int, default=10, help="repetições por variante")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- escolhe a instância: por argumento ou perguntando ---
    inst_name = args.instance
    if inst_name:
        inst_name = inst_name[:-4] if inst_name.lower().endswith(".tsp") else inst_name
        if not os.path.exists(os.path.join(args.data, f"{inst_name}.tsp")):
            print(f"[erro] '{inst_name}.tsp' não encontrado em '{args.data}/'.")
            return
    else:
        inst_name = ask_instance(args.data)
        if inst_name is None:
            print("Encerrado.")
            return

    variants = make_variants()
    if args.variants:
        variants = [v for v in variants if v.label in args.variants]

    path = os.path.join(args.data, f"{inst_name}.tsp")
    inst = read_tsp(path)
    D = build_distance_matrix(inst)
    print(f"\n### Instância {inst.name}  (n={inst.n}, "
          f"EDGE_WEIGHT_TYPE={inst.edge_weight_type}, ótimo={inst.optimum})")

    all_stats = []
    for base_cfg in variants:
        cfg = scale_config(base_cfg, inst.n)
        print(f"\n  Variante {cfg.label}  "
              f"(pop={cfg.pop_size}, gen={cfg.generations}, "
              f"cx={cfg.crossover}, mut={cfg.mutation}, ls={cfg.local_search})")

        # calibra uma estimativa de tempo e pergunta se quer rodar esta variante
        print("    calibrando estimativa de tempo...", flush=True)
        est = estimate_runtime_seconds(D, cfg, n_runs=args.runs)
        print(f"    tempo estimado para {args.runs} repetições: "
              f"{format_duration(est)}")
        if not ask_yes_no(f"    Rodar a variante {cfg.label}?"):
            print(f"    (pulando {cfg.label})")
            continue

        t0 = time.time()
        stats = run_variant(inst, D, cfg, n_runs=args.runs, base_seed=1000)
        dt = time.time() - t0
        print(f"  -> {cfg.label}: {stats.formatted()}  "
              f"(melhor={stats.best:.0f}, tempo real={format_duration(dt)})")

        plot_convergence(stats, args.out)
        plot_route(stats, inst, args.out)
        all_stats.append(stats)

    if not all_stats:
        print("\nNenhuma variante foi executada. Encerrado.")
        return

    print_summary_table(all_stats)
    compare_against_optimum(all_stats)
    csv_path = save_summary_csv(all_stats, args.out)
    print(f"Resumo salvo em: {csv_path}")
    print(f"Gráficos salvos em: {args.out}/")


if __name__ == "__main__":
    main()
