"""
Orquestração dos experimentos exigidos pelo trabalho:

Para cada instância e cada variante do GA:
  1. roda 10 repetições com a MESMA configuração mas sementes diferentes;
  2. guarda a melhor solução (rota e custo) de cada repetição;
  3. calcula custo médio ± desvio-padrão sobre as 10 repetições;
  4. compara com o ótimo global conhecido (gap %);
  5. gera os gráficos de convergência e o gráfico 2D da melhor rota.

Tudo é salvo na pasta de saída (results/ por padrão).
"""

from __future__ import annotations

import copy
import csv
import os
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")            # backend sem janela (só salva arquivos)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .ga import GAConfig, GeneticAlgorithm, GAResult
from .tsplib import TSPInstance, build_distance_matrix


@dataclass
class VariantStats:
    instance: str
    variant: str
    costs: list[float]          # 10 melhores custos (um por repetição)
    mean: float
    std: float
    best: float
    optimum: int | None
    best_route: np.ndarray
    best_history: GAResult      # histórico da repetição que achou a melhor rota

    @property
    def gap_pct(self) -> float | None:
        """Gap percentual do custo médio em relação ao ótimo."""
        if self.optimum:
            return 100.0 * (self.mean - self.optimum) / self.optimum
        return None

    def formatted(self) -> str:
        """Formato pedido no enunciado: custo_médio ± desvio_padrão."""
        return f"{self.mean:.2f} ± {self.std:.2f}"


def estimate_runtime_seconds(
    D: np.ndarray,
    cfg: GAConfig,
    n_runs: int = 10,
    probe_gens: int = 2,
) -> float:
    """
    Estima quanto tempo a variante deve levar (em segundos) para as `n_runs`
    repetições. Faz isso rodando uma "sonda" curta (poucas gerações), mede o
    tempo por geração e extrapola para o nº real de gerações e repetições.

    É uma estimativa aproximada (a máquina e a aleatoriedade variam), mas dá
    uma boa ideia de ordem de grandeza antes de você decidir rodar.
    """
    probe = copy.copy(cfg)
    probe.generations = max(1, min(probe_gens, cfg.generations))

    ga = GeneticAlgorithm(D, probe)
    t0 = time.time()
    ga.run(seed=1000)
    elapsed = time.time() - t0

    # o run faz (probe.generations + 1) avaliações de geração (a inicial + cada geração)
    per_gen = elapsed / (probe.generations + 1)
    per_run = per_gen * (cfg.generations + 1)
    return per_run * n_runs


def format_duration(seconds: float) -> str:
    """Formata segundos em algo legível: '12 s', '~3 min', '~1 h 5 min'."""
    if seconds < 90:
        return f"~{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"~{minutes:.0f} min"
    hours = int(minutes // 60)
    rem = int(minutes % 60)
    return f"~{hours} h {rem} min"


def run_variant(
    inst: TSPInstance,
    D: np.ndarray,
    cfg: GAConfig,
    n_runs: int = 10,
    base_seed: int = 1000,
) -> VariantStats:
    
    """Roda as 10 repetições (sementes base_seed+0 .. base_seed+9) de uma variante."""
    ga = GeneticAlgorithm(D, cfg)
    costs: list[float] = []
    best_overall_cost = float("inf")
    best_overall_route = None
    best_overall_result = None

    for run in range(n_runs):
        seed = base_seed + run
        res = ga.run(seed)
        costs.append(res.best_cost)
        if res.best_cost < best_overall_cost:
            best_overall_cost = res.best_cost
            best_overall_route = res.best_route
            best_overall_result = res
        print(f"    [{cfg.label}] run {run + 1:2d}/{n_runs}  "
              f"seed={seed}  custo={res.best_cost:.2f}")

    costs_arr = np.array(costs)
    return VariantStats(
        instance=inst.name,
        variant=cfg.label,
        costs=costs,
        mean=float(costs_arr.mean()),
        std=float(costs_arr.std(ddof=1)),     # desvio amostral (n-1)
        best=float(costs_arr.min()),
        optimum=inst.optimum,
        best_route=best_overall_route,
        best_history=best_overall_result,
    )


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def plot_convergence(stats: VariantStats, out_dir: str) -> str:
    """Gráfico da evolução do melhor custo e do custo médio por geração."""
    h = stats.best_history
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(h.history_best, label="Melhor custo", linewidth=2)
    ax.plot(h.history_mean, label="Custo médio da população",
            linewidth=1, alpha=0.7)
    if stats.optimum:
        ax.axhline(stats.optimum, color="red", linestyle="--",
                   label=f"Ótimo ({stats.optimum})")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Custo da rota")
    ax.set_title(f"Convergência — {stats.instance} — {stats.variant}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"conv_{stats.instance}_{stats.variant}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_route(stats: VariantStats, inst: TSPInstance, out_dir: str) -> str:
    """
    Gráfico 2D da melhor rota encontrada, desenhado com NetworkX
    (conforme sugerido no enunciado).
    """
    route = stats.best_route
    coords = inst.coords
    pos = {i: (coords[i, 0], coords[i, 1]) for i in range(inst.n)}

    G = nx.DiGraph()
    G.add_nodes_from(range(inst.n))
    edges = [(int(route[i]), int(route[(i + 1) % inst.n])) for i in range(inst.n)]
    G.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=(7, 6))
    node_size = 60 if inst.n <= 200 else 8
    nx.draw_networkx_nodes(G, pos, node_size=node_size,
                           node_color="#1f77b4", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.0,
                           edge_color="#444", arrows=False, ax=ax)
    if inst.n <= 60:
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    title = f"Melhor rota — {stats.instance} — {stats.variant}\ncusto = {stats.best:.0f}"
    if stats.optimum:
        title += f"  (ótimo = {stats.optimum})"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(out_dir, f"route_{stats.instance}_{stats.variant}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Saída tabular
# --------------------------------------------------------------------------- #
def save_summary_csv(all_stats: list[VariantStats], out_dir: str) -> str:
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instancia", "variante", "media", "desvio_padrao",
                    "melhor", "otimo", "gap_medio_%", "custo_medio_fmt"])
        for s in all_stats:
            w.writerow([
                s.instance, s.variant, f"{s.mean:.2f}", f"{s.std:.2f}",
                f"{s.best:.2f}", s.optimum if s.optimum else "",
                f"{s.gap_pct:.2f}" if s.gap_pct is not None else "",
                s.formatted(),
            ])
    return path


def print_summary_table(all_stats: list[VariantStats]) -> None:
    print("\n" + "=" * 78)
    print(f"{'Instância':<10}{'Variante':<22}{'Custo médio ± dp':<24}"
          f"{'Melhor':<10}{'Gap%':<8}")
    print("-" * 78)
    for s in all_stats:
        gap = f"{s.gap_pct:.1f}" if s.gap_pct is not None else "-"
        print(f"{s.instance:<10}{s.variant:<22}{s.formatted():<24}"
              f"{s.best:<10.0f}{gap:<8}")
    print("=" * 78 + "\n")
