"""
Algoritmo Genético (GA) para o Problema do Caixeiro Viajante (TSP).

Representação
-------------
Cada indivíduo (cromossomo) é uma PERMUTAÇÃO dos índices das cidades
[0, 1, ..., n-1]. A ordem dos genes é a ordem de visita; a rota é fechada
(volta da última cidade para a primeira). 

Fitness
-------
Queremos MINIMIZAR o custo da rota (soma das distâncias do ciclo). O "fitness"
aqui é simplesmente o custo: quanto menor, melhor.

Operadores (todos parametrizáveis, para permitir explorar variações)
--------------------------------------------------------------------
- Inicialização: aleatória, ou semeada com vizinho-mais-próximo (NN).
- Seleção:       torneio de tamanho k.
- Crossover:     OX (Order Crossover) ou PMX (Partially Mapped Crossover).
- Mutação:       swap (troca duas cidades) ou inversion (inverte um segmento,
                 equivalente a um movimento 2-opt).
- Elitismo:      mantém os melhores indivíduos intactos a cada geração.
- Busca local:   2-opt opcional aplicado aos filhos (GA "memético").

A classe GeneticAlgorithm encapsula tudo e o método .run() devolve a melhor
rota, seu custo e o histórico de convergência (para os gráficos).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Configuração de uma variante do GA
# --------------------------------------------------------------------------- #
@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 500
    crossover: str = "OX"          # "OX" ou "PMX"
    mutation: str = "inversion"    # "swap" ou "inversion"
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2     # prob. de mutar cada filho
    tournament_size: int = 3
    elitism: int = 2               # nº de elites preservados por geração
    init: str = "random"           # "random" ou "nearest_neighbor"
    local_search: bool = False     # aplicar 2-opt nos filhos? (memético)
    ls_max_passes: int = 1         # passes de 2-opt por aplicação
    label: str = "GA"              # nome da variante (aparece nos resultados)


@dataclass
class GAResult:
    best_route: np.ndarray
    best_cost: float
    history_best: list[float] = field(default_factory=list)   # melhor custo por geração
    history_mean: list[float] = field(default_factory=list)   # custo médio da população por geração


# --------------------------------------------------------------------------- #
# Funções utilitárias de custo
# --------------------------------------------------------------------------- #
def route_cost(route: np.ndarray, D: np.ndarray) -> float:
    """Custo de um ciclo fechado: soma das arestas + retorno ao início."""
    return float(D[route, np.roll(route, -1)].sum())


def population_costs(pop: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Custo de toda a população de uma vez (vetorizado)."""
    nxt = np.roll(pop, -1, axis=1)
    return D[pop, nxt].sum(axis=1)


# --------------------------------------------------------------------------- #
# Inicialização
# --------------------------------------------------------------------------- #
def _nearest_neighbor_route(D: np.ndarray, start: int, rng: np.random.Generator) -> np.ndarray:
    """Constrói uma rota gulosa: sempre vai para a cidade não visitada mais próxima."""
    n = D.shape[0]
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda c: D[current, c])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return np.array(route, dtype=np.int64)


def _init_population(cfg: GAConfig, D: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = D.shape[0]
    pop = np.empty((cfg.pop_size, n), dtype=np.int64)
    for i in range(cfg.pop_size):
        if cfg.init == "nearest_neighbor" and i == 0:
            # apenas um indivíduo guloso, para dar um "empurrão" inicial;
            # o resto aleatório preserva a diversidade da população.
            pop[i] = _nearest_neighbor_route(D, rng.integers(n), rng)
        else:
            pop[i] = rng.permutation(n)
    return pop


# --------------------------------------------------------------------------- #
# Seleção (torneio)
# --------------------------------------------------------------------------- #
def _tournament(costs: np.ndarray, k: int, rng: np.random.Generator) -> int:
    aspirants = rng.integers(0, len(costs), size=k)
    return int(aspirants[np.argmin(costs[aspirants])])


# --------------------------------------------------------------------------- #
# Crossovers
# --------------------------------------------------------------------------- #
def _order_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    OX (Order Crossover): copia um segmento de p1 para o filho e preenche o
    resto com a ordem das cidades de p2, pulando as já presentes.
    """
    n = len(p1)
    a, b = sorted(rng.integers(0, n, size=2))
    child = -np.ones(n, dtype=np.int64)
    child[a:b + 1] = p1[a:b + 1]
    taken = set(p1[a:b + 1].tolist())

    fill = [c for c in p2 if c not in taken]   # ordem relativa de p2
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child


def _pmx_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    PMX (Partially Mapped Crossover): copia um segmento de p1 e resolve os
    conflitos restantes seguindo o mapeamento p1<->p2 dentro do segmento.
    """
    n = len(p1)
    a, b = sorted(rng.integers(0, n, size=2))
    child = -np.ones(n, dtype=np.int64)
    child[a:b + 1] = p1[a:b + 1]
    mapping = {p1[i]: p2[i] for i in range(a, b + 1)}

    for i in range(n):
        if a <= i <= b:
            continue
        gene = p2[i]
        # se o gene de p2 já está no segmento copiado, segue o mapeamento
        while gene in mapping:
            gene = mapping[gene]
        child[i] = gene
    return child


def _crossover(cfg: GAConfig, p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if cfg.crossover == "PMX":
        return _pmx_crossover(p1, p2, rng)
    return _order_crossover(p1, p2, rng)


# --------------------------------------------------------------------------- #
# Mutações
# --------------------------------------------------------------------------- #
def _mutate_swap(route: np.ndarray, rng: np.random.Generator) -> None:
    i, j = rng.integers(0, len(route), size=2)
    route[i], route[j] = route[j], route[i]


def _mutate_inversion(route: np.ndarray, rng: np.random.Generator) -> None:
    """Inverte um segmento aleatório (mutação 2-opt)."""
    i, j = sorted(rng.integers(0, len(route), size=2))
    route[i:j + 1] = route[i:j + 1][::-1]


def _mutate(cfg: GAConfig, route: np.ndarray, rng: np.random.Generator) -> None:
    if cfg.mutation == "swap":
        _mutate_swap(route, rng)
    else:
        _mutate_inversion(route, rng)


# --------------------------------------------------------------------------- #
# Busca local: 2-opt (para o GA memético)
# --------------------------------------------------------------------------- #
def two_opt(route: np.ndarray, D: np.ndarray, max_passes: int = 1) -> np.ndarray:
    """
    Melhora a rota com 2-opt (first-improvement): tenta inverter segmentos e
    aceita a inversão sempre que ela reduz o custo. Para após `max_passes`
    varreduras completas ou quando nenhuma melhoria é encontrada.

    Obs.: 2-opt é O(n²) por passe; para instâncias grandes (u1432) use poucos
    passes ou desligue a busca local, senão fica caro.
    """
    n = len(route)
    r = route.copy()
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a, b = r[i], r[i + 1]
            for j in range(i + 2, n):
                c = r[j]
                d = r[(j + 1) % n]
                if d == a:
                    continue
                # ganho de trocar arestas (a,b)+(c,d) por (a,c)+(b,d)
                delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                if delta < -1e-9:
                    r[i + 1:j + 1] = r[i + 1:j + 1][::-1]
                    b = r[i + 1]
                    improved = True
        if not improved:
            break
    return r


# --------------------------------------------------------------------------- #
# Algoritmo Genético principal
# --------------------------------------------------------------------------- #
class GeneticAlgorithm:
    def __init__(self, D: np.ndarray, cfg: GAConfig):
        self.D = D
        self.cfg = cfg
        self.n = D.shape[0]

    def run(self, seed: int) -> GAResult:
        cfg, D = self.cfg, self.D
        rng = np.random.default_rng(seed)

        pop = _init_population(cfg, D, rng)
        costs = population_costs(pop, D)

        best_idx = int(np.argmin(costs))
        best_route = pop[best_idx].copy()
        best_cost = float(costs[best_idx])

        hist_best, hist_mean = [best_cost], [float(costs.mean())]

        for _ in range(cfg.generations):
            new_pop = np.empty_like(pop)

            # ---- Elitismo: copia os melhores diretamente ----
            if cfg.elitism > 0:
                elite_idx = np.argsort(costs)[:cfg.elitism]
                new_pop[:cfg.elitism] = pop[elite_idx]

            # ---- Gera o restante por seleção + crossover + mutação ----
            for i in range(cfg.elitism, cfg.pop_size):
                p1 = pop[_tournament(costs, cfg.tournament_size, rng)]
                if rng.random() < cfg.crossover_rate:
                    p2 = pop[_tournament(costs, cfg.tournament_size, rng)]
                    child = _crossover(cfg, p1, p2, rng)
                else:
                    child = p1.copy()

                if rng.random() < cfg.mutation_rate:
                    _mutate(cfg, child, rng)

                if cfg.local_search:
                    child = two_opt(child, D, cfg.ls_max_passes)

                new_pop[i] = child

            pop = new_pop
            costs = population_costs(pop, D)

            gen_best_idx = int(np.argmin(costs))
            if costs[gen_best_idx] < best_cost:
                best_cost = float(costs[gen_best_idx])
                best_route = pop[gen_best_idx].copy()

            hist_best.append(best_cost)
            hist_mean.append(float(costs.mean()))

        return GAResult(
            best_route=best_route,
            best_cost=best_cost,
            history_best=hist_best,
            history_mean=hist_mean,
        )
