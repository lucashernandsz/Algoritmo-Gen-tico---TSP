"""
Leitura de instâncias TSPLIB'95 e cálculo da matriz de distâncias.

A TSPLIB define o campo EDGE_WEIGHT_TYPE em cada arquivo .tsp, que diz COMO
a distância entre duas cidades deve ser calculada. As cinco instâncias do
trabalho usam três tipos diferentes:

    instância   EDGE_WEIGHT_TYPE   forma de cálculo
    --------------------------------------------------
    burma14     GEO                distância geográfica (lat/long, km)
    att48       ATT                distância "pseudo-euclidiana" (Att)
    eil101      EUC_2D             distância euclidiana 2D arredondada
    kroA200     EUC_2D             distância euclidiana 2D arredondada
    u1432       EUC_2D             distância euclidiana 2D arredondada

As fórmulas seguem exatamente o documento oficial (Reinelt, TSPLIB95
Documentation, seção "The distance functions"). Usar a fórmula errada faz o
custo da rota não bater com o ótimo publicado, então isso é crucial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Ótimos globais publicados na TSPLIB'95 (usados só para comparação no relatório).
KNOWN_OPTIMA = {
    "burma14": 3323,
    "att48": 10628,
    "eil101": 629,
    "kroA200": 29368,
    "u1432": 152970,
}


@dataclass
class TSPInstance:
    """Guarda os dados crus de uma instância já parseada."""
    name: str
    edge_weight_type: str
    coords: np.ndarray          # shape (n, 2): coluna 0 = x, coluna 1 = y
    n: int

    @property
    def optimum(self) -> int | None:
        return KNOWN_OPTIMA.get(self.name)


# --------------------------------------------------------------------------- #
# Parsing do arquivo .tsp
# --------------------------------------------------------------------------- #
def read_tsp(path: str) -> TSPInstance:
    name = ""
    ewt = ""
    dimension = 0
    coords: list[tuple[float, float]] = []
    reading_coords = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "EOF":
                continue

            if reading_coords:
                parts = line.split()
                # linha de coordenada: "id x y"  (id é descartado, usamos a ordem)
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key, value = key.strip().upper(), value.strip()
                if key == "NAME":
                    name = value
                elif key == "EDGE_WEIGHT_TYPE":
                    ewt = value.upper()
                elif key == "DIMENSION":
                    dimension = int(value)
            elif line.upper().startswith("NODE_COORD_SECTION"):
                reading_coords = True

    coords_arr = np.asarray(coords, dtype=float)
    n = dimension or len(coords_arr)
    # nome curto (ex.: "burma14") a partir do NAME ou do caminho
    short = (name or path).split("/")[-1].split("\\")[-1].replace(".tsp", "")
    return TSPInstance(name=short, edge_weight_type=ewt, coords=coords_arr, n=n)


# --------------------------------------------------------------------------- #
# Funções de distância (uma para cada EDGE_WEIGHT_TYPE)
# --------------------------------------------------------------------------- #
def _nint(x: np.ndarray) -> np.ndarray:
    """Arredondamento para o inteiro mais próximo (nint), como na TSPLIB."""
    return np.floor(x + 0.5)


def _dist_euc_2d(coords: np.ndarray) -> np.ndarray:
    """EUC_2D: d = nint( sqrt(dx² + dy²) )."""
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    return _nint(np.sqrt(dx * dx + dy * dy))


def _dist_att(coords: np.ndarray) -> np.ndarray:
    """
    ATT (pseudo-euclidiana): r = sqrt((dx² + dy²) / 10); t = nint(r);
    se t < r então d = t + 1, senão d = t.
    """
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    r = np.sqrt((dx * dx + dy * dy) / 10.0)
    t = _nint(r)
    return np.where(t < r, t + 1.0, t)


# A TSPLIB define PI com esse valor truncado (não math.pi) na fórmula GEO.
_TSPLIB_PI = 3.141592


def _to_geo_radians(coord_1d: np.ndarray) -> np.ndarray:
    """
    Converte as coordenadas (que estão em graus.minutos, formato DDD.MM) para
    radianos, exatamente como a TSPLIB manda.
    """
    deg = np.trunc(coord_1d)                       # parte inteira = graus
    minutes = coord_1d - deg                        # parte decimal = minutos
    return _TSPLIB_PI * (deg + 5.0 * minutes / 3.0) / 180.0


def _dist_geo(coords: np.ndarray) -> np.ndarray:
    """GEO: distância geográfica sobre a Terra (raio RRR = 6378.388 km)."""
    RRR = 6378.388
    lat = _to_geo_radians(coords[:, 0])             # x = latitude
    lon = _to_geo_radians(coords[:, 1])             # y = longitude

    q1 = np.cos(lon[:, None] - lon[None, :])
    q2 = np.cos(lat[:, None] - lat[None, :])
    q3 = np.cos(lat[:, None] + lat[None, :])
    # clip evita NaN por erro numérico no acos quando o argumento passa de ±1
    arg = np.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
    return np.floor(RRR * np.arccos(arg) + 1.0)


def build_distance_matrix(inst: TSPInstance) -> np.ndarray:
    """
    Monta a matriz n×n de distâncias inteiras de acordo com o EDGE_WEIGHT_TYPE
    da instância. A diagonal fica 0.
    """
    ewt = inst.edge_weight_type
    if ewt == "EUC_2D":
        d = _dist_euc_2d(inst.coords)
    elif ewt == "ATT":
        d = _dist_att(inst.coords)
    elif ewt == "GEO":
        d = _dist_geo(inst.coords)
    else:
        raise ValueError(
            f"EDGE_WEIGHT_TYPE '{ewt}' não suportado. "
            "As instâncias do trabalho usam EUC_2D, ATT ou GEO."
        )
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float64)
