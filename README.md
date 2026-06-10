# GA para o TSP — SI805 (Computação Bioinspirada)

Algoritmo Genético aplicado ao Problema do Caixeiro Viajante (TSP) sobre cinco
instâncias da TSPLIB'95, seguindo a metodologia experimental do trabalho.

## 1. Instalação

```bash
pip install -r requirements.txt
```

## 2. Baixar as instâncias

Baixe os arquivos `.tsp` da TSPLIB e coloque-os na pasta `data/`:
<https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/>

Arquivos necessários (nomes exatos):

| arquivo        | n     | EDGE_WEIGHT_TYPE | ótimo   |
|----------------|-------|------------------|---------|
| `burma14.tsp`  | 14    | GEO              | 3323    |
| `att48.tsp`    | 48    | ATT              | 10628   |
| `eil101.tsp`   | 101   | EUC_2D           | 629     |
| `kroA200.tsp`  | 200   | EUC_2D           | 29368   |
| `u1432.tsp`    | 1432  | EUC_2D           | 152970  |

(Já vem um `data/burma14.tsp` de exemplo para testar.)

## 3. Rodar

```bash
python main.py                              # todas as instâncias e variantes (10 repetições)
python main.py --instances burma14 att48    # só algumas instâncias
python main.py --variants V2_OX_inv V4_memetic
python main.py --runs 10 --out results
```

## 4. Saídas (pasta `results/`)

- `conv_<inst>_<variante>.png` — convergência (melhor custo e custo médio por geração).
- `route_<inst>_<variante>.png` — gráfico 2D da melhor rota (NetworkX).
- `summary.csv` — custo médio ± desvio-padrão, melhor custo, ótimo e gap %.
- Tabela-resumo também impressa no terminal.

## 5. Estrutura do código

| arquivo               | papel                                                        |
|-----------------------|--------------------------------------------------------------|
| `src/tsplib.py`       | leitura dos `.tsp` e cálculo de distâncias (EUC_2D/ATT/GEO). |
| `src/ga.py`           | núcleo do GA: operadores, 2-opt e o laço evolutivo.          |
| `src/experiment.py`   | 10 repetições, estatísticas e geração dos gráficos.          |
| `main.py`             | configura as variantes, escala parâmetros e orquestra tudo.  |

## 6. Variantes comparadas (edite em `main.py`)

| label         | crossover | mutação    | busca local | inicialização |
|---------------|-----------|------------|-------------|---------------|
| `V1_OX_swap`  | OX        | swap       | não         | aleatória     |
| `V2_OX_inv`   | OX        | inversion  | não         | aleatória     |
| `V3_PMX_inv`  | PMX       | inversion  | não         | aleatória     |
| `V4_memetic`  | OX        | inversion  | 2-opt       | vizinho + ale.|

> **u1432**: a busca local (V4) com 2-opt completo é cara em 1432 cidades; em
> `scale_config` os parâmetros são reduzidos lá. Ajuste conforme o tempo da sua
> máquina (ou rode V4 só nas instâncias menores).
