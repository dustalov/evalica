# Code to Reproduce Experiments from the COLING 2025 Paper

- Ustalov, D. [Reliable, Reproducible, and Really Fast Leaderboards with Evalica](https://aclanthology.org/2025.coling-demos.6). 2025. Proceedings of the 31st International Conference on Computational Linguistics: System Demonstrations. 46&ndash;53. arXiv: [2412.11314 [cs.CL]](https://arxiv.org/abs/2412.11314).

## Prerequisites

- Chatbot Arena's Dump (August 2024): <https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json>
- LLMFAO Dataset: <https://raw.githubusercontent.com/dustalov/llmfao/refs/heads/master/crowd-comparisons.csv> &rarr; `llmfao.csv`

### `requirements.txt`

```
evalica==0.3.2
numpy==2.2.0
pandas==2.2.3
pyarrow==18.1.0
scikit-learn==1.6.0
tqdm==4.67.1
```

## Table 1: [chatbot_arena.csv](chatbot_arena.csv)

```shell
python3 -m chatbot_arena
```

## Table 2: [rust_python.csv](rust_python.csv)

```shell
python3 -m rust_python
```

## Figure 3: [scale.csv](scale.csv)

```shell
python3 -m scale_data
python3 -m scale_compute
```
