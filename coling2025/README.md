# Code to Reproduce Experiments from the COLING 2025 Paper

- Ustalov, D. [Reliable, Reproducible, and Really Fast Leaderboards with Evalica](https://arxiv.org/abs/2412.11314). 2024. arXiv: [2412.11314 [cs.CL]](https://arxiv.org/abs/2412.11314).

## Prerequisites

- [`requirements.txt`](requirements.txt)
- Chatbot Arena's Dump (August 2024): <https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json>
- LLMFAO Dataset: <https://raw.githubusercontent.com/dustalov/llmfao/refs/heads/master/crowd-comparisons.csv>

## Table 1: [chatbot_arena.csv](chatbot_arena.csv)

```shell
python3 -m chatbot_arena
```

## Table 2: [rust_python.csv](rust_python.csv)

```shell
python3 -m rust_python
```

## Table 3: [scale.csv](scale.csv)

```shell
python3 -m scale_data
python3 -m scale_compute
```
