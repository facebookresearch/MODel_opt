# OLLA

OLLA (Optimizing the Lifetime and Location of Arrays) makes it possible to train larger deep neural networks on existing hardware. OLLA optimizes the order in which the neural network operators are executed to minimize peak memory usage. Furthermore OLLA eliminates memory fragmentation to ensure that no memory is wasted.

## Approach

Our approach is described in detail on the [OLLA arXiv paper](https://arxiv.org/abs/2210.12924)

## Getting Started

Install
```
conda create --name olla python=3.9
conda activate olla
conda install pytorch torchvision torchaudio torchtext -c pytorch
pip install pandas intervaltree networkx graphviz
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
pip install pydot
```

Other tools to install:
```
# MacOS
brew install graphviz

# Linux
# TBD
```

Run benchmarks:
```
python benchmarks.py
```

Run tests:
```
python -m unittest tests/*
```

Expected Status of each test:
```
python -m unittest tests/dataflow_graph_test.py # PASS
python -m unittest tests/torch_graph_importer_test.py # PASS
python -m unittest tests/torch_graph_importer_test_vision.py # PASS
python -m unittest tests/torch_graph_importer_test_transformers.py # PASS
python -m unittest tests/simulator_test.py # PASS
python -m unittest tests/utils_test.py # PASS
python -m unittest tests/max_cut_test.py # PASS
python -m unittest tests/fx_profiler_test.py # PASS
python -m unittest tests/fx_optimizer_test.py # PASS
python -m unittest tests/torch_scheduler_test.py # PASS
python -m unittest tests/memory_planner_test.py # 1 tests fail
python -m unittest tests/spill_profiler_test.py # PASS
python -m unittest tests/training_graph_optimizer_test.py # PASS
python -m unittest tests/ilp_solver_test.py # PASS
python -m unittest tests/training_graph_optimizer_large_test.py # 1 test fails
python -m unittest tests/scheduler_test.py $ 4 out of 8 tests fails
python -m unittest tests/defragmenter_test.py # 1 out of 4 tests fails
```

## Citation

If you use OLLA, please cite us with:

```text
@article{steiner2022olla,
  title={OLLA: Optimizing the Lifetime and Location of Arrays to Reduce the Memory Usage of Neural Networks},
  author={Steiner, Benoit and Elhoushi, Mostafa and Kahn, Jacob, and Hegarty, James},
  doi = {10.48550/arXiv.2210.12924},
  year={2022},
}
```
