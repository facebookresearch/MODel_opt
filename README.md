# OLLA

[![CircleCI](https://img.shields.io/circleci/build/github/facebookresearch/OLLA?label=CircleCI)](https://app.circleci.com/pipelines/github/facebookresearch/OLLA)

*OLLA* (Optimizing the Lifetime and Location of Arrays) enables training larger deep neural networks on existing hardware. It accomplishes this with a few techniques:
- *Operator order optimization* — reodering tensor operators to reduce peak memory usage
- *Fragmentation reduction* — dynamic memory profiling and scheduling to better-utilize memory.

Our approach is described in detail on the [OLLA arXiv paper](https://arxiv.org/abs/2210.12924). See [citing](#citation) below to attribute the work.

## Getting Started
The following steps are required to run the optimizer:
```
pip install networkx graphviz intervaltree pandas pydot
```
[Gurobi](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-) is currently required as the default ILP solver. OLLA has been tested with Gurobi 9.1.1:
```
conda install gurobi=9.1.1 -c gurobi
```
but other Gurobi versions are likely to work.

[Graphviz](https://graphviz.org/download/) may also be required for generating/outputting some traced Tensor operation graphs.

### Using with PyTorch
For use with PyTorch, install PyTorch >= 1.12 (`functorch` must be included or installed separately if not present). The following example with CUDA 11.3:
```
pip install torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/nightly/cu113
```

### Benchmarks
To run benchmarks, `torchtext` is required:
```
pip install torchtext
```
Run benchmarks:
```
python benchmarks.py
```

### Running Tests
Run all unit tests with:
```
python -m unittest discover -s tests --pattern "*_test.py"
```

Run unit tests that are skipped with by setting `RUN_SKIPPED=1` in the environment before the command.

## Citation

If you use OLLA, please use the below BibTex for citing:
```text
@article{steiner2022olla,
  title={OLLA: Optimizing the Lifetime and Location of Arrays to Reduce the Memory Usage of Neural Networks},
  author={Steiner, Benoit and Elhoushi, Mostafa and Kahn, Jacob, and Hegarty, James},
  doi = {10.48550/arXiv.2210.12924},
  year={2022},
}
```
