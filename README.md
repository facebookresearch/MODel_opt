# MODel

[![CircleCI](https://img.shields.io/circleci/build/github/facebookresearch/MODel_opt?label=CircleCI)](https://app.circleci.com/pipelines/github/facebookresearch/MODel_opt)

*MODel* (Memory Optimizations for Deep Learning)[^1] [Accepted in ICML 2023] enables training larger deep neural networks on existing hardware. It accomplishes this with a few techniques:

- *Operator order optimization* — reodering tensor operators to reduce peak memory usage
- *Fragmentation reduction* — dynamic memory profiling and scheduling to better-utilize memory.

Our approach is described in detail on the [MODel arXiv paper](https://arxiv.org/abs/2210.12924). See [citing](#citation) below to attribute the work.

## Quickstart

Installing MODel in your Python environment is simple:

```bash
git clone https://github.com/facebookresearch/MODel_opt
pip install . [--extra-index-url <url>]
```

**Note**:

- The above install will attempt to install `torch`, `torchaudio`, `torchvision`, and `torchtext` based on default distributions. To install for your CUDA version/OS, see the [PyTorch Getting Started](https://pytorch.org/get-started/locally/) documentation, appending the `--extra-index-url` flag and value to the above install command as needed.
- MODel is tested with Gurobi 9.5.2; use your own license or version as needed.

### Benchmarks

To run benchmarks:

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

If you use MODel, please use the below BibTex for citing:

```text
@article{steiner2023model,
  title={MODel: Memory Optimizations for Deep Learning},
  author={Steiner, Benoit and Elhoushi, Mostafa and Kahn, Jacob, and Hegarty, James},
  journal={Accepted in International Conference on Machine Learning, ICML 2023}
  doi={10.48550/arXiv.2210.12924},
  year={2023},
}
```

[^1]: The work of this repo and associated paper was previously named *OLLA* (Optimizing the Lifetime and Location of Arrays)