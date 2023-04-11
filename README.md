# OLLA

[![CircleCI](https://img.shields.io/circleci/build/github/facebookresearch/OLLA?label=CircleCI)](https://app.circleci.com/pipelines/github/facebookresearch/OLLA)

*OLLA* (Optimizing the Lifetime and Location of Arrays) enables training larger deep neural networks on existing hardware. It accomplishes this with a few techniques:
- *Operator order optimization* — reodering tensor operators to reduce peak memory usage
- *Fragmentation reduction* — dynamic memory profiling and scheduling to better-utilize memory.

Our approach is described in detail on the [OLLA arXiv paper](https://arxiv.org/abs/2210.12924). See [citing](#citation) below to attribute the work.

## Quickstart
Installing OLLA in your Python environment is simple:
```bash
git clone https://github.com/facebookresearch/olla
cd olla
conda create --name olla python=3.10
conda activate olla
pip install .
```
**Note**:
- The above install will attempt to install `torch`, `torchaudio`, `torchvision`, and `torchtext` based on default distributions. To install for your CUDA version/OS, see the [PyTorch Getting Started](https://pytorch.org/get-started/locally/) documentation, appending the `--extra-index-url` flag and value to the above install command as needed.
- OLLA is tested with Gurobi 9.1.1; use your own license or version as needed.

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

If you use OLLA, please use the below BibTex for citing:
```text
@article{steiner2022olla,
  title={OLLA: Optimizing the Lifetime and Location of Arrays to Reduce the Memory Usage of Neural Networks},
  author={Steiner, Benoit and Elhoushi, Mostafa and Kahn, Jacob, and Hegarty, James},
  doi = {10.48550/arXiv.2210.12924},
  year={2022},
}
```
