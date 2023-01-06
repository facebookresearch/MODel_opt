# OLLA

[![CircleCI](https://img.shields.io/circleci/build/github/facebookresearch/OLLA?label=CircleCI)](https://app.circleci.com/pipelines/github/facebookresearch/OLLA)

*OLLA* (Optimizing the Lifetime and Location of Arrays) enables training larger deep neural networks on existing hardware. It accomplishes this with a few techniques:
- *Operator order optimization* — reodering tensor operators to reduce peak memory usage
- *Fragmentation reduction* — dynamic memory profiling and scheduling to better-utilize memory.

Our approach is described in detail on the [OLLA arXiv paper](https://arxiv.org/abs/2210.12924). See [citing](#citation) below to attribute the work.

## Getting Started
The following steps are required to run the optimizer:
```
conda create --name olla_1.11 python=3.8
conda activate olla_1.11
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

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
