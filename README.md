# MODeL

[![CircleCI](https://img.shields.io/circleci/build/github/facebookresearch/MODel_opt?label=CircleCI)](https://app.circleci.com/pipelines/github/facebookresearch/MODel_opt)

*MODeL* (Memory Optimizations for Deep Learning)[^1] [Accepted in ICML 2023] enables training larger deep neural networks on existing hardware. It accomplishes this with a few techniques:

- *Operator order optimization* — reodering tensor operators to reduce peak memory usage
- *Fragmentation reduction* — dynamic memory profiling and scheduling to better-utilize memory.

Our approach is described in detail on the [MODeL arXiv paper](https://arxiv.org/abs/2210.12924). See [citing](#citation) below to attribute the work.

## Quickstart
### Setup
Installing MODeL in your Python environment is simple:

```bash
git clone https://github.com/facebookresearch/MODeL_opt
cd MODEL_opt
conda create --name model_opt python=3.10
conda activate model_opt
pip install . gurobipy==9.5.2
```

**Notes**:

- Regarding PyTorch dependencies:
  - The above `pip install` command will attempt to install `torch`, `torchaudio`, `torchvision`, and `torchtext` based on default distributions. To install for your CUDA version/OS, see the [PyTorch Getting Started](https://pytorch.org/get-started/locally/) documentation, appending the `--extra-index-url` flag and value to the above install command as needed: `pip install . gurobipy==9.5.2 --extra-index-url <url>`
- Regarding Gurobi dependency:
  - MODeL is tested with Gurobi 9.5.2; use your own license or version as needed: `pip install . gurobipy==<your Gurobi version here>`. 
  - Different Girobi versions are slightly different numerically. Hence using a Gurobu version different from what we are using in this repo could lead to different solutions in our benchmarks and test cases, although those different solutions should have the same objective value.
  - You can view the steps below to setup Gurobi's license:

<details>
<summary><b>Steps to Setup Gurobi License</b></summary>

1. Make an academic account with Gurobi at: https://pages.gurobi.com/registration
2. Request an acadmic license at: https://www.gurobi.com/downloads/end-user-license-agreement-academic/
3. Install the license by running the `grbgetkey` command at the end of the page. If you save the license to a non-default location (outside your home directory), you will need to export the `GRB_LICENSE_FILE` variable with the path to the licence.
4. In your `~/.bashrc` you can setup the following environment variables:
```
# Gurobi
export OLLA_GUROBI_ISV_NAME=...
export OLLA_GUROBI_ISV_APP_NAME=...
export OLLA_GUROBI_ISV_EXPIRATION=...
export OLLA_GUROBI_ISV_CODE=...
```

</details>

### Benchmarks
To run a specific benchmark:
```
python benchmarks.py --model alexnet --batch-size 32 --mode train
```
You could expect an output log like this:
<details>
<summary><b>Command log</b></summary>

```
LOADING MODEL alexnet IN train MODE WITH BATCH SIZE 32
MODEL STATS: #RAW NODES=144, #RAW EDGES=125
MODEL STATS: #ACTUAL OPERATORS=76, #ACTUAL TENSORS=72
  CANONICALIZING MODEL
  CONSTRAINING WEIGHT UPDATES
  CONSTRAINING TENSOR GENERATORS
  CHECKING GRAPH
BENCHMARKING MODEL alexnet IN train MODE WITH BATCH SIZE 32
  SIMULATED PEAK MEM USAGE IS 0.5618 GB
  PERFORM NODE REORDERING
Set parameter OutputFlag to value 1
/private/home/melhoushi/miniconda3/envs/model_opt/lib/python3.10/site-packages/torch/fx/node.py:242: UserWarning: Trying to prepend a node to itself. This behavior has no effect on the graph.
  warnings.warn("Trying to prepend a node to itself. This behavior has no effect on the graph.")
  REORDERED NODES IN 0.1s. SIMULATED PEAK MEMORY USAGE WAS 0.4783 GB (SAVED 14.860221%)
```

</details>

- We support a large number of vision and language models, including: `alexnet`, `mobilenet`, `vit`, `bert`, `vit`, OPT models (from `opt-125m` and `opt-350m` all the way to `opt-66b`).
- To view more options, run `python benchmarks.py --help`

To run all benchmarks and reproduce the results in the paper:
```
bash benchmarks.sh
```

### Running Tests

Run all unit tests with:
```
python -m unittest discover -s tests --pattern "*_test.py"
```

Run tests in a specific file:
```
python -m unittest tests/ilp_solver_test.py
```

Run specific unit test:
```
python -m unittest tests.ilp_solver_test.ILPSolverTest.testSimpleProblem
```

Run unit tests that are skipped with by setting `RUN_SKIPPED=1` in the environment before the command.

## Citation

If you use MODeL, please use the below BibTex for citing:

```text
@article{steiner2023model,
  title={MODeL: Memory Optimizations for Deep Learning},
  author={Steiner, Benoit and Elhoushi, Mostafa and Kahn, Jacob, and Hegarty, James},
  journal={Accepted in International Conference on Machine Learning, ICML 2023}
  doi={10.48550/arXiv.2210.12924},
  year={2023},
}
```

[^1]: The work of this repo and associated paper was previously named *OLLA* (Optimizing the Lifetime and Location of Arrays)
