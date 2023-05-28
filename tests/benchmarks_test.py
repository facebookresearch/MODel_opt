
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shlex
import subprocess
import unittest
from typing import List

import pandas as pd
import torch


class BenchmarksTest(unittest.TestCase):
    def setUp(self):
        pass

    def run_model_benchmarks(
        self,
        model: str,
        modes: List[str] = None,
        batch_sizes: List[str] = None,
        solver_timeout: int = 120,
        time_profile: bool = True,
        gpu_profile: bool = None,
        warm_up_iters=0,
        profile_iters=1,
        additional_args: str = "",
    ) -> pd.DataFrame:
        # set default values
        if modes is None:
            modes = ["eval", "train"]
        if batch_sizes is None:
            batch_sizes = [1, 32]
        if gpu_profile is None:
            gpu_profile = torch.cuda.is_available()

        # create arg strings
        log_path = f"/tmp/opt4ml_{model}_benchmarks.csv"
        arg_modes = " ".join(modes)
        arg_batch_sizes = " ".join(map(str, batch_sizes))
        arg_time_profile = "--time-profile" if time_profile else ""

        # Run benchmark
        # TODO: Call functions in benchmarks.py instead of calling buck command
        subprocess.check_call(
            shlex.split(
                f"python benchmarks.py --model={model} -b {arg_batch_sizes} --mode {arg_modes} --solver-timeout={solver_timeout} --log-path={log_path} {arg_time_profile} --warm-up-iters={warm_up_iters} --profile-iters={profile_iters} {additional_args}"
            )
        )

        # Load CSV file and make some verifications common to all benchmarks
        df = pd.read_csv(log_path)
        self.verify_log(df, model, modes, batch_sizes, additional_args)

        return df

    def verify_log(
        self,
        df: pd.DataFrame,
        model: str,
        modes: List[str] = None,
        batch_sizes: List[str] = None,
        additional_args: str = "",
    ):
        self.assertEqual(df.shape[0], len(modes) * len(batch_sizes))
        self.assertTrue(all([model_name == model for model_name in df["model"]]))
        if "skip-simulation" not in additional_args:
            self.assertTrue(
                all(
                    [
                        simulated_peak_mem_usage > 0
                        for simulated_peak_mem_usage in df["simulated.peak_mem_usage"]
                    ]
                )
            )
        if "skip-node-ordering" not in additional_args:
            self.assertTrue(
                all(
                    [
                        node_ordering_solver_time > 0
                        for node_ordering_solver_time in df["node_ordering.solver_time"]
                    ]
                )
            )
            self.assertTrue(
                all(
                    [
                        node_ordering_peak_mem_usage > 0
                        for node_ordering_peak_mem_usage in df[
                            "node_ordering.peak_mem_usage"
                        ]
                    ]
                )
            )
        if "verify-node-ordering" in additional_args:
            self.assertTrue(
                all(
                    [
                        node_ordering_verification == "SUCCESS"
                        for node_ordering_verification in df[
                            "node_ordering.verification"
                        ]
                    ]
                )
            )

    def testAlexNetBenchmarks(self):
        self.run_model_benchmarks(
            "alexnet", additional_args="--verify-node-ordering --generate-addresses"
        )

    def testBertBenchmarks(self):
        # run each mode/batch size separately to avoid OOM error
        self.run_model_benchmarks("bert", batch_sizes=[1], modes=["eval"], additional_args="--generate-addresses")
        self.run_model_benchmarks("bert", batch_sizes=[32], modes=["eval"], additional_args="--generate-addresses")
        self.run_model_benchmarks("bert", batch_sizes=[1], modes=["train"], additional_args="--generate-addresses")
        self.run_model_benchmarks("bert", batch_sizes=[32], modes=["train"], additional_args="--generate-addresses")


    @unittest.skip("fx can't trace the model")
    def testConformerBenchmarks(self):
        self.run_model_benchmarks("conformer", additional_args="--generate-addresses")

    def testDeepLabBenchmarks(self):
        self.run_model_benchmarks(
            "deeplab", modes=["eval"], additional_args="--generate-addresses"
        )

    def testEfficientNetBenchmarks(self):
        # run each mode/batch size separately to avoid OOM error
        self.run_model_benchmarks("efficientnet", batch_sizes=[1], modes=["eval"], additional_args="--generate-addresses")
        self.run_model_benchmarks("efficientnet", batch_sizes=[32], modes=["eval"], additional_args="--generate-addresses")
        self.run_model_benchmarks("efficientnet", batch_sizes=[1], modes=["train"], additional_args="--generate-addresses")
        self.run_model_benchmarks("efficientnet", batch_sizes=[32], modes=["train"], additional_args="--generate-addresses")

    @unittest.skip("fx can't trace the model")
    def testEmformerBenchmarks(self):
        self.run_model_benchmarks("emformer", additional_args="--generate-addresses")

    def testGoogleNetBenchmarks(self):
        self.run_model_benchmarks("googlenet", additional_args="--generate-addresses")

    def testInceptionBenchmarks(self):
        self.run_model_benchmarks("inception", additional_args="--generate-addresses")

    def testMNASNetBenchmarks(self):
        self.run_model_benchmarks("mnasnet", additional_args="--generate-addresses")

    def testMobileNetBenchmarks(self):
        self.run_model_benchmarks("mobilenet", additional_args="--generate-addresses")

    @unittest.skip("fx can't trace the model")
    def testRaftBenchmarks(self):
        self.run_model_benchmarks("raft", additional_args="--generate-addresses")

    def testResNet18Benchmarks(self):
        self.run_model_benchmarks("resnet", additional_args="--generate-addresses")

    def testResNet50Benchmarks(self):
        self.run_model_benchmarks("resnet50", additional_args="--generate-addresses")

    def testResNet3DBenchmarks(self):
        self.run_model_benchmarks("resnet3d", additional_args="--generate-addresses")

    def testSqueezeNetBenchmarks(self):
        self.run_model_benchmarks("squeezenet", additional_args="--generate-addresses")

    @unittest.skip("FIXME")
    def testSSDBenchmarks(self):
        self.run_model_benchmarks(
            "ssd", modes=["eval"], additional_args="--generate-addresses"
        )

    @unittest.skip("FIXME")
    def testSwinBenchmarks(self):
        self.run_model_benchmarks(
            "swin", modes=["eval"], additional_args="--generate-addresses"
        )

    def testTranformerBenchmarks(self):
        self.run_model_benchmarks("transformer", batch_sizes=[1], modes=["eval"], additional_args="--generate-addresses")
        self.run_model_benchmarks("transformer", batch_sizes=[32], modes=["eval"], additional_args="--generate-addresses")
        self.run_model_benchmarks("transformer", batch_sizes=[1], modes=["train"], additional_args="--generate-addresses")
        self.run_model_benchmarks("transformer", batch_sizes=[32], modes=["train"], additional_args="--generate-addresses")

    def testTranformerDFLTBenchmarks(self):
        self.run_model_benchmarks("transformer_dflt", batch_sizes=[1], modes=["eval"], additional_args="--generate-addresses")
        self.run_model_benchmarks("transformer_dflt", batch_sizes=[32], modes=["eval"], additional_args="--generate-addresses")
        self.run_model_benchmarks("transformer_dflt", batch_sizes=[1], modes=["train"], additional_args="--generate-addresses")
        self.run_model_benchmarks("transformer_dflt", batch_sizes=[32], modes=["train"], additional_args="--generate-addresses")

    def testVGGBenchmarks(self):
        self.run_model_benchmarks("vgg", additional_args="--generate-addresses")

    def testVGG16Benchmarks(self):
        self.run_model_benchmarks("vgg16", additional_args="--generate-addresses")

    def testVGG19Benchmarks(self):
        self.run_model_benchmarks("vgg19", additional_args="--generate-addresses")

    def testVITBenchmarks(self):
        self.run_model_benchmarks("vit", additional_args="--generate-addresses")

    @unittest.skip("FIXME")
    def testXLMRBenchmarks(self):
        self.run_model_benchmarks("xlmr", additional_args="--generate-addresses")

    def testOPT350MBenchmarks(self):
        self.run_model_benchmarks("opt-350m", batch_sizes=[1], modes=["eval"], additional_args="--generate-addresses")
        self.run_model_benchmarks("opt-350m", batch_sizes=[1], modes=["train"], additional_args="--generate-addresses")
