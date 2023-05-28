
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import time
import traceback
from collections import OrderedDict
from multiprocessing import Process

import pandas as pd

import torch
import torch.fx
import torchaudio
import torchtext
import torchvision

from olla import simulator, training_graph_optimizer, utils, visualizer
from olla.torch import fx_profiler, torch_graph_importer
from olla.torch.fx_optimizer import FXOptimizer

# Fix the environment to enable graphviz to work.
# del os.environ["LD_LIBRARY_PATH"]

KB = 2**10
MB = 2**20
GB = 2**30

class Benchmark:
    def load_model(
        self,
        model_name,
        mode,
        batch_size=32,
        device="cpu",
        distributed=False,
        profile=None,
        warm_up_iters=0,
        profile_iters=1,
        render_model=False,
        infer_trace=False,
    ):
        if model_name == "alexnet":
            model = torchvision.models.alexnet()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "conformer":
            input_dim = 80
            model = torchaudio.models.Conformer(
                input_dim=input_dim,
                num_heads=4,
                ffn_dim=128,
                num_layers=4,
                depthwise_conv_kernel_size=31,
            )
            lengths = torch.randint(1, 400, (batch_size,))  # (batch,)
            inp = torch.rand(
                batch_size, int(lengths.max()), input_dim
            )  # (batch, num_frames, input_dim)
            inputs = (inp, lengths)
        elif model_name == "deeplab":
            # Also try deeplabv3_mobilenet_v3_large()
            model = torchvision.models.segmentation.deeplabv3_resnet50()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "efficientnet":
            model = torchvision.models.efficientnet_b0()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "emformer":
            model = torchaudio.models.Emformer(
                512, 8, 2048, 20, 4, right_context_length=1
            )
            inp = torch.rand(batch_size, 400, 512)  # batch, num_frames, feature_dim
            lengths = torch.randint(1, 200, (batch_size,))  # batch
            inputs = (inp, lengths)
        elif model_name == "googlenet":

            class GoogleNetWrapper(torch.nn.Module):
                def __init__(self):
                    super(GoogleNetWrapper, self).__init__()
                    self.model = torchvision.models.googlenet()

                def forward(self, x):
                    rslt = self.model(x)
                    if self.model.training:
                        return torch.concat((rslt[0], rslt[1], rslt[2]), dim=-1)
                    else:
                        return rslt

            model = GoogleNetWrapper()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "inception":

            class InceptionWrapper(torch.nn.Module):
                def __init__(self):
                    super(InceptionWrapper, self).__init__()
                    self.model = torchvision.models.inception_v3()

                def forward(self, x):
                    rslt = self.model(x)
                    if self.model.training:
                        return torch.concat((rslt.logits, rslt.aux_logits), dim=-1)
                    else:
                        return rslt

            model = InceptionWrapper()
            # Need batch size > 1 when training
            min_batch_size = max(batch_size, 2) if mode == "train" else batch_size
            inputs = (torch.randn((min_batch_size, 3, 299, 299)),)
        elif model_name == "mnasnet":
            model = torchvision.models.mnasnet0_5()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "mobilenet":
            model = torchvision.models.mobilenet_v2()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "mobilenetv1":
            import torch.nn as nn
            class MobileNetV1(torch.nn.Module):
                def __init__(self, ch_in, n_classes):
                    super(MobileNetV1, self).__init__()

                    def conv_bn(inp, oup, stride):
                        return nn.Sequential(
                            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                            nn.BatchNorm2d(oup),
                            nn.ReLU(inplace=True)
                            )

                    def conv_dw(inp, oup, stride):
                        return nn.Sequential(
                            # dw
                            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                            nn.BatchNorm2d(inp),
                            nn.ReLU(inplace=True),

                            # pw
                            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(oup),
                            nn.ReLU(inplace=True),
                            )

                    self.model = nn.Sequential(
                        conv_bn(ch_in, 32, 2),
                        conv_dw(32, 64, 1),
                        conv_dw(64, 128, 2),
                        conv_dw(128, 128, 1),
                        conv_dw(128, 256, 2),
                        conv_dw(256, 256, 1),
                        conv_dw(256, 512, 2),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 1024, 2),
                        conv_dw(1024, 1024, 1),
                        nn.AdaptiveAvgPool2d(1)
                    )
                    self.fc = nn.Linear(1024, n_classes)

                def forward(self, x):
                    x = self.model(x)
                    x = x.view(-1, 1024)
                    x = self.fc(x)
                    return x
            model = MobileNetV1(ch_in=3, n_classes=1000)
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "raft":
            model = torchvision.models.optical_flow.raft_small()
            inputs = (
                torch.randn((batch_size, 3, 520, 960)),
                torch.randn((batch_size, 3, 520, 960)),
            )
        elif model_name == "resnet" or model_name == "resnet18":
            model = torchvision.models.resnet18()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "resnet50":
            model = torchvision.models.resnet50()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "resnet101":
            model = torchvision.models.resnet101()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "resnet152":
            model = torchvision.models.resnet152()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "resnet3d":
            model = torchvision.models.video.r3d_18()
            inputs = (torch.randn((batch_size, 3, 1, 112, 112)),)
        elif model_name == "bert":
            bert_base = torchtext.models.ROBERTA_BASE_ENCODER
            model = bert_base.get_model()
            transform = bert_base.transform()
            max_seq_len = 256
            seq_len = args.seq_len if args.seq_len else max_seq_len

            word = "Hello"
            # Repeat text to fill maximum sequence length of model
            sentence = " ".join([word] * seq_len)
            batch_sentences = [sentence] * batch_size
            inputs = (
                torchtext.functional.to_tensor(transform(batch_sentences), padding_value=1),
            )
        elif model_name == "squeezenet":
            model = torchvision.models.squeezenet1_0()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "ssd":
            model = torchvision.models.detection.ssd300_vgg16()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "swin":
            model = torchvision.models.swin_t()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "transformer":
            model = torch.nn.Transformer(
                d_model=512, nhead=1, num_encoder_layers=1, num_decoder_layers=1
            )
            inputs = (
                torch.rand((10, batch_size, 512)),
                torch.rand((20, batch_size, 512)),
            )
        elif model_name == "transformer_dflt":
            model = torch.nn.Transformer()
            inputs = (
                torch.rand((10, batch_size, 512)),
                torch.rand((20, batch_size, 512)),
            )
        elif model_name == "vgg" or model_name == "vgg11":
            model = torchvision.models.vgg11()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "vgg16":
            model = torchvision.models.vgg16()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "vgg19":
            model = torchvision.models.vgg19()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "vit":
            model = torchvision.models.vit_b_16()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "xlmr":
            xlmr_base = torchtext.models.XLMR_BASE_ENCODER
            model = xlmr_base.get_model()
            transform = xlmr_base.transform()
            max_seq_len = 256
            seq_len = args.seq_len if args.seq_len else max_seq_len

            word = "Hello"
            # Repeat text to fill maximum sequence length of model
            sentence = " ".join([word] * seq_len)
            batch_sentences = [sentence] * batch_size

            inputs = (
                torchtext.functional.to_tensor(transform(batch_sentences), padding_value=1),
            )
        elif model_name.startswith("opt"):
            from transformers import AutoTokenizer, OPTModel
            class OPTWrapper(torch.nn.Module):
                def __init__(self):
                    super(OPTWrapper, self).__init__()
                    self.model = OPTModel.from_pretrained(f"facebook/{model_name}")

                def forward(self, input_ids, attention_mask):
                    return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
            max_seq_len = 2047
            seq_len = args.seq_len if args.seq_len else max_seq_len

            word = "Hello"
            # Repeat text to fill maximum sequence length of model
            sentence = " ".join([word] * seq_len)
            batch_sentences = [sentence] * batch_size

            inputs = list(tokenizer(batch_sentences, return_tensors="pt").values())
            model = OPTWrapper()
        elif model_name == "gpt2":
            # TODO: Fix error when loading GPT2
            from transformers import GPT2Tokenizer, GPT2Model
            class GPT2Wrapper(torch.nn.Module):
                def __init__(self):
                    super(GPT2Wrapper, self).__init__()
                    self.model = GPT2Model.from_pretrained('gpt2')

                def forward(self, input_ids, attention_mask):
                    return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            max_seq_len = 1024
            seq_len = args.seq_len if args.seq_len else max_seq_len

            word = "Hello"
            # Repeat text to fill maximum sequence length of model
            batch_sentences = " ".join([word] * seq_len)
            input_batch = [batch_sentences] * batch_size
	    
            inputs = list(tokenizer(input_batch, return_tensors="pt").values())
            model = GPT2Wrapper()

        if mode == "eval":
            model.eval()

        if device != "cpu":
            if distributed:
                model = torch.nn.DataParallel(model)

            model.to(device)
            # convert tuple to list so that we can modify it
            inputs = list(inputs)
            for idx, input in enumerate(inputs):
                inputs[idx] = input.to(device)
            inputs = tuple(inputs)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        importer = torch_graph_importer.TorchGraphImporter()
        (g, pt_node_order, fx_graph, fx_to_df_map) = importer.import_via_aotautograd(
            model,
            *inputs,
            optimizer=optimizer,
            loss_fn=torch.nn.CrossEntropyLoss(),
            mode=mode,
            profile=profile,
            warm_up_iters=warm_up_iters,
            profile_iters=profile_iters,
            return_node_ordering=True,
            return_fx_graph=True,
        )
        g.name = f"{model_name}_{batch_size}_{mode}"

        outputs = None
        if infer_trace:
            print("  INFER ORIGINAL FX TRACE", flush=True)
            fx_graph.recompile()
            with torch.no_grad():
                torch.manual_seed(0)
                output = fx_graph.forward(
                    inputs,
                    params=dict(model.named_parameters()),
                    buffers=dict(model.named_buffers()),
                )
            outputs = [output] if type(output) is not list else output

        # Prevent Pytorch from leaking memory
        # del model
        del importer.fx_trace
        del importer
        torch.cuda.empty_cache()

        assert g.is_valid(verbose=True)

        # Dump the graph in the background
        if render_model:

            def dump_model():
                print("  PRINTING MODEL IN THE BACKGROUND", flush=True)
                with open(
                    "/tmp/"
                    + model_name
                    + "_"
                    + str(batch_size)
                    + "_raw_"
                    + mode
                    + ".txt",
                    mode="w",
                ) as f:
                    f.write(str(g))

                g.dump(
                    "/tmp/" + model_name + "_" + str(batch_size) + "_raw_" + mode,
                    format="svg",
                )

            p = Process(target=dump_model, name="dump_" + model_name, daemon=False)
            p.start()

        print("  CANONICALIZING MODEL", flush=True)
        g.canonicalize()
        print("  CONSTRAINING WEIGHT UPDATES", flush=True)
        g.constrain_weight_updates()
        print("  CONSTRAINING TENSOR GENERATORS", flush=True)
        g.constrain_tensor_generators()

        print("  CHECKING GRAPH", flush=True)
        assert g.is_valid(verbose=True)

        # model_name = model.__class__.__name__
        # g.dump("/tmp/" + model_name + "_" + mode, format="svg")

        # TODO: instead of passing around many objects, perhaps we should encapsulate this function and the variables in the class.
        return g, pt_node_order, fx_graph, fx_to_df_map, model, inputs, outputs

    def measure_pt_alloc_time(self, node_ordering, num_times=100):
        class MemLoc:
            def __init__(self, size):
                self.size = size
                self.address = None

            def run(self):
                if self.address:
                    torch.cuda.caching_allocator_delete(self.address)
                    self.address = None
                else:
                    self.address = torch.cuda.caching_allocator_alloc(self.size)

        edge_ref_counts = {}
        mem_sequence = []
        mem_locs = {}
        for n in node_ordering:
            for fanout in n.fanout:
                if fanout.size > 0:
                    edge_ref_counts[fanout] = len(fanout.sinks)
                    tensor = MemLoc(fanout.size)
                    mem_sequence.append(tensor)
                    mem_locs[fanout] = tensor

            for fanin in n.fanin:
                if fanin.size == 0:
                    continue
                edge_ref_counts[fanin] -= 1
                if edge_ref_counts[fanin] == 0:
                    tensor = mem_locs[fanin]
                    mem_sequence.append(tensor)

        start = time.time()
        for _ in range(num_times):
            for op in mem_sequence:
                op.run()
        stop = time.time()
        num_alloc_dealloc_pairs = num_times * len(mem_sequence) / 2
        return (stop - start, num_alloc_dealloc_pairs)

    # TODO: should we have run_profile() as a function here that measures fragmentation, instead of calculating or measuring fragmentation inside TorchGraphImporter? This would decouple profiling from TorchGraphImporter and hence make it easier to run the same script on AWS

    def verify_fx_trace(self, fx_graph, model, outputs_orig):
        with torch.no_grad():
            torch.manual_seed(0)
            output = fx_graph.forward(
                inputs,
                params=dict(model.named_parameters()),
                buffers=dict(model.named_buffers()),
            )

        outputs = [output] if type(output) is not list else output
        for orig, after in zip(outputs_orig, outputs):
            assert torch.allclose(orig, after)

    def run_simulation(self, g, node_order):
        start = time.time()
        s = simulator.Simulator(g)
        stop = time.time()
        simulated_peak_mem_usage, mem_per_timestep = s.Simulate(node_order)
        return (simulated_peak_mem_usage, stop - start)

    def run_node_ordering(self, g, fx_graph, fx_to_df_map):
        start = time.time()
        s = training_graph_optimizer.Scheduler(
            g, rel_stop=0.005, timeout_s=args.solver_timeout
        )
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=False,
            max_spills=0,
        )
        stop = time.time()

        assert utils.validate_timeline(schedule)
        assert utils.validate_node_ordering(g, schedule)
        assert summary["peak_mem_usage"] == summary["required_memory"]
        # assert summary["peak_mem_usage"] <= simulated_peak_mem_usage
        assert summary["total_data_swapped"] == 0

        node_ordering = utils.extract_node_ordering(g, schedule)

        try:
            fx_opt = FXOptimizer(fx_graph, fx_to_df_map)
            fx_opt.Reorder(node_ordering)
            fx_graph_opt = fx_opt.fx_trace
        except Exception as e:
            print(f"  FAILED TO EXPORT NODE REORDERD SOLUTON FOR {model} TO FX:\n{traceback.format_exc()}")
            fx_graph_opt = None

        return (summary["peak_mem_usage"], node_ordering, fx_graph_opt, stop - start)

    def run_address_generation(self, g, node_order):
        start = time.time()
        s = training_graph_optimizer.Scheduler(
            g,
            rel_stop=0.001,
            timeout_s=args.solver_timeout,
            print_relaxation=True,
        )
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=False,
            max_spills=0,
            account_for_fragmentation=True,
            user_schedule=node_order,
        )
        stop = time.time()
        peak_mem_usage = summary["required_memory"]
        fragmentation = (peak_mem_usage - summary["peak_mem_usage"]) / peak_mem_usage
        assert utils.validate_timeline(schedule)
        assert utils.validate_address_allocation(mem_loc)
        assert summary["peak_mem_usage"] <= summary["required_memory"]
        assert summary["total_data_swapped"] == 0

        visualizer.draw_schedule(schedule, img_path="/tmp/" + g.name + ".png")

        return (peak_mem_usage, fragmentation, stop - start)

    def run_rematerialization(self, g, memory_budget):
        start = time.time()
        s = training_graph_optimizer.Scheduler(
            g, rel_stop=0.01, timeout_s=args.solver_timeout
        )
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=False,
            allow_rematerialization=True,
            mem_limit=memory_budget,
        )
        stop = time.time()
        extra_runtime = summary["rematerialization_time"]
        model_runtime = 0
        for n in g.nodes.values():
            if not n.time:
                continue
            model_runtime += n.time
        assert utils.validate_timeline(schedule)
        assert summary["peak_mem_usage"] == summary["required_memory"]
        assert summary["total_data_swapped"] == 0
        return (extra_runtime / model_runtime, stop - start)

    def run_spilling(self, g, memory_budget):
        start = time.time()
        s = training_graph_optimizer.Scheduler(
            g, rel_stop=0.01, timeout_s=args.solver_timeout
        )
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=True,
            allow_rematerialization=False,
            mem_limit=memory_budget,
        )
        stop = time.time()
        # TODO: replace bytes swapped to actual estimate of how much time it takes to
        # spill the data
        extra_runtime = summary["total_data_swapped"] / 16.0e9
        model_runtime = 0
        for n in g.nodes.values():
            if not n.time:
                continue
            model_runtime += n.time
        # print(f"MODEL TIME {model_runtime} vs SPILLING TIME {extra_runtime}")
        assert utils.validate_timeline(schedule)
        assert summary["peak_mem_usage"] == summary["required_memory"]
        return (extra_runtime / model_runtime, stop - start)


BENCHMARKS = {
    "alexnet": ["eval", "train"],
    "bert": ["eval", "train"],
    # "conformer": ["eval", "train"],  # fx can't trace the model
    # "deeplab": ["eval"],  # Train mode doesn't load
    "efficientnet": ["eval", "train"],
    # "emformer": ["eval", "train"],  # fx can't trace the model
    "googlenet": ["eval", "train"],
    "inception": ["eval", "train"],
    "mnasnet": ["eval", "train"],
    "mobilenet": ["eval", "train"],
    # "raft": ["eval", "train"],  # Model fails to load
    "resnet": ["eval", "train"],
    "resnet50": ["eval", "train"],
    "resnet3d": ["eval", "train"],
    "squeezenet": ["eval", "train"],
    # "ssd": ["eval"],  # Needs target in train mode
    # "swin": ["eval"],  # Model fails sanity checks
    "transformer": ["eval", "train"],
    "transformer_dflt": ["eval", "train"],
    "vgg": ["eval", "train"],
    "vgg16": ["eval", "train"],
    "vgg19": ["eval", "train"],
    "vit": ["eval", "train"],
    "xlmr": ["eval", "train"],
    "opt-125m": ["eval", "train"],
    "opt-350m": ["eval", "train"],
    "opt-1.3b": ["eval", "train"],
    "opt-2.7b": ["eval", "train"],
    "opt-6.7b": ["eval", "train"],
    "opt-13b": ["eval", "train"],
    "opt-30b": ["eval", "train"],
    "opt-66b": ["eval", "train"],
}


import argparse

# fmt: off
parser = argparse.ArgumentParser(description="MemOpt Benchmarks")
parser.add_argument("-b", "--batch-size", "--batch-sizes", nargs="+", type=int, default=[1, 32])
parser.add_argument("-m", "--model", "--models", nargs="+", type=str, default=BENCHMARKS.keys())
parser.add_argument("--mode", "--modes", nargs="+", type=str, choices=["eval", "train"], default=None)
parser.add_argument("--distributed", action="store_true", help="Distribute among GPUs")

parser.add_argument("--seq-len", type=int, default=None, help="Sequence length for text/speech/sequence models. If not specified, use the model's maximum length")

parser.add_argument("--solver-timeout", type=int, default=1800, help="ILP solver timeout in seconds")
parser.add_argument("--render-models", action="store_true")
parser.add_argument("--memory-profile", action="store_true")
parser.add_argument("--time-profile", action="store_true")
parser.add_argument("--warm-up-iters", type=int, default=None, help="Warm up iterations before profiling time or memory.")
parser.add_argument("--profile-iters", type=int, default=None, help="Number of iterations to profile time or memory.")
parser.add_argument("--profile-alloc-time", action="store_true")
parser.add_argument("--skip-simulation", action="store_true")

parser.add_argument("--skip-node-ordering", action="store_true")
parser.add_argument("--verify-node-ordering", action="store_true")
parser.add_argument("--memory-profile-node-ordering", action="store_true")

parser.add_argument("--generate-addresses", action="store_true")

parser.add_argument("--rematerialization", action="store_true")

parser.add_argument("--spilling", action="store_true")

parser.add_argument("--log-path", "--log_path", default="/tmp/opt4ml_benchmarks.csv")
parser.add_argument("-a", "--append-log", action="store_true")
parser.add_argument(
    '-d', '--debug',
    help="Log debugging statements",
    action="store_const", dest="log_level", const=logging.DEBUG,
    default=logging.WARNING,
)
parser.add_argument(
    '-v', '--verbose',
    help="Log verbose info",
    action="store_const", dest="log_level", const=logging.INFO,
)
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    print(f"Running with args {args}")

    b = Benchmark()

    results = []

    for model in args.model:
        modes = BENCHMARKS[model] if not args.mode else args.mode
        for mode in modes:
            for batch_size in args.batch_size:
                result = OrderedDict(
                    [("model", model), ("mode", mode), ("batch_size", batch_size)]
                )
                print(
                    f"\nLOADING MODEL {model} IN {mode} MODE WITH BATCH SIZE {batch_size}",
                    flush=True,
                )
                device = "cpu"
                distributed = args.distributed
                profile = []
                warm_up_iters = 1 if args.warm_up_iters is None else args.warm_up_iters
                profile_iters = 10 if args.profile_iters is None else args.profile_iters
                if args.time_profile or args.rematerialization or args.spilling:
                    profile.append("time")
                if args.memory_profile:
                    torch.cuda.empty_cache()
                    profile.append("memory")
                if args.memory_profile or args.memory_profile_node_ordering:
                    warm_up_iters = (
                        0 if args.warm_up_iters is None else args.warm_up_iters
                    )
                    profile_iters = (
                        300 if args.profile_iters is None else args.profile_iters
                    )
                    device = "cuda"

                try:
                    (
                        graph,
                        pt_node_order,
                        fx_graph,
                        fx_to_df_map,
                        torch_model,
                        inputs,
                        outputs,
                    ) = b.load_model(
                        model,
                        mode,
                        batch_size,
                        device=device,
                        distributed=distributed,
                        profile=profile,
                        warm_up_iters=warm_up_iters,
                        profile_iters=profile_iters,
                        render_model=args.render_models,
                        infer_trace=args.verify_node_ordering,
                    )
                except Exception as e:
                    print(f"  FAILED TO LOAD {model}, SKIPPING TO NEXT MODEL:\n{traceback.format_exc()}")
                    result["load_model.error"] = str(e).replace("\n", " ")
                    continue

                print(
                    f"BENCHMARKING MODEL {model} IN {mode} MODE WITH BATCH SIZE {batch_size}",
                    flush=True,
                )
                if args.memory_profile:
                    print(
                        f"PROFILED MAX MEMORY FRAGMENTATION IS {graph.max_mem_fragmentation:%} AND PROFILED PEAK MEMORY IS {graph.peak_reserved_bytes/GB:.4f} GB, PROFILED ALLOCATED MEMORY AT PEAK IS {graph.allocated_mem_at_peak/GB:.4f} GB"
                    )
                    result[
                        "profile.max_mem_fragmentation"
                    ] = graph.max_mem_fragmentation
                    result["profile.peak_mem_usage"] = graph.peak_reserved_bytes / GB
                    result[
                        "profile.allocated_mem_at_peak"
                    ] = graph.allocated_mem_at_peak / GB

                if not args.skip_simulation:
                    simulated_peak_mem_usage, _ = b.run_simulation(
                        graph,
                        pt_node_order,
                    )
                    print(
                        f"  SIMULATED PEAK MEM USAGE IS {simulated_peak_mem_usage / GB:.4f} GB",
                        flush=True,
                    )
                    result["simulated.peak_mem_usage"] = simulated_peak_mem_usage / GB

                if args.profile_alloc_time:
                    torch.cuda.empty_cache()
                    runtime, alloc_count = b.measure_pt_alloc_time(pt_node_order)
                    print(f"RAN {alloc_count} ALLOC/DEALLOC IN {runtime:.1f}s")
                    print(
                        f"PROFILED AVERAGE TIME IS {1e6*runtime/alloc_count} USEC PER ALLOC/DEALLOC"
                    )
                    result["profile.alloc_runtime"] = runtime
                    result["profile.alloc_count"] = alloc_count

                if not args.skip_node_ordering:
                    assert (
                        not args.skip_simulation
                    ), "Simulation is required to run node ordering"
                    try:
                        print("  PERFORM NODE REORDERING", flush=True)
                        (
                            peak_mem_usage,
                            node_ordering,
                            fx_graph_opt,
                            solver_time,
                        ) = b.run_node_ordering(graph, fx_graph, fx_to_df_map)

                        print(
                            f"  REORDERED NODES IN {solver_time:.1f}s. SIMULATED PEAK MEMORY USAGE WAS {peak_mem_usage / GB:.4f} GB (SAVED {(simulated_peak_mem_usage - peak_mem_usage) / simulated_peak_mem_usage:%})",
                            flush=True,
                        )

                        result["node_ordering.solver_time"] = solver_time
                        result["node_ordering.peak_mem_usage"] = peak_mem_usage / GB

                        if args.verify_node_ordering:
                            print("  INFER FX TRACE AFTER NODE REORDERING", flush=True)
                            try:
                                b.verify_fx_trace(fx_graph_opt, torch_model, outputs)
                                result["node_ordering.verification"] = "SUCCESS"
                            except Exception as e:
                                print(
                                    f"  FAILED TO VERIFY REORDERED NODES:\n{traceback.format_exc()}",
                                    flush=True,
                                )
                                result["node_ordering.verification"] = "FAIL"
                                result["node_ordering.verification.error"] = str(
                                    e
                                ).replace("\n", " ")

                        if args.memory_profile_node_ordering:
                            print(
                                "  PROFILE FX TRACE AFTER NODE REORDERING", flush=True
                            )
                            try:
                                torch.cuda.empty_cache()
                                profiler = fx_profiler.ProfilingInterpreter(
                                    fx_graph_opt,
                                    profile_time=False,
                                    profile_memory=True,
                                    warm_up_iters=warm_up_iters,
                                    profile_iters=profile_iters,
                                )
                                with torch.no_grad():
                                    profiler.run(
                                        *inputs,
                                        *dict(torch_model.named_parameters()).values(),
                                        *dict(torch_model.named_buffers()).values(),
                                    )
                                result["node_ordering.profile"] = "SUCCESS"

                                print(
                                    f"AFTER NODE ORDERING: PROFILED MAX MEMORY FRAGMENTATION IS {profiler.get_max_mem_fragmentation():%} AND PROFILED PEAK MEMORY IS {profiler.get_peak_reserved_bytes()/GB:.4f} GB, PROFILED ALLOCATED MEMORY AT PEAK IS {profiler.get_allocated_mem_at_peak()/GB:.4f} GB"
                                )
                                result[
                                    "node_ordering.profile.max_mem_fragmentation"
                                ] = profiler.max_mem_fragmentation
                                result[
                                    "node_ordering.profile.peak_reserved_bytes"
                                ] = profiler.peak_reserved_bytes / GB
                                result[
                                    "node_ordering.profile.allocated_mem_at_peak"
                                ] = profiler.allocated_mem_at_peak / GB
                            except Exception as e:
                                print(
                                    f"  FAILED TO PROFILE REORDERED NODES:\n{traceback.format_exc()}",
                                    flush=True,
                                )
                                result["node_ordering.profile"] = "FAIL"
                                result["node_ordering.profile.error"] = str(e).replace(
                                    "\n", " "
                                )

                    except Exception as e:
                        print(f"  FAILED TO REORDER NODES:\n{traceback.format_exc()}", flush=True)
                        result["node_ordering.error"] = str(e).replace("\n", " ")
                        continue

                if args.generate_addresses:
                    assert (
                        not args.skip_simulation
                    ), "Simulation is required to run address generation"
                    assert (
                        not args.skip_node_ordering
                    ), "Node ordering is required to run address generation"
                    try:
                        (
                            peak_mem_usage,
                            fragmentation,
                            solver_time,
                        ) = b.run_address_generation(graph, node_ordering)
                        print(
                            f"  GENERATED ADDRESSES IN {solver_time:.1f}s. PEAK MEM USAGE WAS {peak_mem_usage / GB:.4f} GB, FRAGMENTATION WAS {fragmentation:%}",
                            flush=True,
                        )
                        result["address_generation.solver_time"] = solver_time
                        result["address_generation.fragmentation"] = fragmentation
                        result["address_generation.peak_mem_usage"] = peak_mem_usage / GB
                    except Exception as e:
                        print(f"  FAILED TO GENERATE ADDRESSES:\n{traceback.format_exc()}", flush=True)
                        result["address_generation.error"] = str(e).replace("\n", " ")
                        traceback.print_exc()

                if args.rematerialization:
                    assert (
                        not args.skip_simulation
                    ), "Simulation is required to run rematerialization"
                    assert (
                        not args.skip_node_ordering
                    ), "Node ordering is required to run rematerialization"
                    try:
                        s = training_graph_optimizer.Scheduler(graph)
                        min_memory, _ = s.ComputeMinimumMemoryRequired()
                        for savings in [0.1, 0.25, 0.5, 0.75, 1.0]:
                            done = False
                            memory_budget = peak_mem_usage * (1.0 - savings)
                            if memory_budget < min_memory:
                                memory_budget = min_memory
                                savings = 1.0 - memory_budget / peak_mem_usage
                                done = True
                            overhead, solver_time = b.run_rematerialization(
                                graph, memory_budget
                            )
                            print(
                                f"  PLANNED REMATERIALIZATION TO SAVE {savings:%} MEMORY IN {solver_time:.1f}s. INCREASED MODEL LATENCY BY {overhead:%})",
                                flush=True,
                            )
                            result[
                                f"rematerialization.savings_{savings}.overhead"
                            ] = overhead
                            result[
                                f"rematerialization.savings_{savings}.solver_time"
                            ] = solver_time
                            if done:
                                break
                    except Exception as e:
                        print(
                            f"  FAILED TO PLAN REMATERIALIZATION TO SAVE {savings:%} MEMORY:\n{traceback.format_exc()}",
                            flush=True,
                        )
                        result[f"rematerialization.savings_{savings}.error"] = str(
                            e
                        ).replace("\n", " ")
                        traceback.print_exc()

                if args.spilling:
                    assert (
                        not args.skip_simulation
                    ), "Simulation is required to run spilling"
                    assert (
                        not args.skip_node_ordering
                    ), "Node ordering is required to run address spilling"

                    try:
                        graph.constrain_relative_ordering(node_ordering, linearize=True)
                        s = training_graph_optimizer.Scheduler(graph)
                        min_memory, _ = s.ComputeMinimumMemoryRequired()
                        for savings in [0.1, 0.25, 0.5, 0.75, 1.0]:
                            done = False
                            memory_budget = peak_mem_usage * (1.0 - savings)
                            if memory_budget < min_memory:
                                memory_budget = min_memory
                                savings = 1.0 - memory_budget / peak_mem_usage
                                done = True
                            overhead, solver_time = b.run_spilling(graph, memory_budget)
                            print(
                                f"  PLANNED SPILLING TO SAVE {savings:%} MEMORY IN {solver_time:.1f}s. INCREASED MODEL LATENCY BY {overhead:%})",
                                flush=True,
                            )
                            result[f"spilling.savings_{savings}.overhead"] = overhead
                            result[
                                f"spilling.savings_{savings}.solver_time"
                            ] = solver_time
                            if done:
                                break
                    except Exception as e:
                        print(
                            f"  FAILED TO PLAN SPILLING TO SAVE {savings:%} MEMORY:\n{traceback.format_exc()}",
                            flush=True,
                        )
                        result[f"spilling.savings_{savings}.error"] = str(e).replace(
                            "\n", " "
                        )
                        traceback.print_exc()

                # Log result
                results.append(result)
                pd.DataFrame(results).fillna("").to_csv(
                    args.log_path,
                    mode="a" if args.append_log else "w",
                    header=not args.append_log,
                    index=False,
                    float_format="%.4g",
                )
