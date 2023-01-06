import os
from collections.abc import Iterable
import pandas as pd
import fx_profiler1
import torch
import torch.nn.utils._stateless as stateless
import torchaudio
import torchtext
import torchvision
from functorch import make_fx
from fx_profiler1 import allocated_key

gpu_log_dir = f"{os.path.expanduser('~')}/profiling/profile/logs/"
class TorchGraphImporter:
    def __init__(self):
        self.fx: torch.fx.GraphModule = None
        self.profiler: fx_profiler1 = None
        self.profiler_table: pd.DataFrame = None
    def assert_type(self, mode):
        assert mode in ["eval", "train"], "Invalid mode provided"
    def _load_profile_data_to_graph(self, profiler_table):
        if (
            "reserved_bytes.all.current" in profiler_table.columns
            and allocated_key in profiler_table.columns
        ):

            # deduce maximum memory fragmentation
            idx = profiler_table["reserved_bytes.all.current"].idxmax()
            row = profiler_table.iloc[idx]
            max_mem_fragmentation = (
                row["reserved_bytes.all.current"] - row[allocated_key]
            ) / row["reserved_bytes.all.current"]
            peak_reserved_bytes = row["reserved_bytes.all.current"]
            return max_mem_fragmentation, peak_reserved_bytes

    def import_via_aotautograd(
        self,
        model,
        *inputs,
        mode="train",
        optimizer=None,
        profile=None,
        warm_up_iters=0,
        profile_iters=1,
        return_node_ordering=True,
    ):
        self.assert_type(mode)
        # TODO: if mode is train, assert that inputs have `required_grad = True`
        def fn_model_wrapper(args, params, buffers):
            if not isinstance(args, Iterable):
                args = [args]
            params_and_buffers = {**params, **buffers}
            out = stateless.functional_call(model, params_and_buffers, args)
            if mode == "eval":
                return out
            elif mode == "train":
                out.sum().backward()
                if optimizer is True:
                    return [p - p.grad for p in params.values()]
                    # return [p.sub_(1e-4 * p.grad) for p in params.values()]
                elif optimizer is not None:
                    optimizer.step()
                # TODO: this causes graph to show an output with many incoming edges. Shall we try `return None` or simply don't return?
                return list(params.values())
        def detach_decomposition(x):
            return x
        fx_trace = make_fx(
            fn_model_wrapper,
            #decomposition_table={torch.ops.aten.detach.default: detach_decomposition},
            decomposition_table={torch.ops.aten.detach: detach_decomposition}, # for PyTorch 1.11
        )(inputs, dict(model.named_parameters()), dict(model.named_buffers()))
        # print("SYMBOLIC TRACE: \n" + str(fx_trace.graph))
        # print("CODE FOR TRACE: \n" + str(fx_trace.code))
        # aotautograd performs its own shape inference so we don't need to perform ours
        return self.import_from_fx(
            fx_trace,
            *inputs,
            *dict(model.named_parameters()).values(),
            *dict(model.named_buffers()).values(),
            cleanup=True,
            shape_inference=False,
            profile=profile,
            warm_up_iters=warm_up_iters,
            profile_iters=profile_iters,
            return_node_ordering=return_node_ordering,
        )

    def import_from_fx(
        self,
        fx_trace,
        *inputs,
        cleanup=False,
        shape_inference=True,
        profile=None,
        warm_up_iters=0,
        profile_iters=1,
        return_node_ordering=False,
    ):
        # TODO: we are profiling the model twice: once to measure execution time and once for shape propagation. So we need to somehow combine both ShapeProp into our custom ProfilingInterpreter so that we only profileo once
        # run profiler
        if profile:
            assert isinstance(profile, list) and (
                "time" in profile or "memory" in profile
            ), "profile argument should either be None or a list with either 'time' or 'memory' string elements"
            self.profiler = fx_profiler1.ProfilingInterpreter(
                fx_trace,
                profile_time="time" in profile,
                profile_memory="memory" in profile,
                warm_up_iters=warm_up_iters,
                profile_iters=profile_iters,
            )
            self.profiler.run(*inputs)
            self.profiler_table = self.profiler.summary()
        if profile:
            return self._load_profile_data_to_graph(self.profiler_table)

class Benchmark:
    def load_model(
        self,
        model_name,
        mode,
        batch_size=32,
        device="cpu",
        profile=None,
        warm_up_iters=0,
        profile_iters=1,
        log_gpu_profile=False,
        render_model=False,
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
        elif model_name == "raft":
            model = torchvision.models.optical_flow.raft_small()
            inputs = (
                torch.randn((batch_size, 3, 520, 960)),
                torch.randn((batch_size, 3, 520, 960)),
            )
        elif model_name == "resnet":
            model = torchvision.models.resnet18()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "resnet3d":
            model = torchvision.models.video.r3d_18()
            inputs = (torch.randn((batch_size, 3, 1, 112, 112)),)
        elif model_name == "squeezenet":
            model = torchvision.models.squeezenet1_0()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "roberta":
            xlmr_base = torchtext.models.XLMR_BASE_ENCODER
            model = xlmr_base.get_model()
            transform = xlmr_base.transform()
            input_batch = ["Hello world"] * batch_size
            inputs = (
                torchtext.functional.to_tensor(transform(input_batch), padding_value=1),
            )
        elif model_name == "ssd":
            model = torchvision.models.detection.ssd300_vgg16()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "swin":
            model = torchvision.models.swin_t()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "transformer":
            model = torch.nn.Transformer(
                nhead=1, num_encoder_layers=1, num_decoder_layers=1
            )
            inputs = (
                torch.rand((10, batch_size, 512)),
                torch.rand((20, batch_size, 512)),
            )
        elif model_name == "vgg":
            model = torchvision.models.vgg11()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        elif model_name == "vit":
            model = torchvision.models.vit_b_16()
            inputs = (torch.randn((batch_size, 3, 224, 224)),)
        if mode == "eval":
            model.eval()
        if device != "cpu":
            model.to(device)
            # convert tuple to list so that we can modify it
            inputs = list(inputs)
            for idx, input in enumerate(inputs):
                inputs[idx] = input.to(device)
            inputs = tuple(inputs)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        importer = TorchGraphImporter()
        max_mem_fragmentation, peak_reserved_bytes = importer.import_via_aotautograd(
            model,
            *inputs,
            optimizer=optimizer,
            mode=mode,
            profile=profile,
            warm_up_iters=warm_up_iters,
            profile_iters=profile_iters,
        )
        if log_gpu_profile:
            assert profile
            os.makedirs(gpu_log_dir, exist_ok=True)
            importer.profiler_table.to_csv(
                f"{gpu_log_dir}/{model_name}_{batch_size}_{mode}.csv"
            )
            with open(f"{gpu_log_dir}/max_fragmentation.csv", "a+") as f:
                f.write(
                    f"{model_name},{batch_size},{mode},{max_mem_fragmentation},{peak_reserved_bytes}\n"
                )
            print(
                f"{model_name},{batch_size},{mode},{max_mem_fragmentation},{peak_reserved_bytes}"
            )
BENCHMARKS = {
    "alexnet": ["train"],
}
import argparse
parser = argparse.ArgumentParser(description="MemOpt Benchmarks")
parser.add_argument("--generate-addresses", action="store_true")
parser.add_argument("--rematerialization", action="store_true")
parser.add_argument("--render-models", action="store_true")
parser.add_argument("--gpu-profile", action="store_true")
parser.add_argument("--log-gpu-profile", action="store_true")
parser.add_argument("--skip-simulation", action="store_true")
parser.add_argument("--skip-node-ordering", action="store_true")
if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with args {args}")
    b = Benchmark()
    if args.log_gpu_profile:
        os.makedirs(gpu_log_dir, exist_ok=True)
        with open(f"{gpu_log_dir}/max_fragmentation.csv", "w+") as f:
            f.write(
                "model_name,batch_size,mode,g.max_mem_fragmentation,peak_reserved_bytes\n"
            )
    for model, modes in BENCHMARKS.items():
        for mode in modes:
            for batch_size in [64]:
                print(
                    f"\nLOADING MODEL {model} IN {mode} MODE WITH BATCH SIZE {batch_size}",
                    flush=True,
                )
                device = "cpu"
                profile = []
                warm_up_iters = 0
                profile_iters = 1
                if args.rematerialization or args.gpu_profile:
                    profile.append("time")
                if args.gpu_profile:
                    torch.cuda.empty_cache()
                    profile.append("memory")
                    warm_up_iters = 0
                    profile_iters = 300
                    device = "cuda"
                if True: # try:
                    b.load_model(
                        model,
                        mode,
                        batch_size,
                        device=device,
                        profile=profile,
                        warm_up_iters=warm_up_iters,
                        profile_iters=profile_iters,
                        log_gpu_profile=args.log_gpu_profile,
                        render_model=args.render_models,
                    )
                else: # except Exception as e:
                    print(f"  FAILED TO LOAD {model}, SKIPPING TO NEXT MODEL: {e}")
                    continue
