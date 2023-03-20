
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
import tensorboard
import tensorflow_models as tfm
import official as tfo
import tensorflow_text as tf_text
import official.vision.modeling.backbones as tfo_backbones
from tf_models import alexnet, vgg11
from pathlib import Path
import os
import fire


def get_model(model_name, batch_size=32):
    # Vision
    if model_name == "alexnet":
        model = alexnet(input_shape=(224, 224, 3), num_classes=1000)
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "resnet_deeplab":
        model = tfo_backbones.resnet_deeplab.DilatedResNet(model_id=50)
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "efficientnet":
        model = tf.keras.applications.efficientnet.EfficientNetB0()
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "mnasnet":
        model = tf.keras.applications.nasnet.NASNetMobile()
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "mobilenet":
        model = tfm.vision.backbones.MobileNet(model_id="MobileNetV2")
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "resnet" or model_name == "resnet18":
        model = tfm.vision.backbones.ResNet(model_id=18)
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "resnet50":
        model = tfm.vision.backbones.ResNet(model_id=50)
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "resnet3d":
        raise ValueError("resnet3d not yet supported")
        # tfm is missing a resnet3d 18 variant
        # model = tfm.vision.backbones.ResNet3D(
        #     model_id=backbone_cfg.model_id,
        #     temporal_strides=temporal_strides,
        #     temporal_kernel_sizes=temporal_kernel_sizes)
        inputs = (tf.random.normal((batch_size, 3, 1, 112, 112)),)
    elif model_name == "vgg" or model_name == "vgg11":
        model = vgg11(input_shape=(224, 224, 3), num_classes=1000)
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "vgg16":
        model = tf.keras.applications.vgg16.VGG16()
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "vgg19":
        model = tf.keras.applications.vgg19.VGG19()
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)
    elif model_name == "vit":
        model = tfm.vision.backbones.VisionTransformer(
            input_specs=tf.keras.layers.InputSpec(shape=[None, 224, 224, 3])
        )
        inputs = (tf.random.normal((batch_size, 224, 224, 3)),)

    # NLP
    elif model_name == "bert":
        from transformers import TFRobertaModel

        max_seq_len = 512
        inputs = (
            tf.cast(
                (tf.random.normal((batch_size, max_seq_len)) + 1) * 1000, dtype=tf.int64
            ),
            tf.cast(
                (tf.random.normal((batch_size, max_seq_len)) + 1) * 1000, dtype=tf.int64
            ),
        )
        model = TFRobertaModel.from_pretrained("roberta-base")
    elif model_name == "transformer":

        class Transformer(tf.Module):
            def __init__(self):
                super(Transformer, self).__init__()
                self.encoder = tfm.nlp.layers.TransformerEncoderBlock(
                    inner_dim=512, num_attention_heads=1, inner_activation="relu"
                )
                self.decoder = tfm.nlp.layers.TransformerDecoderBlock(
                    intermediate_size=512,
                    num_attention_heads=1,
                    intermediate_activation="relu",
                )

            def __call__(self, inputs):
                context, x = inputs
                context = self.encoder(context)
                # attention masks are None
                x = self.decoder(inputs=(x, context, None, None))
                # don't project to a vocab size
                return x

        model = Transformer()
        # context, x
        seq_length = 512
        inputs = (
            tf.random.normal((batch_size, seq_length, 10)),
            tf.random.normal((batch_size, seq_length, 20)),
        )
    elif model_name == "xlmr":
        from transformers import AutoTokenizer, TFXLMRobertaModel

        max_seq_len = 512  # remaining seq will be padded
        inputs = (
            tf.cast(
                (tf.random.normal((batch_size, max_seq_len)) + 1) * 1000, dtype=tf.int64
            ),
            tf.cast(
                (tf.random.normal((batch_size, max_seq_len)) + 1) * 1000, dtype=tf.int64
            ),
        )
        model = TFXLMRobertaModel.from_pretrained("xlm-roberta-base")
    else:
        raise ValueError(f"Model name {model_name} invalid")

    return model, inputs


def main(
    model_name: str,
    batch_size: int,
    out_dir: str,
    use_xla: bool = False,
    memory_growth: bool = False,
):
    print(
        f"Profiling model {model_name} with batch size {batch_size} with use_xla={use_xla} memory_growth={memory_growth}"
    )

    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device=device, enable=memory_growth)

    # to run the Tensorboard profiler, add libcupti to the library path, i.e.:
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/public/apps/cuda/11.6/extras/CUPTI/lib64/
    # export XLA_FLAGS=--xla_gpu_cuda_data_dir=/public/apps/cuda/11.6/
    model, inputs = get_model(model_name, batch_size=batch_size)

    @tf.function(jit_compile=use_xla)
    def run(model, inputs):
        output = model(inputs)
        return output

    _logdir = Path(out_dir) / f"{model_name}-{batch_size}"
    os.makedirs(_logdir, exist_ok=True)
    tf.profiler.experimental.start(logdir=str(_logdir))
    output = run(model, inputs)
    tf.profiler.experimental.stop()

    # This measure may not agree too closely with
    memory_info = tf.config.experimental.get_memory_info(device="GPU:0")
    print(f"Memory info: {memory_info}")
    with open(_logdir / "memory.log", "w") as f:
        f.write(
            'Result of tf.config.experimental.get_memory_info(device="GPU:0"): {memory_info}'
        )


if __name__ == "__main__":
    fire.Fire(main)
