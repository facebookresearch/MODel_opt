#! /usr/bin/sh
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

set -o xtrace
trap : SIGTERM SIGINT

BENCHMARKS=(
  "alexnet,eval,1"
  "alexnet,eval,32"
  "alexnet,train,1"
  "alexnet,train,32"
  "bert,eval,1"
  "bert,eval,32"
  "bert,train,1"
  "bert,train,32"
  #"conformer,eval,1"
  #"conformer,eval,32"
  #"conformer,train,1"
  #"conformer,train,32"
  #"deeplab,eval,1"
  #"deeplab,eval,32"
  #"deeplab,train,1"
  #"deeplab,train,32"
  "efficientnet,eval,1"
  "efficientnet,eval,32"
  "efficientnet,train,1"
  "efficientnet,train,32"
  #"emformer,eval,1"
  #"emformer,eval,32"
  #"emformer,train,1"
  #"emformer,train,32"
  "googlenet,eval,1"
  "googlenet,eval,32"
  "googlenet,train,1"
  "googlenet,train,32"
  "inception,eval,1"
  "inception,eval,32"
  "inception,train,1"
  "inception,train,32"
  "mnasnet,eval,1"
  "mnasnet,eval,32"
  "mnasnet,train,1"
  "mnasnet,train,32"
  "mobilenet,eval,1"
  "mobilenet,eval,32"
  "mobilenet,train,1"
  "mobilenet,train,32"
  #"raft,eval,1"
  #"raft,eval,32"
  #"raft,train,1"
  #"raft,train,32"
  "resnet,eval,1"
  "resnet,eval,32"
  "resnet,train,1"
  "resnet,train,32"
  "resnet50,eval,1"
  "resnet50,eval,32"
  "resnet50,train,1"
  "resnet50,train,32"
  "resnet3d,eval,1"
  "resnet3d,eval,32"
  "resnet3d,train,1"
  "resnet3d,train,32"
  "squeezenet,eval,1"
  "squeezenet,eval,32"
  "squeezenet,train,1"
  "squeezenet,train,32"
  #"ssd,eval,1"
  #"ssd,eval,32"
  #"ssd,train,1"
  #"ssd,train,32"
  #"swin,eval,1"
  #"swin,eval,32"
  #"swin,train,1"
  #"swin,train,32"
  "transformer,eval,1"
  "transformer,eval,32"
  "transformer,train,1"
  "transformer,train,32"
  "transformer_dflt,eval,1"
  "transformer_dflt,eval,32"
  "transformer_dflt,train,1"
  "transformer_dflt,train,32"
  "vgg,eval,1"
  "vgg,eval,32"
  "vgg,train,1"
  "vgg,train,32"
  "vgg16,eval,1"
  "vgg16,eval,32"
  "vgg16,train,1"
  "vgg16,train,32"
  "vgg19,eval,1"
  "vgg19,eval,32"
  "vgg19,train,1"
  "vgg19,train,32"
  "vit,eval,1"
  "vit,eval,32"
  "vit,train,1"
  "vit,train,32"
  "xlmr,eval,1"
  "xlmr,eval,32"
  "xlmr,train,1"
  "xlmr,train,32"
  "alexnet,eval,64"
  "alexnet,eval,128"
  "alexnet,train,64"
  "alexnet,train,128"
  "bert,eval,64"
  "bert,eval,128"
  "bert,train,64"
  "bert,train,128"
  #"conformer,eval,64"
  #"conformer,eval,128"
  #"conformer,train,64"
  #"conformer,train,128"
  #"deeplab,eval,64"
  #"deeplab,eval,128"
  #"deeplab,train,64"
  #"deeplab,train,128"
  "efficientnet,eval,64"
  "efficientnet,eval,128"
  "efficientnet,train,64"
  "efficientnet,train,128"
  #"emformer,eval,64"
  #"emformer,eval,128"
  #"emformer,train,64"
  #"emformer,train,128"
  "googlenet,eval,64"
  "googlenet,eval,128"
  "googlenet,train,64"
  "googlenet,train,128"
  "inception,eval,64"
  "inception,eval,128"
  "inception,train,64"
  "inception,train,128"
  "mnasnet,eval,64"
  "mnasnet,eval,128"
  "mnasnet,train,64"
  "mnasnet,train,128"
  "mobilenet,eval,64"
  "mobilenet,eval,128"
  "mobilenet,train,64"
  "mobilenet,train,128"
  #"raft,eval,64"
  #"raft,eval,128"
  #"raft,train,64"
  #"raft,train,128"
  "resnet,eval,64"
  "resnet,eval,128"
  "resnet,train,64"
  "resnet,train,128"
  "resnet50,eval,64"
  "resnet50,eval,128"
  "resnet50,train,64"
  "resnet50,train,128"
  "resnet3d,eval,64"
  "resnet3d,eval,128"
  "resnet3d,train,64"
  "resnet3d,train,128"
  "squeezenet,eval,64"
  "squeezenet,eval,128"
  "squeezenet,train,64"
  "squeezenet,train,128"
  #"ssd,eval,64"
  #"ssd,eval,128"
  #"ssd,train,64"
  #"ssd,train,128"
  #"swin,eval,64"
  #"swin,eval,128"
  #"swin,train,64"
  #"swin,train,128"
  "transformer,eval,64"
  "transformer,eval,128"
  "transformer,train,64"
  "transformer,train,128"
  "transformer_dflt,eval,64"
  "transformer_dflt,eval,128"
  "transformer_dflt,train,64"
  "transformer_dflt,train,128"
  "vgg,eval,64"
  "vgg,eval,128"
  "vgg,train,64"
  "vgg,train,128"
  "vgg16,eval,64"
  "vgg16,eval,128"
  "vgg16,train,64"
  "vgg16,train,128"
  "vgg19,eval,64"
  "vgg19,eval,128"
  "vgg19,train,64"
  "vgg19,train,128"
  "vit,eval,64"
  "vit,eval,128"
  "vit,train,64"
  "vit,train,128"
  "xlmr,eval,64"
  "xlmr,eval,128"
  "xlmr,train,64"
  "xlmr,train,128"
  "alexnet,eval,256"
  "alexnet,eval,512"
  "alexnet,train,256"
  "alexnet,train,512"
  "bert,eval,256"
  "bert,eval,512"
  "bert,train,256"
  "bert,train,512"
  #"conformer,eval,256"
  #"conformer,eval,512"
  #"conformer,train,256"
  #"conformer,train,512"
  #"deeplab,eval,256"
  #"deeplab,eval,512"
  #"deeplab,train,256"
  #"deeplab,train,512"
  "efficientnet,eval,256"
  "efficientnet,eval,512"
  "efficientnet,train,256"
  "efficientnet,train,512"
  #"emformer,eval,256"
  #"emformer,eval,512"
  #"emformer,train,256"
  #"emformer,train,512"
  "googlenet,eval,256"
  "googlenet,eval,512"
  "googlenet,train,256"
  "googlenet,train,512"
  "inception,eval,256"
  "inception,eval,512"
  "inception,train,256"
  "inception,train,512"
  "mnasnet,eval,256"
  "mnasnet,eval,512"
  "mnasnet,train,256"
  "mnasnet,train,512"
  "mobilenet,eval,256"
  "mobilenet,eval,512"
  "mobilenet,train,256"
  "mobilenet,train,512"
  #"raft,eval,256"
  #"raft,eval,512"
  #"raft,train,256"
  #"raft,train,512"
  "resnet,eval,256"
  "resnet,eval,512"
  "resnet,train,256"
  "resnet,train,512"
  "resnet50,eval,256"
  "resnet50,eval,512"
  "resnet50,train,256"
  "resnet50,train,512"
  "resnet3d,eval,256"
  "resnet3d,eval,512"
  "resnet3d,train,256"
  "resnet3d,train,512"
  "squeezenet,eval,256"
  "squeezenet,eval,512"
  "squeezenet,train,256"
  "squeezenet,train,512"
  #"ssd,eval,256"
  #"ssd,eval,512"
  #"ssd,train,256"
  #"ssd,train,512"
  #"swin,eval,256"
  #"swin,eval,512"
  #"swin,train,256"
  #"swin,train,512"
  "transformer,eval,256"
  "transformer,eval,512"
  "transformer,train,256"
  "transformer,train,512"
  "transformer_dflt,eval,256"
  "transformer_dflt,eval,512"
  "transformer_dflt,train,256"
  "transformer_dflt,train,512"
  "vgg,eval,256"
  "vgg,eval,512"
  "vgg,train,256"
  "vgg,train,512"
  "vgg16,eval,256"
  "vgg16,eval,512"
  "vgg16,train,256"
  "vgg16,train,512"
  "vgg19,eval,256"
  "vgg19,eval,512"
  "vgg19,train,256"
  "vgg19,train,512"
  "vit,eval,256"
  "vit,eval,512"
  "vit,train,256"
  "vit,train,512"
  "xlmr,eval,256"
  "xlmr,eval,512"
  "xlmr,train,256"
  "xlmr,train,512"
  "alexnet,eval,1024"
  "alexnet,eval,2048"
  "alexnet,train,1024"
  "alexnet,train,2048"
  "bert,eval,1024"
  "bert,eval,2048"
  "bert,train,1024"
  "bert,train,2048"
  #"conformer,eval,1024"
  #"conformer,eval,2048"
  #"conformer,train,1024"
  #"conformer,train,2048"
  #"deeplab,eval,1024"
  #"deeplab,eval,2048"
  #"deeplab,train,1024"
  #"deeplab,train,2048"
  "efficientnet,eval,1024"
  "efficientnet,eval,2048"
  "efficientnet,train,1024"
  "efficientnet,train,2048"
  #"emformer,eval,1024"
  #"emformer,eval,2048"
  #"emformer,train,1024"
  #"emformer,train,2048"
  "googlenet,eval,1024"
  "googlenet,eval,2048"
  "googlenet,train,1024"
  "googlenet,train,2048"
  "inception,eval,1024"
  "inception,eval,2048"
  "inception,train,1024"
  "inception,train,2048"
  "mnasnet,eval,1024"
  "mnasnet,eval,2048"
  "mnasnet,train,1024"
  "mnasnet,train,2048"
  "mobilenet,eval,1024"
  "mobilenet,eval,2048"
  "mobilenet,train,1024"
  "mobilenet,train,2048"
  #"raft,eval,1024"
  #"raft,eval,2048"
  #"raft,train,1024"
  #"raft,train,2048"
  "resnet,eval,1024"
  "resnet,eval,2048"
  "resnet,train,1024"
  "resnet,train,2048"
  "resnet50,eval,1024"
  "resnet50,eval,2048"
  "resnet50,train,1024"
  "resnet50,train,2048"
  "resnet3d,eval,1024"
  "resnet3d,eval,2048"
  "resnet3d,train,1024"
  "resnet3d,train,2048"
  "squeezenet,eval,1024"
  "squeezenet,eval,2048"
  "squeezenet,train,1024"
  "squeezenet,train,2048"
  #"ssd,eval,1024"
  #"ssd,eval,2048"
  #"ssd,train,1024"
  #"ssd,train,2048"
  #"swin,eval,1024"
  #"swin,eval,2048"
  #"swin,train,1024"
  #"swin,train,2048"
  "transformer,eval,1024"
  "transformer,eval,2048"
  "transformer,train,1024"
  "transformer,train,2048"
  "transformer_dflt,eval,1024"
  "transformer_dflt,eval,2048"
  "transformer_dflt,train,1024"
  "transformer_dflt,train,2048"
  "vgg,eval,1024"
  "vgg,eval,2048"
  "vgg,train,1024"
  "vgg,train,2048"
  "vgg16,eval,1024"
  "vgg16,eval,2048"
  "vgg16,train,1024"
  "vgg16,train,2048"
  "vgg19,eval,1024"
  "vgg19,eval,2048"
  "vgg19,train,1024"
  "vgg19,train,2048"
  "vit,eval,1024"
  "vit,eval,2048"
  "vit,train,1024"
  "vit,train,2048"
  "xlmr,eval,1024"
  "xlmr,eval,2048"
  "xlmr,train,1024"
  "xlmr,train,2048"
)

first_time=1
for benchmark in "${BENCHMARKS[@]}"; do
  while IFS=',' read -r model mode batch_size; do
    append_log=$([ "${first_time}" == 1 ] && echo || echo "--append-log")

    python benchmarks1.py -b "${batch_size}" --model "${model}" --mode "${mode}" ${append_log} $@ &

    # kill python run if user kills shell script, then terminate script
    FIND_PID=$!
    wait $FIND_PID
    if [[ $? -gt 128 ]]
    then
        kill $FIND_PID
        exit 1
    fi

    first_time=0
  done <<< "${benchmark}"
done
