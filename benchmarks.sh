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
)

first_time=1
for benchmark in "${BENCHMARKS[@]}"; do
  while IFS=',' read -r model mode batch_size; do
    append_log=$([ "${first_time}" == 1 ] && echo || echo "--append-log")

    python benchmarks.py -b "${batch_size}" --model "${model}" --mode "${mode}" ${append_log} &

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
