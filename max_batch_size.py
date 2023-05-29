
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import model_opt
import pandas as pd
import torch
import torchvision
import traceback
import logging

import concurrent.futures

logging.basicConfig(level=logging.CRITICAL)

def create_model():
    return torchvision.models.mobilenet_v2().cuda()

def create_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)

loss_fn = torch.nn.MSELoss()
is_training = True

def get_max(fn, *args, **kwargs):
    start = 1
    end = None
    val = start
    while True:
        if try_model_in_process(fn, val, *args, **kwargs):
            val *= 2
        else:
            break
    
    start = val // 2
    end = val
    mid = None
    while start <= end:
        mid = (start + end) // 2
        if try_model_in_process(fn, mid, *args, **kwargs):
            if not try_model_in_process(fn, mid+1, *args, **kwargs):
                return mid
            else:
                start = mid + 1
        else:
            if try_model_in_process(fn, mid-1, *args, **kwargs):
                return mid - 1
            else:
                end = mid - 1
    return None

def try_model(fn, bs, *args, **kwargs):
    logging.info(f"Trying batch size: {bs}")
    input = torch.Tensor(bs, 3, 224, 224).cuda()
    try:
        output = fn(input, *args, **kwargs)
    except Exception as e:
        logging.debug(str(e))
        logging.debug(traceback.format_exc())
        return False
    return True

def try_model_in_process(fn, bs, *args, **kwargs):
    return run_in_process(try_model, fn, bs, *args, **kwargs)

def run_in_process(fn, *args, **kwargs):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        fn = executor.submit(fn, *args, **kwargs)
        ret = fn.result()  # will rethrow any exceptions
        return ret

def run_eager(input, create_model, is_training=False, create_optimizer=None, loss_fn=None):
    model = create_model()
    
    if is_training:
        model.train()
        optimizer = create_optimizer(model)
    else:
        model.eval()
    
    output = model(input)
    if is_training:
        target = torch.zeros_like(output)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

def run_fx(input, create_model):
    model = create_model()
    model.eval()

    from torch.fx import symbolic_trace
    model = symbolic_trace(model)

    model(input)

def run_olla(input, create_model, is_training=False, create_optimizer=None, loss_fn=None):
    model = create_model()

    if is_training:
        model.train()
        model_olla = model_opt.optimize(model, input, optimizer=create_optimizer(model), loss_fn=loss_fn)
    else:
        model.eval()
        model_olla = model_opt.optimize(model, input)

    model_olla(input)

print("Eager:")
max_eager = get_max(run_eager, create_model, is_training=is_training, create_optimizer=create_optimizer, loss_fn=loss_fn)
print(f"max_eager is {max_eager}")

# TODO: support training for fx
print("fx:")
max_fx = get_max(run_fx, create_model)
print(f"max_fx is {max_fx}")

print("model_opt:")
max_olla = get_max(run_olla, create_model, is_training=is_training, create_optimizer=create_optimizer, loss_fn=loss_fn)
print(f"max_olla is {max_olla}")
