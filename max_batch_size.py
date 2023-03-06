
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import olla
import pandas as pd
import torch
import torchvision
import traceback
import logging

import concurrent.futures

logging.basicConfig(level=logging.CRITICAL)

def get_max(fn):
    start = 1
    end = None
    val = start
    while True:
        if try_model_in_process(fn, val):
            val *= 2
        else:
            break
    
    start = val // 2
    end = val
    mid = None
    while start <= end:
        mid = (start + end) // 2
        if try_model_in_process(fn, mid):
            if not try_model_in_process(fn, mid+1):
                return mid
            else:
                start = mid + 1
        else:
            if try_model_in_process(fn, mid-1):
                return mid - 1
            else:
                end = mid - 1
    return None

def try_model(fn, bs):
    logging.info(f"Trying batch size: {bs}")
    input = torch.Tensor(bs, 3, 224, 224).cuda()
    try:
        output = fn(input)
    except Exception as e:
        logging.debug(str(e))
        return False
    return True


def try_model_in_process(fn, bs):
    return run_in_process(try_model, fn, bs)

def run_in_process(fn, *args, **kwargs):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        fn = executor.submit(fn, *args, **kwargs)
        ret = fn.result()  # will rethrow any exceptions
        return ret

def run_eager(input):
    model = torchvision.models.resnet18().cuda()
    model.eval()

    model(input)

def run_fx(input):
    model = torchvision.models.resnet18().cuda()
    model.eval()

    from torch.fx import symbolic_trace
    model = symbolic_trace(model)

    model(input)

def run_olla(input):
    model = torchvision.models.resnet18().cuda()
    model.eval()

    model_olla = olla.optimize(model, input)
    model_olla(input)

print("Eager:")
max_eager = get_max(run_eager)
print(f"max_eager is {max_eager}")

print("fx:")
max_fx = get_max(run_fx)
print(f"max_fx is {max_fx}")

print("olla:")
max_olla = get_max(run_olla)
print(f"max_olla is {max_olla}")


# PyTorch Eager
"""
max_batch_eager = None
bs = 1
while True:
    input = torch.Tensor(bs, 3, 224, 224).cuda()

    try:
        logging.info(f"Trying batch size: {bs}")
        output = model(input)
    except Exception as e:
        logging.debug(str(e))
        break

    bs *= 2

bs //= 2
# TODO: use binary search
while True:
    input = torch.Tensor(bs, 3, 224, 224).cuda()

    try:
        logging.info(f"Trying batch size: {bs}")
        output = model(input)
    except Exception as e:
        logging.debug(str(e))
        break

    max_batch_eager = bs
    bs += 1

print(f"Eager Mode: Max batch size: {max_batch_eager}")
"""

"""
# Torch.Fx

from torch.fx import symbolic_trace
model = symbolic_trace(model)

max_batch_fx = None
bs = 1
while True:
    input = torch.Tensor(bs, 3, 224, 224).cuda()

    try:
        logging.info(f"Trying batch size: {bs}")
        output = model(input)
    except Exception as e:
        logging.debug(str(e))
        break

    bs *= 2

bs //= 2
# TODO: use binary search
while True:
    input = torch.Tensor(bs, 3, 224, 224).cuda()

    try:
        logging.info(f"Trying batch size: {bs}")
        output = model(input)
    except Exception as e:
        logging.debug(str(e))
        break

    max_batch_fx = bs
    bs += 1

print(f"Torch.Fx Mode: Max batch size: {max_batch_fx}")
"""

"""
# Olla

import olla

max_batch_olla = None
model = None
model_olla = None
input = None
bs = 1
while True:
    torch.cuda.empty_cache()
    del model
    del model_olla
    del input
    model = None
    model_olla = None
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    input = torch.Tensor(bs, 3, 224, 224).cuda()

    try:
        logging.info(f"Trying batch size: {bs}")
        model = torchvision.models.resnet18().cuda()
        model.eval()
        model_olla = olla.optimize(model, input)
        output = model_olla(input)
    except Exception as e:
        logging.debug(str(e))
        break

    bs *= 2

bs //= 2
# TODO: use binary search
while True:
    torch.cuda.empty_cache()
    del model
    del model_olla
    del input
    model = None
    model_olla = None
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    input = torch.Tensor(bs, 3, 224, 224).cuda()

    try:
        logging.info(f"Trying batch size: {bs}")
        model = torchvision.models.resnet18().cuda()
        model.eval()
        model_olla = olla.optimize(model, input)
        model_olla = olla.optimize(model, input)
        output = model_olla(input)
    except Exception as e:
        logging.debug(str(e))
        break

    max_batch_olla = bs
    bs += 1

print(f"Olla Mode: Max batch size: {max_batch_olla}")
"""