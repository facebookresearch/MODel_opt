import torch


class TraceOptimizer:
    def __init__(self, fx_trace):
        self.fx_trace = fx_trace

    def Reorder(self, optimizer_result):
        pass
