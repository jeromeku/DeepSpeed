from contextlib import ExitStack, contextmanager

import torch


@contextmanager
def mark(label: str):
    """
    Utility context manager for annotating ranges for both torch.profiler and nvtx.
    """
    with ExitStack() as stack:
        stack.enter_context(torch.profiler.record_function(label))  # enter fn1 context
        stack.enter_context(torch.cuda.nvtx.range(label))  # enter fn2 context
        handle = torch.cuda.nvtx.range_start(label)
        yield
        torch.cuda.nvtx.range_end(handle)
