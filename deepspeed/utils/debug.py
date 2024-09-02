# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import datetime
import os
import pdb

import deepspeed.comm as dist

BREAKPOINT_ENV_VAR="PYTHONBREAKPOINT"
GLOBAL_LOG_DIR = "deepspeed_logs"

log_dir = None
fhandles = {}
# For lazy import with printflock()
fcntl = None

# for debug purposes map module and param objects to their fully qualified names
module_names = {}
param_names = {}


def debug_clear_module_and_param_names():
    global module_names
    global param_names
    module_names = {}
    param_names = {}


def debug_extract_module_and_param_names(model):
    # extract the fully qualified names as soon as the model is acquired
    global module_names
    global param_names
    # XXX: can probably make a map of param2module and vice-versa
    module_names = {module: name for name, module in model.named_modules()}
    param_names = {param: name for name, param in model.named_parameters()}


def debug_module2name(module):
    if module in module_names:
        return module_names[module]
    else:
        return "unknown"


def debug_module2name_id(module):
    return f"name={debug_module2name(module)} id={module.id}"


def debug_module2name_class(module):
    return f"name={debug_module2name(module)} {module.__class__.__name__}"


def debug_param2name(param):
    if param in param_names:
        return param_names[param]
    else:
        return "unknown"


def debug_param2name_id(param):
    return f"name={debug_param2name(param)} id={param.ds_id}"


def debug_param2name_id_shape(param):
    return f"name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape}"


def debug_param2name_id_shape_device(param):
    return f"name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape} device={param.device}"


def debug_param2name_id_numel(param):
    return f"name={debug_param2name(param)} id={param.ds_id} numel={param.numel()}"


def debug_param2name_id_shape_status(param):
    return f"name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape} status={param.ds_status}"


def printflock(*msgs):
    """

    For printing messages for all concurrent gpus w/o getting interleaved text.

    This is useful when debugging issues where multi-gpus don't sync.

    1. Enable the force debug in say partitioning and zero3 files
    2. Override the usual versions with ::

        def print_rank_0(message, debug=False, force=False):
            rank = deepspeed.comm.get_rank()
            printflock(f"[{rank}] {message}")
    3. run the program and you get both logs non-interleaved

    But this makes it very difficult to make sense of the output, so the ``log_rank_file`` helper
    function might be more useful, as it's easier to send each log stream into a separate file and
    then compare those.

    """
    global fcntl
    if fcntl is None:
        import fcntl

    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


fh = None

def set_breakpoint():
    if os.environ.get(BREAKPOINT_ENV_VAR, "0") == "1":
        pdb.set_trace()
    
def log_rank_file(rank, *msgs, log_path=None, mode='a'):
    """
    Print to a log file of the given rank

    This is useful for debugging hanging in sync processes. Here is a possible workflow:

    1. Enable the force debug in say partitioning and zero3 files
    2. Override the usual versions of print_rank_0 in those files with ::

        def print_rank_0(message, debug=False, force=False):
            rank = deepspeed.comm.get_rank()
            log_rank_file(rank, message)

    3. run the program
    4. fix up the expected differences, e.g. different cuda numbers ::

        perl -pi -e 's|cuda:1|cuda:0|' log_rank_*

    5. now diff and see where names and ids diverge - you will find where the gpus don't do the same
    work (e.g. when some layers get conditionally skipped on one gpu but not all)

        diff -u log_rank_0.txt log_rank_1.txt | less

    """
    global log_dir
    global fhandles
    
    if log_dir is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_dir = os.path.join(GLOBAL_LOG_DIR, ts)
        if not os.path.exists(log_dir) and dist.get_local_rank() == 0:
            os.makedirs(log_dir, exist_ok=True)
        dist.barrier()
   
    if log_path is None:
        # global fh
        # #get formatted time string to minute
            
        # if fh is None:
        #     ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        #     log_dir = os.path.join(GLOBAL_LOG_DIR, ts)
        #     if not os.path.exists(log_dir):
        #         os.makedirs(log_dir)
        log_path = os.path.join(log_dir, f"log_rank_{rank}.txt")
        
    else:
        log_path = os.path.join(log_dir, log_path)
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path, exist_ok=True)
        # if 
        # fh = open(os.path.join(GLOBAL_LOG_DIR, log_path), mode=mode)
    if log_path not in fhandles:
        fhandles[log_path] = open(log_path, mode=mode)        
    
    fh = fhandles[log_path]
    
    for m in msgs:
        fh.write(f"{m}\n")
    fh.flush()


def print_backward_tensors(tensor):

    def _print_bwd_tensors(grad_fn):
        print(f"Backward tensors in {grad_fn}")
        for funcs in grad_fn.next_functions:
            if funcs[0]:
                try:
                    tensor = getattr(funcs[0], 'variable')
                    print(funcs[0])
                    print(f"Tensor - id: {id(tensor)}, shape: {tensor.shape}, data: {tensor}, grad: {tensor.grad}")
                except AttributeError as e:
                    _print_bwd_tensors(funcs[0])

    if hasattr(tensor, 'grad_fn'):
        _print_bwd_tensors(tensor.grad_fn)
