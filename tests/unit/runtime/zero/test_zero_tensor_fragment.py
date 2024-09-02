# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys

sys.path.insert(0, "/home/jeromeku/DeepSpeed/tests")

import json
import os
from collections import defaultdict

import pytest
import torch
import wandb
from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import SimpleModel, random_dataloader
from unit.util import bf16_required_version_check

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.aio import AsyncIOBuilder
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.utils import (
    safe_get_full_fp32_param,
    safe_get_full_grad,
    safe_get_full_optimizer_state,
    safe_get_local_fp32_param,
    safe_get_local_grad,
    safe_get_local_optimizer_state,
    safe_set_full_fp32_param,
    safe_set_full_optimizer_state,
    safe_set_local_fp32_param,
    safe_set_local_optimizer_state,
)
from deepspeed.utils.debug import log_rank_file

WEIGHT_KEY = "weight"
FIRST_ORDER_KEY = "exp_avg"
SECOND_ORDER_KEY = "exp_avg_sq"


def validate_tensor(model, api_type, opt_states):
    assert api_type in ["full", "local"]
    for _, lp in model.named_parameters():
        param_list = []
        if opt_states:
            param_list.append(
                safe_get_full_optimizer_state(lp, "exp_avg")
                if api_type == "full"
                else safe_get_local_optimizer_state(lp, "exp_avg")
            )
            param_list.append(
                safe_get_full_optimizer_state(lp, "exp_avg_sq")
                if api_type == "full"
                else safe_get_local_optimizer_state(lp, "exp_avg_sq")
            )
        else:
            param_list.append(
                safe_get_full_fp32_param(lp)
                if api_type == "full"
                else safe_get_local_fp32_param(lp)
            )
            param_list.append(
                safe_get_full_grad(lp)
                if api_type == "full"
                else safe_get_local_grad(lp)
            )
        if lp.requires_grad:
            assert all([p is not None for p in param_list])
        else:
            assert all([p is None for p in param_list])


class MyModel(torch.nn.Module):
    def __init__(self, hidden_dim, frozen_weights):
        super(MyModel, self).__init__()
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_dim, 1),
                torch.nn.Linear(1, 1),
                torch.nn.Linear(1, hidden_dim),
            ]
        )
        
        self.act = torch.nn.ReLU()
        self.cel = torch.nn.CrossEntropyLoss()
     
        if frozen_weights:
            self.linears[0].weight.requires_grad = False
            self.linears[0].bias.requires_grad = False

    def forward(self, x, y):
        for l in self.linears:
            x = l(x)
            x = self.act(x)
        return self.cel(x, y)

class DebugModel(torch.nn.Module):
    def __init__(self, hidden_dim, frozen_weights=False, bias=False):
        super().__init__()
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_dim, hidden_dim, bias=bias),
                torch.nn.Linear(hidden_dim, hidden_dim, bias=bias),
            ]
        )
#        self.act = torch.nn.ReLU()
        # self.cel = torch.nn.CrossEntropyLoss()
     
        if frozen_weights:
            self.linears[0].weight.requires_grad = False
            self.linears[0].bias.requires_grad = False

    def forward(self, x):
        for l in self.linears:
            x = l(x)
 #           x = self.act(x)
        return x

def create_grad_dict(grad, stats=["mean", "min", "max"]):
    grad_dict = {}
    for stat in stats:
        grad_dict[stat] = getattr(torch, stat)(grad).item()
    return grad_dict

def init_wandb(project="grads", group_name="test", name="tensor_fragment", entity=None):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    from wandb.sdk.wandb_run import Run
    run: Run = wandb.init(
                project=project,
                group=group_name,
                name=name,
                save_code=False,
                force=False,
                entity=entity,
            )
    return run

def collect_grads(model, step, grad_types=["global"], flatten=True):
    grad_dict = defaultdict(lambda: defaultdict(dict))
    for mod_name, mod in model.named_children():
        for param_name, param in mod.named_parameters():
            #   param_dict = {}
            # if len(param) > 0:
            log_rank_file(dist.get_rank(), f"step {step}: {param_name} {param.ds_summary()}")
           
            if "global" in grad_types:
                global_grads = safe_get_full_grad(param)
                log_rank_file(dist.get_rank(), f"global_grads at step {step}: {param_name} {global_grads}")
                grad_dict[mod_name][param_name]["global"] = global_grads
            if "local" in grad_types:
                local_grads = safe_get_local_grad(param)
                grad_dict[mod_name][param_name]["local"] = local_grads
    return flatten_dict(grad_dict, parent_key=f"rank-{dist.get_rank()}") if flatten else grad_dict

def flatten_dict(d, parent_key=None, sep='/'):
    items = []
    
    if parent_key is None:
        parent_key = f"rank-{dist.get_rank()}"
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def log_grads(grad_dict, step, use_wandb=True):
    if not use_wandb:
        rank = dist.get_rank()
        key = f"{rank=}-{step=}"
        keyed_grads = {key: grad_dict}  # {key: grad_dict}

        msg = json.dumps(keyed_grads, indent=4)
        log_rank_file(rank, msg, log_path=f"rank-{rank}-grads.log")
    else:
        wandb.log(grad_dict)
def log_histogram(grad_dict, step, num_bins=128, use_wandb=True, log_local=True, suffix="grad_hist"):
    grad_hist = {k: wandb.Histogram(v.cpu(), num_bins=num_bins) for k, v in grad_dict.items()}
    hist_data = {k: v.to_json() for k, v in grad_hist.items()}
    if log_local:
        # grad_hist = {k: hist.to_json() for k, hist in grad_hist.items()}
        rank = dist.get_rank()
        # key = f"{rank=}-{step=}"
        # keyed_grads = {key: grad_hist}  # {key: grad_dict}
        msg = json.dumps({step: hist_data}, indent=4)
        log_rank_file(rank, msg, log_path=f"rank-{rank}-{suffix}.log")
    
    if use_wandb:
        wandb.log(grad_hist, step=step)
        wandb.log(hist_data, step=step)

def get_grad_stats(flattened_grad_dict, stats=["mean", "median", "min", "max"]):
    grad_stats = {}
    for k, v in flattened_grad_dict.items():
        grad_stats[k] = { stat: getattr(torch, stat)(v).cpu().item() for stat in stats}
    return grad_stats
        
def run_fragmented_model(
    model, config_dict, hidden_dim, dtype, validate_after_bwd, validate_after_step
):
    model: deepspeed.DeepSpeedEngine
    model, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=config_dict
    )
    rank = dist.get_rank()
    run = init_wandb(project="grads", group_name="debug", name=f"rank-{rank}")


    data_loader = random_dataloader(
        model=model,
        total_samples=25,
        hidden_dim=hidden_dim,
        device=model.device,
        dtype=dtype,
        batch_size=5
    )

    dist.barrier()
    test_batch = next(iter(data_loader))
    y = model(test_batch[0])
    dy = torch.randn_like(y)
    
    for n, batch in enumerate(data_loader):
        loss = model(batch[0])
        model.backward(loss.sum())
        grad_dict = collect_grads(model, step=n, grad_types=["global"])
        print(f"GRAD DICT at {n}: {grad_dict.keys()}")

        log_histogram(grad_dict, step=n, num_bins=128, use_wandb=False, log_local=True)
        # log_histogram(grad_dict, step=n, num_bins=128, use_wandb=False)
        # grad_stats = get_grad_stats(grad_dict)
        # log_grads(grad_stats, step=n, use_wandb=False)
        # log_grads(grad_stats, step=n, use_wandb=True)
        validate_after_bwd(model)
        model.step()
        validate_after_step(model)

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()

def get_named_parameters_by_module(model):
    module_param_dict = [
        {k: [n for n, p in v.named_parameters()]} for k, v in model.named_children()
    ]
    return module_param_dict


# @pytest.mark.parametrize('frozen_weights', [True, False])
class TestTensorFragmentGet(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2
    reuse_dist_env = True

    @pytest.mark.parametrize("api_type", ["local", "full"])
    @pytest.mark.parametrize("zero_stage", [3])
    @pytest.mark.parametrize(
        "offload_device",
        [
            OffloadDeviceEnum.none,
        ],
    )  # OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    def test_zero_fragments(
        self, tmpdir, api_type, zero_stage, offload_device, frozen_weights
    ):
        if offload_device == OffloadDeviceEnum.nvme:
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")
            if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
                pytest.skip("Skip tests since async-io is not compatible")

        if api_type == "local" and zero_stage != 3:
            pytest.skip(
                f"Local APIs only for zero stage 3 but current stage is {zero_stage}"
            )

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 1e-6}},
            "zero_optimization": {
                "stage": zero_stage,
            },
        }
        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 2}
        elif get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}

        if offload_device == OffloadDeviceEnum.cpu:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device
            }
        elif offload_device == OffloadDeviceEnum.nvme:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device,
                "nvme_path": str(tmpdir),
            }

        hidden_dim = 128

        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                model = DebugModel(hidden_dim, frozen_weights=False)
        else:
            model = MyModel(hidden_dim, frozen_weights)

        validate_after_bwd = lambda model: validate_tensor(
            model, api_type, opt_states=False
        )
        validate_after_step = lambda model: validate_tensor(
            model, api_type, opt_states=True
        )

        run_fragmented_model(
            model,
            config_dict,
            hidden_dim,
            preferred_dtype(),
            validate_after_bwd,
            validate_after_step,
        )
        # [{k: [n for n,p in v.named_parameters()]} for k,v in module.named_children()]

    def test_bf16_fragments(self, frozen_weights):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet.")
        if frozen_weights:
            pytest.skip(
                "TODO: Frozen weights not currently supported by BF16 Optimizer"
            )

        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 1e-6}},
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 0,
            },
        }

        hidden_dim = 128
        model = MyModel(hidden_dim, frozen_weights)

        api_type = "full"
        validate_after_bwd = lambda model: validate_tensor(
            model, api_type, opt_states=False
        )
        validate_after_step = lambda model: validate_tensor(
            model, api_type, opt_states=True
        )

        run_fragmented_model(
            model,
            config_dict,
            hidden_dim,
            torch.bfloat16,
            validate_after_bwd,
            validate_after_step,
        )


def create_random_values(model, key_list, group, use_cuda=True):
    param_values = {}
    for n, lp in model.named_parameters():
        param_shape = lp.ds_shape if hasattr(lp, "ds_id") else lp.shape
        param_values[n] = {}
        for key in key_list:
            rand_value = torch.rand(
                param_shape, dtype=torch.float32, device=model.device
            )
            dist.broadcast(rand_value, src=0, group=group)
            param_values[n][key] = rand_value
    return param_values


def set_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, value_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                safe_set_full_fp32_param(lp, value_tensor)
            else:
                safe_set_full_optimizer_state(lp, value_tensor, key)


def validate_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, expected_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                actual_tensor = safe_get_full_fp32_param(lp)
            else:
                actual_tensor = safe_get_full_optimizer_state(lp, key)
            assert torch.equal(expected_tensor, actual_tensor)


def create_random_values_for_local(model, key_list, group, use_cuda=True):
    param_values = {}
    for n, lp in model.named_parameters():
        param_shape = lp.ds_tensor.shape
        param_values[n] = {}
        for key in key_list:
            device = model.device if use_cuda else "cpu"
            rand_value = torch.rand(param_shape, dtype=torch.float32, device=device)
            # dist.broadcast(rand_value, src=0, group=group)
            param_values[n][key] = rand_value
    return param_values


def set_local_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, value_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                safe_set_local_fp32_param(lp, value_tensor)
            else:
                safe_set_local_optimizer_state(lp, value_tensor, key)


def validate_local_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, expected_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                actual_tensor = safe_get_local_fp32_param(lp)
            else:
                actual_tensor = safe_get_local_optimizer_state(lp, key)
            assert torch.equal(expected_tensor, actual_tensor)


helper_funcs_mapping = {
    "full": {
        "create_random_values": create_random_values,
        "set_param_values_with_dict": set_param_values_with_dict,
        "validate_param_values_with_dict": validate_param_values_with_dict,
    },
    "local": {
        "create_random_values": create_random_values_for_local,
        "set_param_values_with_dict": set_local_param_values_with_dict,
        "validate_param_values_with_dict": validate_local_param_values_with_dict,
    },
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
class TestTensorFragmentUpdate(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2
    reuse_dist_env = True

    @pytest.mark.parametrize("api_type", ["local", "full"])
    @pytest.mark.parametrize("zero_stage", [1, 2, 3])
    @pytest.mark.parametrize(
        "offload_device",
        [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme],
    )
    def test_zero_fragments(self, tmpdir, api_type, zero_stage, offload_device, dtype):
        if dtype == torch.bfloat16 and not bf16_required_version_check(
            accelerator_check=False
        ):
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if api_type == "local" and zero_stage != 3:
            pytest.skip(
                f"Local APIs only for zero stage 3 but current stage is {zero_stage}"
            )

        if offload_device == OffloadDeviceEnum.nvme:
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")
            if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
                pytest.skip("Skip tests since async-io is not compatible")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 1e-6}},
            "zero_optimization": {
                "stage": zero_stage,
            },
        }

        if offload_device == OffloadDeviceEnum.cpu:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device
            }
        elif offload_device == OffloadDeviceEnum.nvme:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device,
                "nvme_path": str(tmpdir),
            }

        if dtype == torch.float16:
            if not get_accelerator().is_fp16_supported():
                pytest.skip("fp16 is not supported")
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        hidden_dim = 128
        if zero_stage == 3:
            config_dict["zero_optimization"]["param_persistence_threshold"] = hidden_dim
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                model = SimpleModel(hidden_dim)
        else:
            model = SimpleModel(hidden_dim)

        world = dist.get_world_size()
        group = dist.new_group(ranks=list(range(world)))

        dist.barrier()

        def validate_func(model):
            optim_keys = [WEIGHT_KEY, FIRST_ORDER_KEY, SECOND_ORDER_KEY]
            helper_funcs = helper_funcs_mapping[api_type]
            optim_state_values = helper_funcs["create_random_values"](
                model,
                optim_keys,
                group,
                use_cuda=offload_device == OffloadDeviceEnum.none,
            )
            helper_funcs["set_param_values_with_dict"](model, optim_state_values)
            helper_funcs["validate_param_values_with_dict"](model, optim_state_values)

        run_fragmented_model(
            model, config_dict, hidden_dim, dtype, lambda _: None, validate_func
        )
        

def test_wandb(project="grads", group_name="test", name="tensor_fragment", entity=None):
    run = init_wandb(project=project, group_name=group_name, name=name, entity=entity)
    hidden_dim = 128
    dtype = torch.float32
    model = DebugModel(128, False).to(device="cuda")
    
    data_loader = random_dataloader(
        model=model,
        total_samples=5,
        hidden_dim=hidden_dim,
        device="cuda",
        dtype=dtype,
        batch_size=1,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    batch = next(iter(data_loader))
    x = batch[0]
    y = model(x)
    dy = torch.randn_like(y)
    
    for n, batch in enumerate(data_loader):
        y = model(batch[0])
        #model.backward(loss)
        y.backward(dy)
        grad_dict = collect_grads(model, stats=None)
        flattened_grads = flatten_dict(grad_dict, parent_key = "rank0", sep="/")
        grad_vals = {k: v.cpu() for k, v in flattened_grads.items()}
        
        grad_hist = {k: wandb.Histogram(v, num_bins=128) for k, v in grad_vals.items()}
        # print(f"grad dict {n} = {json.dumps(grad_dict, indent=4)}")
        # print(f"flattened grads {n} = {json.dumps(flattened_grads, indent=4)}")   
        json_hist = {k: v.to_json() for k, v in grad_hist.items()}
        print(f"grad hist {n} = {json.dumps(json_hist, indent=4)}")     
        wandb.log(grad_hist, step=n)
        optimizer.step()
    run.finish()
    print(run.history())
    print(run.summary)
    print(run.config)

if __name__ == "__main__":
    # test_wandb(group_name="clean", name="histogram")
    test_suite = TestTensorFragmentGet()

    with open("test_tensor_fragment_get.log", "w") as tmpdir:
        test_suite.test_zero_fragments(
            tmpdir=tmpdir,
            api_type="full",
            zero_stage=3,
            offload_device=OffloadDeviceEnum.none,
            frozen_weights=False,
        )
