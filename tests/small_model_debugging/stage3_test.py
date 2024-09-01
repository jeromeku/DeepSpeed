# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

import deepspeed
from deepspeed.utils.debug import log_rank_file
import deepspeed.comm as dist
from deepspeed.utils.logging import log_dist
###################################
# Setup
###################################
from deepspeed.runtime.zero import partition_parameters

partition_parameters.print_rank_0
INPUT_DIM = HIDDEN_DIM = OUTPUT_DIM = 4

class VerboseLinear(torch.nn.Linear):

    def __init__(self, **kwargs):
        print(f'Begin VerboseLinear.__init__')
        super().__init__(**kwargs)
        print(f'End VerboseLinear.__init__')


class LinearStack(torch.nn.Module):

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_layer = VerboseLinear(in_features=self.input_dim, out_features=self.hidden_dim, bias=False)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)
            for x in range(num_layers)
        ])
        self.output_layer = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim, bias=False)
        # self.identity = torch.nn.Identity()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        # x = self.identity(x)
        return x


###################################
# DRIVER
###################################


def test_driver():
    print()
    print('BUILDING MODEL')
    with deepspeed.zero.Init():
        model = LinearStack()
    print()

    # parted = [name for (name, p) in model.named_parameters() if p._partitioned]
    # not_parted = [name for (name, p) in model.named_parameters() if not p._partitioned]
    # print('partitioned: ', parted)
    # print('full: ', not_parted)
    # print()
    ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 500000000
            },
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "fp16": {
                "enabled": True,
                # "auto_cast": false,
                "loss_scale": 0,
                "initial_scale_power": 0,
                # "loss_scale_window": 1000,
                # "hysteresis": 2,
                # "consecutive_hysteresis": false,
                # "min_loss_scale": 1
}
        }
    
    # breakpoint()
    out = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())
    print(f"DEBUG: {out}")
    model: deepspeed.DeepSpeedEngine = out[0]
    optimizer = out[1]
    # model.train()
    
    print(f"DEBUG: {type(model.optimizer)}")
    if dist.is_initialized():
        device = f"cuda:{dist.get_rank()}"
    else:
        device = "cuda"
    print(f"DEBUG: Is dist initialized: {dist.is_initialized()} {dist.get_world_size()} {dist.get_rank()}")
    print(f"DEBUG: {model.optimizer.fp16_groups}")        
    test_input = torch.randn(1, model.input_dim, dtype=torch.half, device=device)
    grad_output = torch.randn(1, model.output_dim, dtype=torch.half, device=device)

    grad_output.requires_grad = False
    test_input.requires_grad = False

    print()
    print('BEGINNING FORWARD')
    print()
    # breakpoint()
    
    for _ in range(5):
        model.tput_timer.start()
        output = model(test_input)    
        print(f"DEBUG OUTPUT: {output.abs().max()} {output.sum()}")
        model.backward(output.sum())
        print(f"DEBUG GRAD_NORM: {model.optimizer._get_norm_groups()}")
        print(f"DEBUG: {model.optimizer.averaged_gradients}")
        print(f"DEBUG: AVERAGED GRADIENTS: {model.optimizer.averaged_gradients}")
        model.step()
     
        model.tput_timer.stop(global_step=True)
    # print(f"THROUGHPUT: {model.tput_timer.avg_samples_per_sec():.3f} samples/sec")
    
    # parted = [name for (name, p) in model.named_parameters() if p._partitioned]
    # not_parted = [name for (name, p) in model.named_parameters() if not p._partitioned]
    # print('partitioned: ', parted)
    # print('full:' , not_parted)
    # print()

    #samyamspeed.disable()


test_driver()
