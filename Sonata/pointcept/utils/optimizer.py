"""
Optimizer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")

# Only enable param with keyword in finetuning_param_list to have gradients
def filter_grads(model, finetuning_param_list):
    for n, p in model.named_parameters():
        flag = False
        for i in range(len(finetuning_param_list)):
            if finetuning_param_list[i] in n:
                p.requires_grad = True
                flag = True
                break
        if not flag:
            p.requires_grad = False
            print(n)
    return model

def build_optimizer(cfg, model, param_dicts=None, finetuning_param_list=None):
    if param_dicts is None:
        if finetuning_param_list is not None:
            cfg.params = model.parameters()
        else:
            filter_grads(model, finetuning_param_list)
            cfg.params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        # Filter gradients
        if finetuning_param_list is not None:
            filter_grads(model, finetuning_param_list)
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]

        # Add param_dicts to cfg.params
        for i in range(len(param_dicts)):
            param_group = dict(names=[], params=[])
            if "lr" in param_dicts[i].keys():
                param_group["lr"] = param_dicts[i].lr
            if "momentum" in param_dicts[i].keys():
                param_group["momentum"] = param_dicts[i].momentum
            if "weight_decay" in param_dicts[i].keys():
                param_group["weight_decay"] = param_dicts[i].weight_decay
            cfg.params.append(param_group)

        # Assign parameters to different groups
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i].keyword in n:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

        # Print the parameters
        logger = get_root_logger()
        for i in range(len(cfg.params)):
            param_names = cfg.params[i].pop("names")
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            logger.info(f"Params Group {i+1} -{message} Params: {param_names}.")
    return OPTIMIZERS.build(cfg=cfg)
