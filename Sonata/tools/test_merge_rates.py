"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.hooks import TESTERS
# from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch
import torch.distributed as dist
from pointcept.utils.logger import get_root_logger

from pointcept.engines.test import SemSegTester
from pointcept.utils.logger import logger_initialized
from pointcept.utils.logger import close_logger

import numpy as np
import math
import os

def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = SemSegTester(cfg)
    tester.test()
    logger_initialized.clear()
    close_logger(tester.logger)

def wait_for_previous_worker():
    """
    Wait until the previous launch worker has finished.
    This function uses a distributed barrier to synchronize all processes.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    dist.barrier()

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    
    # rates = list(np.arange(0.5, 0.96, 0.05))
    rates = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    save_path = cfg.save_path
    
    for rate in rates:
        stride = int(math.ceil(1 / (1.0-rate)))
        cfg.model.backbone["additional_info"]["r"] = rate
        cfg.model.backbone["additional_info"]["stride"] = stride    
        cfg.save_path = os.path.join(save_path, f"rate_{rate:.2f}_stride_{stride}")
        os.makedirs(cfg.save_path, exist_ok=True)
        
        # wait_for_previous_worker()
        print(f"Testing with rate: {rate}, stride: {stride}")
        main_worker(cfg)
        
        

if __name__ == "__main__":
    main()
