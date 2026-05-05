import os
import numpy as np
import torch
import torch.distributed as dist
import random
import deepspeed

def pipp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # initialize the process group
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    # dist.init_process_group(backend="nccl",
    #                         rank=rank,
    #                         world_size=world_size,)
    # Set random seed
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    # enable fp32
    # torch.set_float32_matmul_precision("high")
    
    # torch._dynamo.config.optimize_ddp = False

def pipp_cleanup():
    dist.destroy_process_group()