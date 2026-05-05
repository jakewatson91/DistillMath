import torch
import torch.multiprocessing as mp
from dist_distill.dist_main import dist_main
import os
import random
import numpy as np
import argparse

# os.environ["WANDB_MODE"] = "disabled"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" # MUST use one device.
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "3600"  # seconds

def get_args():
    parser = argparse.ArgumentParser(description="Distill ability from large model to small model.")

    parser.add_argument("--resume", dest="resume", action="store_true", help="Continue distillation.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    args = get_args()

    world_size = torch.cuda.device_count()
    print(f"detected {world_size} GPUs.")
    print("spawn processes...")

    mp.spawn(dist_main, args=(world_size, args.resume), nprocs=world_size)
    # dist_main(world_size, False)