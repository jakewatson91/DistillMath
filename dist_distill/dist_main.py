import gc
import os

import torch
from dist_distill.dist_utils import dist_setup, dist_cleanup
from dist_distill.dist_distiller import Dist_Distiller
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataset.data_loader import DistillDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, SequenceParallel, ColwiseParallel, RowwiseParallel
from torch.distributed.tensor import Replicate
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import hf_hub_download, login
from dotenv import load_dotenv
import yaml
import math
from tqdm import tqdm
import numpy as np
import torch.distributed as dist

HF_DATASET_REPO = "jakewatson/mathdistill-traces"
HF_DATASET_FILES = [
    "collect_openai_gsm8k_math.json",
    "collect_meta-math_MetaMathQA_math.json",
]


def _hf_login():
    load_dotenv()
    token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)


def _download_dataset_files():
    return [
        hf_hub_download(repo_id=HF_DATASET_REPO, filename=fn, repo_type="dataset")
        for fn in HF_DATASET_FILES
    ]

def get_teacher_tp_plan():
    return {
        # output_layouts=Replicate() ensures the sharded embedding output 
        # is all-gathered before hitting the first LayerNorm.
        "model.embed_tokens": ColwiseParallel(output_layouts=Replicate()),

        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),

        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),

        # We omit Norm layers; RowwiseParallel already returns replicated 
        # tensors for the next layer's Norm to use.
        
        "lm_head": ColwiseParallel(output_layouts=Replicate())
    }

class pad_and_truncate:
    def __init__(self, pad_token_id, max_len=8192):
        self.pad_id = pad_token_id
        self.max_len = max_len

    def __call__(self, batch):
        max_len = 8192
        batch_max_len = -1
        for item in batch:
            assert item[0].shape == item[1].shape == item[2].shape
            if item[0].shape[0] > batch_max_len: batch_max_len = item[0].shape[0]
        if batch_max_len > max_len: print(f"truncate! from {batch_max_len} to {max_len}") #, data:{batch[0]}")
        batch_align_len = min(batch_max_len, max_len)
        batch_padded = []
        for item in batch:
            if batch_align_len > item[0].shape[0]: # pad to fixed length.
                padding_len = batch_align_len - item[0].shape[0]
                input_padding, mask_padding, label_padding = np.array([self.pad_id] * padding_len), np.array([0] * padding_len), np.array([-100] * padding_len)
                batch_padded.append((np.append(item[0], input_padding)[:-1], np.append(item[1], mask_padding)[1:], np.append(item[2], label_padding)[1:])) # shifted in dataloader
            else: # truncate
                batch_padded.append((item[0][0:batch_align_len][:-1], item[1][0:batch_align_len][1:], item[2][0:batch_align_len][1:])) # shifted in dataloader
        return default_collate(batch_padded)

def dist_main(rank: int, world_size: int, resume: bool):
    print("dist setup...")
    dist_setup(rank, world_size)
    print("dist setup done.")
    _hf_login()
    # configure
    with open("distill_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    teacher_group_list = config['distill']['teacher_group']
    student_group_list = config['distill']['student_group']
    assert len(teacher_group_list) == len(student_group_list), "teacher and student group number mismatch!!!"
    teacher_groups, student_groups, broadcast_groups = [], [], []
    for tg in teacher_group_list: teacher_groups.append(dist.new_group(ranks=tg))
    for sg in student_group_list: student_groups.append(dist.new_group(ranks=sg))
    for tg, sg in zip(teacher_groups, student_groups): # ??? Need to investigate: we must initialize thie group very early, why we can't move it to Dist_Distiller's init function ???
        ranks_to_combine = sorted(set(dist.get_process_group_ranks(tg)) | set(dist.get_process_group_ranks(sg)))
        broadcast_groups.append(dist.new_group(ranks=ranks_to_combine))
    # distinguish identity.
    isTeacher, isStudent = False, False
    group_num = -1
    for i, tg in enumerate(teacher_groups):
        if rank in dist.get_process_group_ranks(tg):
            isTeacher = True
            group_num = i
            break
    for i, sg in enumerate(student_groups):
        if rank in dist.get_process_group_ranks(sg):
            isStudent = True
            group_num = i
            break
    assert isTeacher != isStudent and group_num >= 0, "either teacher or student!!!"
    print(f"Rank {rank} is {'Teacher' if isTeacher else 'Student'} of group {group_num}.")
    # load corresponding models
    teacher_model_name, student_model_name = "Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/Qwen2.5-Math-1.5B"
    teacher_tokenizer, teacher_model, student_tokenizer, student_model = None, None, None, None
    dataloader = None
    if isTeacher:
        print(f"Rank {rank}: Loading Teacher...")
        student_tokenizer, student_model = None, None
        pg_rank_list = dist.get_process_group_ranks(teacher_groups[group_num])
        n_teacher_gpus = len(pg_rank_list)
        # tp_mesh = init_device_mesh("cuda", (n_teacher_gpus,))
        # print(f'{rank} teacher pg ranks: {pg_rank_list}')
        tp_mesh = DeviceMesh("cuda", pg_rank_list)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        teacher_model = parallelize_module(teacher_model, tp_mesh, get_teacher_tp_plan()).to(f'cuda:{rank}')
        teacher_model.to(f'cuda:{rank}')
        teacher_model.eval()
        if rank == pg_rank_list[0]:  # head GPU of the group, master rank of a teacher group.
            print(f"Rank {rank}: Preparing dataset and dataloader for Teacher{group_num}, Total # of teacher: {len(teacher_groups)}")
            collate_fn = pad_and_truncate(teacher_tokenizer.pad_token_id)
            dataset_list = _download_dataset_files()
            dataset = DistillDataset(dataset_list, teacher_tokenizer)
            sampler = DistributedSampler(
                dataset,
                num_replicas=len(teacher_groups),   # one replica per group, not per GPU
                rank=group_num,             # group index, not global rank
                shuffle=True,
            )
            dataloader = DataLoader(dataset,
                                    batch_size=config['distill']['batch_size_per_gpu'],
                                    pin_memory=True,
                                    shuffle=False, # must be False when using DDP
                                    num_workers=4,
                                    drop_last=True,
                                    prefetch_factor=2,
                                    sampler=sampler,
                                    collate_fn=collate_fn)
    elif isStudent:
        print(f"Rank {rank}: Loading Student...")
        teacher_tokenizer, teacher_model = None, None
        pg_rank_list = dist.get_process_group_ranks(student_groups[group_num])
        n_student_gpus = len(pg_rank_list)
        # tp_mesh = init_device_mesh("cuda", (n_student_gpus,))
        # print(f'{rank} student pg ranks: {pg_rank_list}')
        tp_mesh = DeviceMesh("cuda", pg_rank_list)
        student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        student_tokenizer.save_pretrained("saved_model/tokenizer")
        student_model = AutoModelForCausalLM.from_pretrained(student_model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        # student_model = parallelize_module(student_model, tp_mesh, get_teacher_tp_plan())
        student_model.to(f'cuda:{rank}')
        student_model.train()
        # student_optimizer = ZeroRedundancyOptimizer(student_model.parameters(),
        #                                     optimizer_class=torch.optim.AdamW,
        #                                     lr=config['distill']['learning_rate'],
        #                                     betas=(0.9, 0.95),
        #                                     weight_decay=config['distill']['weight_decay'])
        student_optimizer = torch.optim.AdamW(student_model.parameters(),
                                            lr=config['distill']['learning_rate'],
                                            betas=(0.9, 0.95),
                                            weight_decay=config['distill']['weight_decay'],
                                            foreach=False,
                                            fused=False)
    else:
        raise ValueError("wrong rank!")
    # test dataloader
    # max_len = -1
    # pg_rank_list = dist.get_process_group_ranks(teacher_groups[group_num])
    # if rank == pg_rank_list[0]:
    #     for i, data in tqdm(enumerate(dataloader)):
    #         x, mask = data
    #         assert x.shape == mask.shape
    #         if x.shape[1] > max_len: max_len = x.shape[1]
    # print(f'the max length in dataset is {max_len}')
    dist.barrier() # complete model loading.
    print("Model loading complete.")
    ################################
    # while True:
    #     pass
    dist_distiller = Dist_Distiller(train_data_loader=dataloader,
                                    master_ranks=[dist.get_process_group_ranks(tg)[0] for tg in teacher_groups],
                                    isTeacher=isTeacher,
                                    isStudent=isStudent,
                                    teacher_groups=teacher_groups,
                                    student_groups=student_groups,
                                    broadcast_groups=broadcast_groups,
                                    group_num=group_num,
                                    teacher_model=teacher_model,
                                    student_model=student_model,
                                    optimizer=student_optimizer if isStudent else None,
                                    grad_accum_steps=config['distill']['batch_size']//config['distill']['batch_size_per_gpu'],
                                    config=config)
    dist_distiller.distill()
    dist_cleanup()