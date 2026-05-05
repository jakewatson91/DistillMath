import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.distributed.tensor import DTensor
# from torch.nn.parallel import DistributedDataParallel as DDP
import os
import math
# from LiveCodeBench.lcb_runner.runner.parser import get_args
# from LiveCodeBench.lcb_runner.utils.scenarios import Scenario
# from LiveCodeBench.lcb_runner.lm_styles import LanguageModelStore
# from LiveCodeBench.code_test import test_model
import gc
from gsm8k_test import GSM8K_Test

import wandb
from tqdm import tqdm

class Dist_Distiller:
    def __init__(self, train_data_loader: DataLoader,
                 master_ranks: list,
                 isTeacher: bool,
                 isStudent: bool,
                 teacher_groups: list,
                 student_groups: list,
                 broadcast_groups: list,
                 group_num: int,
                 teacher_model: torch.nn.Module,
                 student_model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 grad_accum_steps: int,
                 config: dict):
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        self.device = f'cuda:{self.rank}'
        self.config = config
        self.save_path = self.config['path']['save']
        self.master_ranks, self.isTeacher, self.isStudent = master_ranks, isTeacher, isStudent
        self.teacher_groups, self.student_groups, self.group_num = teacher_groups, student_groups, group_num
        self.broadcast_group = broadcast_groups[self.group_num]
        # sanity check
        if self.isTeacher:
            assert teacher_model is not None, "teacher model is None!!!"
            assert student_model is None, "student model is not None!!!"
            self.teacher_model, self.student_model = teacher_model, None
            self.optimizer, self.grad_accum_steps = None, None
            if self.rank in self.master_ranks:
                assert train_data_loader is not None, "train data loader is None for master rank!!!"
                self.train_data_loader = train_data_loader
            else:
                assert train_data_loader is None, "train data loader is not None for non-master teacher ranks!!!"
                self.train_data_loader = None
        elif self.isStudent:
            assert teacher_model is None, "teacher model is not None!!!"
            assert student_model is not None, "student model is None!!!"
            self.teacher_model, self.student_model = None, student_model
            assert optimizer is not None and grad_accum_steps is not None, "optimizer or grad_accum_steps is None for student ranks!!!"
            self.optimizer, self.grad_accum_steps = optimizer, grad_accum_steps
            assert train_data_loader is None, "train data loader is not None for student ranks!!!"
            self.train_data_loader = None
            self.n_forward, self.step = 0, 0 # record training.
            self.ce_loss_fn = torch.nn.CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-100)
            # Creates a GradScaler for mixed precision training.
            self.scaler = torch.GradScaler()
            self.avg_loss = torch.zeros((1,), device=self.device)
            self.avg_topk_mismatch_loss = torch.zeros((1,), device=self.device)
            self.avg_topk_inner_loss = torch.zeros((1,), device=self.device)
            self.avg_kl_loss = torch.zeros((1,), device=self.device)
            self.avg_ce_loss = torch.zeros((1,), device=self.device)
            self.avg_grad_norm = torch.zeros((1,), device=self.device)
            self.n_token_trained = torch.zeros((1,), device=self.device)
            self.optimizer.zero_grad(set_to_none = True)
            self.avg_loss.zero_()
            self.avg_grad_norm.zero_()
            self.n_token_trained.zero_()
        else:
            raise ValueError("rank not in teacher or student ranks!!!")
        # sanity check end
        # wandb
        if self.rank == dist.get_process_group_ranks(self.student_groups[0])[0]:
            wandb.init(project="Distill", entity="zhiyang")

    def distill(self):
        while True:
            if self.isTeacher:
                if self.rank in self.master_ranks:
                    for epoch in range(0, 2000000000000):
                        print(f'training epoch: {epoch}', flush=True)
                        self.train_data_loader.sampler.set_epoch(epoch)
                        for i, data in tqdm(enumerate(self.train_data_loader)):
                            # dist.barrier(group=self.broadcast_group) # wait for teacher to prepare data and broadcast.
                            x, mask, label4ce = data
                            assert x.shape == mask.shape == label4ce.shape
                            self.teacher_model.eval()
                            batch_len = torch.tensor([x.shape[1]], dtype=torch.long, device=self.device)
                            dist.broadcast(batch_len, src=self.rank, group=self.broadcast_group)
                            # all ranks are prepared to receive data.
                            x, mask, label4ce = x.to(self.device), mask.to(self.device), label4ce.to(self.device)
                            dist.broadcast(x, src=self.rank, group=self.broadcast_group)
                            with torch.no_grad():
                                teacher_output = self.teacher_model(x).logits[..., 0:151936].contiguous()
                            for student_rank in dist.get_process_group_ranks(self.student_groups[self.group_num]):
                                dist.send(tensor=teacher_output, dst=student_rank, tag=0)
                                dist.send(tensor=mask, dst=student_rank, tag=1)
                                dist.send(tensor=label4ce, dst=student_rank, tag=2)
                else:
                    self.teacher_model.eval()
                    batch_len = torch.zeros((1,), dtype=torch.long, device=self.device)
                    dist.broadcast(batch_len, src=dist.get_process_group_ranks(self.teacher_groups[self.group_num])[0], group=self.broadcast_group)
                    x = torch.empty((self.config['distill']['batch_size_per_gpu'], batch_len.item()), dtype=torch.long, device=self.device)
                    dist.broadcast(x, src=dist.get_process_group_ranks(self.teacher_groups[self.group_num])[0], group=self.broadcast_group)
                    with torch.no_grad():
                        teacher_output = self.teacher_model(x).logits.contiguous()
            elif self.isStudent:
                # initial test
                if self.step == 0 and self.n_forward == 0:
                    self.student_model.eval()
                    print('saving model...', flush=True)
                    self.student_model.save_pretrained("/home/zhiyang/projects/distill/saved_model/model")
                    # test model
                    gsm8k_test = GSM8K_Test()
                    print("testing model...", flush=True)
                    test_acc = gsm8k_test.test_pass1(model=self.student_model, device=self.device, batch_size=self.config['distill']['test_batch_size'])
                    print(f'GSM8K Test Accuracy: {test_acc}', flush=True)
                    if self.rank == dist.get_process_group_ranks(self.student_groups[0])[0]:
                        wandb.log({"GSM8K Test": test_acc}, step=self.step, commit=False)
                # continue
                # dist.barrier(group=self.broadcast_group) # wait for teacher to prepare data and broadcast.
                self.student_model.train()
                batch_len = torch.zeros((1,), dtype=torch.long, device=self.device)
                dist.broadcast(batch_len, src=dist.get_process_group_ranks(self.teacher_groups[self.group_num])[0], group=self.broadcast_group)
                x = torch.empty((self.config['distill']['batch_size_per_gpu'], batch_len.item()), dtype=torch.long, device=self.device)
                dist.broadcast(x, src=dist.get_process_group_ranks(self.teacher_groups[self.group_num])[0], group=self.broadcast_group)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    student_output = self.student_model(x).logits
                teacher_output = torch.empty_like(student_output, dtype=torch.bfloat16, requires_grad=False, device=self.device)
                mask = torch.empty_like(x, dtype=torch.long, requires_grad=False, device=self.device)
                label4ce = torch.empty_like(x, dtype=torch.long, requires_grad=False, device=self.device)
                dist.recv(tensor=teacher_output, src=dist.get_process_group_ranks(self.teacher_groups[self.group_num])[0], tag=0)
                dist.recv(tensor=mask, src=dist.get_process_group_ranks(self.teacher_groups[self.group_num])[0], tag=1)
                dist.recv(tensor=label4ce, src=dist.get_process_group_ranks(self.teacher_groups[self.group_num])[0], tag=2)
                # train student...have gotten student_output, teacher_output, mask, label4ce
                # 1. kl divergence loss
                eps = 1e-6
                teacher_probs_no_temp = torch.nn.functional.softmax(teacher_output, dim=-1)
                teacher_probs_temp = torch.nn.functional.softmax(teacher_output / self.config['distill']['temperature'], dim=-1)
                # (i). top-k mismatch loss
                # teacher_topk_probs_no_temp, teacher_topk_idx_no_temp = teacher_probs_no_temp.topk(self.config['distill']['topk'], dim=-1)
                # teacher_topk_probs_no_temp_sum = teacher_topk_probs_no_temp.sum(dim=-1, keepdim=True).clamp(min=eps, max=1-eps)
                # teacher_topk_probs_no_temp_one_minus_sum = 1 - teacher_topk_probs_no_temp_sum
                # teacher_topk_probs_bern_no_temp = torch.concatenate([teacher_topk_probs_no_temp_sum, teacher_topk_probs_no_temp_one_minus_sum], dim=-1)
                # student_probs_no_temp_sum = torch.softmax(student_output, dim=-1).gather(-1, teacher_topk_idx_no_temp).sum(dim=-1, keepdim=True).clamp(min=eps, max=1-eps)
                # student_probs_no_temp_one_minus_sum = 1 - student_probs_no_temp_sum
                # student_probs_bern_no_temp = torch.concatenate([student_probs_no_temp_sum, student_probs_no_temp_one_minus_sum], dim=-1)
                # topk_mismatch_loss = torch.nn.functional.kl_div(torch.log(student_probs_bern_no_temp + eps), teacher_topk_probs_bern_no_temp, reduction='none').sum(dim=-1) * mask
                # topk_mismatch_loss = topk_mismatch_loss.sum() / mask.sum()
                topk_mismatch_loss = torch.zeros((1,), device=self.device)
                # (ii). top-k inner distribution loss
                teacher_topk_probs_temp, teacher_topk_idx_temp = teacher_probs_temp.topk(self.config['distill']['topk'], dim=-1)
                teacher_topk_probs_temp = teacher_topk_probs_temp / teacher_topk_probs_temp.sum(dim=-1, keepdim=True)
                log_student_topk_probs_temp = torch.log_softmax(student_output.gather(-1, teacher_topk_idx_temp) / self.config['distill']['temperature'], dim=-1)
                topk_inner_loss = torch.nn.functional.kl_div(log_student_topk_probs_temp, teacher_topk_probs_temp, reduction='none').sum(dim=-1) * mask
                topk_inner_loss = (topk_inner_loss.sum() / mask.sum()) * (self.config['distill']['temperature'] ** 2)
                # topk_mismatch_loss, topk_inner_loss = torch.zeros((1,), device=self.device), torch.zeros((1,), device=self.device)
                kl_loss = topk_mismatch_loss + topk_inner_loss
                # 2. ce loss
                voc_size = student_output.shape[-1]
                ce_loss = self.ce_loss_fn(student_output.view(-1, voc_size), label4ce.view(-1))
                # ce_loss = torch.zeros((1,), device=self.device)
                # total loss
                total_loss = (self.config['distill']['kl_loss_weight'] * kl_loss + (1 - self.config['distill']['kl_loss_weight']) * ce_loss)
                # total_loss = kl_loss
                total_loss = total_loss / self.grad_accum_steps
                self.avg_loss[0] += total_loss.detach().item()
                self.avg_topk_mismatch_loss[0] += topk_mismatch_loss.detach().item() / self.grad_accum_steps
                self.avg_topk_inner_loss[0] += topk_inner_loss.detach().item() / self.grad_accum_steps
                self.avg_kl_loss[0] += kl_loss.detach().item() / self.grad_accum_steps
                self.avg_ce_loss[0] += ce_loss.detach().item() / self.grad_accum_steps
                self.n_token_trained += mask.sum().item() / 1e6
                self.n_forward += 1
                if self.n_forward % self.grad_accum_steps == 0:
                    # backprop the accumulated gradients
                    total_loss.backward()
                    # self.scaler.scale(total_loss).backward()
                    # self.scaler.unscale_(self.optimizer)
                    # ####### 1. clip grad norm #######
                    # grad_sq_sum = torch.tensor([0.0], device=self.device)
                    # for param in self.student_model.parameters():
                    #     if param.grad is not None:
                    #         grad_sq_sum += torch.sum(param.grad.detach().pow(2))
                    # ####### 2. all-reduce the global grad norm #######
                    # dist.all_reduce(grad_sq_sum, op=dist.ReduceOp.SUM, group=self.student_groups[self.group_num], async_op=False)
                    # global_grad_norm = torch.sqrt(grad_sq_sum)
                    # ####### 3. clip #######
                    # clip_coef = 1.0 / (global_grad_norm + eps)
                    # if clip_coef < 1.0:
                    #     for param in self.student_model.parameters():
                    #         if param.grad is not None:
                    #             param.grad.detach().mul_(clip_coef)
                    global_grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0) # ONLY used under single GPU training!!!
                    self.avg_grad_norm[0] = global_grad_norm.item()
                    self.optimizer.step()
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    self.step += 1
                    self.optimizer.zero_grad(set_to_none = True)
                    # do test
                    test_interval = 20
                    if (self.step % test_interval == 0) and (self.n_forward % self.grad_accum_steps == 0):
                        self.student_model.eval()
                        print('saving model...', flush=True)
                        self.student_model.save_pretrained("/home/zhiyang/projects/distill/saved_model/model")
                        # test model
                        gsm8k_test = GSM8K_Test()
                        print("testing model...", flush=True)
                        test_acc = gsm8k_test.test_pass1(model=self.student_model, device=self.device, batch_size=self.config['distill']['test_batch_size'])
                        print(f'GSM8K Test Accuracy: {test_acc}', flush=True)
                        if self.rank == dist.get_process_group_ranks(self.student_groups[0])[0]:
                            wandb.log({"GSM8K Test": test_acc}, step=self.step, commit=False)
                    # collect training logs
                    if self.rank == dist.get_process_group_ranks(self.student_groups[0])[0]:
                        # wandb.log({"epoch": epoch}, step=step, commit = False)
                        wandb.log({"grad_norm": self.avg_grad_norm[0].item()}, step=self.step, commit = False)
                        # wandb.log({"lr": self.scheduler.get_last_lr()[0]}, step=step, commit = False)
                        wandb.log({"topk_mismatch_loss": self.avg_topk_mismatch_loss[0].item()}, step=self.step, commit=False)
                        wandb.log({"topk_inner_loss": self.avg_topk_inner_loss[0].item()}, step=self.step, commit=False)
                        wandb.log({"kl_loss": self.avg_kl_loss[0].item()}, step=self.step, commit=False)
                        wandb.log({"ce_loss": self.avg_ce_loss[0].item()}, step=self.step, commit=False)
                        wandb.log({"loss": self.avg_loss[0].item()}, step=self.step, commit=False)
                        wandb.log({"n_token_trained(M)": self.n_token_trained[0].item()}, step=self.step, commit=True)
                    self.avg_loss.zero_()
                    self.avg_topk_mismatch_loss.zero_()
                    self.avg_topk_inner_loss.zero_()
                    self.avg_kl_loss.zero_()
                    self.avg_ce_loss.zero_()
                    self.avg_grad_norm.zero_()
                    # if self.step % test_interval == 1:
                    # #     del total_loss, kl_loss, ce_loss, topk_mismatch_loss, topk_inner_loss, x, mask, label4ce, teacher_output, student_output
                    # #     self.student_model.eval()
                    #       print('saving model...', flush=True)
                    #       self.student_model.save_pretrained("/home/zhiyang/projects/distill/saved_model/model")
                    #       # test model
                    #       gsm8k_test = GSM8K_Test()
                    #       print("testing model...", flush=True)
                    #       test_acc = gsm8k_test.test_pass1(model=self.student_model, device=self.device, batch_size=self.config['distill']['test_batch_size'])
                    #       if self.rank == dist.get_process_group_ranks(self.student_groups[0])[0]:
                    #           wandb.log({"GSM8K Test": test_acc}, step=self.step, commit=True)
                    #     torch.cuda.synchronize()
                    #     torch.cuda.empty_cache()
                    #     self.student_model = self.student_model.to('cpu')
                    #     for state in self.optimizer.state.values():
                    #         for k, v in state.items():
                    #             if torch.is_tensor(v):
                    #                 state[k] = v.cpu()
                    #     gc.collect()
                    #     torch.cuda.synchronize()
                    #     torch.cuda.empty_cache()
                    #     if self.rank == dist.get_process_group_ranks(self.student_groups[self.group_num])[0]:
                    #         print('testing model...', flush=True)
                    #         # Print every live GPU tensor and its size
                    #         # for obj in gc.get_objects():
                    #         #     try:
                    #         #         if torch.is_tensor(obj) and obj.is_cuda:
                    #         #             print(type(obj), obj.shape, obj.dtype,
                    #         #                 obj.element_size() * obj.nelement() / 1e6, "MB")
                    #         #     except:
                    #         #         pass
                    #         args = get_args()
                    #         args.scenario = Scenario.codegeneration
                    #         args.evaluation = True
                    #         args.debug = False
                    #         args.not_fast = False
                    #         args.cot_code_execution = False
                    #         args.tensor_parallel_size = 1
                    #         args.gpu_memory_utilization = 0.75
                    #         args.start_date = "2024-07-01"
                    #         args.end_date = "2024-11-30"
                    #         model_style = LanguageModelStore["Qwen/Qwen2.5-Coder-7B-Instruct"]
                    #         test_model(args, model_style,
                    #                 "/home/zhiyang/projects/distill/saved_model/tokenizer",
                    #                 "/home/zhiyang/projects/distill/saved_model/model",
                    #                 "/home/zhiyang/projects/distill/eval_report/eval.json")
                    #         print('test done.', flush=True)
                    #     dist.barrier(group=self.student_groups[self.group_num]) # wait for test to complete.
                    #     self.student_model = self.student_model.to(self.device)
                    #     for state in self.optimizer.state.values():
                    #         for k, v in state.items():
                    #             if torch.is_tensor(v):
                    #                 state[k] = v.cuda()
                    #     self.student_model.train()
                else:
                    # self.scaler.scale(total_loss).backward()
                    total_loss.backward()
                # print(f"Rank {self.rank}: {teacher_output.shape}, {student_output.logits.shape}, {mask.shape}", flush=True)
            else:
                raise ValueError("wrong rank!")