import os
import shutil

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
)
from torch.nn.utils import clip_grad_norm_
from rouge import Rouge
from sacrebleu.metrics import BLEU
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import math
import numpy as np
import logging

from .data_process import MyDataset
from .utils import get_writer, get_logger

logger = get_logger('log.txt')

rouge = Rouge()
bleu = BLEU()
class T5ForFewshotGeneration(nn.Module):
    def __init__(self, args):
        super(T5ForFewshotGeneration, self).__init__()
        self.args = args
        self.tkn = T5Tokenizer.from_pretrained(args.model_name_or_path, max_model_length=args.max_source_length)
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).cuda()

    def forward(self, input_ids, attention_mask):        
        return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

    def finetune(self, train_dataset=None, dev_dataset=None, pretrain=True, save_measure=None):
        args = self.args
        writer = get_writer(args.logdir)
        model = self.model
        tkn = self.tkn
        
        summarize_prefix = self.tkn.encode('summarize:', return_tensors='pt', add_special_tokens=False).cuda()
        expand_prefix = self.tkn.encode('expand:', return_tensors='pt', add_special_tokens=False).cuda()

        num_update_steps_per_epoch = math.ceil(
            len(train_dataset) / args.gradient_accumulation_steps
        )
        if args.max_train_steps > 0:
            args.num_train_epochs = math.ceil(
                args.max_train_steps / num_update_steps_per_epoch
            )
        else:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

        optimizer = AdamW(
            params=self.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
            amsgrad=False,
        )
        if args.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.max_train_steps)
        elif args.scheduler_type == 'linear':
            scheduler = LinearLR(optimizer, start_factor=1., end_factor=0.)
        else:
            raise ValueError("scheduler_type must be cosine or linear")

        global_steps = 0
        if args.ckp_path is not None:
            self._load(args.ckp_path, optimizer, scheduler)
            pretrain = False
            global_steps = args.init_step

        if args.use_amp:
            scaler = GradScaler(init_scale=args.scale_loss)
        
        gen_kwargs_a2b = {
            "max_length": args.max_target_length,
            "length_penalty": 1.0,
            "early_stopping": True,
        }
        gen_kwargs_b2a = {
            "max_length": args.max_source_length,
            "length_penalty": 0.0,
            "early_stopping": True,
        }

        self.log("********** Running training **********")
        self.log(f"  Num examples = {len(train_dataset.data)}")
        self.log(f"  Num Epochs = {args.num_train_epochs}")
        self.log(f"  Instantaneous train batch size = {args.train_batch_size}")
        self.log(f"  Instantaneous eval batch size = {args.eval_batch_size}")
        self.log(f"  Total train batch size (w. accumulation) = {total_batch_size}")
        self.log(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        self.log(f"  Total optimization steps = {args.max_train_steps}")
        
        self.gold_batches = train_dataset.get_batches(args.pretrain_steps)
        train_dataset.shuffle(seed=args.seed)
        # temperature = 1.0
        if pretrain:
            self.log("********** Pretraining **********")
            self.train()
            for step, batch in enumerate(self.gold_batches):
                with autocast(args.use_amp):
                    source_ids, source_mask, source_label, target_ids, target_mask, target_label = batch
                    inputs = self.prefixed_samples(expand_prefix, source_ids, source_mask, source_label)
                    loss_idt_a = model(**inputs)[0]
                    inputs = self.prefixed_samples(summarize_prefix, target_ids, target_mask, target_label)
                    loss_idt_b = model(**inputs)[0]
                    inputs = self.prefixed_samples(summarize_prefix, source_ids, source_mask, target_label)
                    loss_gold_b = model(**inputs)[0]
                    inputs = self.prefixed_samples(expand_prefix, target_ids, target_mask, source_label)
                    loss_gold_a = model(**inputs)[0]

                    inputs = self.prefixed_samples(summarize_prefix, source_ids, source_mask, None)
                    fake_tgt_ids = model.generate(
                        inputs=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        **gen_kwargs_a2b,
                    )
                    inputs = self.prefixed_samples(expand_prefix, target_ids, target_mask, None)
                    fake_src_ids = model.generate(
                        inputs=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        **gen_kwargs_b2a,
                    )
                    inputs = self.prefixed_samples(expand_prefix, fake_tgt_ids, self.gen_mask(fake_tgt_ids), source_label)
                    loss_cyc_a = model(**inputs)[0]
                    inputs = self.prefixed_samples(summarize_prefix, fake_src_ids, self.gen_mask(fake_src_ids), target_label)
                    loss_cyc_b = model(**inputs)[0]

                    loss_gold = loss_gold_a + loss_gold_b
                    loss_total = (loss_idt_a + loss_idt_b) * args.lambda_idt + (loss_cyc_a + loss_cyc_b) * args.lambda_cyc
                    loss = loss_gold + loss_total
                    loss /= args.gradient_accumulation_steps

                if args.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (
                    (step+1) % args.gradient_accumulation_steps == 0
                    or step == len(train_dataset) - 1
                ):
                    clip_grad_norm_(self.parameters(), args.max_grad_norm)
                    if args.use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    losses = {
                        'total': loss_total.item(),
                        'gold_a': loss_gold_a.item(),
                        'gold_b': loss_gold_b.item(),
                        'idt_a': loss_idt_a.item(),
                        'idt_b': loss_idt_b.item(),
                        'cyc_a': loss_cyc_a.item(),
                        'cyc_b': loss_cyc_b.item(),
                    }
                    for k, v in losses.items():
                        writer.add_scalar(f"pretrain/{k}", v, step//args.gradient_accumulation_steps)
            
            output_dir = os.path.join(args.output_dir, "pretrain")
            os.makedirs(output_dir, exist_ok=True)
            self._save(output_dir, optimizer, scheduler)
            self.log("********** Pretraining finished**********")

        progress_bar = tqdm(range(args.max_train_steps))

        gold_idx = 0
        last_step = global_steps
        progress_bar.update(last_step)
        found_last = last_step == 0
        
        best_score = 0.
        saves = []
        for _ in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataset):
                if not found_last:
                    if step / args.gradient_accumulation_steps < last_step:
                        continue
                    else:
                        found_last = True
                self.train()

                with autocast(args.use_amp):
                    source_ids, source_mask, source_label, target_ids, target_mask, target_label = batch
                    inputs = self.prefixed_samples(expand_prefix, source_ids, source_mask, source_label)
                    loss_idt_a = model(**inputs)[0]
                    inputs = self.prefixed_samples(summarize_prefix, target_ids, target_mask, target_label)
                    loss_idt_b = model(**inputs)[0]

                    inputs = self.prefixed_samples(summarize_prefix, source_ids, source_mask, None)
                    fake_tgt_ids = model.generate(
                        inputs=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        **gen_kwargs_a2b,
                    )
                    inputs = self.prefixed_samples(expand_prefix, target_ids, target_mask, None)
                    fake_src_ids = model.generate(
                        inputs=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        **gen_kwargs_b2a,
                    )
                    inputs = self.prefixed_samples(expand_prefix, fake_tgt_ids, self.gen_mask(fake_tgt_ids), source_label)
                    loss_cyc_a = model(**inputs)[0]
                    inputs = self.prefixed_samples(summarize_prefix, fake_src_ids, self.gen_mask(fake_src_ids), target_label)
                    loss_cyc_b = model(**inputs)[0]

                    loss_total = (loss_idt_a + loss_idt_b) * args.lambda_idt + (loss_cyc_a + loss_cyc_b) * args.lambda_cyc
                    loss = loss_total

                    if (global_steps+1) % args.gold_steps == 0:
                        source_ids, source_mask, source_label, target_ids, target_mask, target_label = self.gold_batches[gold_idx]
                        gold_idx = (gold_idx + 1) % len(self.gold_batches)
                        inputs = self.prefixed_samples(summarize_prefix, source_ids, source_mask, target_label)
                        loss_gold_b = model(**inputs)[0]
                        inputs = self.prefixed_samples(expand_prefix, target_ids, target_mask, source_label)
                        loss_gold_a = model(**inputs)[0]
                        loss_gold = loss_gold_a + loss_gold_b
                        loss += loss_gold
                    loss /= args.gradient_accumulation_steps

                if args.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (
                    (step+1) % args.gradient_accumulation_steps == 0
                    or step == len(train_dataset) - 1
                ):
                    if args.use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    progress_bar.update(1)
                    global_steps += 1
                    writer.add_scalar("train/total", loss_total.item(), global_steps)
                    writer.add_scalar("train/idt_a", loss_idt_a.item(), global_steps)
                    writer.add_scalar("train/idt_b", loss_idt_b.item(), global_steps)
                    writer.add_scalar("train/cyc_a", loss_cyc_a.item(), global_steps)
                    writer.add_scalar("train/cyc_b", loss_cyc_b.item(), global_steps)
                    
                    if global_steps % args.gold_steps == 0:
                        writer.add_scalar("train/gold_a", loss_gold_a.item(), global_steps)
                        writer.add_scalar("train/gold_b", loss_gold_b.item(), global_steps)
                        
                    if args.logging_steps > 0 and global_steps% args.logging_steps == 0 :
                        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_steps)
                        writer.add_scalar("loss", loss.item(), global_steps)
                        self.log(
                            "global_steps {} - lr: {:.10f}  loss: {:.8f}".format(
                                global_steps,
                                optimizer.param_groups[0]['lr'],
                                loss.item(),
                            )
                        )

                    if args.save_steps > 0 and global_steps % args.save_steps == 0:
                        self.log("********** Running evaluating **********")
                        self.log(f"********** Step {global_steps} **********")
                        output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                        os.makedirs(output_dir, exist_ok=True)
                        self._save(output_dir, optimizer, scheduler)
                        saves.append(output_dir)
                        if len(saves) > args.num_saves_kept:
                            shutil.rmtree(saves.pop(0))
                            
                        eval_results = self.evaluate(dev_dataset, args.num_eval_batch)
                        scores = list(eval_results.values())
                        if save_measure is None:
                            score = np.mean(scores)
                        else:
                            measures = []
                            for i in save_measure:
                                measures.append(scores[i])
                            score = np.mean(measures)
                            
                        self.debug(eval_results)
                        self._save_metric(output_dir, eval_results)
                        
                        if score > best_score:
                            self.log(f"********** Saving best result in step {global_steps} **********")
                            best_score = score
                            best_dir = os.path.join(args.output_dir, "best")
                            os.makedirs(best_dir, exist_ok=True)
                            self._save(best_dir, optimizer, scheduler)
                            self._save_metric(best_dir, eval_results)
                            
                        for k, v in eval_results.items():
                            writer.add_scalar(f"eval/{k}", v, global_steps)
                        self.log("********** Evaluating Done **********")

                if global_steps >= args.max_train_steps:
                    self.log("********** Running evaluating **********")
                    self.log(f"********** Step {global_steps} **********")
                    output_dir = os.path.join(args.output_dir, f"final-step-{global_steps}")
                    os.makedirs(output_dir, exist_ok=True)
                    self._save(output_dir, optimizer, scheduler)
                    eval_results = self.evaluate(dev_dataset, -1)
                    self.debug(eval_results)
                    self._save_metric(output_dir, eval_results)
                    for k, v in eval_results.items():
                        writer.add_scalar(f"eval/{k}", v, global_steps)
                    self.log("********** Evaluating Done **********")
                    self.log("********** Training Done **********")
                    return


    @torch.no_grad()
    def evaluate(self, data_loader=None, num_batch=100, split='validation'):
        if data_loader is None:
            data_loader = MyDataset(self.tkn, self.args.eval_batch_size, split=split)
           
        summarize_prefix = self.tkn.encode('summarize:', return_tensors='pt', add_special_tokens=False).cuda()
        self.eval()
        model = self.model
        tkn = self.tkn

        gen_kwargs = {
            "max_length": self.args.max_target_length,
            "num_beams": self.args.num_beams,
            "length_penalty": 1.0,
            "early_stopping": True,
        }
        decoded_preds = []
        decoded_labels = []        
        progress_bar = tqdm(range(num_batch if num_batch > 0 else len(data_loader)))
        for i, batch in enumerate(data_loader):
            if num_batch > 0 and i >= num_batch:
                break
            source_ids, source_mask, _, target_ids, _, target_label = batch
            inputs = self.prefixed_samples(summarize_prefix, source_ids, source_mask, None)
            generated_tokens = model.generate(
                inputs=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **gen_kwargs,
            )
            labels = np.where(target_label.cpu().numpy() != -100, target_label.cpu().numpy(), tkn.pad_token_id)

            decoded_preds.extend(tkn.batch_decode(generated_tokens.cpu().numpy(), skip_special_tokens=True))
            decoded_labels.extend(tkn.batch_decode(labels, skip_special_tokens=True))
            progress_bar.update(1)

        self.debug(decoded_preds[:3])
        self.debug(decoded_labels[:3])
        if self.args.is_zh:
            f = lambda s: ' '.join(s)
            decoded_preds = list(map(f, decoded_preds))
            decoded_labels = list(map(f, decoded_labels))

        scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        for k in scores.keys():
            scores[k] = scores[k]['f'] * 100
        scores['bleu'] = bleu.corpus_score(decoded_preds, [decoded_labels]).score
        # return decoded_preds, decoded_labels
        return scores
    
    @torch.no_grad()
    def gen_mask(self, ids):
        mask = torch.zeros_like(ids, dtype=torch.int64)
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                if ids[i,j] == self.tkn.pad_token_id:
                    break
                mask[i,j] = 1
        return mask

    @torch.no_grad()
    def prefixed_samples(self, prefix, ids, mask, labels):
        outputs = {
            'input_ids': torch.cat(
                [
                    prefix.repeat(ids.shape[0], 1), 
                    ids
                ],
                dim=-1
            ).cuda(),
            'attention_mask': torch.cat(
                [
                    torch.ones(ids.shape[0], prefix.shape[-1]).cuda(), 
                    mask
                ],
                dim=-1
            ).cuda(),
            'labels': labels.cuda() if labels is not None else labels
        }
        return outputs

    def _save(self, output_dir, optimizer, scheduler):
        state = {
            'model': self.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(output_dir, "model.pth"))
        return
        
    def _load(self, ckp_dir, optimizer, scheduler):
        state = torch.load(os.path.join(ckp_dir, "model.pth"))
        self.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        return
    
    def _save_metric(self, output_dir, eval_results):
        with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
            f.write(str(eval_results))
        return
    
    def debug(self, var):
        print(var)
        logger.debug(var)

    def log(self, info):
        print(info)
        logger.info(info)