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
class T5Normal(nn.Module):
    def __init__(self, args):
        super(T5Normal, self).__init__()
        self.args = args
        self.tkn = T5Tokenizer.from_pretrained(args.model_name_or_path, max_model_length=args.max_source_length)
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).cuda()

    def forward(self, input_ids, attention_mask):        
        return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

    def finetune(self, train_dataset=None, dev_dataset=None):
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
        self.log("********** Pretraining **********")
        self.train()
        for step, batch in enumerate(self.gold_batches):
            with autocast(args.use_amp):
                source_ids, source_mask, source_label, target_ids, target_mask, target_label = batch
                inputs = self.prefixed_samples(summarize_prefix, source_ids, source_mask, target_label)
                loss = model(**inputs)[0]
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
                writer.add_scalar("loss", loss.item(), step//args.gradient_accumulation_steps)

        output_dir = os.path.join(args.output_dir, "pretrain")
        os.makedirs(output_dir, exist_ok=True)
        self._save(output_dir, optimizer, scheduler)
        self.log("********** Pretraining finished**********")
        eval_results = self.evaluate(dev_dataset, -1)
        score = np.mean(list(eval_results.values()))
        self.debug(eval_results)
        self._save_metric(output_dir, eval_results)
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
            scores[k] = scores[k]['f']
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