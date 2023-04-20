import os
import paddle
import paddle.nn as nn
# import paddle.nn.functional as F
from paddle.amp import GradScaler
from paddle.optimizer import AdamW
from paddlenlp.metrics import Rouge1, Rouge2, RougeL, BLEU
from paddlenlp.transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import math
import numpy as np
from paddlehub.utils.log import logger
import logging

from .data_process import (
    get_train_dataloader,
    get_dev_dataloader,
)
from .utils import get_writer, get_scheduler
from .layers import EmbeddingLayer
# from .t5 import T5ForConditionalGeneration

locallogger = logging.getLogger(__name__)
locallogger.setLevel(logging.DEBUG)
logfile = 'log.txt'
fh = logging.FileHandler(logfile, mode='a')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
locallogger.addHandler(fh)

rouge1 = Rouge1()
rouge2 = Rouge2()
class T5ForUnsupervisedGeneration(nn.Layer):
    def __init__(self, args):
        super(T5ForUnsupervisedGeneration, self).__init__()
        self.args = args
        self.tkn = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        emb = model.get_input_embeddings()
        emb = EmbeddingLayer(emb)
        self.emb = emb
        model.set_input_embeddings(emb)
        self.t5_a2b = model
        self.encoder = model.get_encoder()
        self.decoder_b = model.get_decoder()
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model.set_input_embeddings(emb)
        model.t5.encoder = self.encoder
        self.t5_b2a = model
        self.decoder_a = model.get_decoder()
        # self.discriminator = Discriminator()

    def forward(self, input_ids, attention_mask):        
        return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

    def finetune(self, pretrain=True):
        args = self.args
        writer = get_writer(args.logdir)
        t5_a2b, t5_b2a = self.t5_a2b, self.t5_b2a
        tkn = self.tkn
        # get dataloader
        train_dataloader = get_train_dataloader(tkn, args)
        dev_dataloader = get_dev_dataloader(tkn, args)
        self.train_dl = train_dataloader
        self.dev_dl = dev_dataloader

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps > 0:
            args.num_train_epochs = math.ceil(
                args.max_train_steps / num_update_steps_per_epoch
            )
        else:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

        scheduler = get_scheduler(args.learning_rate, args.scheduler_type, args.warmup_steps, args.max_train_steps)
        optimizer = AdamW(
            learning_rate=scheduler,
            beta1=0.9,
            beta2=0.999,
            epsilon=args.adam_epsilon,
            parameters=self.parameters(),
            grad_clip = paddle.nn.ClipGradByNorm(clip_norm=args.max_grad_norm)
        )

        global_steps = 0
        if args.ckp_path is not None:
            self._load(args.ckp_path, optimizer)
            scheduler = optimizer._learning_rate
            pretrain = False
            global_steps = args.init_step


        if args.use_amp:
            scaler = GradScaler(init_loss_scaling=args.scale_loss)
        
        gen_kwargs_a2b = {
            "max_length": args.max_target_length,
            "length_penalty": 1.0,
            "early_stopping": True,
            # "repetition_penalty": 1.5,
            # "num_beams": 2,
            "decode_strategy": "greedy_search"#"beam_search"
        }
        gen_kwargs_b2a = {
            "max_length": args.max_source_length,
            "length_penalty": 1.0,
            # "repetition_penalty": 1.5,
            "early_stopping": True,
            # "num_beams": 2,
            "decode_strategy": "greedy_search"#"beam_search"
        }

        self.log("********** Running training **********")
        self.log(f"  Num examples = {len(train_dataloader.dataset)}")
        self.log(f"  Num Epochs = {args.num_train_epochs}")
        self.log(f"  Instantaneous train batch size = {args.train_batch_size}")
        self.log(f"  Instantaneous eval batch size = {args.eval_batch_size}")
        self.log(f"  Total train batch size (w. accumulation) = {total_batch_size}")
        self.log(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        self.log(f"  Total optimization steps = {args.max_train_steps}")
        
        self.gold_batches = []
        for step, batch in enumerate(train_dataloader):
            if step >= args.pretrain_steps:
                break
            self.gold_batches.append(batch)
        # temperature = 1.0
        if pretrain:
            self.log("********** Pretraining **********")
            self.train()

            for step, batch in enumerate(self.gold_batches):
                if step >= args.pretrain_steps:
                    break
                source_ids, source_mask, source_label, target_ids, target_mask, target_label = batch

                loss_idt_a = t5_b2a(input_ids=source_ids, attention_mask=source_mask, labels=source_label)[0]
                loss_idt_b = t5_a2b(input_ids=target_ids, attention_mask=target_mask, labels=target_label)[0]
                loss_gold_b = t5_a2b(input_ids=source_ids, attention_mask=source_mask, labels=target_label)[0]
                loss_gold_a = t5_b2a(input_ids=target_ids, attention_mask=target_mask, labels=source_label)[0]
                fake_tgt_ids = t5_a2b.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    **gen_kwargs_a2b,
                )[0]
                fake_src_ids = t5_b2a.generate(
                    input_ids=target_ids,
                    attention_mask=target_mask,
                    **gen_kwargs_b2a,
                )[0]
                loss_cyc_a = t5_b2a(input_ids=fake_tgt_ids, attention_mask=self.gen_mask(fake_tgt_ids), labels=source_label)[0]
                loss_cyc_b = t5_a2b(input_ids=fake_src_ids, attention_mask=self.gen_mask(fake_src_ids), labels=target_label)[0]

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
                    or step == len(train_dataloader) - 1
                ):
                    if args.use_amp:
                        scaler.minimize(optimizer, loss)
                    else:
                        optimizer.minimize(loss)
                    optimizer.clear_grad()
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
            self._save(output_dir, optimizer)
            self.log("********** Pretraining finished**********")

        progress_bar = tqdm(range(args.max_train_steps))

        gold_idx = 0
        last_step = global_steps
        progress_bar.update(last_step)
        found_last = False
        for _ in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                if not found_last:
                    if step / args.gradient_accumulation_steps < last_step:
                        continue
                    else:
                        found_last = True
                self.train()

                source_ids, source_mask, source_label, target_ids, target_mask, target_label = batch
                
                loss_idt_a = t5_b2a(input_ids=source_ids, attention_mask=source_mask, labels=source_label)[0]
                loss_idt_b = t5_a2b(input_ids=target_ids, attention_mask=target_mask, labels=target_label)[0]
                fake_tgt_ids = t5_a2b.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    **gen_kwargs_a2b,
                )[0]
                fake_src_ids = t5_b2a.generate(
                    input_ids=target_ids,
                    attention_mask=target_mask,
                    **gen_kwargs_b2a,
                )[0]
                loss_cyc_a = t5_b2a(input_ids=fake_tgt_ids, attention_mask=self.gen_mask(fake_tgt_ids), labels=source_label)[0]
                loss_cyc_b = t5_a2b(input_ids=fake_src_ids, attention_mask=self.gen_mask(fake_src_ids), labels=target_label)[0]

                loss_total = (loss_idt_a + loss_idt_b) * args.lambda_idt + (loss_cyc_a + loss_cyc_b) * args.lambda_cyc
                loss = loss_total

                if (global_steps+1) % args.gold_steps == 0:
                    source_ids, source_mask, source_label, target_ids, target_mask, target_label = self.gold_batches[gold_idx]
                    gold_idx = (gold_idx + 1) % len(self.gold_batches)
                    loss_gold_b = t5_a2b(input_ids=source_ids, attention_mask=source_mask, labels=target_label)[0]
                    loss_gold_a = t5_b2a(input_ids=target_ids, attention_mask=target_mask, labels=source_label)[0]
                    loss_gold = loss_gold_a + loss_gold_b
                    loss += loss_gold
                loss /= args.gradient_accumulation_steps

                if args.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (
                    (step+1) % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    if args.use_amp:
                        scaler.minimize(optimizer, loss)
                    else:
                        optimizer.minimize(loss)
                    optimizer.clear_grad()
                    scheduler.step()

                    progress_bar.update(1)
                    global_steps += 1
                    writer.add_scalar("train/total", loss_total.item(), global_steps)
                    writer.add_scalar("train/idt_a", loss_idt_a.item(), global_steps)
                    writer.add_scalar("train/idt_b", loss_idt_b.item(), global_steps)
                    writer.add_scalar("train/cycle_a", loss_cyc_a.item(), global_steps)
                    writer.add_scalar("train/cycle_b", loss_cyc_b.item(), global_steps)
                    if global_steps % args.gold_steps == 0:
                        writer.add_scalar("train/gold_a", loss_gold_a.item(), global_steps)
                        writer.add_scalar("train/gold_b", loss_gold_b.item(), global_steps)
                    if args.logging_steps > 0 and global_steps% args.logging_steps == 0 :
                        writer.add_scalar("lr", optimizer.get_lr(), global_steps)
                        writer.add_scalar("loss", loss.item(), global_steps)
                        self.log(
                            "global_steps {} - lr: {:.10f}  loss: {:.8f}".format(
                                global_steps,
                                optimizer.get_lr(),
                                loss.item(),
                            )
                        )

                    if args.save_steps > 0 and global_steps % args.save_steps == 0:
                        self.log("********** Running evaluating **********")
                        self.log(f"********** Step {global_steps} **********")
                        output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                        os.makedirs(output_dir, exist_ok=True)
                        self._save(output_dir, optimizer)
                        eval_results = self.evaluate(dev_dataloader, args.num_eval_batch)
                        self.debug(eval_results)
                        self._save_metric(output_dir, eval_results)
                        for k, v in eval_results.items():
                            writer.add_scalar(f"eval/{k}", v, global_steps)
                        self.log("********** Evaluating Done **********")

                if global_steps >= args.max_train_steps:
                    self.log("********** Running evaluating **********")
                    self.log(f"********** Step {global_steps} **********")
                    output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                    os.makedirs(output_dir, exist_ok=True)
                    self._save(output_dir, optimizer)
                    eval_results = self.evaluate(t5_a2b, dev_dataloader, -1)
                    self.debug(eval_results)
                    self._save_metric(output_dir, eval_results)
                    for k, v in eval_results.items():
                        writer.add_scalar(f"eval/{k}", v, global_steps)
                    self.log("********** Evaluating Done **********")
                    self.log("********** Training Done **********")
                    return


    def _cycle_loss(self, input_ids, input_mask, input_labels, t5, t5_rev, max_length):
        fake_tgt_ids = self.generate(
            t5,
            input_ids=input_ids,
            max_length=max_length,
        )
        loss_cycle, _, _, encoder_output = t5_rev(
            input_ids=fake_tgt_ids,
            attention_mask=self.gen_mask(fake_tgt_ids),
            labels=input_labels
        )
        return loss_cycle, encoder_output[:, 0]


    @paddle.no_grad()
    def evaluate(self, data_loader, num_batch=100):
        self.eval()
        model = self.t5_a2b
        tkn = self.tkn

        gen_kwargs = {
            "max_length": self.args.max_target_length,
            "num_beams": self.args.num_beams,
            "length_penalty": 1.0,
            "early_stopping": True,
            "decode_strategy": "beam_search"
        }
        decoded_preds = []
        decoded_labels = []        
        progress_bar = tqdm(range(num_batch if num_batch > 0 else len(data_loader)))
        for i, batch in enumerate(data_loader):
            if num_batch > 0 and i >= num_batch:
                break
            source_ids, source_mask, _, target_ids, _, target_label = batch
            generated_tokens = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                **gen_kwargs,
            )[0]
            labels = np.where(target_label.numpy() != -100, target_label.numpy(), tkn.pad_token_id)

            decoded_preds.extend(tkn.batch_decode(generated_tokens.numpy(), skip_special_tokens=True))
            decoded_labels.extend(tkn.batch_decode(labels, skip_special_tokens=True))
            progress_bar.update(1)

        self.debug(decoded_preds[:3])
        self.debug(decoded_labels[:3])

        score1 = rouge1.score(decoded_preds, decoded_labels)
        score2 = rouge2.score(decoded_preds, decoded_labels)        
        rougeL = RougeL()
        bleu = BLEU()
        for e, r in zip(decoded_preds, decoded_labels):
            rougeL.add_inst(r, [e])
            bleu.add_inst(r, [e])
        score3 = rougeL.score()
        score4 = bleu.score()
        result = {'rouge1': score1, 'rouge2': score2, 'rougeL': score3, 'bleu': score4}

        return result
    
    @paddle.no_grad()
    def gen_mask(self, ids):
        if len(ids.shape) > 2:
            ids = ids.argmax(-1)
        mask = paddle.zeros(ids.shape, dtype='int64')
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                if ids[i,j] == self.tkn.pad_token_id:
                    break
                mask[i,j] = 1
        return mask

    def _save(self, output_dir, optimizer):        
        paddle.save(self.state_dict(), os.path.join(output_dir, "model.pdparams"))
        paddle.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pdopt"))
        self.tkn.save_pretrained(output_dir)
        return

    def _load(self, ckp_dir, optimizer):        
        model_state_dict = paddle.load(os.path.join(ckp_dir, "model.pdparams"))
        opt_state_dict = paddle.load(os.path.join(ckp_dir, "optimizer.pdopt"))

        self.set_state_dict(model_state_dict)
        optimizer.set_state_dict(opt_state_dict)
        return
    
    def _save_metric(self, output_dir, eval_results):
        with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
            f.write(str(eval_results))
        return
    
    def debug(self, var):
        logger.debug(var)
        locallogger.debug(var)

    def log(self, info):
        logger.info(info)
        locallogger.info(info)