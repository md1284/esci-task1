import re
import os
import sys
import math
import json
import torch
import random
import pickle
import argparse
import datetime

import numpy as np
import pandas as pd
import sentencepiece
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import  TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, ModelSummary
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from typing import Any, Optional, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT

from model import T5FineTuner, T5NegFineTuner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args, train_params):
    sys.setrecursionlimit(10000)
    set_seed(args.seed)
    
    if args.model_mode == "gr":
        model = T5FineTuner(args)
    elif args.model_mode == "grcl":
        model = T5NegFineTuner(args)
        train_params["num_sanity_val_steps"] = 0

        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if torch.cuda.current_device() == 0:
        print('='*80)
        print(f"# of trainable parameters: {trainable_params}\n# of total parameters: {total_params}")
        print('='*80)

    trainer = pl.Trainer(**train_params)
    
    if args.do_train:
        if torch.cuda.current_device() == 0:
            now = datetime.datetime.now()
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Start Training...")

        if args.resume_from_checkpoint is None:
            trainer.fit(model)
        else:
            print(f"@@@ Resume Training from {args.resume_from_checkpoint}")
            trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
        now = datetime.datetime.now()
        print(
            f"{torch.cuda.current_device()} // [{now.strftime('%Y-%m-%d %H:%M:%S')}] Done Training..."
        )
    if args.do_test:
        if torch.cuda.current_device() == 0:
            now = datetime.datetime.now()
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Start Testing...")
        trainer.test(model)
        now = datetime.datetime.now()
        print(
            f"{torch.cuda.current_device()} // [{now.strftime('%Y-%m-%d %H:%M:%S')}] Done Testing... "
        )
    return args.output_dir

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, required=True, type=str)
    arg_ = parser.parse_args()

    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    if hparam.tensorboard_log and hparam.do_train:
        tensorboard_logger = TensorBoardLogger(
            save_dir=os.path.join("tensorboard", hparam.tensorboard_project), name=hparam.tensorboard_run_name
        )
    else:
        tensorboard_logger = None

    args_dict = dict(
        seed=42,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        early_stop_callback=False,
        output_dir=hparam.output_dir,
        dataset=hparam.dataset,
        model_name_or_path=hparam.model,
        model_mode=hparam.model_mode,
        test_model_path=hparam.test_model_path,
        test_name=hparam.test_name,
        tensorboard_log=hparam.tensorboard_log,
        tensorboard_project=hparam.tensorboard_project,
        tensorboard_run_name=hparam.tensorboard_run_name,

        max_input_length=hparam.max_input_length,
        max_output_length=hparam.max_output_length,
        max_beamsearch_length=hparam.max_beamsearch_length,
        val_beam_size=hparam.val_beam_size,

        learning_rate=hparam.learning_rate,
        lr_scheduler=hparam.lr_scheduler,
        accelerator=hparam.accelerator,
        num_train_epochs=hparam.num_train_epochs,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.eval_batch_size,
        test_batch_size=hparam.test_batch_size,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.n_gpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint,
        check_val_every_n_epoch=hparam.check_val_every_n_epoch,

        do_train=hparam.do_train,
        do_test=hparam.do_test,
        
        train_file=hparam.train_file,
        dev_file=hparam.dev_file,
        test_file=hparam.test_file,
        queries=hparam.queries,
        collection=hparam.collection,
        pid_to_productid=hparam.pid_to_productid,
        pid_to_product=hparam.pid_to_product,
        qid_to_queryid=hparam.qid_to_queryid,
        qid_to_candidate_pid=hparam.qid_to_candidate_pid,
        qid_to_esci_pid=hparam.qid_to_esci_pid,
        qid_to_trie=hparam.qid_to_trie,
    )
    args = argparse.Namespace(**args_dict)
    
    ### constraint check ###
    assert not (args.do_train and args.do_test), "Choose between train|test"
    assert args.model_mode in ["gr", "grcl"]

    if torch.cuda.current_device() == 0:
        print("#" * 80)
        print(args)
        print("#" * 80)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor="val ndcg",
        mode="max",
        dirpath=args.output_dir,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=5,
    )
    callbacks.append(checkpoint_callback)

    if args.lr_scheduler == "constant" and torch.cuda.current_device() == 0:
        print(f"@@@ Not Using Learning Rate Scheduler")
    else:
        lr_callback = LearningRateMonitor()
        callbacks.append(lr_callback)

    if args.accelerator == "ddp":
        plugins = DDPStrategy(find_unused_parameters=False)
        fp_16 = False
        args.fp16 = False
        if torch.cuda.current_device() == 0:
            print(f"@@@ Using DDP without FP16")
    elif args.accelerator == "deepspeed":
        plugins = DeepSpeedStrategy(stage=2, load_full_weights=True)
        
        fp_16 = True
        args.fp16 = True
        if torch.cuda.current_device() == 0:
            print(f"@@@ Using Deepspeed stage2 with FP16")
    else:
        raise NotImplementedError("** accelerator: Choose between (ddp|deepspeed)")

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        strategy=plugins,
        max_epochs=args.num_train_epochs,
        precision=16 if fp_16 else 32,
        default_root_dir=args.output_dir,
        logger=tensorboard_logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        num_sanity_val_steps=1,
    )
    main(args, train_params)