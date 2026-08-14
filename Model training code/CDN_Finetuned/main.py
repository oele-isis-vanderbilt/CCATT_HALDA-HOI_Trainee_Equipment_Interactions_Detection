
import argparse
import time
import datetime
import random
from pathlib import Path
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model

from collections import OrderedDict

# === PEFT / LoRA ===
from peft.tuners.lora import LoraLayer, LoraModel
from peft import PeftModel, LoraConfig


# --------------------------------------------------------------------------------------
# Utilities for saving / loading LoRA adapters (consolidated to single definitions)
# --------------------------------------------------------------------------------------

def save_all_lora_adapters(model: torch.nn.Module, save_dir: str) -> None:
    """Save all LoRA adapters discovered under the given model tree.
    Works with PEFT-injected modules (LoraLayer) regardless of nesting depth.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving LoRA adapters to: {save_dir}")

    # Walk the module tree; when we find a LoraLayer, ascend until we reach the
    # adapter container (module that implements `save_adapter`).
    from peft.tuners.lora.layer import LoraLayer as _LoraLayer
    for name, module in model.named_modules():
        if isinstance(module, _LoraLayer):
            parent = module
            while not hasattr(parent, "save_adapter") and hasattr(parent, "model"):
                parent = parent.model
            if hasattr(parent, "save_adapter"):
                adapter_name = getattr(parent, "active_adapter", "default")
                out_path = os.path.join(save_dir, f"adapter_{name.replace('.', '_')}")
                try:
                    parent.save_adapter(out_path, adapter_name=adapter_name)
                    print(f"✅ Saved LoRA adapter for {name} → {out_path}")
                except Exception as e:
                    print(f"⚠️ Could not save adapter for {name}: {e}")


def load_lora_adapters(model: torch.nn.Module, adapter_root: str) -> None:
    """Load LoRA adapters back into their corresponding submodules when available."""
    print(f"📂 Loading LoRA adapters from: {adapter_root}")
    for name, module in model.named_modules():
        path = os.path.join(adapter_root, f"adapter_{name.replace('.', '_')}")
        if os.path.exists(path):
            try:
                if hasattr(module, "load_adapter"):
                    module.load_adapter(path, adapter_name="default")
                    print(f"✅ Loaded LoRA adapter into: {name}")
                else:
                    # In some wrappers, the container that has load_adapter is `module.model`
                    container = getattr(module, "model", None)
                    if container is not None and hasattr(container, "load_adapter"):
                        container.load_adapter(path, adapter_name="default")
                        print(f"✅ Loaded LoRA adapter into container of: {name}")
                    else:
                        print(f"⚠️ Module {name} has no `load_adapter()`; skipping.")
            except Exception as e:
                print(f"❌ Failed to load adapter for {name}: {e}")


# --------------------------------------------------------------------------------------
# Arg parser
# --------------------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers_hopd', default=3, type=int,
                        help="Number of hopd decoding layers in the transformer")
    parser.add_argument('--dec_layers_interaction', default=3, type=int,
                        help="Number of interaction decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_matching', action='store_true',
                        help="Use obj/sub matching 2class loss in first decoder, default not use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_matching', default=1, type=float,
                        help="Sub and obj box matching coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--matching_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # decoupling training parameters
    parser.add_argument('--freeze_mode', default=0, type=int)
    parser.add_argument('--obj_reweight', action='store_true')
    parser.add_argument('--verb_reweight', action='store_true')
    parser.add_argument('--use_static_weights', action='store_true',
                        help='use static weights or dynamic weights, default use dynamic')
    parser.add_argument('--queue_size', default=4704*1.0, type=float,
                        help='Maxsize of queue for obj and verb reweighting, default 1 epoch')
    parser.add_argument('--p_obj', default=0.7, type=float,
                        help='Reweighting parameter for obj')
    parser.add_argument('--p_verb', default=0.7, type=float,
                        help='Reweighting parameter for verb')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    # LoRA controls
    parser.add_argument('--use_lora', action='store_true', help='Enable LoRA for attention and MLP layers')
    parser.add_argument('--lora_rank', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=float)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    parser.add_argument('--adapter_dir', default=None, type=str)

    return parser


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # Device & rank
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])  # provided by torchrun
        torch.cuda.set_device(local_rank)
        args.gpu = local_rank
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    # Reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1) Build base model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # 2) (Optional) Load backbone/heads from checkpoint BEFORE injecting LoRA so key names match
    model_without_ddp = model
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        ckpt = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True) if args.resume.startswith('https') else torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'], strict=False)
        if not args.eval and all(k in ckpt for k in ['optimizer', 'lr_scheduler', 'epoch']):
            # Will restore after optimizer is created
            resume_pkg = ckpt
        else:
            resume_pkg = None
    else:
        resume_pkg = None

    if args.pretrained and not args.resume:
        # Resize classifier heads if user-specified class counts differ, then load
        ckpt = torch.load(args.pretrained, map_location='cpu')
        state_dict = ckpt['model']
        print("Loading pretrained weights from:", args.pretrained)
        
        # === Adjust object classifier ===
        if 'obj_class_embed.weight' in state_dict and args.num_obj_classes is not None:
            old_w = state_dict['obj_class_embed.weight']
            old_b = state_dict['obj_class_embed.bias']
            C, D = old_w.shape
            if C != args.num_obj_classes + 1:  # account for background if applicable
                new_C = args.num_obj_classes + 1
                new_w = torch.zeros(new_C, D)
                new_b = torch.zeros(new_C)
                copy_C = min(C, new_C)
                new_w[:copy_C] = old_w[:copy_C]
                new_b[:copy_C] = old_b[:copy_C]
                mean_w, mean_b = old_w.mean(0), old_b.mean(0)
                if new_C > copy_C:
                    new_w[copy_C:] = mean_w + 0.01 * torch.randn(new_C - copy_C, D)
                    new_b[copy_C:] = mean_b + 0.01 * torch.randn(new_C - copy_C)
                state_dict['obj_class_embed.weight'] = new_w
                state_dict['obj_class_embed.bias'] = new_b

        # === Adjust verb classifier ===
        if 'verb_class_embed.weight' in state_dict and args.num_verb_classes is not None:
            old_w = state_dict['verb_class_embed.weight']
            old_b = state_dict['verb_class_embed.bias']
            C, D = old_w.shape
            if C != args.num_verb_classes:
                new_C = args.num_verb_classes
                new_w = torch.zeros(new_C, D)
                new_b = torch.zeros(new_C)
                copy_C = min(C, new_C)
                new_w[:copy_C] = old_w[:copy_C]
                new_b[:copy_C] = old_b[:copy_C]
                mean_w, mean_b = old_w.mean(0), old_b.mean(0)
                if new_C > copy_C:
                    new_w[copy_C:] = mean_w + 0.01 * torch.randn(new_C - copy_C, D)
                    new_b[copy_C:] = mean_b + 0.01 * torch.randn(new_C - copy_C)
                state_dict['verb_class_embed.weight'] = new_w
                state_dict['verb_class_embed.bias'] = new_b

        model_without_ddp.load_state_dict(state_dict, strict=False)
        print("Weights loaded from pretrained weights:", args.pretrained)

    # 3) Inject LoRA (before DDP) and set requires_grad according to freeze_mode
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        # target_modules is not used because we wrap each transformer block with LoraModel below
    )
    def _count_lora_trainable_params(mod: torch.nn.Module) -> int:
        total = 0
        for m in mod.modules():
            if isinstance(m, LoraLayer):
                for n, p in m.named_parameters():
                    if p.requires_grad:
                        total += p.numel()
        return total
    
    def _inject_decoder_layers(mod):
        for i, layer in enumerate(mod.transformer.decoder.layers):
            mod.transformer.decoder.layers[i] = LoraModel(mod.transformer.decoder.layers[i], lora_cfg, adapter_name="default")

    def _inject_encoder_layers(mod):
        for i, layer in enumerate(mod.transformer.encoder.layers):
            mod.transformer.encoder.layers[i] = LoraModel(layer, lora_cfg, adapter_name="default")

    def _inject_interaction_decoder_layers(mod):
        for i, layer in enumerate(mod.transformer.interaction_decoder.layers):
            mod.transformer.interaction_decoder.layers[i] = LoraModel(layer, lora_cfg, adapter_name="default")

    # Freeze logic + LoRA injection
    # Start by freezing everything; then selectively unfreeze
    for p in model_without_ddp.parameters():
        p.requires_grad = False

    if args.freeze_mode == 0:
        # Full finetuning, no LoRA required
        for p in model_without_ddp.parameters():
            p.requires_grad = True
    elif args.freeze_mode in [1]:
        # Train decoders and heads
        for name, p in model_without_ddp.named_parameters():
            if ('decoder' in name) or ('verb_class_embed' in name) or ('obj_class_embed' in name) or \
               ('sub_bbox_embed' in name) or ('obj_bbox_embed' in name) or (args.use_matching and 'matching_embed' in name):
                p.requires_grad = True
    elif args.freeze_mode in [11]:
        for p in model_without_ddp.parameters():
            p.requires_grad = True

        # Train decoders and heads
        for name, p in model_without_ddp.named_parameters():
            if ('interaction_decoder' in name) :
                p.requires_grad = False

        total_params = sum(p.numel() for p in model_without_ddp.parameters())
        trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params} / {total_params} ({trainable_params / total_params:.2%})")

    elif args.freeze_mode in [2, 3, 4, 5, 201]:
        # LoRA on decoder (+ optionally encoder / interaction_decoder) + heads
        if args.use_lora:
            _inject_decoder_layers(model_without_ddp)
            if args.freeze_mode in [3, 4, 5, 201]:
                _inject_encoder_layers(model_without_ddp)
            if args.freeze_mode in [4, 5, 201]:
                _inject_interaction_decoder_layers(model_without_ddp)

        # Unfreeze heads and query embeddings
        for name, p in model_without_ddp.named_parameters():
            if ('verb_class_embed' in name) or ('obj_class_embed' in name) or \
               ('sub_bbox_embed' in name) or ('obj_bbox_embed' in name) or ('query_embed' in name):
                p.requires_grad = True

        # Unfreeze only LoRA params
        for module in model_without_ddp.modules():
            if isinstance(module, LoraLayer):
                for pname, param in module.named_parameters():
                    if 'lora' in pname.lower():
                        param.requires_grad = True

        if args.freeze_mode == 5:
            # additionally unfreeze last ResNet block + input proj
            for name, p in model_without_ddp.named_parameters():
                if 'backbone.0.body.layer4' in name or 'input_proj' in name:
                    p.requires_grad = True

        if args.freeze_mode == 201:
            # Special case: keep encoder LoRA frozen
            for name, p in model_without_ddp.named_parameters():
                if 'encoder' in name:
                    p.requires_grad = False

    # Flag if we actually have any LoRA layers
    lora_found = any(isinstance(m, LoraLayer) for m in model_without_ddp.modules())
    if not lora_found and args.use_lora and args.freeze_mode in [2, 3, 4, 5, 201]:
        print("⚠️ No LoRA layers found after injection; check configuration.")

    # 4) Wrap with DDP (after all mutations). Then create optimizer using model_without_ddp
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # 5) Optimizer / LR
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Restore optimizer / scheduler if resuming
    if 'resume_pkg' in locals() and resume_pkg is not None and not args.eval:
        try:
            optimizer.load_state_dict(resume_pkg['optimizer'])
            lr_scheduler.load_state_dict(resume_pkg['lr_scheduler'])
            args.start_epoch = resume_pkg['epoch'] + 1
            print(f"Resumed optimizer/scheduler at epoch {args.start_epoch}")
        except Exception as e:
            print(f"⚠️ Failed to restore optimizer/scheduler: {e}")

    # 6) Datasets / Samplers / Loaders
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # 7) Optional eval-only paths (ensure ranks stay in lock-step)
    output_dir = Path(args.output_dir)

    if args.eval and args.adapter_dir is not None:
        load_lora_adapters(model_without_ddp, args.adapter_dir)
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args)
        if args.distributed:
            torch.distributed.barrier()
        print(test_stats)
        return
    elif args.eval:
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args)
        if args.distributed:
            torch.distributed.barrier()
        print(test_stats)
        return

    # 8) Training
    

    total_params = sum(p.numel() for p in model_without_ddp.parameters())
    trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params} / {total_params} ({trainable_params / total_params:.2%})")

    print("Start training")
    start_time = time.time()
    best_metric = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()

        # Save last checkpoint at final epoch
        if epoch == args.epochs - 1:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
            if lora_found:
                save_all_lora_adapters(model_without_ddp, os.path.join(output_dir, "lora_adapters_all"))

        # Evaluation cadence (keep your prior logic)
        if args.freeze_mode == 0 and epoch < args.lr_drop and epoch % 5 != 0:
            continue
        elif args.freeze_mode == 0 and epoch >= args.lr_drop and epoch % 2 == 0:
            continue

        test_stats = evaluate_hoi(
            args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args
        )

        if args.distributed:
            torch.distributed.barrier()

        if args.dataset_file == 'hico':
            performance = test_stats.get('mAP', 0.0)
        elif args.dataset_file == 'vcoco':
            performance = test_stats.get('mAP_all', 0.0)
        else:
            performance = float(test_stats.get('mAP', 0.0))

        if performance > best_metric:
            best_metric = performance
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
            if lora_found:
                save_all_lora_adapters(model_without_ddp, os.path.join(output_dir, "lora_adapters_all"))
            print(f"New best performance: {performance:.3f} at epoch {epoch}")

        # Logging
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_lora_params': _count_lora_trainable_params(model_without_ddp),
            'trainable_params_percentage': (trainable_params / total_params),
        }

        if args.output_dir and utils.is_main_process():
            (output_dir).mkdir(parents=True, exist_ok=True)
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
