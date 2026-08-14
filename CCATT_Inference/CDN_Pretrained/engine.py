import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]

        outputs = model(samples)
        #print(targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import pandas as pd

def save_preds(preds):
    hoi_preds = []
    for p in preds:
        # print("print compete p",p)
        labels = p['labels'].tolist()
        boxes = p['boxes'].tolist()
        sub_ids = p['sub_ids'].tolist()
        obj_ids = p['obj_ids'].tolist()
        verb_scores = p['verb_scores'].tolist()  # 2D: [pair][117]
        obj_scores = p['obj_scores'].tolist()  # 2D: [pair][117]
        matching_scores = p['matching_scores'].tolist() if p['matching_scores'] else None # 2D: [pair][117]
        verb_scores_index_decoder=p['verb_scores_index_decoder'].tolist() 
        # print("Debugging check 89 line engine file, verb_scores_index_decoder",len(verb_scores_index_decoder),len(verb_scores_index_decoder[0]), verb_scores_index_decoder)
        for i in range(len(sub_ids)):
            subj_box = boxes[sub_ids[i]]
            obj_box = boxes[obj_ids[i]]
            for verb_id, score in enumerate(verb_scores[i]):
                if score > 0.00:  # Threshold
                    hoi_preds.append({
                        'filename': p.get('filename', ''),
                        'subject_box': subj_box,
                        'object_box': obj_box,
                        'subject_class': labels[sub_ids[i]],
                        'object_class': labels[obj_ids[i]],
                        'verb_class': verb_id,
                        'score': score,
                        'obj_scores': obj_scores[i],
                        # 'matching_scores': matching_scores,
                        'verb_scores_index_decoder':verb_scores_index_decoder[i][verb_id]
                    })
                    # print("Debugging check 107 line engine file, verb_scores_index_decoder",verb_scores_index_decoder[i][verb_id])
    print("Moving")
    df_preds = pd.DataFrame(hoi_preds)
    import csv
    print("df_preds",df_preds.shape,df_preds.head())
    df_preds.to_csv("df_preds.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print("Moved")
@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    # print("Debugging check 115 line engine file, running save preds")
    print(f"Total number of batches: {len(data_loader)}")

    total_images = 0
    for samples, targets in data_loader:
        total_images += len(targets)
    print(f"Total images processed: {total_images}")

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        #print("="*20,"outputs",outputs.keys(),outputs)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)
        for i in range(len(results)):
            results[i]['filename'] = targets[i]['filename']
        #print("="*20,"results",type(results),len(results),results)
        local_filenames = [res['filename'] for res in results]
        # print(local_filenames)
        # for res in results:
        #     del res['filename']
        
        # preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gathered_results = list(itertools.chain.from_iterable(utils.all_gather(results)))

        # Only your process sees local filenames; assign them directly back
        for i in range(len(local_filenames)):
            gathered_results[i]['filename'] = local_filenames[i]
        #print("136 local_filenames",local_filenames,"="*10,"Debugging check",gathered_results)
        preds.extend(gathered_results)
        # print(preds)
        # return 1
        #print("="*20,type(preds[0]),preds)
        
        #gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))
        #print("="*20,type(gts[0]),gts)
        
        # exit()

    # save_preds(preds)
    if utils.is_main_process():
        save_preds(preds)

    print("="*20, "predictions saved")
    return 1

    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)

    stats = evaluator.evaluate()

    return stats
