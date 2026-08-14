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
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


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

    # print("df_preds",df_preds.shape,df_preds.head())

    df_preds.to_csv("df_preds.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    print("Moved")

def build_hoi_predictions(p):
    hoi_list = []
    labels = p['labels']
    boxes = p['boxes']
    sub_ids = p['sub_ids']
    obj_ids = p['obj_ids']
    verb_scores = p['verb_scores']

    for i in range(len(sub_ids)):
        subj_id = sub_ids[i]
        obj_id = obj_ids[i]
        for verb_id, score in enumerate(verb_scores[i]):
            if score > 0.001:  # Threshold
                hoi_list.append({
                    'verb_id': verb_id,
                    'subject_id': subj_id,
                    'object_id': obj_id,
                    'score': score,
                })
    return hoi_list


@torch.no_grad()
def evaluate_hoi_v1(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))


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
        print("="*20,"results",type(results),len(results),results)
        local_filenames = [res['filename'] for res in results]
        # print(local_filenames)
        # for res in results:
        #     del res['filename']
        
        # preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gathered_results = list(itertools.chain.from_iterable(utils.all_gather(results)))

        # Only your process sees local filenames; assign them directly back
        for i in range(len(local_filenames)):
            gathered_results[i]['filename'] = local_filenames[i]
        # print("136 local_filenames",local_filenames,"="*10,"Debugging check",gathered_results)
        preds.extend(gathered_results)
        # print(preds)
        # return 1
        #print("="*20,type(preds[0]),preds)
        
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))
        #print("="*20,type(gts[0]),gts)
        
        # exit()

    # save_preds(preds)
    # Debugging: changes added, for inference only
    if(getattr(data_loader.dataset, "annotations", None) is None and args.eval):
        if utils.is_main_process():
            save_preds(preds)

    print("="*20, "predictions saved")
    # return 1

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


# Debugging changes start
@torch.no_grad()
def evaluate_hoi_v3(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds, gts = [], []
    preds2=[]
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)
        # for i in range(len(results)):
        #     results[i]['filename'] = targets[i]['filename']

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        
        for i in range(len(results)):
            results[i]['filename'] = targets[i]['filename']

        # All-gather results
        gathered_results = list(itertools.chain.from_iterable(utils.all_gather(results)))

        # Reassign filenames manually (only on main process)
        if utils.is_main_process():
            for i, res in enumerate(gathered_results):
                res['filename'] = results[i]['filename'] if i < len(results) else f"img_{i}.jpg"

        preds2.extend(gathered_results)
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

    print("="*20,"results",type(results),len(results),results)
    local_filenames = [res['filename'] for res in results]
    print(local_filenames)
    # for res in results:
    #     del res['filename']

    metric_logger.synchronize_between_processes()

    if args.eval and utils.is_main_process() and args.adapter_dir=='save_all':
        print("="*50, 'Debugging check program engine line 367')
        save_preds(preds2)
    elif args.eval and utils.is_main_process():
        print("="*50, 'Debugging check program engine line 367')
        save_preds(preds2[:20])

    img_ids = [g['id'] for g in gts]
    _, unique_idx = np.unique(img_ids, return_index=True)
    preds = [p for i, p in enumerate(preds) if i in unique_idx]
    preds2 = [p for i, p in enumerate(preds2) if i in unique_idx]
    gts = [g for i, g in enumerate(gts) if i in unique_idx]
    ##========= evaluate complete HICo === model
    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)

    stats = evaluator.evaluate()
    # stats={}
    # === Filter predictions and GTs ===
    # Add 'hoi_prediction' to each prediction
    for p in preds:
        p['hoi_prediction'] = build_hoi_predictions(p)
    
    # Add 'hoi_annotation' to GTs if needed
    for g in gts:
        print(g)
        if 'hoi_annotation' not in g:
            # g['hoi_annotation'] = g['annotations']  # assuming same as what CDN expects
            if 'hois' in g:
                hois_tensor = g['hois']
                hoi_annotation = []
                for hoi_triplet in hois_tensor:
                    hoi_annotation.append({
                        'subject_id': hoi_triplet[0].item(),
                        'object_id': hoi_triplet[1].item(),
                        'verb_id': hoi_triplet[2].item()
                    })
                g['hoi_annotation'] = hoi_annotation
            else:
                print("Missing 'hois' in ground truth")
        # exit()
        # else:
        #     print("Check the GT annoyations column name")
        #     #exit()
    target_verbs = {37}#, 58}
    target_objs = {39, 63, 79}
    target_objs = set() 
    for p in preds:
        for hoi in p['hoi_prediction']:
            if hoi['verb_id'] in target_verbs:
                target_objs.add(hoi['object_id'])

    target_objs.discard(0)

    filtered_preds, filtered_gts = [], []
    for p, g in zip(preds, gts):
        valid_preds = [hoi for hoi in p['hoi_prediction'] if hoi['verb_id'] in target_verbs and hoi['object_id'] in target_objs]
        valid_gts = [hoi for hoi in g['hoi_annotation'] if hoi['verb_id'] in target_verbs and hoi['object_id'] in target_objs]
        filtered_preds.append({'hoi_prediction': valid_preds, 'id': p['filename']})
        filtered_gts.append({'hoi_annotation': valid_gts, 'id': g['filename']})

    # # === Call evaluator on filtered data ===
    # if dataset_file == 'hico':
    #     evaluator = HICOEvaluator(filtered_preds, filtered_gts,
    #                               data_loader.dataset.rare_triplets,
    #                               data_loader.dataset.non_rare_triplets,
    #                               data_loader.dataset.correct_mat,
    #                               args=args)
    # else:
    #     raise ValueError("Only HICO dataset is supported for this evaluation script.")

    # stats = evaluator.evaluate()

    # === Custom F1 score analysis ===
    y_true, y_pred, y_score = [], [], []
    object_wise = {obj: {'y_true': [], 'y_pred': []} for obj in target_objs}

    for g, p in zip(filtered_gts, filtered_preds):
        print("Debugging check 410 filtered_gts, filtered_preds",filtered_gts, filtered_preds)
        for gt in g['hoi_annotation']:
            label = 1 if gt['verb_id'] == 37 else 0
            y_true.append(label)
            object_wise[gt['object_id']]['y_true'].append(label)
        for pred in p['hoi_prediction']:
            pred_label = 1 if pred['verb_id'] == 37 else 0
            y_pred.append(pred_label)
            y_score.append(pred['score'])
            object_wise[pred['object_id']]['y_pred'].append(pred_label)

    # Global binary F1 (verb 37 vs 58)
    if len(y_true) > 0 and len(y_pred) > 0:
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        map_score = average_precision_score(y_true, y_score)
    else:
        print("length of y_true or y_pred is less than zero")
        p_macro, r_macro, f1_macro, map_score = 0, 0, 0, 0

    # Per-object F1 scores
    object_f1s = {}
    for obj_id in target_objs:
        print(" Debugging check line 432 running")
        obj_true = object_wise[obj_id]['y_true']
        obj_pred = object_wise[obj_id]['y_pred']
        if len(obj_true) > 0 and len(obj_pred) > 0:
            p_obj, r_obj, f1_obj, _ = precision_recall_fscore_support(obj_true, obj_pred, average='binary')
            print("for obj_id- p_obj, r_obj, f1_obj, _ ",p_obj, r_obj, f1_obj, _ )
        else:
            p_obj, r_obj, f1_obj = 0, 0, 0
            print("all values zero"," len(obj_true) > 0 and len(obj_pred) > 0 failed")
        object_f1s[f'object_{obj_id}_f1'] = f1_obj
        object_f1s[f'object_{obj_id}_precision'] = p_obj
        object_f1s[f'object_{obj_id}_recall'] = r_obj

    # Logging
    print(f"\n==== Filtered mAP & Custom F1 Scores ====")
    print(f"Binary AP (verb 37 vs 58): {map_score:.4f}")
    print(f"F1: {f1_macro:.4f} | Precision: {p_macro:.4f} | Recall: {r_macro:.4f}")
    for obj_id in target_objs:
        print(f"Object {obj_id} → F1: {object_f1s[f'object_{obj_id}_f1']:.4f}, "
              f"Precision: {object_f1s[f'object_{obj_id}_precision']:.4f}, "
              f"Recall: {object_f1s[f'object_{obj_id}_recall']:.4f}")

    # Combine all stats
    stats.update({
        'filtered_binary_map': map_score,
        'filtered_binary_f1': f1_macro,
        'filtered_binary_precision': p_macro,
        'filtered_binary_recall': r_macro,
    })
    stats.update(object_f1s)

    return stats



@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    # preds2=[]
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
        # print("136 local_filenames",local_filenames,"="*10,"Debugging check",gathered_results)
        preds.extend(gathered_results)

                # All-gather results
        
        # gathered_results2 = list(itertools.chain.from_iterable(utils.all_gather(results)))

        # # Reassign filenames manually (only on main process)
        # if utils.is_main_process():
        #     for i, res in enumerate(gathered_results2):
        #         res['filename'] = results[i]['filename'] if i < len(results) else f"img_{i}.jpg"

        # preds2.extend(gathered_results2)
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # print(preds)
        # return 1
        #print("="*20,type(preds[0]),preds)
        
        #gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))
        #print("="*20,type(gts[0]),gts)
        
        # exit()

    # save_preds(preds)
    # if utils.is_main_process():
    #     save_preds(preds)
    if args.eval and utils.is_main_process() and args.adapter_dir=='save_all':
        print("="*50, 'Debugging check program engine line 367')
        save_preds(preds)
    elif args.eval and utils.is_main_process():
        print("="*50, 'Debugging check program engine line 367')
        save_preds(preds[:20])

    print("="*20, "predictions saved")
    # return 1

    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    # print("Debugging check program engine liine 547",preds)
    _, indices = np.unique(img_ids, return_index=True)
    # print("indices",indices)
    # print("done")
    # for i, img_preds in enumerate(preds):
    #     if i in indices:
    #         print("i in indices",i, img_preds)
    #     else:
    #         print("i not in indices",i, img_preds['filename'])

    # print("preds line 557",preds) 
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    # print("preds line 557",preds) 
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]
    # print("Debugging check program engine liine 551","preds.labels",)
    # print(type(preds[0]), preds[0].keys(), preds[0]['labels'])
    # print("gts",gts[0]['labels'])
    # exit()
    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)

    stats = evaluator.evaluate()

    #============= New code 
#     for p in preds:
#         p['hoi_prediction'] = build_hoi_predictions(p)

#    # Add 'hoi_annotation' to GTs if needed
#     for g in gts:
#         print(g)
#         if 'hoi_annotation' not in g:
#             # g['hoi_annotation'] = g['annotations']  # assuming same as what CDN expects
#             if 'hois' in g:
#                 hois_tensor = g['hois']
#                 hoi_annotation = []
#                 for hoi_triplet in hois_tensor:
#                     hoi_annotation.append({
#                         'subject_id': hoi_triplet[0].item(),
#                         'object_id': hoi_triplet[1].item(),
#                         'verb_id': hoi_triplet[2].item()
#                     })
#                 g['hoi_annotation'] = hoi_annotation
#             else:
#                 print("Missing 'hois' in ground truth")
#         # exit()
#         # else:
#         #     print("Check the GT annoyations column name")
#         #     #exit()
#     target_verbs = {37}#, 58}
#     # target_objs = {40, 64, 70}
#     target_objs = set() 
#     for p in preds:
#         for hoi in p['hoi_prediction']:
#             if hoi['verb_id'] in target_verbs:
#                 target_objs.add(hoi['object_id'])
#             else:
#                 print("Skipped verb is",hoi['verb_id'])
#     target_objs.discard(0)

#     filtered_preds, filtered_gts = [], []
#     for p, g in zip(preds, gts):
#         valid_preds = [hoi for hoi in p['hoi_prediction'] if hoi['verb_id'] in target_verbs and hoi['object_id'] in target_objs]
#         valid_gts = [hoi for hoi in g['hoi_annotation'] if hoi['verb_id'] in target_verbs and hoi['object_id'] in target_objs]
#         filtered_preds.append({'hoi_prediction': valid_preds, 'id': p['filename']})
#         filtered_gts.append({'hoi_annotation': valid_gts, 'id': g['filename']})

#     print("Debugging check 602","preds",preds,"gts",gts,"filtered_preds",filtered_preds,"filtered_gts",filtered_gts)
#     # === Call evaluator on filtered data ===
#     if dataset_file == 'hico':
#         evaluator = HICOEvaluator(filtered_preds, filtered_gts,
#                                   data_loader.dataset.rare_triplets,
#                                   data_loader.dataset.non_rare_triplets,
#                                   data_loader.dataset.correct_mat,
#                                   args=args)
#     else:
#         raise ValueError("Only HICO dataset is supported for this evaluation script.")

#     stats = evaluator.evaluate()
#     # return stats

#     # === Custom F1 score analysis ===
#     y_true, y_pred, y_score = [], [], []
#     object_wise = {obj: {'y_true': [], 'y_pred': []} for obj in target_objs}

#     for g, p in zip(filtered_gts, filtered_preds):
#         for gt in g['hoi_annotation']:
#             label = 1 if gt['verb_id'] == 37 else 0
#             y_true.append(label)
#             object_wise[gt['object_id']]['y_true'].append(label)
#         for pred in p['hoi_prediction']:
#             pred_label = 1 if pred['verb_id'] == 37 else 0
#             y_pred.append(pred_label)
#             y_score.append(pred['score'])
#             object_wise[pred['object_id']]['y_pred'].append(pred_label)

#     # Global binary F1 (verb 37 vs 58)
#     if len(y_true) > 0 and len(y_pred) > 0:
#         p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
#         map_score = average_precision_score(y_true, y_score)
#     else:
#         p_macro, r_macro, f1_macro, map_score = 0, 0, 0, 0

#     # Per-object F1 scores
#     object_f1s = {}
#     for obj_id in target_objs:
#         obj_true = object_wise[obj_id]['y_true']
#         obj_pred = object_wise[obj_id]['y_pred']
#         if len(obj_true) > 0 and len(obj_pred) > 0:
#             p_obj, r_obj, f1_obj, _ = precision_recall_fscore_support(obj_true, obj_pred, average='binary')
#         else:
#             p_obj, r_obj, f1_obj = 0, 0, 0
#         object_f1s[f'object_{obj_id}_f1'] = f1_obj
#         object_f1s[f'object_{obj_id}_precision'] = p_obj
#         object_f1s[f'object_{obj_id}_recall'] = r_obj

#     # Logging
#     print(f"\n==== Filtered mAP & Custom F1 Scores ====")
#     print(f"Binary AP (verb 37 vs 58): {map_score:.4f}")
#     print(f"F1: {f1_macro:.4f} | Precision: {p_macro:.4f} | Recall: {r_macro:.4f}")
#     for obj_id in target_objs:
#         print(f"Object {obj_id} → F1: {object_f1s[f'object_{obj_id}_f1']:.4f}, "
#               f"Precision: {object_f1s[f'object_{obj_id}_precision']:.4f}, "
#               f"Recall: {object_f1s[f'object_{obj_id}_recall']:.4f}")

#     # Combine all stats
#     stats.update({
#         'filtered_binary_map': map_score,
#         'filtered_binary_f1': f1_macro,
#         'filtered_binary_precision': p_macro,
#         'filtered_binary_recall': r_macro,
#     })
#     stats.update(object_f1s)

    

    return stats
# Debugging changes end
