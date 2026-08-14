"""
python3 /Users/divyamereddy/Documents/Vanderbilt/OELELab/Ccat/CCAT_Primary_Action_Recognition_p2/pretrained_models/Tools_Data_Processing_Generic_forp2_DivyaCCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation_Advanced_R/Main_Code/advPhase2_prepare_nms_input.py \
  --input_csv "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_labels.csv" \
  --output_csv "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_for_nms.csv"

python3 /Users/divyamereddy/Documents/Vanderbilt/OELELab/Ccat/CCAT_Primary_Action_Recognition_p2/pretrained_models/Tools_Data_Processing_Generic_forp2_DivyaCCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation_Advanced_R/Main_Code/5.5_remove_hoi_duplicates.py \
  -i "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_for_nms.csv" \
  -o "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_nms.csv" \
  --person-iou 0.5 --roi-iou 0.5 --score-field verb_score --drop-duplicates

"""

#Outputs 

# Input reviewed:
# /Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2/low_conf_review_bundle/sampled_uncertain_labels.csv
# Intermediate:
# /Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2/low_conf_review_bundle/sampled_uncertain_for_nms.csv
# Final deduped:
# /Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2/low_conf_review_bundle/sampled_uncertain_nms.csv