scp -r mereddd@hyper13.isis.vanderbilt.edu:"/home/mereddd/CCAT_Opensource_work/Advanced/phase2_cdn_autolabel_v2_phase4/" "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/"

#=== for minimal downloads

# mkdir -p "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle"

# scp "mereddd@hyper13.isis.vanderbilt.edu:/home/mereddd/CCAT_Opensource_work/Advanced/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_labels.csv" \
#     "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/"

# scp -r "mereddd@hyper13.isis.vanderbilt.edu:/home/mereddd/CCAT_Opensource_work/Advanced/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/images" \
#        "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/"

"""
python3 Main_Code/5.3_visualize_phase2_annotations.py \
  --state_dir "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4" \
  --frames_dir "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/images" \
  --labels_csv "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_labels.csv" \
  --output_dir "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/review_viz_nms/sampled_uncertain"
"""