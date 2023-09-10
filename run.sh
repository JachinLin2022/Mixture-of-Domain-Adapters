task_name=cola
shot=16
seed=999

# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/mlm4096/model20230908-202139.pt0" "" "" "" cola mlm_${shot}_${seed} ${seed} 0
# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/mlm4096/model20230908-202139.pt0,/root/SimCSE/result/test4096/model_4000.pt" "" "" "" cola add_one_mlm_${shot}_${seed} ${seed} 0

# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 2 pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/mlm4096/model20230908-202139.pt0,/root/SimCSE/result/test4096/model_4000.pt" "" "" "" cola add_one_mlm_${shot}_${seed} ${seed} 0
# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/mlm4096/model20230908-202139.pt0" "" "" "" ffff mlm_${shot}_${seed} ${seed} 0
# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 5 pfeiffer "" "" "" "" ffff pfeiffer_${shot}_${seed} ${seed} 0


# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/mlm4096/model20230908-202139.pt0,/root/SimCSE/result/test4096/model_4000.pt" "" "" "" cola cl_mlm_${shot}_${seed} ${seed} 0
bash stage_two.sh 1 ${shot} ${task_name} 1e-5 100 5 finetune "" "" "" "" ffff finetune_${shot}_${seed} ${seed} 0

# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/mlm4096/model20230908-202139.pt0,/root/SimCSE/result/test4096/model_1000.pt" "" "" "" cola test_random_${shot}_${seed} ${seed} 0
# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "/root/SimCSE/result/test4096/model_1000.pt" "" "" "" cola cl_no_gating_${shot}_${seed} ${seed} 0





