task_name=csqa
shot=64
seed=42

# echo ${seed}
# bash stage_two.sh 4 64 cola 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt" "" "" "" mult-task3 origin_mlm_64_42 42

# bash stage_two.sh 4 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt" "" "" "" mult-task3 origin_mlm_${shot}_${seed} ${seed}
# bash stage_two.sh 4 ${shot} ${task_name} 4e-4 100 5 pfeiffer-mixda "model/simcse/model_16000.pt" "" "" "" test-project cl_${shot}_${seed} ${seed}
# bash stage_two.sh 4 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt,model/simcse/model_16000.pt" "" "" "" mult-task3 cl+mlm_${shot}_${seed} ${seed}

# bash stage_two.sh 1 ${shot} ${task_name} 1e-5 100 5 finetune "" "" "" "" mult-task3 finetune_${shot}_${seed} ${seed}
bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt,/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task3 mlm+cl_${shot}_${seed} ${seed}

# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 60 5 pfeiffer-mixda "model/mlm/model.pt" "" "" "" mult-task3 mlm_${shot}_${seed} ${seed}
# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 60 5 pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/simcse/model_16000.pt" "" "" "" mult-task3 cl_${shot}_${seed} ${seed}
# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 60 5 pfeiffer-mixda "/root/SimCSE/result/test2/model_1.pt" "" "" "" mult-task3 cl_${shot}_${seed} ${seed}
# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 60 5 pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/mlm/model20230903-163435.pt9,/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task3 cl_${shot}_${seed} ${seed}
# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 60 5 pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/mlm/model20230903-163435.pt5,/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task3 cl_${shot}_${seed} ${seed}





# bash stage_two.sh 1 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "/root/SimCSE/result/test2/model_5.pt,model/mlm/model.pt" "" "" "" mult-task3 cl_${shot}_${seed} ${seed}





# bash stage_two.sh 4 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt" "" "" "" test-project origin_mlm_${shot}_${seed} ${seed}
# bash stage_two.sh 4 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/simcse/model_16000.pt" "" "" "" test-project cl_${shot}_${seed} ${seed}
# bash stage_two.sh 4 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt,model/simcse/model_16000.pt" "" "" "" test-project cl+mlm_${shot}_${seed} ${seed}