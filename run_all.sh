# task_name=cola
# shot=64
shots=(16)
seeds=(42 1024 999)
# tasks=(cola rte wnli csqa sst2 qnli mnli stsb)
tasks=(cola csqa rte wnli qnli sst2)
lrs=(1e-4 5e-4)
batchs=(2 4)
device=1
# for seed in "${seeds[@]}"; do
#     for shot in "${shots[@]}"; do
#         for task_name in "${tasks[@]}"; do
#             # echo ${seed}
#             bash stage_two.sh ${device} ${shot} ${task_name} 1e-5 100 5 finetune "" "" "" "" mult-task4 finetune_${shot}_${seed} ${seed}
#             bash stage_two.sh ${device} ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt" "" "" "" mult-task4 origin_mlm_${shot}_${seed} ${seed}
#             bash stage_two.sh ${device} ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task4 cl_${shot}_${seed} ${seed}
#             # bash stage_two.sh ${device} ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt,/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task4 mlm+cl_${shot}_${seed} ${seed}
#             # bash stage_two.sh 1 ${shot} ${task_name} 1e-4 60 5 pfeiffer-mixda "model/mlm/model.pt,/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task3 cl+mlm_${shot}_${seed} ${seed}
#         done
#     done
# done



for task_name in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        shot=16
        lr=1e-4
        batch=2
        # bash stage_two.sh 1 ${shot} ${task_name} 1e-5 100 5 finetune "" "" "" "" duibi finetune_${shot}_${seed} ${seed} 0
        # bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt,/root/SimCSE/result/test2/model_5.pt" "" "" "" duibi cl_ml_${lr}_${batch}_${shot}_${seed} ${seed} 0
        bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "/root/Mixture-of-Domain-Adapters/model/model.pt,model/mlm/model.pt" "" "" "" duibi csqa_ml_${lr}_${batch}_${shot}_${seed} ${seed} 0
    done
done
# for lr in "${lrs[@]}"; do
#     for batch in "${batchs[@]}"; do
#             shot=16
#             # task_name=sst2
#             seed=1024
#             # bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt" "" "" "" mult-task-5 origin_mlm_${lr}_${batch}_${shot}_${seed} ${seed}
#             bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt,/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task-5 cl_ml_${lr}_${batch}_${shot}_${seed} ${seed}
#         done
#     done
# done
# 不同lr和batch

# for lr in "${lrs[@]}"; do
#     for batch in "${batchs[@]}"; do
#             shot=16
#             task_name=cola
#             seed=1024
#             # bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt" "" "" "" mult-task-5 origin_mlm_${lr}_${batch}_${shot}_${seed} ${seed}
#             bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt,/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task-3 cl_ml_att_${lr}_${batch}_${shot}_${seed} ${seed}
#         done
#     done
# done


# for lr in "${lrs[@]}"; do
#     for batch in "${batchs[@]}"; do
#             shot=64
#             task_name=cola
#             seed=1024
#             # bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt" "" "" "" mult-task-5 origin_mlm_${lr}_${batch}_${shot}_${seed} ${seed}
#             bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt,/root/SimCSE/result/test2/model_5.pt" "" "" "" cola mine_${lr}_${batch}_${shot}_${seed} ${seed}
#         done
#     done
# done

# for task_name in "${tasks[@]}"; do
#     for lr in "${lrs[@]}"; do
#         for batch in "${batchs[@]}"; do
#                 shot=16
#                 # task_name=sst2
#                 seed=1024
#                 # bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt" "" "" "" mult-task-5 origin_mlm_${lr}_${batch}_${shot}_${seed} ${seed}
#                 bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt,/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task-5 cl_ml_${lr}_${batch}_${shot}_${seed} ${seed}
#             done
#         done
#     done
# done

# for task_name in "${tasks[@]}"; do
#     for lr in "${lrs[@]}"; do
#         for batch in "${batchs[@]}"; do
#                 shot=64
#                 # task_name=sst2
#                 seed=1024
#                 # bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt" "" "" "" mult-task-5 origin_mlm_${lr}_${batch}_${shot}_${seed} ${seed}
#                 bash stage_two.sh ${device} ${shot} ${task_name} ${lr} 100 ${batch} pfeiffer-mixda "model/mlm/model.pt,/root/SimCSE/result/test2/model_5.pt" "" "" "" mult-task-5 cl_ml_${lr}_${batch}_${shot}_${seed} ${seed}
#             done
#         done
#     done
# done
# 不同seed

# bash stage_two.sh 4 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt" "" "" "" test-project origin_mlm_${shot}_${seed} ${seed}
# bash stage_two.sh 4 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/simcse/model_16000.pt" "" "" "" test-project cl_${shot}_${seed} ${seed}
# bash stage_two.sh 4 ${shot} ${task_name} 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt,model/simcse/model_16000.pt" "" "" "" test-project cl+mlm_${shot}_${seed} ${seed}