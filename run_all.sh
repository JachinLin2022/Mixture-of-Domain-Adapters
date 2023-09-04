bash stage_two.sh 1 64 cola 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt" "" "" "" test-project origin_mlm
bash stage_two.sh 1 64 cola 1e-4 100 5 pfeiffer-mixda "model/simcse/model_16000.pt" "" "" "" test-project cl
bash stage_two.sh 1 64 cola 1e-4 100 5 pfeiffer-mixda "model/mlm/model.pt,model/simcse/model_16000.pt" "" "" "" test-project cl+mlm