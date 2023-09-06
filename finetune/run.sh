export TASK_NAME=cola

python run_glue.py \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train \
  --evaluation_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 256 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_train_samples 64 \
  --seed 1 \
  --num_train_epochs 100 \
  --overwrite_output_dir \
  --output_dir /tmp/$TASK_NAME/