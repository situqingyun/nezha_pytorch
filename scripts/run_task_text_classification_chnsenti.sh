cd ..
CURRENT_DIR=`pwd`
echo $CURRENT_DIR
export MODEL_DIR=$CURRENT_DIR/pretrained_models/nezha-cn-base
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=websentiment

export MODEL_TYPE=nezha

#-----------training-----------------
python task_text_classification_chnsenti.py \
  --model_type=$MODEL_TYPE \
  --model_path=$MODEL_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_predict \
  --gpu=0 \
  --adv_K=3 \
  --patience=4 \
  --eval_all_checkpoints \
  --monitor=eval_acc \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=256 \
  --eval_max_seq_length=256 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=1e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

#python task_text_classification_chnsenti.py --model_type=bert --model_path=bert-base-uncased
#--task_name=ABSA --do_train --do_predict --gpu=0 --monitor=eval_acc
#--data_dir=dataset/ --train_max_seq_length=128 --eval_max_seq_length=128
#--per_gpu_train_batch_size=16 --per_gpu_eval_batch_size=32 --learning_rate=3e-5
#--num_train_epochs=10.0 --logging_steps=-1 --save_steps=-1 --output_dir=output/output/ --overwrite_output_dir --seed=42