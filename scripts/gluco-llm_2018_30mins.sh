model_name=TimeLLM
train_epochs=30
learning_rate=0.01
llama_layers=16

master_port=01188
num_process=4
batch_size=2
d_model=8
d_ff=32

comment='TimeLLM-Glucose'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 588_bbm_train.csv \
  --test_data_path 588_bbm_test.csv \
  --model_id 588 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 591_bbm_train.csv \
  --test_data_path 591_bbm_test.csv \
  --model_id 591 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

  accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 575_bbm_train.csv \
  --test_data_path 575_bbm_test.csv \
  --model_id 575 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

  accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 570_bbm_train.csv \
  --test_data_path 570_bbm_test.csv \
  --model_id 570 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 563_bbm_train.csv \
  --test_data_path 563_bbm_test.csv \
  --model_id 563 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

  accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 559_bbm_train.csv \
  --test_data_path 559_bbm_test.csv \
  --model_id 559 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment