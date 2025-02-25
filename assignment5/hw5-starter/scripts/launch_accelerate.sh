port=$((RANDOM % (23000 - 20000 + 1) + 20000))

path_to_config=$1

accelerate launch --main_process_port=$port --num_processes=4 --multi_gpu train_lm.py $1

