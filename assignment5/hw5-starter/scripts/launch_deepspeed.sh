port=$((RANDOM % (23000 - 20000 + 1) + 20000))

path_to_config=$1 #this is a general config under configs, not a deepspeed config directly.


# to use deepspeed, your config file should have a deepspeed argument, pointing to a json file under deepspeed.
deepspeed --include localhost:0,1,2,3 --master_port $port train_lm.py $path_to_config


