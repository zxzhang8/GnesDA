dataset_list=(uniprot uniref)
dist_type_list=(ed nw)
seed_list=(666 555 444)
for dataset in ${dataset_list[@]}; do
    for seed in ${seed_list[@]}; do
        for dist_type in ${dist_type_list[@]}; do
            train_flag=${dataset}_${dist_type}_${seed}
            echo ${train_flag}
            nohup python main.py --data_type protein --dataset ${dataset} --dist_type ${dist_type} --shuffle-seed ${seed} --epochs 300 --conv_layers 5 --nt 1000 > train_log/${train_flag} &
            PID0=$!
            wait $PID0
        done
    done
done
