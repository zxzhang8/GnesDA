dataset_list=(geolife porto)
dist_type_list=(dtw edr)
seed_list=(666 555 444)
for dataset in ${dataset_list[@]}; do
    for seed in ${seed_list[@]}; do
        for dist_type in ${dist_type_list[@]}; do
            train_flag=${dataset}_${dist_type}_${seed}
            echo ${train_flag}
            nohup python train.py --data_type traj --dist_type ${dist_type} --shuffle-seed ${seed} --epochs 300 --conv_layers 3 --nt 3000 > train_log/${train_flag} &
            PID0=$!
            wait $PID0
        done
    done
done
