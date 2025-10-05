python3 train.py --dataset cifar10 --bias 0.5 --net resnet --gpu 5 --seed 1 --epochs 1500 --lr 0.01 \
        --batchsize 64 --nworkers 100 --nbyz 20 --b 20 --byz_type edge --aggregation mean --ckpt_interval 10 \
        --client_update_path client_update_params/ --global_model_path global_model_params/ --scaling_factor 1.0 \
        --advanced_backdoor 0

python3 test.py --dataset cifar10 --bias 0.5 --net resnet --gpu 5 --seed 1 --epochs 1500 --lr 0.01 \
        --batchsize 64 --nworkers 100 --nbyz 20 --b 20 --byz_type edge --aggregation mean --ckpt_interval 10 \
        --client_update_path client_update_params/ --global_model_path global_model_params/ --scaling_factor 1.0 \
        --advanced_backdoor 0