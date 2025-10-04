# Tracing Back the Malicious Clients in Poisoning Attacks to Federated Learning

This repository contains the official code for our **NeurIPS 2025** paper:  
**[“Tracing Back the Malicious Clients in Poisoning Attacks to Federated Learning”](https://arxiv.org/pdf/2407.07221)**

---

## 🚀 Environment Setup

We recommend using **conda** for environment management.

```bash
conda env create -f environment.yml
conda activate flforensics
````

---

## 🏋️ Training

To train the model and save the checkpoints, use the following command (example for **Scaling Attack**):

```bash
python3 train.py --dataset cifar10 --bias 0.5 --net resnet --gpu 5 --seed 1 --epochs 1500 --lr 0.01 \
        --batchsize 64 --nworkers 100 --nbyz 20 --b 20 --byz_type scale --aggregation mean --ckpt_interval 10 \
        --client_update_path client_update_params/ --global_model_path global_model_params/ --scaling_factor 1.0 \
        --advanced_backdoor 0
````

### Attack Settings

* **Scaling Attack**:
  `--byz_type scale --advanced_backdoor 0`
* **ALIE Attack**:
  `--byz_type scale --advanced_backdoor 1`
* **Edge Attack**:
  `--byz_type edge --advanced_backdoor 0`


---

## 🔎 Trace Back

To trace back using saved checkpoints:

```bash
python3 test.py --dataset cifar10 --bias 0.5 --net resnet --gpu 1 --seed 1 --epochs 1500 --lr 0.01 \
        --batchsize 64 --nworkers 100 --nbyz 20 --b 20 --byz_type scale --aggregation mean --ckpt_interval 10 \
        --client_update_path client_update_params/ --global_model_path global_model_params/ --scaling_factor 1.0 \
        --advanced_backdoor 0
```

---

## 📖 Citation

If you use this code in your work, please kindly cite the following paper:

```bibtex
@inproceedings{jia2025flforensics,
  title={Tracing Back the Malicious Clients in Poisoning Attacks to Federated Learning},
  author={Jia, Yuqi and Fang, Minghong and Liu, Hongbin and Zhang, Jinghuai and Gong, Neil},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

