# Replicate Table 1

This code replicates the experiment result for Table 1: Transfering policy on HalfCheetah-v2 and Ant-v2 environments using our model.

Env Halfcheetah: source (arma0.1), target (arma0.5)

Env Ant: source (crip0), target (crip3)

## Replicate using our pretrained model and our sampled data
**1. Create 4 folders in current directory, and name them as `results_transfer_action`, `data`, `model` and `policy`. These folders are for log, sampled data, pretrained forward and transfer model, and pretrained TD3 policies.**

**2. Download the pretrained model and sampled data from the following link.**

HalfCheetah-v2[https://drive.google.com/drive/folders/1RRp12E4A4PeRuv4yHJo-rNQS4izp6QiQ?usp=sharing]

Ant-v2[https://drive.google.com/drive/folders/1TZS59KtEgrQUnmfffwzv9BSzJTUevyqM?usp=sharing]

Put all the files in the corresponding folder as they are.

**3. Run the following command to start training.**

HalfCheetah-v2:
```
CUDA_VISIBLE_DEVICES=0 python3 train_transfer_action.py --env HalfCheetah-v2 --seed 0
```

Ant-v2:
```
CUDA_VISIBLE_DEVICES=0 python3 train_transfer_action.py --env Ant-v2 --seed 0
```


## Replicate using your own data and/or own sampled data
**1. Create 4 folders in current directory, and name them as `results_transfer_action`, `data`, `model` and `policy`. These folders are for log, sampled data, pretrained forward and transfer model, and pretrained TD3 policies.**

**2. Run the following command to start training.**

HalfCheetah-v2:
```
CUDA_VISIBLE_DEVICES=0 python3 train_transfer_action.py --env HalfCheetah-v2 --seed 0 --newData True --newForward True --newTransfer True
```

Ant-v2:
```
CUDA_VISIBLE_DEVICES=0 python3 train_transfer_action.py --env Ant-v2 --seed 0 --newData True --newForward True --newTransfer True
```
The argument `newData`, `newForward`, and `newTransfer` are for sampling new data, training new forward model and training new transfer model. Please specify these boolean arguments according to your needs.

