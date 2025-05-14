# MINIGE-MNER

Code and model for “MINIGE-MNER: A Multi-Stage Interaction Network Inspired by Gene Editing for Multimodal Named Entity Recognition”

## Installation
```
Python 3.7.16
torch 1.7.1
CUDA 11.0
```

## Required Environment
To run the codes, you need to install the requirements.
```
pip install -r requirements.txt
```

## Data Preparation
+ Twitter2015 & Twitter2017

You need to download two kinds of data to run the code.

1. We followed [HVPNet](https://github.com/zjunlp/HVPNeT) and [UMGF](https://github.com/TransformersWsz/UMGF) for data processing.
2. The generated images from [TMR](https://github.com/thecharm/TMR), many thanks.

Then you should put folders `twitter2015_images`, `twitter2017_images`, `ner15_diffusion_pic`, `ner17_diffusion_pic`, `twitter2015_aux_images`, and `twitter2017_aux_images` under the "./data" directory.

## Commands to run the code:

CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name twitter15 --num_epochs=30 --batch_size=8 --lr=3e-5 --warmup_ratio=0.03 --eval_begin_epoch=1 --ignore_idx=0 --max_seq=128 --log_name twitter15_model --do_train

CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name twitter17 --num_epochs=30 --batch_size=8 --lr=3e-5 --warmup_ratio=0.03 --eval_begin_epoch=1 --ignore_idx=0 --max_seq=128  --log_name twitter17_model --do_train


## Acknowledgements

+ Thanks to Dr.Lyu and his team for contributing the [CSPResNet](https://github.com/yf-lyu/VG-MNER)
