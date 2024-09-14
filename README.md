# TreeSBA

This repository is the official implementation of **TreeSBA**.

**[TreeSBA: Tree-Transformer for Self-Supervised Sequential Brick Assembly](https://dreamguo.github.io/projects/TreeSBA/)**
<br/>
[Mengqi Guo](https://dreamguo.github.io/), [Chen Li](https://chaneyddtt.github.io/), [Yuyang Zhao](https://yuyangzhao.com), [Gim Hee Lee](https://www.comp.nus.edu.sg/~leegh/)
<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://dreamguo.github.io/projects/TreeSBA/) [![arXiv](https://img.shields.io/badge/arXiv-2311.14603-b31b1b.svg)](https://arxiv.org/pdf/2407.15648)


## Abstract
> Inferring step-wise actions to assemble 3D objects with primitive bricks from images is a challenging task due to complex constraints and the vast number of possible combinations. Recent studies have demonstrated promising results on sequential LEGO brick assembly through the utilization of LEGO-Graph modeling to predict sequential actions. However, existing approaches are class-specific and require significant computational and 3D annotation resources. In this work, we first propose a computationally efficient breadth-first search (BFS) LEGO-Tree structure to model the sequential assembly actions by considering connections between consecutive layers. Based on the LEGO-Tree structure, we then design a class-agnostic tree-transformer framework to predict the sequential assembly actions from the input multi-view images. A major challenge of the sequential brick assembly task is that the step-wise action labels are costly and tedious to obtain in practice. We mitigate this problem by leveraging synthetic-to-real transfer learning. Specifically, our model is first pre-trained on synthetic data with full supervision from the available action labels. We then circumvent the requirement for action labels in the real data by proposing an action-to-silhouette projection that replaces action labels with input image silhouettes for self-supervision. Without any annotation on the real data, our model outperforms existing methods with 3D supervision by 7.8% and 11.3% in mIoU on the MNIST and ModelNet Construction datasets, respectively.

## 1. Installation

Pull repo.
```sh
git clone git@github.com:dreamguo/TreeSBA.git
```

Create conda environment.
```sh
conda env create -f environment.yml
conda activate TreeSBA
```

## 2. Usage

Download RAD, MNIST-C, ModelNet-C data used in the paper [here](https://huggingface.co/datasets/dreamer001/TreeSBA_Dataset/blob/main/dataset.zip) and put them under `./dataset` folder. 

Please follow the data folder as:
```
├── dataset
|   ├── graph_dat
|   |   └── random13to18.dat   # RAD-S dataset
|   |   └── random15to50.dat   # RAD dataset
|   |   └── random.dat         # RAD-1k dataset
|   |   └── ...
|   ├── tree_actions
|   |   └── random13to18       # RAD-S dataset
|   |   └── ...
|   ├── tree_dep_actions
│   │   └── random13to18       # RAD-S dataset
|   |   └── ...
|   ├── voxel
│   │   └── mnist_all          # MNIST-C dataset
│   │   └── modelnet_all       # ModelNet-C40 dataset
|   |   └── ...
│   ├── voxel_img
│   │   └── random13to18       # RAD-S dataset
|   |   └── ...
├── ...
```

### Training.
Pre-train on RAD dataset.
```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/RAD.txt
```

Pre-train on RAD-S dataset.
```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/RAD-S.txt
```

Fine tune on MNIST-C dataset.
```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/MNIST-C.txt
```

Fine tune on ModelNet-C3 dataset.
```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/ModelNet-C.txt
```


We also upload pre-trained checkpoints [here](https://huggingface.co/datasets/dreamer001/TreeSBA_Dataset/blob/main/pretrained_model.zip), please put them under `./pretrained_model` folder.

### Test.
Test on MNIST-C dataset.
```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/MNIST-C.txt --inference 1 --save_obj 1 --load_model_path pretrained_model/mnist_all.pt
```

Test on ModelNet-C3 dataset.
```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/ModelNet-C.txt --inference 1 --save_obj 1 --load_model_path pretrained_model/modelnet_all3.pt
```

Test on ModelNet-C40 dataset.
```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/ModelNet-C.txt --inference 1 --save_obj 1 --load_model_path pretrained_model/modelnet_all40.pt
```

## 3. Dataset build from scratch
```sh
python prepare_dataset/prepareRAD.py
```
Our code for generting LEGO dataset is based on [Combinatorial-3D-Shape-Generation](https://github.com/POSTECH-CVLab/Combinatorial-3D-Shape-Generation).

## 4. Citation

If you make use of our work, please cite our paper:
```
@article{Guo2024TreeSBA,
  author    = {Guo, Mengqi and Li, Chen and Zhao, Yuyang and Lee, Gim Hee},
  title     = {TreeSBA: Tree-Transformer for Self-Supervised Sequential Brick Assembly},
  journal   = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
}
```

## 5. Ackowledgements

This work is based on [GenerativeLEGO](https://github.com/uoguelph-mlrg/GenerativeLEGO) and [Combinatorial-3D-Shape-Generation](https://github.com/POSTECH-CVLab/Combinatorial-3D-Shape-Generation). If you use this code in your research, please also acknowledge their work.
