# ResNet-based classifier models on pyTorch

## **Environment**
- OS: Ubuntu 18.04
- python: 3.7(conda env)
- GPU: Dual RTX 2080Ti 
- NVIDIA
    - Driver Version: 440.82
    - CUDA Version: 10.2

If you install pyTorch and pandas, you can use this code.

```
#!/bin/bash

# you can use pip or conda to install pyTorch
# on your base or virtual environment

# using conda
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install pandas

# using pip
pip install torch torchvision
```

## **Training**
For trainin, just run trainer.sh file as below:
```
#!/bin/bash

sudo sh trainer.sh
```

**Notice**

Before running the bash file, you need to change the arguments to suit your situation

```
#!/bin/bash
vim trainer.sh
```

Arguments description in trainer.sh:
- batch_size: batch size for training
- classifier: type of classifier (You can choose "resnet18", "resnet50", or "resnet101")
- class_num: class number of dataset
- dataset: name of dataset ('cifar10', 'mnist', 'fmnist')
- data_path: The path of dataset
- epochs: training epochs (100 is enough)
- gpu_ids: set gpu id (if you have just one GPU, jsut use (0))
- image_size: image size for training
- image_channels: input image channels (3 for 'cifar10' 1 for rest)
- save_dir:
    - In training phase: save path for the best model
    - In inference phase: path for loading the best model
- train: if 'true' the model start training else the model run for inference

## **Inference**
Check performance of the model after training

Arguments in the `inference.sh` are same as `trainer.sh`
```
#!/bin/bash
sudo sh inference.sh
```

## **pre-trained weights and performance**

There are model performances and pre-trained weights link.

        | ResNet18 | ResNet50 | ResNet101
 :----: | :------: | :------: | :------: 
 Cifar10| 91%([link](https://drive.google.com/file/d/1S80m3XS367HicRpghKTiIpV8GlR3lPDB/view?usp=sharing))    |   91%([link](https://drive.google.com/file/d/1VOrf6WkwgVZ1E-6JDegoNR0A-vGdNMEh/view?usp=sharing))    |   91%([link](https://drive.google.com/file/d/1zTYJ9R3suK_N0vV0BmntGGOU0DncWcfR/view?usp=sharing))    
 MNIST  |   99%([link](https://drive.google.com/file/d/15yhYjaiQteFI2iXXnaKE5cv4pD0VtVY2/view?usp=sharing))    |   99%([link](https://drive.google.com/file/d/1aNH8qfHuNWCnvPRnIsxi9Mz5F8UFzkP0/view?usp=sharing))    |   99%([link](https://drive.google.com/file/d/1Hh5r0GV1sqYdH2_nhAMOOINQpA1EqSdn/view?usp=sharing))    
 FMNIST |   93%([link](https://drive.google.com/file/d/1Ax48EwUqwwyR6a9v6088bJXuXgBbqCEG/view?usp=sharing))    |   92%([link](https://drive.google.com/file/d/1jb7Kayhd9nMkplFYooWViaUOVG6DUQnR/view?usp=sharing))    |   93%([link](https://drive.google.com/file/d/1dSyys4mszawB-lS2z_v3Xnp8NV7GR8mu/view?usp=sharing))    



