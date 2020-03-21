# GREYC-Internship
## Gating mechanisms in CNNs inspired by memorization and generalization


This is the code repository for work done during my internship at GREYC laboratory from 3st June - 28th August, 2019 under the mentorship of Prof. Frederic Jurie, Dr. Alexis Lechervy and Shivang Aggarwal. The repository contains code implementation of various experiments conducted by applying gating mechanisms on Resnets for cifar 10 and cifar 100 datasets.   

> The file 'key to models in the files.pdf' has a descriptive list of all trained models and their corresponding code files. 

> The file 'logs_key.pdf' describes the naming convention used for naming tensorboard logs and saved checkpoints. The logs have been shared as a separate google drive link. 


### *Directory structuring required to run any main file* :

1. The main file as mentioned in the key 
2. The model file (resnet.py or likewise) should be placed:
    * In the same directory as the main file for all Resnet 34 (Imagenet type) models
    * Inside the ‘models/' subdirectory relative to the main file for all other models 
3. utils.py, in the same directory as the main file 
4. All baselines (for training from baseline initialisation) are placed inside the directory /baselines 
5. Please ensure that the directory for saving trained checkpoints and tensor board logs is present, if not, create it before running the code
6. Change the ‘data_dir’ in the code and specify the directory where you want to download the dataset. In my setup, I had:
    * 'cifar10_kuangliu/data/' for cifar 10 dataset
    * 'cifar100/data' for cifar 100 dataset 
    
    
### *A few pointers* :

1. The dataset gets downloaded automatically if it is not present already
2. The code is flexible wrt to resnet type, dataset type (cifar10/100) and most hyper parameters; the same code can be used to train a range of models by passing the correct training arguments  
3. For each model trained on different Resnet/dataset/initialisation strategy/other hyper parameter combinations, the code differentiates between checkpoints/logs of different models by using model specific names where the checkpoints/logs get saved
4. Any model can be tested on the trained checkpoints by passing the ‘-e’ argument and specifying the absolute path in the ’test_checkpoint’ argument 


### *References* :

The code for main files and imagenet type res34 was initially adapted from these 3 GitHub repositories - 
* https://github.com/kuangliu/pytorch-cifar
* https://github.com/akamaster/pytorch_resnet_cifar10
* https://github.com/chengyangfu/pytorch-vgg-cifar10

The code for cifar type resnets was adapted from -
* https://github.com/facebook/fb.resnet.torch
* https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


