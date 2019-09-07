# GREYC-Internship
## Gating mechanisms in CNNs inspired by memorization and generalization


This is the code repository for work done during my internship at GREYC laboratory from 3st June - 28th August, 2019 under the mentorship of Dr. Frederic Jurie, Dr. Alexis Lechervy and Shivang Aggarwal. The repository contains code implementation of various experiments conducted by applying gating mechanisms on Resnets for cifar 10 and cifar 100 datasets.   



### Instructions for setting up code for training any model in the key file

#### Directory structuring required to run any main file :

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
