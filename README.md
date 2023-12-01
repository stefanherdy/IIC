# unsupervised-invariant-information-clustering
This is a code implementation of the paper ![Invariant Information Clustering for Unsupervised Image Classification and Segmentation](https://arxiv.org/abs/1807.06653).



Installation

$ git clone https://github.com/stefanherdy/unsupervised-invariant-information-clustering.git

Usage

    - First, add your custom datasets to the input_data folder
    - Run sem_seg_test.py
        You can specify the following parameters:
        --learnrate", type=int, default=0.001, help='learn rate of optimizer"
        --epochs", type=int, default=500
        --batch_size", type=int, default=2, help="Batch Size"
        --num_layers", type=int, default=32, help="Number of UNet layers"
        --num_blocks", type=int, default=1, help="Number of UNet blocks"

        Example usage:
        "python3 train.py --batch_size 4 --learnrate 0.0001 
    
    - optimize your hyperparameters

License

This project is licensed under the MIT License. ©️ 2023 Stefan Herdy
