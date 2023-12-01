#!/usr/bin/python

from transformations import Compose
from transformations import MoveAxis
from customdatasets import SegmentationDataSet
from torch.utils.data import DataLoader 
from sklearn.model_selection import train_test_split
import pathlib# root directory
import torch
from matplotlib import pyplot as plt
import argparse

def train(args):
    root = pathlib.Path('./')

    def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
        filenames = [file for file in path.glob(ext) if file.is_file()]
        return filenames

    inputs = get_filenames_of_path(root / './input_flower')

    transforms = Compose([
        MoveAxis(),
        ])

    dataset_train = SegmentationDataSet(inputs=inputs,
                                        targets=inputs,
                                        transform=transforms)

    # dataset validation
    dataset_valid = SegmentationDataSet(inputs=inputs,
                                        targets=inputs,
                                        transform=transforms)

    # dataloader training
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=args.batch_size,
                                    shuffle=True
                                    )

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=args.batch_size,
                                    shuffle=True
                                    )

    x, y = next(iter(dataloader_training))

    from unet import UNet
    from trainer import Trainer
    from lr_schedule import LearningRateFinder
    
    model = UNet(in_channels=3,
                out_channels=2,
                n_blocks=args.num_blocks,
                start_filters=args.num_layers,
                activation='relu',
                normalization='batch',
                conv_mode='same',
                dim=2)


    if torch.cuda.is_available():
        model.cuda()            

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # criterion
    criterion = torch.nn.CrossEntropyLoss()# optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)# trainer
    trainer = Trainer(model=model,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    training_DataLoader=dataloader_training,
                    validation_DataLoader=dataloader_validation,
                    lr_scheduler=None,
                    epochs=args.epochs,
                    epoch=0,
                    notebook=False)# start training

    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    
    fig = plt.figure()
    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.title('Loss')
    plt.legend(('Training', 'Validation'))
    plt.grid()
    plt.savefig('./loss.png')
    
    model_name =  'model.pt'
    torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pytorch Semantic Segmentation")
    parser.add_argument("--learnrate", type=int, default=0.001, help='learn rate of optimizer')
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch Size")
    parser.add_argument("--num_layers", type=int, default=32, help="Number of UNet layers")
    parser.add_argument("--num_blocks", type=int, default=1, help="Number of UNet blocks")
    args = parser.parse_args()
    
    train(args)

 
