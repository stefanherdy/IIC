#!/usr/bin/python
from transformations import Compose
from transformations import MoveAxis
from customdatasets import SegmentationDataSet
from torch.utils.data import DataLoader 
from sklearn.model_selection import train_test_split
import pathlib# root directory
import torch
from matplotlib import pyplot as plt

def loop(learn_rate, epochs, batchsize, layers, blocks):
    root = pathlib.Path('./')

    def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
        """Returns a list of files in a directory/path. Uses pathlib."""
        filenames = [file for file in path.glob(ext) if file.is_file()]
        return filenames# input and target files

    inputs = get_filenames_of_path(root / './input_flower')
    targets = get_filenames_of_path(root / './input_flower')# training transformations and augmentations

    transforms = Compose([
        MoveAxis(),
        ])
        # random seed
    random_seed = 42# split dataset into training set and validation set
    train_size = 0.9  # 80:20 split

    inputs_train, inputs_valid = train_test_split(
        inputs,
        random_state=random_seed,
        train_size=train_size,
        shuffle=False)

    targets_train, targets_valid = train_test_split(
        targets,
        random_state=random_seed,
        train_size=train_size,
        shuffle=False)# dataset training
    dataset_train = SegmentationDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transforms)

    # dataset validation
    dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        transform=transforms)

    # dataloader training
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=batchsize,
                                    shuffle=True
                                    )

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=batchsize,
                                    shuffle=True
                                    )

    x, y = next(iter(dataloader_training))

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

    from unet import UNet
    from trainer import Trainer
    from lr_schedule import LearningRateFinder
    
    model = UNet(in_channels=3,
                out_channels=2,
                n_blocks=blocks,
                start_filters=layers,
                activation='relu',
                normalization='batch',
                conv_mode='same',
                dim=2)

    print(model)

    if torch.cuda.is_available():
        model.cuda()            

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # criterion
    criterion = torch.nn.CrossEntropyLoss()# optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)# trainer
    trainer = Trainer(model=model,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    training_DataLoader=dataloader_training,
                    validation_DataLoader=dataloader_validation,
                    lr_scheduler=None,
                    epochs=epochs,
                    epoch=0,
                    notebook=False)# start training

    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    minloss_train = round(min(training_losses), 2)
    minloss_val = round(min(validation_losses), 2)
    
    fig = plt.figure()
    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.title('Loss')
    plt.legend(('Training', 'Validation'))
    plt.grid()
    plt.savefig('./loss.png')
    
    model_name =  'model.pt'
    torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

loop(0.001, 3000, 4, 32, 1)

 
