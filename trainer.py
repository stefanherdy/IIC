#!/usr/bin/python
import numpy as np
import torch
import random
from IIC import IIC
import cv2
import time
import random
from utils import transform, backtransform, get_colours
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter
            print(' --------------------- ')
            print('Epoch: ' + str(self.epoch))
            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input = x.to(self.device) # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters

            input = input.type(torch.float32)

            length = 5

            batch_size = input.shape[0]

            randlist_all = []
            for b in range(batch_size):
                randlist = []
                for i in range(length):
                    r = random.choice(range(1, 10))
                    randlist.append(r)
                randlist_all.append(randlist)

            target = transform(input, randlist_all)
            
            outinp = self.model(input)
            outtar = self.model(target)

            orig = input.cpu().detach().numpy()[0, :, :, :]
            orig = np.moveaxis(orig, 0, -1).astype('uint8')
            inpnp1 = outinp.cpu().detach().numpy()[0, 0, :, :]
            inpnp2 = outinp.cpu().detach().numpy()[0, 1, :, :]
            diff = abs(inpnp1-inpnp2)
            inpmax = torch.max(outinp, 1)
            inpmax = torch.max(outinp, 1)[1]
            inpmax = torch.max(outinp, 1)[1][0,:,:]
            inpmax = torch.max(outinp, 1)[1][0,:,:].cpu().numpy()
            inpmax = inpmax.astype(np.float32)
            inpmax = cv2.resize(inpmax, (128,128))
            diff = cv2.resize(diff, (128,128))
            kernel = np.ones((3,3),np.uint8)
            postp = cv2.morphologyEx(inpmax, cv2.MORPH_OPEN, kernel)
            postp = cv2.morphologyEx(postp, cv2.MORPH_CLOSE, kernel)

            mask = np.where(diff<0.6, diff, 0)
            ind = np.where(diff<0.6)
            postp[ind] = 0

            colours = get_colours()

            shape = np.shape(inpmax)
            h = int(shape[0])
            w = int(shape[1])
            col = np.zeros((h, w, 3))
            unique = np.unique(inpmax)
            for i, val in enumerate(unique):
                mask = np.where(inpmax == val)
                for j, row in enumerate(mask[0]):
                    x = mask[0][j]
                    y = mask[1][j]
                    col[x, y, :] = colours[int(val)]

            ax1 = plt.subplot(2, 3, 1), plt.imshow(inpnp1, 'gray')
            ax2 = plt.subplot(2, 3, 2), plt.imshow(inpnp2, 'gray')
            ax3 = plt.subplot(2, 3, 3), plt.imshow(orig, 'gray')
            ax4 = plt.subplot(2, 3, 4), plt.imshow(diff, 'gray')
            ax5 = plt.subplot(2, 3, 5), plt.imshow(col, 'gray')
            ax6 = plt.subplot(2, 3, 6), plt.imshow(postp, 'gray')

            ax1[0].set_title('Class 1')
            ax2[0].set_title('Class 2')
            ax3[0].set_title('Orig. Image')
            ax4[0].set_title('Class Diff.')
            ax5[0].set_title('Classification')
            ax6[0].set_title('CL. postprocesssed')
            
            plt.show(block=False)
            plt.pause(2)

            outtar = backtransform(outtar, randlist_all)
            
            size = 64
            outtar = torch.reshape(outtar, (outtar.shape[0]*outtar.shape[2]*outtar.shape[2], 2))
            outinp = torch.reshape(outinp, (outinp.shape[0]*outinp.shape[2]*outinp.shape[2] ,2))

            loss = IIC(outinp, outtar, C=2)
            loss_value = loss.item()

            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

        print('Training Loss: ' + str(loss_value))
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters

            input = input.type(torch.float32)
            target = torch.flip(input, [2])
            
            outinp = self.model(input)
            outtar = self.model(target)
            outtar = torch.flip(outtar, [2])

            outtar = torch.reshape(outtar, (outtar.shape[0]*outtar.shape[2]*outtar.shape[2], 2))
            outinp = torch.reshape(outinp, (outinp.shape[0]*outinp.shape[2]*outinp.shape[2] ,2))

            loss = IIC(outinp, outtar, C=2)

            loss_value = loss.item()
            valid_losses.append(loss_value)

        print('Validation Loss: ' + str(loss_value))
        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()