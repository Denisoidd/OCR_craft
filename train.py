import argparse
import torch

from torchnet.meter import mAPMeter
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from craft import CRAFT
from test import str2bool, copyStateDict
from dataset_loader import CheckDataset
from transformers import Rescale, ToTensor

parser = argparse.ArgumentParser(description="CRAFT Fine Tuning")
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--num_of_epochs', default=10, type=int)
parser.add_argument('--cuda', default=True, type=str2bool)
parser.add_argument('--tr_anns', default='C:/Users/denis/Desktop/probation/train/ann/', type=str)
parser.add_argument('--tr_images', default='C:/Users/denis/Desktop/probation/train/images', type=str)
parser.add_argument('--v_anns', default='C:/Users/denis/Desktop/probation/val/ann/', type=str)
parser.add_argument('--v_images', default='C:/Users/denis/Desktop/probation/val/images', type=str)
parser.add_argument('--b_s', default=16, type=int)
parser.add_argument('--val_b_s', default=32, type=int)
parser.add_argument('--freeze', default=True, type=bool)

args = parser.parse_args()

if __name__ == '__main__':
    # choose device
    device = 'cpu'
    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Your device is", device)

    # load net and move it to device
    net = CRAFT().float()
    net.to(device)
    print("Model successfully loaded")

    # load pretrained model
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        print("Model weights loaded on gpu")
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
        print("Model weights loaded on cpu")

    # freeze all layers without few last of them
    if args.freeze:
        for name, param in net.named_parameters():
            if 'conv_cls' not in name:
                param.requires_grad = False
            print(name, param.requires_grad)

    # create train dataloader
    t_dataset = CheckDataset(args.tr_anns, args.tr_images,
                             transforms.Compose([Rescale((256, 256)), ToTensor()]))
    train_loader = DataLoader(t_dataset, batch_size=args.b_s)
    print("Train dataset is initialized")

    # create valid dataloader
    v_dataset = CheckDataset(args.v_anns, args.v_images,
                             transforms.Compose([Rescale((256, 256)), ToTensor()]))
    valid_loader = DataLoader(v_dataset, batch_size=args.b_s)
    print("Val dataset is initialized")

    # define loss and optimizer
    optimizer = optim.Adam(net.parameters())
    loss_function = torch.nn.MSELoss()

    # initialize train and val losses
    train_loss, val_loss = [], []

    # TODO: write mAP

    # train loop
    print("Start training process")
    for epoch in range(args.num_of_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the data from dataloader and move it to gpu
            inputs, labels = data['image'].to(device), data['heatmap'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y, feature = net(inputs.float())
            loss = loss_function(y[:, :, :, 0], labels.float())
            loss.backward()
            optimizer.step()

            # put loss into list
            train_loss.append(loss.item())

            # print statistics
            running_loss += loss.item()

        # show and save current train loss
        train_loss.append(running_loss / len(train_loader))
        print('%d epoch train loss: %.8f' % (epoch + 1, running_loss / len(train_loader)))

        # calculate validation loss
        with torch.no_grad():
            running_val_loss = 0.0
            for j, v_data in enumerate(valid_loader):
                # get the data from dataloader and move it to gpu
                inputs, labels = data['image'].to(device), data['heatmap'].to(device)

                # forward
                y, feature = net(inputs.float())
                loss = loss_function(y[:, :, :, 0], labels.float())

                # accumulate val loss
                running_val_loss += loss.item()

            # show and save current train loss
            val_loss.append(running_val_loss / len(valid_loader))
            print('%d epoch validation loss: %.8f' % (epoch + 1, running_val_loss / len(valid_loader)))

    print('Finished Training')
