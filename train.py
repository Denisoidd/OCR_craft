import argparse
import torch

from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from craft import CRAFT
from test import str2bool, copyStateDict
from dataset_loader import CheckDataset
from transformers import Rescale, ToTensor
from refinenet import RefineNet
from combined_net import CombineNet

parser = argparse.ArgumentParser(description="CRAFT Fine Tuning")
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--num_of_epochs', default=15, type=int)
parser.add_argument('--cuda', default=True, type=str2bool)
parser.add_argument('--tr_anns', default='C:/Users/denis/Desktop/probation/train/ann/', type=str)
parser.add_argument('--tr_images', default='C:/Users/denis/Desktop/probation/train/images', type=str)
parser.add_argument('--v_anns', default='C:/Users/denis/Desktop/probation/val/ann/', type=str)
parser.add_argument('--v_images', default='C:/Users/denis/Desktop/probation/val/images', type=str)
parser.add_argument('--t_anns', default='C:/Users/denis/Desktop/probation/test/ann/', type=str)
parser.add_argument('--t_images', default='C:/Users/denis/Desktop/probation/test/images', type=str)
parser.add_argument('--b_s', default=16, type=int)
parser.add_argument('--val_b_s', default=16, type=int)
parser.add_argument('--freeze', default=True, type=bool)
parser.add_argument('--net_save_path', default='experiments/experiment_15ep_16bs_105lr/net.pth', type=str)
parser.add_argument('--refine_net_save_path', default='experiments/experiment_15ep_16bs_105lr/refine_net.pth', type=str)
parser.add_argument('--create_test_mAP', default=True, type=bool)
parser.add_argument('--gt_dir_mAP', default='input/ground-truth', type=str)
parser.add_argument('--refine', default=True, type=bool)
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    # choose device
    device = 'cpu'
    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Your device is", device)

    # combined net
    c_net = CombineNet().float()

    # load net and move it to device
    c_net.to(device)
    # net = CRAFT().float()
    # net.to(device)

    # refine net
    # refine_net = RefineNet().float()
    # refine_net.to(device)
    print("Model successfully loaded")

    # load pretrained model
    if args.cuda:
        # base net
        c_net.base_net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        # refine net
        c_net.refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        print("Model weights loaded on gpu")
    else:
        # base net
        c_net.base_net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
        # refine net
        c_net.refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
        print("Model weights loaded on cpu")

    # freeze all layers without few last of them
    if args.freeze:
        for name, param in c_net.named_parameters():
            if 'base_net' in name:
                param.requires_grad = False
            print(name, param.requires_grad)

    # if args.freeze:
    #     for name, param in refine_net.named_parameters():
    #         print(name, param.requires_grad)

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
    optimizer = optim.Adam(c_net.parameters(), lr=1e-5)
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
            # base net
            # y, feature = net(inputs.float())
            # refine net
            y_refiner = c_net(inputs.float())
            loss = loss_function(y_refiner[:, :, :, 0], labels.float())
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
                # y, feature = net(inputs.float())
                y_refiner = c_net(inputs.float())
                loss = loss_function(y_refiner[:, :, :, 0], labels.float())

                # accumulate val loss
                running_val_loss += loss.item()

            # show and save current train loss
            val_loss.append(running_val_loss / len(valid_loader))
            print('%d epoch validation loss: %.8f' % (epoch + 1, running_val_loss / len(valid_loader)))

        print("Saving the model")
        # saving the model
        torch.save(c_net.base_net.state_dict(), args.net_save_path)
        torch.save(c_net.refine_net.state_dict(), args.refine_net_save_path)

    print('Finished Training')
