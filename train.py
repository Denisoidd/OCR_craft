import argparse
import torch

from craft import CRAFT
from test import str2bool, copyStateDict
from dataset_loader import CheckDataset

parser = argparse.ArgumentParser(description="CRAFT Fine Tuning")
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--num_of_epochs', default=2, type=int)
parser.add_argument('--cuda', default=True, type=str2bool)
parser.add_argument('--step_to_show', default=2000, type=int)
parser.add_argument('--anns', default='C:/Users/denis/Desktop/probation/ann', type=str)
parser.add_argument('--images', default='C:/Users/denis/Desktop/probation/train', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    # load net
    net = CRAFT()

    # load pretrained model
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    # create dataloader
    # TODO: Create dataloader by creating another class. Then use Dataloader for batching
    trainloader = CheckDataset(args.anns, args.images)

    # define loss and optimizer
    # TODO: Create loss function mb class
    optimizer = None
    loss_function = None

    # TODO: freeze first layers
    # TODO: write mAP

    # train loop
    for epoch in range(args.num_of_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % args.step_to_show == (args.step_to_show - 1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / args.step_to_show))
                running_loss = 0.0

    print('Finished Training')
