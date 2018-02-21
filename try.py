'''
author: Yilun Zhang
'''


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataloader import FaceLandmarksDataset
from torch.nn import functional as F




# Hyper Parameters
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    num_epochs = 100
    batch_size = 1
    learning_rate = 0.0001


    trans_im = transforms.Compose(
                       [transforms.ToPILImage(),
                        transforms.Scale((40,40)),
                        transforms.ToTensor(),
                        transforms.Normalize((89.93/255, 99.5/255, 119.78/255), (1., 1., 1.))])

    trans_label=transforms.Compose(
                       [
                        transforms.Scale((40,40))])
                        # transforms.ToTensor(),])



    up_label = nn.Upsample(size=(40, 40), mode='bilinear')
    train_dataset = FaceLandmarksDataset(csv_file='./training.txt',
                                        root_dir='MTFL/', trans_im=trans_im, trans_label=trans_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True,num_workers=2)

    test_dataset = FaceLandmarksDataset(csv_file='./testing.txt',
                                         root_dir='MTFL/', trans_im=trans_im, trans_label=trans_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)


    # CNN Model (2 conv layer)
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5,stride=1,padding=2),
                nn.BatchNorm2d(num_features=64,momentum=0.9),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(num_features=64, momentum=0.9),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0))
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5,stride=1,padding=2),
                nn.BatchNorm2d(num_features=128, momentum=0.9),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(num_features=128, momentum=0.9),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0))
            self.layer3 = nn.Sequential(
                nn.Conv2d(128,384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=384, momentum=0.9),
                nn.ReLU(),
                nn.Conv2d(384,384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=384, momentum=0.9),
                nn.ReLU(),
                nn.Conv2d(384,5, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=5, momentum=0.9),
                nn.Sigmoid())

            self.layer4= nn.Sequential(
                nn.Upsample(size=(40,40), mode='bilinear')
            )






        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            # out = out.view(out.size(0), -1)

            return out

    # #
    # cnn = CNN()
    #
    # cnn.load_state_dict(torch.load('cnn64.pkl'))
    # cnn.cuda()
    # # Loss and Optimizer
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum = 0.99, weight_decay= 5e-4)
    #
    # # Train the Model
    # for epoch in range(65,num_epochs):
    #     for i,[images, labels] in enumerate(train_loader):
    #         images = Variable(images).cuda()
    #         labels = Variable(labels).cuda()
    #         # print(1)
    #         # Forward + Backward + Optimize
    #         optimizer.zero_grad()
    #         outputs = cnn(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         if (i + 1) % 100 == 0:
    #             print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
    #                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
    #     torch.save(cnn.state_dict(), 'cnn%d.pkl'%(epoch))

    # Test the Model
    cnn = CNN()
    cnn.load_state_dict(torch.load('cnn78.pkl'))
    # cnn = torch.load('cnn4.pkl')
    cnn.cuda()
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        outputs = cnn(images)
        print (1)


    # print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

    # Save the Trained Model
    # torch.save(cnn.state_dict(), 'cnn.pkl')