"""
@author: Nian Wu and Miaomiao Zhang

This file is the code for the training of 2D NeurEPDiff.
The NeurEPDiff is trained in low-dimensional bandlimited space.
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/wn/hzfcode/fourier_neural_operator-master/NeurDPDiff")
from utilities import *

from timeit import default_timer
from tensorboardX import SummaryWriter
from Adam import Adam

torch.manual_seed(0);np.random.seed(0)

class H_GobalConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(H_GobalConv, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = 0.01
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 2*self.modes1, 2*self.modes2-2))
        
    # Complex multiplication
    def compl_mul2d(self, input, weights): #input:[1, 20, 12, 12]   weights:[20, 20, 12, 12]
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)  #[20, 20, 12, 12]

    def forward(self, x): #[20, 20, 64, 64]
        """ 
        truncated the signals to fixed modes before complex convolution;
        the truncated signals are approximate to zero
        """
        x_mode1 = x[:,:,:self.modes1, :self.modes1]
        x_mode2 = x[:,:,-self.modes1:, :self.modes1]
        x_mode = torch.cat((x_mode1,x_mode2), dim=2)


        """ complex convolution: multiply two signals in a real space """
        x_mode = torch.fft.irfft2(x_mode) #出：[20, 20, 16, 14]
        x_mode = self.compl_mul2d(x_mode, self.weights)     #出：[20, 20, 16, 14]
        x_mode = torch.fft.rfft2(x_mode) #出：[20, 20, 16, 8]
        

        """ recover the truncated signals by padding zeros to """
        x_pad = torch.zeros_like(x)
        x_pad[:,:,:self.modes1, :self.modes1] = x_mode[:,:,:self.modes1, :self.modes1]
        x_pad[:,:,-self.modes1:, :self.modes1] = x_mode[:,:,-self.modes1:, :self.modes1]
        x = x_pad
        return x

class NeurEPDiff2D(nn.Module):
    def __init__(self, modes1=8, modes2=8, width=20):
        super(NeurEPDiff2D, self).__init__()

        """
        The overall network of Gθ. 
        Gθ := Q˜ ◦ σ(W˜J , H˜J) ◦ · · · ◦ σ(W˜1, H˜1) ◦ P˜

        input: bandlimited initial velocity
        input shape: (batchsize, x=64, y=64, c=2)
        output: velocity in the next step
        output shape: (batchsize, x=64, y=64, c=1)

        1. Encoder P˜: Lift the input to the high channel dimension
        2. Iterative evolution layer
        3. Decoder Q˜: Project from high channel space to the original space
        """
        self.fcgrid = nn.Linear(2, 2)
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.scale = 0.5

        # encoder P˜
        self.fc0 = nn.Parameter(self.scale * torch.rand(2, self.width, 1, 1, dtype=torch.cfloat))
        
        # a non-local convolution kernel H˜
        self.conv0 = H_GobalConv(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = H_GobalConv(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = H_GobalConv(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = H_GobalConv(self.width, self.width, self.modes1, self.modes2)

        # a local linear transform W˜
        self.w0 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, dtype=torch.cfloat))
        self.w1 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, dtype=torch.cfloat))
        self.w2 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, dtype=torch.cfloat))
        self.w3 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, dtype=torch.cfloat))

        # decoder Q˜
        self.fc1 = nn.Parameter(self.scale * torch.rand(self.width, 128,  1, 1, dtype=torch.cfloat))
        self.fc2 = nn.Parameter(self.scale * torch.rand(128, 2, 1, 1, dtype=torch.cfloat))

    def forward(self, x):  #[b, 2, 2*modes, modes]
        ##### Encoder P˜: lift to high channel space   #####
        x = torch.einsum("bi...,io...->bo...", x, self.fc0)   #  [b, 4, 16, 8] x [4,20] = [b, 20, 16, 8]


        #####    Iterative evolution layer   #####
        ##########  1   ##########
        x1 = self.conv0(x)                                    #[b, 20, 16, 8]                  
        x2 = torch.einsum("bi...,io...->bo...", x, self.w0)  #[b, 20, 16, 8]                         
        x = x1 + x2
        # x = x*SmoothOper
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        x = x*SmoothOper
        ##########   2   ##########
        x1 = self.conv1(x)                     
        x2 = torch.einsum("bi...,io...->bo...", x, self.w1)                       
        x = x1 + x2
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        x = x*SmoothOper
        ##########   3   ##########
        x1 = self.conv2(x)                     
        x2 = torch.einsum("bi...,io...->bo...", x, self.w2)                        
        x = x1 + x2
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        x = x*SmoothOper
        ##########   4   ##########
        x1 = self.conv3(x)                     
        x2 = torch.einsum("bi...,io...->bo...", x, self.w3)                         
        x = x1 + x2                                          #[b, 20, 16, 8]


        ##### Decoder P˜: project back to original channel space   #####      
        x = torch.einsum("bi...,io...->bo...", x, self.fc1) #[b, 128, 16, 8]
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        x = x*SmoothOper
        x = torch.einsum("bi...,io...->bo...", x, self.fc2) #[b, 2, 16, 8]             
        return x


################################################################
##########                  configs                   ##########
################################################################
modes = 8
###########  SmoothOper ###########
gamma = 3.0;alpha = 3.0;lpow=3.0
EP = Epdiff(device, (16,16,1), (64,64,1),alpha,gamma,lpow)
SmoothOper, SharpOper = EP.SmoothOper (modes, 1)  #[1, 16, modes]


ntrain = 1000;ntest = 100;epochs = 1000
learning_rate = 0.04; scheduler_step = 100; scheduler_gamma = 0.5; 
T_in = 1; T = 10; imageSize = 64
print(epochs, learning_rate, scheduler_step, scheduler_gamma)

snapshot_path = '/newdisk/wn/DataSet/NP/model/FDD2_64_'+str(T)+'_Velocity'
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)


MODE = "train";batch_size = 20
myloss = LpLossF(size_average=False)
path_model = snapshot_path+ '/Data1000-_alpha{}_lpow{}_gamma{}_lr{}_ep{}-smooth1-nogrid.pth'.format(alpha, lpow, gamma, learning_rate, epochs)  
final_model =  snapshot_path+ '/Data1000-_alpha{}_lpow{}_gamma{}_lr{}_ep{}-smooth1-nogrid.pth'.format(alpha, lpow, gamma, learning_rate, epochs)


################################################################
# load data
################################################################
TRAIN_PATH = '/newdisk/wn/DataSet/NP/EyeBigDiff/velocity64_data_1200.mat'
reader = MatReader(TRAIN_PATH)
gedata = reader.read_field('v')  #[1200, 64, 64, 3, 11]
##################     train    ################################
train_a = gedata[:ntrain,:,:,0:2,:T_in]  #torch.Size([1000, 64, 64, 2, 1])
train_u = gedata[:ntrain,:,:,0:2,T_in:]  #torch.Size([1000, 64, 64, 2, 10])
##################     test     ################################
test_a = gedata[-ntest:,:,:,0:2,:T_in]   #[200, 64, 64, 2, 1]
test_u = gedata[-ntest:,:,:,0:2,T_in:]  #[200, 64, 64, 2, 10]
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

################################################################
# mode configuration
################################################################
model = NeurEPDiff2D().cuda()
print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
minloss = 9999999999



if MODE == "train":
    writer = SummaryWriter(snapshot_path + '/log/smooth1_alpha{}_lpow{}_gamma{}_lr{}_ep{}'.format(alpha, lpow, gamma, learning_rate, epochs))
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy in train_loader:  #xx: [20, 64, 64, 2, 1]  yy:[20, 64, 64, 2, 10]
            loss = 0
            # full dimensional velocity
            xx = xx.squeeze(-1).to(device)  #输入：             [20, 64, 64, 2]
            yy = yy.to(device)  #GroundTruth        [20, 64, 64, 2, 10]

            #get initial velocity: bandlimited fourier velocity
            xx = xx.permute(0, 3, 1, 2)   #[20, 2, 64, 64]
            x_ft = torch.fft.rfft2(xx)/(imageSize*imageSize)
            x_trun1 = x_ft[:, :, :modes, :modes]  #[1, 2, modes, modes]
            x_trun2 = x_ft[:, :, -modes:, :modes]  #[1, 2, modes, modes]
            xx = torch.cat((x_trun1,x_trun2),dim=-2)       #[1, 2, 2*modes, modes]   [20, 2, 16, 8]

            #supervise: v1 -> vt
            for t in range(0, T):
                y = yy[..., t]  #yy:[20, 64, 64, 2, 10]   y:[20, 64, 64, 2]
        
                #ground truth: get the bandlimited fourier velocity
                y = y.permute(0, 3, 1, 2)   #[20, 2, 64, 64]
                y_ft = torch.fft.rfft2(y)/(imageSize*imageSize)
                y_trun1 = y_ft[:, :, :modes, :modes]  #[1, 2, modes, modes]
                y_trun2 = y_ft[:, :, -modes:, :modes]  #[1, 2, modes, modes]
                y = torch.cat((y_trun1,y_trun2),dim=-2)       #[1, 2, 2*modes, modes] [20, 2, 16, 8]

                # predict
                im = model(xx)           #[20, 2, 16, 8]
                loss += myloss(im.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1))
                xx = im

            train_l2_step += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        """ if train_l2_step < minloss:
            minloss = train_l2_step
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path_model) """

        test_l2_step = 0
        with torch.no_grad():
            model.eval()

            for xx, yy in test_loader:
                loss = 0
                # full dimensional velocity
                xx = xx.squeeze(-1).to(device)
                yy = yy.to(device)

                #get initial velocity: bandlimited fourier velocity
                xx = xx.permute(0, 3, 1, 2)   #[20, 2, 64, 64]
                x_ft = torch.fft.rfft2(xx)/(imageSize*imageSize)
                x_trun1 = x_ft[:, :, :modes, :modes]  #[1, 2, modes, modes]
                x_trun2 = x_ft[:, :, -modes:, :modes]  #[1, 2, modes, modes]
                xx = torch.cat((x_trun1,x_trun2),dim=-2)       #[1, 2, 2*modes, modes]

                for t in range(0, T):
                    y = yy[..., t]
                    # full dimensional velocity
                    y = y.permute(0, 3, 1, 2)   #[20, 2, 64, 64]

                    #ground truth: bandlimited fourier velocity
                    y_ft = torch.fft.rfft2(y)/(imageSize*imageSize)
                    y_trun1 = y_ft[:, :, :modes, :modes]  #[1, 2, modes, modes]
                    y_trun2 = y_ft[:, :, -modes:, :modes]  #[1, 2, modes, modes]
                    y = torch.cat((y_trun1,y_trun2),dim=-2)       #[1, 2, 2*modes, modes] [20, 2, 16, 8]

                    im = model(xx)
                    loss += myloss(im.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1))
                    xx = im

                test_l2_step += loss.item()


        t2 = default_timer()
        scheduler.step()

        #一个epoch结束后绘制图  +  保存模型
        writer.add_scalar('info/lr', scheduler.get_last_lr()[0], ep)
        writer.add_scalar('info/train_loss', train_l2_step / ntrain / T, ep)
        writer.add_scalar('info/ts_loss', test_l2_step / ntest / T, ep)

        """ print(ep, t2 - t1, train_l2_step / ntrain / T, test_l2_step / ntest / T)
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, final_model) """


