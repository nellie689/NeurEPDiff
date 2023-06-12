"""
@author: Nian Wu and Miaomiao Zhang

This file is the code for the training of 3D NeurEPDiff.
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

################################################################
# fourier layer
################################################################
class H_GobalConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(H_GobalConv, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 0.01               #1/(20*20) = 0.0025
        self.scale = 0.005              #1/(30*30) = 0.0011
        
        #2023
        #weight1 -------------------------------------
        self.scale = 0.001
        # #weight2 -------------------------------------
        # self.scale = 0.0005


        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 2*self.modes1, 2*self.modes2, 2*self.modes3-2))


    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):  #x: [20, 12, 16, 16, 8]
        """ x = torch.fft.irfftn(x, dim=(-3,-2,-1)) #出：[20, 12, 16, 16, 14]
        x = self.compl_mul3d(x, self.weights)
        x = torch.fft.rfftn(x, dim=(-3,-2,-1)) #出：[20, 12, 16, 16, 8]
        return x """


        x_mode1 = x[:, :, :self.modes1, :self.modes2, :self.modes3]  #[1, 2, modes, modes]
        x_mode2 = x[:, :, -self.modes1:, :self.modes2, :self.modes3]  #[1, 2, modes, modes]
        x_mode3 = x[:, :, :self.modes1, -self.modes2:, :self.modes3]  #[1, 2, modes, modes]
        x_mode4 = x[:, :, -self.modes1:, -self.modes2:, :self.modes3]  #[1, 2, modes, modes]
        x_mode11 = torch.cat((x_mode1,x_mode2),dim=-3) 
        x_mode22 = torch.cat((x_mode3,x_mode4),dim=-3) 
        x_mode = torch.cat((x_mode11,x_mode22),dim=-2)

        x_mode = torch.fft.irfftn(x_mode, dim=(-3,-2,-1)) #出：[20, 12, 16, 16, 14]
        x_mode = self.compl_mul3d(x_mode, self.weights)
        x_mode = torch.fft.rfftn(x_mode, dim=(-3,-2,-1)) #出：[20, 12, 16, 16, 8]

        x_pad = torch.zeros_like(x)
        x_pad[:, :, :self.modes1, :self.modes2, :self.modes3] = x_mode[:, :, :self.modes1, :self.modes2, :self.modes3]
        x_pad[:, :, -self.modes1:, :self.modes2, :self.modes3] = x_mode[:, :, -self.modes1:, :self.modes2, :self.modes3]
        x_pad[:, :, :self.modes1, -self.modes2:, :self.modes3] = x_mode[:, :, :self.modes1, -self.modes2:, :self.modes3]
        x_pad[:, :, -self.modes1:, -self.modes2:, :self.modes3] = x_mode[:, :, -self.modes1:, -self.modes2:, :self.modes3]
        x = x_pad

        return x

class NeurEPDiff3D(nn.Module):
    def __init__(self, modes1=8, modes2=8, modes3=8, width=20):
        super(NeurEPDiff3D, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        # self.modes1 = modes1
        # self.modes2 = modes2
        # self.modes3 = modes3

        self.modes1 = 8
        self.modes2 = 8
        self.modes3 = 8

        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fcgrid = nn.Linear(3, 3)


        self.scale = 0.1
        self.scale = 0.05

        #2023
        #weight1 -------------------------------------
        # self.scale = 0.01   #开头不好
        self.scale = 0.1
        # #weight2 -------------------------------------
        # self.scale = 0.05

        # self.fcgrid = nn.Parameter(self.scale * torch.rand(3, 3, 1, 1, 1, dtype=torch.cfloat))
        # self.fc0 = nn.Parameter(self.scale * torch.rand(6, self.width, 1, 1, 1, dtype=torch.cfloat))
        self.fc0 = nn.Parameter(self.scale * torch.rand(3, self.width, 1, 1, 1, dtype=torch.cfloat))

        self.conv0 = H_GobalConv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = H_GobalConv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = H_GobalConv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = H_GobalConv(self.width, self.width, self.modes1, self.modes2, self.modes3)
     

        # self.w0 = nn.Conv3d(self.width, self.width, 1)
        # self.w1 = nn.Conv3d(self.width, self.width, 1)
        # self.w2 = nn.Conv3d(self.width, self.width, 1)
        # self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.w0 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, 1, dtype=torch.cfloat))
        self.w1 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, 1, dtype=torch.cfloat))
        self.w2 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, 1, dtype=torch.cfloat))
        self.w3 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, 1, dtype=torch.cfloat))
       


        # self.modRelu0 = modRelu(self.width)
        # self.modRelu1 = modRelu(self.width)
        # self.modRelu2 = modRelu(self.width)
        # self.modRelu3 = modRelu(128)

        # self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 3)
        self.fc1 = nn.Parameter(self.scale * torch.rand(self.width, 128,  1, 1, 1, dtype=torch.cfloat))
        self.fc2 = nn.Parameter(self.scale * torch.rand(128, 3, 1, 1, 1, dtype=torch.cfloat))

    def forward(self, x):  #previous : x:[20, 3, 16, 16, 8]
        
        """ ######  11 grid3+fc  ###########
        grid = self.get_grid3(x.permute(0,2,3,4,1).shape, x.device)  #[20, 16, 16, 8, 3]->[20, 3, 16, 16, 8]
        grid = self.fcgrid(grid).permute(0,4,1,2,3)
        grid = torch.complex(grid, torch.zeros(grid.shape,device=grid.device, requires_grad=True))

        x = torch.cat((x, grid), dim=1)    #[20, 6, 16, 16, 8] """

        #####   lift to high dimension   #####
        x = torch.einsum("bi...,io...->bo...", x, self.fc0)
        


        #####   1   #####
        x1 = self.conv0(x)                                    #[b, 20, 16, 16, 8]                  
        x2 = torch.einsum("bi...,io...->bo...", x, self.w0)  #[b, 20, 16, 16, 8]                     
        x = x1 + x2
        x = x*SmoothOper
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        # x = self.modRelu0(x)

        # #####   2   #####
        x1 = self.conv1(x)                     
        x2 = torch.einsum("bi...,io...->bo...", x, self.w1)                       
        x = x1 + x2
        x = x*SmoothOper
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        # x = self.modRelu1(x)
        
        # #####   3   #####
        x1 = self.conv2(x)                     
        x2 = torch.einsum("bi...,io...->bo...", x, self.w2)                        
        x = x1 + x2
        x = x*SmoothOper
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        # x = self.modRelu2(x)
        
        #####   4  #####
        x1 = self.conv3(x)                     
        x2 = torch.einsum("bi...,io...->bo...", x, self.w3)                         
        x = x1 + x2
        x = x*SmoothOper
    
        ##[b, 20, 16, 8]
        # x = x.permute(0, 2, 3, 1)               #[b, 20, 16, 8]                        
        x = torch.einsum("bi...,io...->bo...", x, self.fc1) #[b, 128, 16, 16, 8]
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        # x = self.modRelu3(x)

        x = torch.einsum("bi...,io...->bo...", x, self.fc2) #[b, 3, 16, 16, 8]


        return x

    def get_grid(self, shape, device):  #shape : [20, 64, 64, 64, 3]
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    def get_grid2(self, shape, device, gridsize=64):  #shape : [20, 64, 64, 64, 3]
        batchsize, size_x, size_y, size_z = shape[0], gridsize, gridsize, gridsize
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1).to(device)
        trun1 = grid[:, :self.modes1, :self.modes2, :self.modes3, :]  #[1, 2, modes, modes]
        trun2 = grid[:, -self.modes1:, :self.modes2, :self.modes3, :]  #[1, 2, modes, modes]
        trun3 = grid[:, :self.modes1, -self.modes2:, :self.modes3, :]  #[1, 2, modes, modes]
        trun4 = grid[:, -self.modes1:, -self.modes2:, :self.modes3, :]  #[1, 2, modes, modes]
        yy1 = torch.cat((trun1,trun2),dim=-4) 
        yy2 = torch.cat((trun3,trun4),dim=-4) 
        trunr = torch.cat((yy1,yy2),dim=-3)
        return trunr
        trunI = torch.zeros(trunr.shape, dtype=trunr.dtype).to(device)

        return torch.complex(trunr, trunI)

    def get_grid3(self, shape, device, gridsize=64):  #shape : [20, 64, 64, 64, 3]
        batchsize, size_x, size_y, size_z = shape[0], gridsize, gridsize, gridsize
        gridx = torch.fft.fftfreq(size_x)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])

        gridy = torch.fft.fftfreq(size_y)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])

        gridz = torch.fft.fftfreq(size_z)

        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1).to(device)

        trun1 = grid[:, :self.modes1, :self.modes2, :self.modes3, :]  #[1, 2, modes, modes]
        trun2 = grid[:, -self.modes1:, :self.modes2, :self.modes3, :]  #[1, 2, modes, modes]
        trun3 = grid[:, :self.modes1, -self.modes2:, :self.modes3, :]  #[1, 2, modes, modes]
        trun4 = grid[:, -self.modes1:, -self.modes2:, :self.modes3, :]  #[1, 2, modes, modes]
        yy1 = torch.cat((trun1,trun2),dim=-4) 
        yy2 = torch.cat((trun3,trun4),dim=-4) 
        trunr = torch.cat((yy1,yy2),dim=-3)
        return trunr
        trunI = torch.zeros(trunr.shape, dtype=trunr.dtype).to(device)
        return torch.complex(trunr, trunI)


    def get_grid_embedding(self, gridsize=64):  #shape : [20, 64, 64, 64, 3]
        batchsize, size_x, size_y, size_z = 1, gridsize, gridsize, gridsize
        gridx = torch.fft.fftfreq(size_x)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])

        gridy = torch.fft.fftfreq(size_y)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])

        gridz = torch.fft.fftfreq(size_z)

        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1).to(device)

        trun1 = grid[:, :self.modes1, :self.modes2, :self.modes3, :]  #[1, 2, modes, modes]
        trun2 = grid[:, -self.modes1:, :self.modes2, :self.modes3, :]  #[1, 2, modes, modes]
        trun3 = grid[:, :self.modes1, -self.modes2:, :self.modes3, :]  #[1, 2, modes, modes]
        trun4 = grid[:, -self.modes1:, -self.modes2:, :self.modes3, :]  #[1, 2, modes, modes]
        yy1 = torch.cat((trun1,trun2),dim=-4) 
        yy2 = torch.cat((trun3,trun4),dim=-4) 
        trunr = torch.cat((yy1,yy2),dim=-3)
        return trunr

################################################################
##########                  configs                   ##########
################################################################
modes = 8
###########  SmoothOper ###########
gamma = 1.0;alpha = 2.0;lpow=2.0
EP = Epdiff(device, (16, 16, 16), (64, 64, 64),  alpha,gamma,lpow)
SmoothOper, SharpOper = EP.SmoothOper (modes, 1)  #[1, 16, 16, modes]

ntrain = 1500;ntest = 100;epochs = 1000
learning_rate = 0.006;scheduler_step = 100;scheduler_gamma = 0.8
T = 10; imageSize = 64
print(epochs, learning_rate, scheduler_step, scheduler_gamma)


snapshot_path = '/newdisk/wn/DataSet/NP/model/FDD3_64_'+str(T)+'_Velocity'
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)


MODE = "train";batch_size = 20
myloss = LpLossF(size_average=False)
path_model =  snapshot_path+ '/lr{}_{}_{}_ep{}-nogrid-smooth2_{}_{}.pth'.format(learning_rate, scheduler_gamma, scheduler_step, epochs, lpow, alpha)
final_model =  snapshot_path+ '/lr{}_{}_{}_ep{}-nogrid-smooth2_{}_{}.pth'.format(learning_rate, scheduler_gamma, scheduler_step, epochs, lpow, alpha)


################################################################
# load data
################################################################
train_data = Field_dataset("/newdisk/wn/DataSet/NP/Brain64", "train", "velocity", ntrain)
test_data = Field_dataset("/newdisk/wn/DataSet/NP/Brain64", "test", "velocity", ntest)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)



################################################################
# mode configuration
################################################################
model = NeurEPDiff3D().cuda()
print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
minloss = 9999999999



if MODE == "train":
    writer = SummaryWriter(snapshot_path + '/log/smooth2_alpha{}_lpow{}_gamma{}_lr{}_ep{}'.format(alpha, lpow, gamma, learning_rate, epochs))
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        prefetcher_train = data_prefetcher(train_loader)
        data = prefetcher_train.next()
        while data is not None:
        # for data in train_loader:
            loss = 0
            # full dimensional velocity
            xx = data[...,0].to(device)  #[20, 64, 64, 64, 3]
            yy = data[...,1:].to(device) #[20, 64, 64, 64, 3, 10]

            #get initial velocity: bandlimited fourier velocity
            xx = xx.permute(0, 4, 1, 2, 3)   #[20, 3, 64, 64, 64]
            x_ft = torch.fft.rfftn(xx, dim=(-3,-2,-1))/(64*64*64)
            x_trun1 = x_ft[:, :, :modes, :modes, :modes]  #[1, 3, modes, modes, modes]
            x_trun2 = x_ft[:, :, -modes:, :modes, :modes]  #[1, 3, modes, modes, modes]
            x_trun3 = x_ft[:, :, :modes, -modes:, :modes]  #[1, 3, modes, modes, modes]
            x_trun4 = x_ft[:, :, -modes:, -modes:, :modes]  #[1, 3, modes, modes, modes]
            xx1 = torch.cat((x_trun1, x_trun2),dim=-3) 
            xx2 = torch.cat((x_trun3, x_trun4),dim=-3) 
            xx = torch.cat((xx1, xx2),dim=-2)       #[1, 3, 2*modes, 2*modes, modes]

            #supervise: v1 -> vt
            for t in range(0, T):
                y = yy[..., t]  #y: [20, 64, 64, 64, 3]
        
                #ground truth: get the bandlimited fourier velocity
                y = y.permute(0, 4, 1, 2, 3)   #[20, 3, 64, 64, 64]
                y_ft = torch.fft.rfftn(y, dim=(-3,-2,-1))/(imageSize*imageSize*imageSize)
                y_trun1 = y_ft[:, :, :modes, :modes, :modes]  #[1, 2, modes, modes]
                y_trun2 = y_ft[:, :, -modes:, :modes, :modes]  #[1, 2, modes, modes]
                y_trun3 = y_ft[:, :, :modes, -modes:, :modes]  #[1, 2, modes, modes]
                y_trun4 = y_ft[:, :, -modes:, -modes:, :modes]  #[1, 2, modes, modes]
                yy1 = torch.cat((y_trun1,y_trun2),dim=-3) 
                yy2 = torch.cat((y_trun3,y_trun4),dim=-3) 
                y = torch.cat((yy1,yy2),dim=-2)       #[1, 2, 2*modes, 2*modes, modes]


                # predict
                im = model(xx)           #[20, 3, 16, 16, 8]
                loss += myloss(im.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1))
                
                xx = im

            train_l2_step += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            data = prefetcher_train.next()

        """ if train_l2_step < minloss:
            minloss = train_l2_step
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path_model) """

        test_l2_step = 0
        with torch.no_grad():
            model.eval()

            for data in test_loader:
                loss = 0
                # full dimensional velocity
                xx = data[...,0].to(device)  #[20, 64, 64, 64, 3]
                yy = data[...,1:].to(device)


                #get initial velocity: bandlimited fourier velocity
                xx = xx.permute(0, 4, 1, 2, 3)   #[20, 3, 64, 64, 64]
                x_ft = torch.fft.rfftn(xx, dim=(-3,-2,-1))/(64*64*64)
                x_trun1 = x_ft[:, :, :modes, :modes, :modes]  #[1, 3, modes, modes, modes]
                x_trun2 = x_ft[:, :, -modes:, :modes, :modes]  #[1, 3, modes, modes, modes]
                x_trun3 = x_ft[:, :, :modes, -modes:, :modes]  #[1, 3, modes, modes, modes]
                x_trun4 = x_ft[:, :, -modes:, -modes:, :modes]  #[1, 3, modes, modes, modes]
                
                xx1 = torch.cat((x_trun2,x_trun1),dim=-3) 
                xx2 = torch.cat((x_trun4,x_trun3),dim=-3) 
                xx = torch.cat((xx2,xx1),dim=-2)       #[1, 3, 2*modes, 2*modes, modes]


                for t in range(0, T):
                    y = yy[..., t]
                    #ground truth: bandlimited fourier velocity
                    y = y.permute(0, 4, 1, 2, 3)   #[20, 3, 64, 64, 64]

                    #ground truth: bandlimited fourier velocity
                    y_ft = torch.fft.rfftn(y, dim=(-3,-2,-1))/(64*64*64)
                    y_trun1 = y_ft[:, :, :modes, :modes, :modes]   #[1, 3, modes, modes, modes]
                    y_trun2 = y_ft[:, :, -modes:, :modes, :modes]  #[1, 3, modes, modes, modes]
                    y_trun3 = y_ft[:, :, :modes, -modes:, :modes]  #[1, 3, modes, modes, modes]
                    y_trun4 = y_ft[:, :, -modes:, -modes:, :modes] #[1, 3, modes, modes, modes]
                    yy1 = torch.cat((y_trun2,y_trun1),dim=-3) 
                    yy2 = torch.cat((y_trun4,y_trun3),dim=-3) 
                    y = torch.cat((yy2,yy1),dim=-2)       #[1, 3, 2*modes, 2*modes, modes]

                    im = model(xx)
                    loss += myloss(im.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1))
                    xx = im

                test_l2_step += loss.item()
                

        t2 = default_timer()
        scheduler.step()

        #一个epoch结束后绘制图  +  保存模型
        writer.add_scalar('info/lr', scheduler.get_last_lr()[0], ep)
        writer.add_scalar('info/train_loss', train_l2_step / ntrain / (T), ep)
        writer.add_scalar('info/train_full', train_l2_full / ntrain, ep)
        print(ep, t2 - t1, train_l2_step / ntrain / (T), train_l2_full / ntrain)

    """ torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, final_model) """


