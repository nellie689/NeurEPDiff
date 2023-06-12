"""
@author: Nian Wu and Miaomiao Zhang

This file is NeurEPDiff to perform forward shooting for 3D Brain MRIs, i.e., predict the velocity sequence v1->vt, given an initial velocity. 
All prediction are implemented in bandlimited space.
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/wn/hzfcode/fourier_neural_operator-master")
import SimpleITK as sitk
from utilities import *


class H_GobalConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(H_GobalConv, self).__init__()
        """
        H kernel: complex global convolutional kernel.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 0.001         
        
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 2*self.modes1, 2*self.modes2, 2*self.modes3-2))


    # Complex multiplication
    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        """ 
        truncated the signals to fixed modes before complex convolution;
        the truncated signals are approximate to zero
        """
        x_mode1 = x[:, :, :self.modes1, :self.modes2, :self.modes3]
        x_mode2 = x[:, :, -self.modes1:, :self.modes2, :self.modes3]
        x_mode3 = x[:, :, :self.modes1, -self.modes2:, :self.modes3]
        x_mode4 = x[:, :, -self.modes1:, -self.modes2:, :self.modes3]
        x_mode11 = torch.cat((x_mode1,x_mode2),dim=-3) 
        x_mode22 = torch.cat((x_mode3,x_mode4),dim=-3) 
        x_mode = torch.cat((x_mode11,x_mode22),dim=-2)

        """ complex convolution: multiply two signals in a real space """
        x_mode = torch.fft.irfftn(x_mode, dim=(-3,-2,-1)) #出：[20, 12, 16, 16, 14]
        x_mode = self.compl_mul3d(x_mode, self.weights)
        x_mode = torch.fft.rfftn(x_mode, dim=(-3,-2,-1)) #出：[20, 12, 16, 16, 8]


        """ recover the truncated signals by padding zeros to """                                                                                         
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
        self.fcgrid = nn.Linear(3, 3)
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.scale = 0.1
        
        # Encoder P˜
        self.fc0 = nn.Parameter(self.scale * torch.rand(3, self.width, 1, 1, 1, dtype=torch.cfloat))

        # Non-local convolution kernel H˜
        self.conv0 = H_GobalConv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = H_GobalConv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = H_GobalConv(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = H_GobalConv(self.width, self.width, self.modes1, self.modes2, self.modes3)
    
        # Local linear transform W˜
        self.w0 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, 1, dtype=torch.cfloat))
        self.w1 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, 1, dtype=torch.cfloat))
        self.w2 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, 1, dtype=torch.cfloat))
        self.w3 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, 1, dtype=torch.cfloat))
       
        # Decoder Q˜
        self.fc1 = nn.Parameter(self.scale * torch.rand(self.width, 128,  1, 1, 1, dtype=torch.cfloat))
        self.fc2 = nn.Parameter(self.scale * torch.rand(128, 3, 1, 1, 1, dtype=torch.cfloat))

    def forward(self, x):  #previous : x:[b, 3, 2*modes, 2*modes, modes]
        ##### Encoder P˜: lift to high channel space   #####
        x = torch.einsum("bi...,io...->bo...", x, self.fc0)
        
        
        #####    Iterative evolution layer   #####
        ##########  1   ##########  
        x1 = self.conv0(x)                                    #[b, 20, 16, 16, 8]                  
        x2 = torch.einsum("bi...,io...->bo...", x, self.w0)  #[b, 20, 16, 16, 8]                     
        x = x1 + x2
        x = x*SmoothOper
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        

        # #####   2   #####
        x1 = self.conv1(x)                     
        x2 = torch.einsum("bi...,io...->bo...", x, self.w1)                       
        x = x1 + x2
        x = x*SmoothOper
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
       
        
        # #####   3   #####
        x1 = self.conv2(x)                     
        x2 = torch.einsum("bi...,io...->bo...", x, self.w2)                        
        x = x1 + x2
        x = x*SmoothOper
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
       

        #####   4  #####
        x1 = self.conv3(x)                     
        x2 = torch.einsum("bi...,io...->bo...", x, self.w3)                         
        x = x1 + x2
        x = x*SmoothOper
    
        ##[b, 20, 16, 8]
        # x = x.permute(0, 2, 3, 1)               #[b, 20, 16, 8]                        
        x = torch.einsum("bi...,io...->bo...", x, self.fc1) #[b, 128, 16, 16, 8]
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        
        x = torch.einsum("bi...,io...->bo...", x, self.fc2) #[b, 3, 16, 16, 8]
        return x


################################################################
##########                  configs                   ##########
################################################################
device = torch.device('cpu') # device = torch.device('cuda') 
modes = 8;
# modes = 16;
lpow=2.0;alpha = 2.0;gamma=1.0

EP = Epdiff(device, (16, 16, 16), (64, 64, 64),  alpha,gamma,lpow)
SmoothOper, SharpOper = EP.SmoothOper (modes, 1)  #[1, 16, 16, 8]

##############   load model    ##################################
final_model =  '/newdisk/wn/DataSet/NP/model/FDD3_64_10_Velocity/Data1000-_base_lr0.006_0.8_100_ep1000-nogrid-nodecay-diffweight1-smooth2_2.0_2.0.pth'
model_v = NeurEPDiff3D().to(device)
model_v.load_state_dict(torch.load(final_model,map_location='cpu')['model_state_dict'])
TSteps = 10

imagesize = 192
imagesize = 128
imagesize = 64


def initfun(vx,vy,vz,iter):
    ######### initial velocity: full dimensional spatial space ##################
    '''need:    batch_size,imagesize,imagesize,imagesize,3'''
    VS0 = torch.stack((torch.from_numpy(vx),torch.from_numpy(vy),torch.from_numpy(vz)),dim=-1).unsqueeze(0)  #[1, imagesize, imagesize, imagesize, 3]
    ######### transpose the channel ###############
    v0trans = VS0.permute(0,4,1,2,3)  #[1, 3, imagesize, imagesize, imagesize]
    
    # get the bandlimited fourier velocity #
    x_ft = torch.fft.rfftn(v0trans, dim=(-3,-2,-1))/(imagesize*imagesize*imagesize)    #[1, 2, imagesize, imagesize, int(imagesize/2)+1]
    x_trun1 = x_ft[:, :, :modes, :modes, :modes]  #[1, 3, modes, modes, modes]
    x_trun2 = x_ft[:, :, -modes:, :modes, :modes]  #[1, 3, modes, modes, modes]
    x_trun3 = x_ft[:, :, :modes, -modes:, :modes]  #[1, 3, modes, modes, modes]
    x_trun4 = x_ft[:, :, -modes:, -modes:, :modes]  #[1, 3, modes, modes, modes]
    xx1 = torch.cat((x_trun1, x_trun2),dim=-3) 
    xx2 = torch.cat((x_trun3, x_trun4),dim=-3) 
    vf = torch.cat((xx1, xx2),dim=-2).to(device)   ##[1, 3, 2*modes, 2*modes, modes]
    
    reslist=[]
    with torch.no_grad():
        model_v.eval()
        
        #predict: v1 -> vt
        for t in range(0, TSteps):  #T:9  step:1
            vf = model_v(vf)           #vf: [1, 3, 2*modes, 2*modes, modes]
           
            """ 
            For the sake of ease of implementation, we have transformed Fourier velocity into spatial velocity,
            then pass them to FLASH(C++).
            Nevertheless, in all of our codes, we will be working exclusively with bandlimited Fourier signals.
            """ 
            pd_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device=device)  ##[1, 3, 2*modes, 2*modes, modes]
            pd_ft[:, :, :modes, :modes, :modes]  = vf[:, :, :modes, :modes, :]
            pd_ft[:, :, -modes:, :modes, :modes]  = vf[:, :, -modes:, :modes, :]
            pd_ft[:, :, :modes, -modes:, :modes]  = vf[:, :, :modes, -modes:, :]
            pd_ft[:, :, -modes:, -modes:, :modes]  = vf[:, :, -modes:, -modes:, :]
            vs = torch.fft.irfftn(pd_ft*imagesize*imagesize*imagesize, s=(imagesize,imagesize,imagesize), dim=(-3,-2,-1))  #[1, 3, imagesize, imagesize, imagesize]
            if(device.type=="cuda"):
                V_X = vs[0,0,...].contiguous().cpu().numpy()
                V_Y = vs[0,1,...].contiguous().cpu().numpy()
                V_Z = vs[0,2,...].contiguous().cpu().numpy()
            elif(device.type=="cpu"):
                V_X = vs[0,0,...].contiguous().numpy()
                V_Y = vs[0,1,...].contiguous().numpy()
                V_Z = vs[0,2,...].contiguous().numpy()
                V_Z = vs[0,2,...].contiguous().numpy()
            reslist.append(V_X)
            reslist.append(V_Y)
            reslist.append(V_Z)

    return tuple(reslist)

        
