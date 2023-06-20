"""
@author: Nian Wu and Miaomiao Zhang

This file is NeurEPDiff to perform forward shooting for 2D bull eyes, i.e., predict the velocity sequence v1->vt, given an initial velocity. 
All prediction are implemented in bandlimited space.
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("ROOT_DIRECTORY_OF_NeurEPDiff")
import SimpleITK as sitk
from utilities import *


class H_GobalConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(H_GobalConv, self).__init__()
        """
        H kernel: complex global convolutional kernel.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 0.01
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 2*self.modes1, 2*self.modes2-2))
        
    # Complex multiplication
    def compl_mul2d(self, input, weights): 
        return torch.einsum("bixy,ioxy->boxy", input, weights) 
    
    def forward(self, x):
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

        # Encoder P˜
        self.fc0 = nn.Parameter(self.scale * torch.rand(2, self.width, 1, 1, dtype=torch.cfloat))
        
        # Non-local convolution kernel H˜
        self.conv0 = H_GobalConv(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = H_GobalConv(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = H_GobalConv(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = H_GobalConv(self.width, self.width, self.modes1, self.modes2)

        #Local transform W˜
        self.w0 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, dtype=torch.cfloat))
        self.w1 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, dtype=torch.cfloat))
        self.w2 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, dtype=torch.cfloat))
        self.w3 = nn.Parameter(self.scale * torch.rand(self.width, self.width, 1, 1, dtype=torch.cfloat))

        # Decoder Q˜
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
device = torch.device('cpu') # device = torch.device('cuda') 
gamma = 3.0;alpha = 3.0;lpow=3.0
modes = 8; 
# modes = 16
EP = Epdiff(device, (16,16,1), (64,64,1),alpha,gamma,lpow)
SmoothOper, SharpOper = EP.SmoothOper (modes, 1)  #[1, 16, 8]
final_model =  'PATH_OF_2D_MODEL' #.pth format
                                                                       
######################   load model    #########################
model_v = NeurEPDiff2D().to(device)
model_v.load_state_dict(torch.load(final_model,map_location='cpu')['model_state_dict'])
TSteps = 10



imagesize = 192
imagesize = 128
imagesize = 64



def initfun(vx,vy,vz,iter):
    ######### initial velocity: full dimensional spatial space ##################
    '''need:    batch_size,imagesize,imagesize,2'''
    VS0 = torch.stack((torch.from_numpy(vx),torch.from_numpy(vy)),dim=-1)  #[1, imagesize, imagesize, 2] 
    ######### transpose the channel ###############
    v0trans = (VS0.permute(0, 3, 1, 2))[:,0:2,...]  #[1, 2, imagesize, imagesize]

    # get the bandlimited fourier velocity #
    x_ft = torch.fft.rfft2(v0trans)/(imagesize*imagesize)    #[1, 2, imagesize, int(imagesize/2)+1]
    x_trun1 = x_ft[:, :, :modes, :modes]     #[1, 2, modes, modes]
    x_trun2 = x_ft[:, :, -modes:, :modes]    #[1, 2, modes, modes]
    vf = torch.cat((x_trun1,x_trun2),dim=-2).to(device) #[1, 2, 2*modes, modes]

    reslist=[]
    with torch.no_grad():
        model_v.eval()
        
        #predict: v1 -> vt
        for t in range(0, TSteps):
            vf = model_v(vf)           #vf: [1, 2, 2*modes, modes]           

            """ 
            For the sake of ease of implementation, we have transformed Fourier velocity into spatial velocity,
            then pass them to FLASH(C++).
            Nevertheless, in all of our codes, we will be working exclusively with bandlimited Fourier signals.
            """ 

            pd_ft = torch.zeros_like(x_ft)        ##[1, 2, modes, int(modes/2)+1]
            pd_ft[:, :, :modes, :modes]  = vf[:, :, :modes, :] #[1, 2, modes, modes]
            pd_ft[:, :, -modes:, :modes]  = vf[:, :, -modes:, :] #[1, 2, modes, modes]
            vs = torch.fft.irfft2(pd_ft*imagesize*imagesize, s=(imagesize,imagesize))  #[1, 2, imagesize, imagesize]
            if(device.type=="cuda"):
                V_X = vs[0,0,...].contiguous().cpu().numpy()
                V_Y = vs[0,1,...].contiguous().cpu().numpy()
                V_Z = np.zeros(V_Y.shape)
            elif(device.type=="cpu"):
                V_X = vs[0,0,...].contiguous().numpy()
                V_Y = vs[0,1,...].contiguous().numpy()
                # V_Z = np.zeros(V_Y.shape)
                V_Z = np.zeros_like(V_Y)
                

            reslist.append(V_X)
            reslist.append(V_Y)
            reslist.append(V_Z)

    
    return tuple(reslist)

