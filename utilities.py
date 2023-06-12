import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import SimpleITK as sitk
from PIL import Image,ImageDraw,ImageFont
import matplotlib.font_manager as fm
import operator
from functools import reduce
from functools import partial
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import os

#################################################
#
# Utilities
#
#################################################
# torch.cuda.set_device('cuda:0')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


#loss function with rel/abs Lp loss
class LpLossF(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLossF, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):   #x: [20, 4096]    y: [20, 4096]
        # return F.mse_loss(x.real, y.real, reduction='mean')+F.mse_loss(x.imag, y.imag, reduction='mean')
        num_examples = x.size()[0]

        diff_norms = torch.norm(torch.abs(x.reshape(num_examples,-1) - y.reshape(num_examples,-1)), self.p, 1) #torch.Size([50])
        y_norms = torch.norm(torch.abs(y.reshape(num_examples,-1)), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)  #diff_norms:0.3271      y_norms:47.3393     /: 0.0069

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


class Epdiff(nn.Module):
    def __init__(self, device, trunc=(16, 16, 16), iamgeSize=(64,64,64), alpha=3, gamma=1, lpow=6):
        super(Epdiff, self).__init__()
        self.alpha=alpha;self.gamma=gamma; self.lpow=lpow
        self.imgX, self.imgY, self.imgZ, self.iamgeSize = iamgeSize[0],iamgeSize[1],iamgeSize[2],iamgeSize
        self.device = device
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.truncX = trunc[0]
        self.truncY = trunc[1]
        self.truncZ = trunc[2]
        if (self.truncX%2==0):
            self.truncX = self.truncX - 1   #15
        if (self.truncY%2==0):
            self.truncY = self.truncY - 1   #15
        if (self.truncZ%2==0):
            self.truncZ = self.truncZ - 1   #15
        #######   Lcoeff : V˜ → M˜ ∗ maps v to M (momentum)     Kcoeff: M˜ →  V˜
        self.Kcoeff, self.Lcoeff, self.CDcoeff = self.fftOpers (self.alpha, self.gamma, self.lpow, self.truncX, self.truncY, self.truncZ, device)
        # self.Kcoeff2, self.Lcoeff2 = self.fftOpers2 (8)
        

    def SmoothOper(self, mode, batchsize):  #shape : [20, 64, 64, 64, 3]
        size_x, size_y, size_z = self.iamgeSize[0], self.iamgeSize[1], self.iamgeSize[2]

        spx = 1 ##spacing information x 
        spy = 1 ##spacing information y 
        spz = 1 ##spacing information z 

        if(self.truncZ != 1):
            gridx = torch.tensor(np.linspace(0, 1-1/size_x, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
            gridy = torch.tensor(np.linspace(0, 1-1/size_y, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
            gridz = torch.tensor(np.linspace(0, 1-1/size_z, size_z), dtype=torch.float)
            gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
            grid = torch.cat((gridx, gridy, gridz), dim=-1).to(self.device)

            trun1 = grid[:, :mode, :mode, :mode, :]     #[b, modes, modes, modes, 3]
            trun2 = grid[:, -mode:, :mode, :mode, :]    #[b, modes, modes, modes, 3]
            trun3 = grid[:, :mode, -mode:, :mode, :]    #[b, modes, modes, modes, 3]
            trun4 = grid[:, -mode:, -mode:, :mode, :]   #[b, modes, modes, modes, 3]
            yy1 = torch.cat((trun1,trun2),dim=-4)       #[b, 2*modes, modes, modes, 3]      #[b, 16, 8, 8, 3]
            yy2 = torch.cat((trun3,trun4),dim=-4)       #[b, 2*modes, modes, modes, 3]      #[b, 16, 8, 8, 3]
            trunr = torch.cat((yy1,yy2),dim=-3)         #[b, 2*modes, 2*modes, modes, 3]    #[b, 16, 16, 8, 3]

            coeff = (-2.0*torch.cos(2.0 * torch.pi * trunr) + 2.0)/(spx*spx);
            val = pow(self.alpha*(torch.sum(coeff,dim=-1))+self.gamma, self.lpow);

        else:
            gridx = torch.tensor(np.linspace(0, 1-1/size_x, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
            gridy = torch.tensor(np.linspace(0, 1-1/size_y, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
            grid = torch.cat((gridx, gridy), dim=-1).to(self.device)
            trun1 = grid[:, :mode, :mode, :]     #[b, modes, modes, 3]
            trun2 = grid[:, -mode:, :mode, :]    #[b, modes, modes, 3]
            trunr = torch.cat((trun1,trun2),dim=-3)       #[b, 2*modes, modes, 3]      #[b, 16, 8, 2]

            coeff = (-2.0*torch.cos(2.0 * torch.pi * trunr) + 2.0)/(spx*spx)
            val = pow(self.alpha*(torch.sum(coeff,dim=-1))+self.gamma, self.lpow)

            # Lcoeff = torch.stack((val,val),dim=-1)       #[b, 16, 8, 2]   sharp
            # Kcoeff = torch.stack((1/val,1/val),dim=-1)   #[b, 16, 8, 2]   smooth


        resSmooth = (1/val).unsqueeze(1)  #momemtum -> velocity      K operator
        resSharp = val.unsqueeze(1)  #velocity -> momemtum           L operator

        return resSmooth, resSharp   ##[b, 16, 8]

    
    def fftOpers (self, alpha, gamma, lpow, truncX, truncY, truncZ, device):
        fsx = self.iamgeSize[0]
        fsy = self.iamgeSize[1]
        fsz = self.iamgeSize[2]

        size = truncX*truncY*truncZ 
        fsize = fsx * fsy * fsz 
        beginX = int(-(truncX-1) / 2.0);  #-7
        beginY = int(-(truncY-1) / 2.0);  #-7
        beginZ = int(-(truncZ-1) / 2.0);  #-7
        endX = beginX + truncX;           #8
        endY = beginY + truncY;           #8
        endZ = beginZ + truncZ;           #8
        padX = 2 * truncX - 1; 
        padY = 2 * truncY - 1;
        padZ = 2 * truncZ - 1;
        padsize = padX * padY * padZ;

        sX = 2.0 * torch.pi / fsx;
        sY = 2.0 * torch.pi / fsy;
        sZ = 2.0 * torch.pi / fsz;
        id = 0;
        fftLoc = torch.zeros(3 * size).to(device);
        corrLoc = torch.zeros(3 * size).to(device);
        conjLoc = torch.zeros(3 * size).to(device);
        ################################ Kenerls ###################################
        for k in range (beginZ, endZ):
            for p in range (beginY, endY):
                for i in range (beginX, endX):
                    fftZ = k; corrZ = k; conjZ = k;
                    fftY = p; corrY = p; conjY = p; 
                    fftX = i; corrX = i; conjX = i;

                    if(k < 0):
                        fftZ = k + fsz; corrZ = k + padZ; conjZ = fftZ;
                    if(p < 0):
                        fftY = p + fsy; corrY = p + padY; conjY = fftY;
                    if(i < 0): 
                        fftX = i + fsx; corrX = i + padX;
                        conjX = -i ; conjY = -p; conjZ = -k;
                        if (p > 0):
                            conjY = -p + fsy;
                        if (k > 0):
                            conjZ = -k + fsz;
                    fftLoc[3*id] = fftX; corrLoc[3*id] = corrX; conjLoc[3*id] = conjX;
                    fftLoc[3*id+1] = fftY; corrLoc[3*id+1] = corrY; conjLoc[3*id+1] = conjY;
                    fftLoc[3*id+2] = fftZ; corrLoc[3*id+2] = corrZ; conjLoc[3*id+2] = conjZ;
                    id +=1
        Lcoeff = torch.zeros(truncX*truncY*truncZ*3, dtype=torch.cfloat).to(device)
        Kcoeff = torch.zeros(truncX*truncY*truncZ*3, dtype=torch.cfloat).to(device)
        CDcoeff = torch.zeros(truncX*truncY*truncZ*3, dtype=torch.cfloat).to(device)
        spx = 1 ##spacing information x 
        spy = 1 ##spacing information y 
        spz = 1 ##spacing information z 
    




        for id in range (0,size):
            index = 3*id;
            xcoeff = (-2.0*torch.cos(sX*fftLoc[index]) + 2.0)/(spx*spx);
            ycoeff = (-2.0*torch.cos(sY*fftLoc[index+1]) + 2.0)/(spy*spy);
            zcoeff = (-2.0*torch.cos(sZ*fftLoc[index+2]) + 2.0)/(spz*spz);
            val = pow(alpha*(xcoeff + ycoeff + zcoeff)+gamma, lpow);
            
            Lcoeff[index]= val
            Lcoeff[index+1] = val
            Lcoeff[index+2] = val
            Kcoeff[index] = 1/val
            Kcoeff[index+1] = 1/val
            Kcoeff[index+2] = 1/val
            CDcoeff[index]= complex(0,torch.sin(sX*fftLoc[index])/spx)
            CDcoeff[index+1]= complex(0,torch.sin(sX*fftLoc[index+1])/spy)
            CDcoeff[index+2]= complex(0,torch.sin(sX*fftLoc[index+2])/spz)
        
        
        return Kcoeff, Lcoeff, CDcoeff


# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]   #(50, 64, 64, 5000)
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


from torch.utils.data import Dataset
def load_volfile_field(file_field):
    fields = []
    for i in range(10):
        pth = "{}_{}.mhd".format(file_field,i)  
        fields.append(sitk.GetArrayFromImage(sitk.ReadImage(pth)))  #[(64, 64, 64, 3),...]
    
    pth = "{}_{}.mhd".format(file_field,10)
    if os.path.exists(pth):
        fields.append(sitk.GetArrayFromImage(sitk.ReadImage(pth)))
    return np.stack(fields, axis=-1)
class Field_dataset(Dataset):
    def __init__(self, base_dir="/mnt/sda/hzf/wn/2/NP/Brain64", split="train", mode="velocity", ntrain=1000, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.mode = mode
        self.data_dir = base_dir
        self.pefix = "v0" if mode=="velocity" else "phiinv"
        self.ntrain = ntrain
        
    def __len__(self):
        return self.ntrain

    def __getitem__(self, idx):
        if self.split == "test":
            idx += 2100
        else:
            idx += 1200

        if self.mode=="velocity":
            fieldv = load_volfile_field("{}/velocity/v0{}".format(self.data_dir, idx+1)) #(64, 64, 64, 3, 11)
            return torch.from_numpy(fieldv)
        elif(self.mode=="phiinv"):
            fieldv = load_volfile_field("{}/velocity/v0{}".format(self.data_dir, idx+1)) #(64, 64, 64, 3, 11)
            fieldphiinv = load_volfile_field("{}/phiinv/phiinv{}".format(self.data_dir, idx+1)) #(64, 64, 64, 3, 10)
            sample = {'v': fieldv, 'phiinv': fieldphiinv}
            return sample
        



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)
            
    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data