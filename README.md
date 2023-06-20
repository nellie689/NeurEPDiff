************************** NeurEPDiff ************************** 

This repository contains source code and data for NeurEPDiff: Neural Operators to Predict Geodesics in Deformation Spaces (https://arxiv.org/pdf/2303.07115.pdf).


************************** Disclaimer ************************** 

This code is only for research purpose and non-commercial use only, and we request you to cite our research paper if you use it:  
NeurEPDiff: Neural Operators to Predict Geodesics in Deformation Spaces  
Nian Wu and Miaomiao Zhang. International Conference on Information Processing in Medical Imaging (IPMI) 2023.

@article{wu2023neurepdiff,  
  title={NeurEPDiff: Neural Operators to Predict Geodesics in Deformation Spaces},  
  author={Wu, Nian and Zhang, Miaomiao},  
  journal={arXiv preprint arXiv:2303.07115},  
  year={2023}  
}  


************************** Setup ************************** 
* [PyCA](https://bitbucket.org/scicompanat/pyca) 
* [FLASH] (https://bitbucket.org/FlashC/flashc/src/master/)
* python=3.9
* pytorch=1.10.1
* matplotlib
* numpy
* SimpleITK


************************** Usage ************************** 

Below is a *QuickStart* guide on how to use NeurEPDiff for network training and testing.

========================= Training ========================

If you want to train you own model, please run:  
python NeurDPDiff/Train/NeruEPDiff_2D_train.py   
or  
python NeurDPDiff/Train/NeruEPDiff_3D_train.py   

Required Input Data: initial velocity field(v0) and associated forward shooting velocity sequence(v1-vt), i.e., numerical solutions of Fourier EPDiff given v0.  
Please note that all the velocity fields are in bandlimited Fourier sapce, they are generated by FLASH algorithm [1] (https://bitbucket.org/FlashC/flashc).  

Tips: To facilitate running the code, we have uploaded a demo of 2D training data. The demo data can be found in the DATA directory.

========================= Testing ========================

If you want to perform registration with well-learned NeruEPDiff, please run:  
bash NeurDPDiff/Test/RegWithNeurEpdiff2D.sh  
or  
bash NeurDPDiff/Test/RegWithNeurEpdiff3D.sh  

Please note that the registration is based on FLASH algorithm.   
We have provide a revision of the FLASH(C++) (Optimization2d.cxx and Optimization3d.cxx) codebase, designed to interact with NeurEPDiff.   
You should install FLASH if you want to run perform registration with well-learned NeruEPDiff.   



************************** Reference **************************                   
[1]. Finite-dimensional Lie algebras for fast diffeomorphic image registration.  
      Miaomiao Zhang, P. Thomas Fletcher. Information Processing in Medical Imaging (IPMI), 2015.
      
[2]. Frequency diffeomorphisms for efficient image registration.  
      Miaomiao Zhang et al.. Information Processing in Medical Imaging (IPMI), 2017.  
      
[3]. Fast diffeomorphic image registration via Fourier-approximated Lie algebras.  
      Miaomiao Zhang, P. Thomas Fletcher. International Journal of Computer Vision(IJCV), 2019 . 


