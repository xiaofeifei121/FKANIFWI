'''
Modules for Implicit Full Waveform Inversion (IFWI):
    where, 
    - IRN denotes a deep neural network (MLP) for subsurface models,
    - IFWI2D denotes a framework for FWI and IFWI.

Updates:
    - 
siren
@Author: Jian Sun
-- Ocean University of China
-- Jan, 2022

FKAN
@Author: Wenbin Tian
-- China University of Petroleum (Beijing), Karamay Campus
-- May 31, 2026
'''

from __future__ import print_function
import os
import copy
import torch
import torch.utils.data
import numpy as np
from torchvision import transforms

from rnn_fd import rnn2D
from generator import gen_Segment2d
import math
import torch
import torch.nn as nn
import numpy as np
from math import pi
from ChebyKANLayer import ChebyKANLayer


class IRN(torch.nn.Module):
    '''
    An Implicit Representation Network.
    
    neuron:             a list to define number of layers (length of the list) and number of neurons in each layer;
    activation:         'relu', 'tanh' or 'sine',
                        if 'sine' is chosen, then special initilizations are implemented (see SIREN paper).
    outermost_linear:   if True, then not activation function is applied on the last layer.
    '''
    def __init__(self, 
                 neuron=[2, 256, 256, 256, 256, 1], 
                 omega_0=30, 
                 prob=0.2,
                 bias=True, 
                 dropout=False,
                 outermost_linear=False,
                 activation='sine'):
        super(IRN, self).__init__()
        self.omega_0 = omega_0
        self.neuron = neuron
        self.d_flag = dropout
        self.outermost_linear = outermost_linear
        
        self.linear = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        for idx in range(len(neuron) - 1):
            self.linear.append(torch.nn.Linear(neuron[idx], neuron[idx + 1], bias=bias))
            if self.d_flag:
                self.dropout.append(torch.nn.Dropout(prob, inplace=True))
            
        if activation == 'relu':
            self.omega_0 = 1
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.omega_0 = 1
            self.activation = torch.nn.Tanh()
        else:
            self.activation = torch.sin
            self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            self.linear[0].weight.uniform_(-1 / self.neuron[0], 1 / self.neuron[0])
            for ix in range(1, len(self.linear)):
                self.linear[ix].weight.uniform_(-np.sqrt(6 / self.neuron[ix]) / self.omega_0, 
                                                 np.sqrt(6 / self.neuron[ix]) / self.omega_0)
            
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        feature = self.activation(self.omega_0 * self.linear[0](coords))
        for ilayer, layer in enumerate(self.linear[1:]):
            if self.d_flag:
                feature = self.dropout[ilayer](feature)
            feature = layer(feature)
            if not self.outermost_linear or ilayer + 2 < len(self.linear):
                feature = self.activation(self.omega_0 * feature)
        return feature, coords

# ==============================================================
# FourierKANLayer：使用 Fourier 基展开的 KAN 特征提取层
# ==============================================================
class FourierKANLayer(nn.Module):
    def __init__(
        self, input_dim, output_dim, gridsize, addbias=True, smooth_initialization=False
    ):
        """
        input_dim      : 输入维度（坐标维度，如 x,z ）
        output_dim     : 输出特征维度（hidden_features）
        gridsize       : 使用多少个 Fourier 频率（k=1..gridsize）
        addbias        : 是否添加偏置
        smooth_initialization : 是否对高频 coefficient 进行平滑衰减初始化
        """
        super(FourierKANLayer, self).__init__()

        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = input_dim
        self.outdim = output_dim

        # 决定频率归一化尺度（是否平滑）
        grid_norm_factor = (
            (torch.arange(gridsize) + 1) ** 2 if smooth_initialization else np.sqrt(gridsize)
        )

        # fouriercoeffs 形状：(2, out_dim, input_dim, gridsize)
        # 第一维 2 表示 cos 和 sin 的两套系数
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, output_dim, input_dim, gridsize)
            / (np.sqrt(input_dim) * grid_norm_factor)
        )

        # 可选偏置
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        # x 形状：(N,..., input_dim)
        xshp = x.shape

        # 输出形状保留 batch 维，将最后一维改成 outdim
        outshape = xshp[:-1] + (self.outdim,)

        # 展平输入 (N_flat, input_dim)
        x = x.reshape(-1, self.inputdim)

        # 构造频率 k = 1..gridsize，形状 (1,1,1,gridsize)
        k = torch.arange(1, self.gridsize + 1, device=x.device).reshape(1, 1, 1, self.gridsize)

        # 将 x reshape 为 (N_flat, 1, input_dim, 1) 以便广播
        xrshp = x.reshape(x.shape[0], 1, x.shape[1], 1)

        # 计算 cos(kx) 和 sin(kx)，广播生成 (N_flat, 1, input_dim, gridsize)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        # 使用四ier系数加权求和（在 input_dim 和 gridsize 维度上求和）
        y = torch.sum(c * self.fouriercoeffs[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], dim=(-2, -1))

        # 加偏置
        if self.addbias:
            y += self.bias

        # reshape 回原 batch 尺寸
        
        y = y.reshape(outshape)
        return y


# ==============================================================
# FKANLayer：FourierKAN + LayerNorm
# ==============================================================
class FKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid):
        """
        in_features  : 输入维度
        out_features : 特征输出维度
        grid         : Fourier 模式个数
        """
        super(FKANLayer, self).__init__()
        self.fkan = FourierKANLayer(in_features, out_features, grid)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        
        x = self.fkan(x)      # FourierKAN 特征映射
        x = self.norm(x)      # LayerNorm 稳定训练
        return x


# ==============================================================
# SineLayer（名字保留，但实际使用 tanh + sigmoid 门控激活）
# ==============================================================
class SineLayer(nn.Module):
    """
    实际激活公式：
        y = (z + tanh(omega_0 * z)) * sigmoid(z)
    其中 z = W x + b

    • 具有强非线性和高频表达能力
    • 类似 SIREN 初始化，但激活并非 sin
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
        scale=10.0,
        init_weights=True,
    ):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        # 线性映射
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # 初始化权重（模仿 SIREN）
        self.init_weights()

    def init_weights(self):
        # SIREN 风格初始化
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )
    

    def forward(self, input):
        # 线性映射
        z = self.linear(input)
        # scale = self.generate_scale(z)
        # 高频 tanh + 残差 z，再乘 sigmoid 门控
        # return (z + torch.tanh(self.omega_0 * z)) * torch.sigmoid(z)
        return (torch.sin(self.omega_0 * z)) 

## FINER sin⁡(𝑜𝑚𝑒𝑔𝑎∗𝑠𝑐𝑎𝑙𝑒∗(𝑊𝑥+𝑏𝑖𝑎𝑠)) 𝑠𝑐𝑎𝑙𝑒=|𝑊𝑥+𝑏𝑖𝑎𝑠|+1
# class FinerLayer(nn.Module):
#     def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False):
#         super().__init__()
#         self.omega_0 = omega_0
#         self.is_first = is_first
#         self.in_features = in_features
#         self.linear = nn.Linear(in_features, out_features, bias=bias)
#         self.init_weights()
#         self.scale_req_grad = scale_req_grad
#         self.first_bias_scale = first_bias_scale
#         if self.first_bias_scale != None:
#             self.init_first_bias()
    
#     def init_weights(self):
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.weight.uniform_(-1 / self.in_features, 
#                                              1 / self.in_features)      
#             else:
#                 self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
#                                              np.sqrt(6 / self.in_features) / self.omega_0)

#     def init_first_bias(self):
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
#                 # print('init fbs', self.first_bias_scale)

#     def generate_scale(self, x):
#         if self.scale_req_grad: 
#             scale = torch.abs(x) + 1
#         else:
#             with torch.no_grad():
#                 scale = torch.abs(x) + 1
#         return scale
        
#     def forward(self, input):
#         x = self.linear(input)
#         scale = self.generate_scale(x)
#         out = torch.sin(self.omega_0 * scale * x)
#         return out

# class Finer(nn.Module):
#     def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True, 
#                  first_bias_scale=None, scale_req_grad=False):
#         super().__init__()
#         self.net = []
#         self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))

#         for i in range(hidden_layers):
#             self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))

#         final_linear = nn.Linear(hidden_features, out_features)
#         with torch.no_grad():
#             final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
#                                           np.sqrt(6 / hidden_features) / hidden_omega_0)
#         self.net.append(final_linear)
#         self.net = nn.Sequential(*self.net)

#     def forward(self, coords):
#         coords = coords.clone().detach().requires_grad_(True)# 若为 (1,N,d)，变成 (N,d)，方便后续矩阵运算
#         output = self.net(coords)
#         return output, coords        
        
# ==============================================================
# FourierKAN_INR：完整隐式表示网络（INR）
# ==============================================================
class FourierKAN_INR(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, grid):
        """
        in_features      : 输入维度（如坐标 x,z）
        hidden_features  : 隐藏层维度（KAN 输出）
        out_features     : 输出维度（如速度、波场值等）
        grid             : Fourier 模式数量
        """
        super(FourierKAN_INR, self).__init__()

        # 第 1 层：FourierKAN 特征提取
        self.fkan = FKANLayer(in_features, hidden_features, grid)

        # 多层非线性网络（扩大特征维度）
        self.hid1 = SineLayer(hidden_features, 2 * hidden_features)
        self.hid2 = SineLayer(2 * hidden_features, 2 * hidden_features)
        self.hid3 = SineLayer(2 * hidden_features, 2 * hidden_features)
        self.hid4 = SineLayer(2 * hidden_features, 2 * hidden_features)


        

        # 输出层
        self.out = nn.Linear(2 * hidden_features, out_features)

        # SIREN 风格的小范围初始化
        with torch.no_grad():
            const = np.sqrt(6 / hidden_features) / 30
            self.out.weight.uniform_(-const, const)

    def forward(self, coords):
        """
        coords : 输入坐标（例如 (x,z)）
        """
        coords = coords.clone().detach().requires_grad_(True)# 若为 (1,N,d)，变成 (N,d)，方便后续矩阵运算
        x = self.fkan(coords)   # FourierKAN 特征

        y1 = self.hid1(x)
        y2 = self.hid2(y1)
        y3 = self.hid3(y2)
        y4 = self.hid4(y3)

        out = self.out(y4)      # 输出物理量
        return out, coords 

#############################################################################################
# ##                       Implicit Full Waveform Inversion  Model                        ###
#############################################################################################
class IFWI2D():
    """
    Implicit Full Waveform Inversion (for acoustic only) with an implicit repreesentation neural network.
    This module allows three types of network:
        - FWI:          A version full waveform inversion using RNN cell,
                        where each cell acts as a finite-difference operator, 
                        which takes the velocity (could be variable) as input and output shot gather.
        - IRN:          An implicit MLP neural network for image/velocity representation,
                        which takes coordinates as inputs, and output a (normalized) velocity model.
        - IFWI:         A two-step implicit full waveform inversion, including IRN + FWI(RNN),
                        coords -> [NN] -> {vel} -> [RNN] -> shot_pred.
    """
    def __init__(self, 
                 mean=2.6718, 
                 std=0.9738,
                 neuron=[2, 256, 256, 256, 256, 1], 
                 omega_0=30, 
                 prob=0.2,
                 activation='sine', 
                 bias=True, 
                 dropout=False,
                 outermost_linear=False,
                 nz=None,
                 nx=None,
                 zs=None,
                 xs=None,
                 zr=None, 
                 xr=None,
                 dz=None,
                 dt=None,
                 npad=0, 
                 order=2, 
                 vmax=6000,
                 log_para=1e-6,
                 segment_size=100,
                 vpadding=None,
                 freeSurface=True,
                 regularization="TV",
                 dtype=torch.float32,
                 pretrained=None,
                 device='cpu',
                 netOpt='IFWI',
                 method="kan"):
        """
        Args:
            start_channel(int):     the number of channels for the first Conv layer in FCN.
            ns(int):                the number of shot gathers (in_channel for fcn2D class).
            nz(int):                the number of samples in depth for velocity model
            zs(tensor):             in shape of [num_vels, ns].
            xs(tensor):             in shape of [num_vels, ns].
            zr(tensor):             in shape of [num_vels, ns, nx], or an integer
            xr(tensor):             in shape of [num_vels, ns, nx]. or a list/tensor in length of nx.
            dz & dt (float):        represent grid and time interval respectively.
            npad(int):              is the number of grid padding to the velocity for absorbing boundaries.
            order(int);             the order of finite difference is used for wave propagation modeling.
            freeSurface(boolean):   True (default) or False for free surface option in forward modeling (RNN).
            netOpt(str):            one of the str in list ['RNN', 'FCN', 'PGNN'].
            vpadding:               the intention of vpadding is to fix the velocity in the surface layer (such as water layer), 
                                        to stablize the forward propagation.
                                        if it's a tensor, size: [num_grids_1st_layer, nx], 
                                            the first layer padding to velocity models.
                                        elif it's a tuple, (v0, num_grids_1st_layer)
                                        else: None(default), no padding.
        """
        super(IFWI2D, self).__init__()
        assert netOpt in ('FWI', 'IRN', 'IFWI'), "Error: Only ['FWI', 'IRN', 'IFWI'] are supported."
        self.std = std
        self.mean = mean
        self.vmax = vmax / 1000  # self.vmax is in km/s
        self.dtype = dtype
        self.vmodel = torch.empty(1)  # this is for the predict and load_state Funcs in FWI mode.
        self.device = device
        self.netOpt = netOpt
        self.reg_op = regularization
        self.segment_size = segment_size
        self.nx_pad = nx + 2 * npad
        self.nz_pad = nz + npad if freeSurface else nz + 2 * npad

        x = np.arange(0, nx) * dz / 1000
        z = np.arange(0, nz) * dz / 1000
        X, Z = np.meshgrid(x[None, :], z[:, None])
        X = torch.from_numpy(X).type(dtype=dtype).to(device)
        Z = torch.from_numpy(Z).type(dtype=dtype).to(device)
        self.coords = torch.stack([X, Z], dim=-1).to(device)[None, :]  # shape [1, nz, nx, 2]
        # self.transform = transforms.Normalize(mean=(mean), std=(std))

        if torch.is_tensor(vpadding):
            self.vpadding = vpadding.type(dtype).to(device)
        elif isinstance(vpadding, tuple):
            self.vpadding = torch.ones((int(vpadding[1]), nx), dtype=dtype, device=device) * vpadding[0]
        else:
            self.vpadding = vpadding
        
        if netOpt in ['FWI', 'IFWI']:
            rnn = rnn2D(nz, nx, zs, xs, zr, xr, dz, dt, npad, order, vmax, log_para, freeSurface, dtype, device)
            self.rnn = rnn.type(dtype).to(device)
        if netOpt in ['IRN', 'IFWI']:
            if method.lower() == 'kan':
                vel_net = FourierKAN_INR(
                    in_features=2,
                    hidden_features=64,
                    out_features=1,
                    grid=5
                )
                print(method.lower())
            elif method.lower() == 'irn':
                vel_net = IRN(
                    neuron=neuron,
                    omega_0=omega_0,
                    prob=prob,
                    bias=bias,
                    dropout=dropout,
                    activation=activation,
                    outermost_linear=outermost_linear
                )
                print(method.lower())

            else:
                raise ValueError(
                    f"Unknown method '{method}'. "
                    "Supported methods are: ['irn', 'kan']"
                )
            self.vel_net = vel_net.type(dtype).to(device)
            if pretrained is not None:
                self.vel_net.load_state_dict(torch.load(pretrained)['state_dict'])

        # if torch.cuda.device_count() > 1:
        #     print(torch.cuda.device_count(), " GPUs will be used!")
        #     # rnn  = torch.nn.DataParallel(rnn)
        #     vel_net = torch.nn.DataParallel(vel_net)
            
    def predict(self, coords=None, resume_file_name=None, best=False, uncertainty=False, NoSim=100):
        """
        This function only works for IRN or IFWI, performs the predict process of the IRN(MLP).
        
        coords: coordinates for velocity model, which will not be used in FWI mode.
        best:   flag for loading best_loss_model.
        uncertainty: flag for uncertainty analysis in IFWI or IRN mode, in which the dropout option will be activated.
        NoSim:  number of simulation will be performed for uncertainty analysis.
        """
        # assert self.netOpt in ['IRN', 'IFWI'], "Error: Only ['IRN', 'IFWI'] are supported in model.predict option."
        coords = self.coords if coords is None else coords
        if isinstance(resume_file_name, str):
            _, _, _, _, _, _ = self.load_state(resume_file_name, best)
        else:
            print("Failed loading save model, using current model for prediction.")
        if self.netOpt in ['IRN', 'IFWI']:
            self.vel_net.eval()
            with torch.no_grad():
                if uncertainty:
                    vmodel = []
                    for m in self.vel_net.modules():
                        if m.__class__.__name__.startswith('Dropout'):
                            m.train()
                    for ix in range(NoSim):
                        _vpred_, _ = self.vel_net(coords)
                        vmodel.append(_vpred_.detach())
                    vmodel = torch.cat(vmodel, axis=0)
                else:            
                    vmodel, _ = self.vel_net(self.coords if coords is None else coords)
                vmodel = (vmodel * self.std + self.mean) * 1000
                # vmodel[vmodel < 1000] = 1000
                # vmodel[vmodel > self.vmax * 1000] = self.vmax * 1000
        else:
            vmodel = self.vmodel * 1000
        return vmodel, coords

    def train(self, 
              MaxIter=10000, 
              vmodel=None, 
              wavelet=None, 
              shots=None, 
              alpha=0,
              option=0, 
              learning_rate=1e-4,
              log_interval=1, 
              wandb=None,
              resume_file_name=None,
              save_file_name=''):
        """
        Args:
            - MaxIter:              Maximum training iteration
            - learning_rate:        learning rate for Adam optimizer
            - resume_file_name:     resume training from a saved file
            - log_interval:         output log interval
            - alpha:                trade-off between data_loss and regularization loss
                                    default=0, i.e., no regularization
                                    if 'auto', trade-off will be zero before 90% of data loss is reduced,
                                        after that, trade-off will be applied to make regularization be 10% of data loss 
            - vmodel:               in km/s
                                    for the IRN-mode, vmodel is the label data in shape [num_vels, nz, nx, num_neurons_in_output_layers],
                                    for the FWI-mode, vmodel is the initial velocity in shape [num_vels, nz, nx],
                                    for the IFWI-mode, vmodel is not required or used.
            - wavelet:              for the IRN-mode, wavelet is not required,
                                    for the FWI/IFWI-mode, wavelet needs to be provided as a tensor,
                                        and will be used in the forward propagation process.
            - shots:                for the IRN-mode, shots is not required,
                                    for the FWI/IFWI-mode, shots will be used as label data in shape [num_vels, ns, nt, nx].
            - option:               default=0, is not required for the IRN-mode, 
                                        and will be used in segmented forward propagation 
                                        (i.e., trunaced RNN) under the FWI/IFWI-modes.
        """
        # if wandb is not None:
        #     self.run = wandb.init(project=self.netOpt + " Project",
        #                           reinit=True,
        #                           config={"learning_rate": learning_rate})
        
        self.vmodel = vmodel
        if self.netOpt in ['IRN', 'IFWI']:
            self.clip = 0.25
            self.vel_net.train()
            self.params = self.vel_net.parameters() 
        else:
            self.clip = 100
            self.vmodel = torch.nn.Parameter(copy.deepcopy(vmodel))
            self.rnn.register_parameter("vmodel", self.vmodel)
            self.params = self.rnn.parameters()
            self.rnn.train()
        optimizer = torch.optim.Adam(lr=learning_rate, params=self.params)
        
        best_loss = 1e100
        best_loss_model=0
        best_loss_epoch = 0
        resume_from_epoch = 0
        train_loss_history = []
        if isinstance(resume_file_name, str):
            resume_from_epoch, best_loss, best_loss_epoch, best_loss_model, \
                train_loss_history, optimizer = self.load_state(resume_file_name, False, optimizer)
        
        trade_off = 0 if alpha=='auto' else alpha
        for epoch in range(resume_from_epoch, MaxIter):
            vpred, loss = self.train_one_epoch(optimizer, self.vmodel, wavelet, shots, trade_off, option)
            train_loss_history.append(loss)
            if alpha=='auto' and loss[1] / train_loss_history[-1][1] <= 0.1:
                trade_off = 0.01 * loss[1] / loss[2]
            # if wandb is not None:
            #     self.run.log({'Total Loss': loss[0], 'Data Loss': loss[1], 'Regularization Loss': loss[2]})
            if epoch % log_interval == 0 or epoch == MaxIter - 1:
                print("Epoch: {0:5d}, Loss: {1[0]:.4e}, DataLoss: {1[1]:.4e}, RegLoss: {1[2]:.4e}".format(epoch, loss))
                if loss[0] < best_loss:
                    best_loss = loss[0]
                    best_loss_epoch = epoch
                    best_loss_model = copy.deepcopy(self.vel_net.state_dict() if self.netOpt in ['IRN', 'IFWI'] else self.vmodel)
                self.save_state(epoch, 
                                best_loss, 
                                best_loss_epoch, 
                                best_loss_model, 
                                train_loss_history, 
                                optimizer, 
                                log_interval,
                                save_file_name)
        # if wandb is not None:
        #     self.run.finish()
        return train_loss_history, vpred

    def train_one_epoch(self, optimizer, vmodel=None, wavelet=None, shots=None, trade_off=0, option=0):
        """
        This function performs the training for one-epoch, 
            including all possible segmented wavelet/shots for Truncated RNN.
        """
        loss_Reg = 0
        loss_total = 0
        loss_segAll = 0
        loss_regAll = 0
        loss_regALL_AVE =0
        if self.netOpt == 'IRN':
            optimizer.zero_grad()
            
            vpred, _, _, _, _ = self.forward_process(None, None, None, None, option)
            loss = ((vpred - vmodel)**2).mean()
            loss.backward()
            optimizer.step()
            loss_total += loss.detach().cpu().item()
        else:
            shots = shots.to(self.device)
            # prev_state = torch.zeros([shots.shape[0], shots.shape[1], self.nz_pad, self.nx_pad], dtype=self.dtype).to(self.device)
            # curr_state = torch.zeros([shots.shape[0], shots.shape[1], self.nz_pad, self.nx_pad], dtype=self.dtype).to(self.device)
           

            for iseg, (segWavelet, segData) in enumerate(gen_Segment2d(wavelet, shots, segment_size=self.segment_size, option=option)):
                optimizer.zero_grad()
                vpred, vgrad, shot_segPred, _, _ = self.forward_process(vmodel, segWavelet, None, None, option) 


                loss_Seg = ((shot_segPred - segData)**2).sum() / shots.shape[0] / shots.shape[1] / shots.shape[-2] / shots.shape[-1]/10
                if self.reg_op == "TV":
                    loss_Reg = (vgrad**2 + 1e-6).sqrt().mean()/10  # Total-Variation regularization
                    loss_regAll += loss_Reg.detach().cpu().item()
                loss = loss_Seg + trade_off * loss_Reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.clip) 
                optimizer.step()

                loss_total += loss.detach().cpu().item()
                loss_segAll += loss_Seg.detach().cpu().item()
                # loss_regAll += loss_Reg.detach().cpu().item()
                loss_regALL_AVE=loss_regAll / (iseg + 1)
        
        return vpred.detach(), [loss_total, loss_segAll, loss_regALL_AVE]
        
    def forward_process(self, vmodel=None, wavelet=None, prev_state=None, curr_state=None, option=0):  
        """
        This function performs the forward process of IRN/FWI/IFWI, with segmented/full-time-steps wavelet.
        (all inputs need to be tensor except option)
        Args:
            - vmodel:       unit in km/s;
                            the velocity model to be inverted (required by rnn2D): [num_vels, nz, nx],
                            for FWI, it is the tensor of the initial model and requires_grad_(True),
                            for IRN and  IFWI, this is not needed (num_vels can only be 1 in the case of IFWI).
            - wavelet:      segment_wavelet, input for RNN(fd), shape: [num_vels, len_tSeg] or [len_tSeg]
            - prev_state:   Contains initial wavefields for FD, in shape of [num_vels, ns, nz_pad, nx_pad]
            - curr_state:   Contains initial wavefields for FD, in shape of [num_vels, ns, nz_pad, nx_pad]
            - option:       =0 (default), averaging partitioning the input with segement_size.
                            =1, starting point for segments moving forward by step.
                                for even number segment_size: segment_size//2 step.
                                for odd number segment_size: segment_size//2+1 step.
                            =2, starting point for segments at always index=0.
                                For example, segments are:[0->segment_size, 0->2*segment_size, 0->3*segment_size, ...]
        """
        seg_yPred, regularizer = 0, 0

        if self.netOpt == "IRN":
            # In IRN-mode, vmodel is scaled
            vgrad = 0
            vmodel, coords = self.vel_net(self.coords)  # output vmodel shape [num_vels, nz, nx, 1]
            vmodel = (vmodel * self.std + self.mean) * 1000
        elif self.netOpt == "FWI":
            # vmodel.data[vmodel.data < 1] = 1
            # vmodel.data[vmodel.data > self.vmax] = self.vmax
            vmodel = vmodel * 1000
            vgrad = 0
            _, _, seg_yPred, _ = self.rnn(vmodel, wavelet, None, None, option)
            seg_yPred[torch.isnan(seg_yPred)] = 0
            seg_yPred[seg_yPred == float('inf')] = 0
        else:
            vmodel, coords = self.vel_net(self.coords)  # output vmodel shape [num_vels，nz, nx, 1]
            vgrad = 0
            if self.reg_op == "TV": 
                vgrad = self.gradient(vmodel, coords)
                # vlapl = self.laplace(vmodel, coords)

            vmodel = (vmodel.squeeze(dim=-1) * self.std + self.mean) * 1000
            # vmodel[vmodel < 1000] = 1000
            # vmodel[vmodel > self.vmax * 1000] = self.vmax * 1000
            if self.vpadding is not None:
                vmodel = torch.cat((self.vpadding[None, :], vmodel[:, self.vpadding.shape[0]:, :]), dim=1)
            _, _, seg_yPred, _ = self.rnn(vmodel, wavelet, None, None, option)
            seg_yPred[torch.isnan(seg_yPred)] = 0
            seg_yPred[seg_yPred == float('inf')] = 0
        return vmodel, vgrad, seg_yPred, prev_state, curr_state
    
    def gradient(self, y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad
    
    def divergence(self, y, x):
        div = 0.
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
        return div

    def laplace(self, y, x):
        grad = self.gradient(y, x)
        return self.divergence(grad, x)

    def load_state(self, resume_file_name=None, best=False, optimizer=None):
        checkpoint = torch.load(resume_file_name)
        resume_from_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_epoch = checkpoint['best_loss_epoch']
        best_loss_model = checkpoint['best_loss_model']
        train_loss_history = checkpoint['train_loss']
        if best:
            print("Loading the best loss model at Epoch {}".format(best_loss_epoch))
        if self.netOpt in ['IRN', 'IFWI']:
            self.vel_net.load_state_dict(best_loss_model if best else checkpoint['state_dict'])
        else:
            self.vmodel.data = best_loss_model if best else checkpoint['state_dict']
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return resume_from_epoch, best_loss, best_loss_epoch, best_loss_model, train_loss_history, optimizer

    def save_state(self, 
                   epoch, 
                   best_loss, 
                   best_loss_epoch, 
                   best_loss_model, 
                   train_loss_history, 
                   optimizer,
                   log_interval=1,
                   file_name=""):
        state = {'epoch': epoch + 1,
                 'best_loss': best_loss,
                 'best_loss_epoch': best_loss_epoch,
                 'best_loss_model': best_loss_model,
                 'train_loss': train_loss_history,
                 'state_dict': self.vel_net.state_dict() if self.netOpt in ['IRN', 'IFWI'] else self.vmodel,
                 'optimizer': optimizer.state_dict(),
                 }
        torch.save(state, file_name + "checkpoint-{}.pth".format(epoch + 1))
        if os.path.exists(file_name + "checkpoint-{}.pth".format(epoch + 1 - log_interval)) and epoch % 10*log_interval != 0:
            os.remove(file_name + "checkpoint-{}.pth".format(epoch + 1 - log_interval))