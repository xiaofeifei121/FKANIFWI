import numpy as np
import torch
import torch.nn.functional as F
import math

import deepwave
from deepwave import scalar

class rnn2D(torch.nn.Module):
    """
    使用 Deepwave 进行正演模拟，替代原来的 RNN 框架
    """
    def __init__(self, nz, nx, zs, xs, zr, xr, dz, dt, 
                 npad=0, order=2, vmax=6000, 
                 log_para=1e-6,
                 freeSurface=True, 
                 dtype=torch.float32, device='cpu'):
        super(rnn2D, self).__init__()
        
        # 保存基本参数
        self.dtype = dtype
        self.device = device
        self.zs, self.xs = zs, xs
        self.zr, self.xr = zr, xr
        self.dz = dz
        self.dt = dt
        self.nz = nz
        self.nx = nx
        self.npad = npad
        self.freeSurface = freeSurface
        self.order = order
 
   
        
        # Deepwave 使用 PML 边界，npad 会自动处理
        self.nx_pad = nx + 2 * npad
        if freeSurface:
            self.nz_pad = nz + npad
        else:
            self.nz_pad = nz + 2 * npad

    def forward(self, vmodel, segment_wavelet, prev_state=None, curr_state=None, option=0):
        """
        使用 Deepwave 进行正演
        Input:
            vmodel(tensor): [num_vels, nz, nx]
            segment_wavelet(tensor): [num_vels, len_tSeg] or [len_tSeg]
        Output:
            segment_ytPred(tensor): [num_vels, num_shots, nt, num_receivers]
        """
        num_vels = vmodel.shape[0]
        num_shots = self.zs.shape[1]

        # 保证 wavelet 至少有 [num_vels, nt]
        if segment_wavelet.dim() == 1:
            segment_wavelet = segment_wavelet.unsqueeze(0).repeat(num_vels, 1)  # [num_vels, nt]
        nt = segment_wavelet.shape[1]

        # 接收器数量
        if isinstance(self.zr, int):
            num_receivers = self.nx  # 或者你自定义的数量
        else:
            num_receivers = self.xr.shape[2]  # e.g., 288

        all_shots_data = []

        for ivel in range(num_vels):
            shot_data_list = []
            for ishot in range(num_shots):
                # ----- 震源位置，形状 [1, 1, 2] -----
                if self.freeSurface:
                    sz = int(self.zs[ivel, ishot].item())
                    sx = int(self.xs[ivel, ishot].item())
                else:
                    # 使用 deepwave 的 pml_width 时，坐标仍给“物理网格坐标”，不要额外加 npad
                    sz = int(self.zs[ivel, ishot].item())
                    sx = int(self.xs[ivel, ishot].item())

                source_loc = torch.tensor([[sz, sx]], dtype=torch.long, device=self.device)  # [1,2]
                source_loc = source_loc.unsqueeze(0)  # [1,1,2]  —— batch=1, nsrc=1

                # ----- 接收器位置，形状 [1, R, 2] -----
                if isinstance(self.zr, int):
                    receiver_z = torch.full((num_receivers,), int(self.zr), dtype=torch.long, device=self.device)
                    receiver_x = torch.arange(num_receivers, dtype=torch.long, device=self.device)
                else:
                    receiver_z = self.zr[ivel, ishot, :].to(self.device).long()  # [R]
                    receiver_x = self.xr[ivel, ishot, :].to(self.device).long()  # [R]

                # 不要把坐标加 npad；pml 由 pml_width 控制
                receiver_loc = torch.stack([receiver_z, receiver_x], dim=1)  # [R,2]
                receiver_loc = receiver_loc.unsqueeze(0)                     # [1,R,2]

                # ----- 震源时间函数 [1, 1, nt] -----
                # 原来用了 repeat(13,1,1) 会变成 [13, nt]，不符合 batch 维度约定
                source_amplitudes = segment_wavelet[ivel:ivel+1, :].unsqueeze(1)  # [1,1,nt]
                # print(vmodel.shape)
                # ----- Deepwave 正演 -----
                out = scalar(
                    vmodel[0, :, :],   # [1, nz, nx] —— batch=1
                    self.dz,                     # dx 或 (dz, dx)，若各向异性网格用 (self.dz, self.dx)
                    self.dt,
                    source_amplitudes=source_amplitudes,    # [1,1,nt]
                    source_locations=source_loc,            # [1,1,2]
                    receiver_locations=receiver_loc,        # [1,R,2]
                    pml_width=[0 if self.freeSurface else self.npad, self.npad, self.npad, self.npad],
                    accuracy=self.order,
                )

                # Deepwave 接收器数据通常为 [batch, nt, R]；也可能是 [nt, R]
                receiver_data = out[-1]
                
                if receiver_data.dim() == 3:
                    receiver_data = receiver_data[0]   # [nt, R]
                # 统一转为 [R, nt]
                # receiver_data = receiver_data.transpose(0, 1).contiguous()
                receiver_data = receiver_data.contiguous()
                shot_data_list.append(receiver_data)   # [R, nt]

            # [num_shots, R, nt]
            shots_for_this_vel = torch.stack(shot_data_list, dim=0)
            all_shots_data.append(shots_for_this_vel)

        # [num_vels, num_shots, R, nt] -> [num_vels, num_shots, nt, R]
        segment_ytPred = torch.stack(all_shots_data, dim=0).permute(0, 1, 3, 2)

        avg_regularizer = torch.tensor([[0]], dtype=self.dtype, device=self.device)
        return None, None, segment_ytPred, avg_regularizer
