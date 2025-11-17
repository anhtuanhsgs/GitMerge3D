# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def do_nothing(x, mode=None):
    return x

def bsm(
    metric: torch.Tensor,
    ratio:float=1.0,    
    class_token: bool = False,
) -> Tuple[Callable, Callable]:
    
    protected = 0
    if class_token:
        protected += 1
    if len(metric.shape) == 2:
        metric = metric[None,...]

    # We can only reduce by a maximum of 50% tokens
    T = metric.shape[1]
    
    if ratio < 1.0:
        r = math.floor(T- T*ratio)
    else:
        return do_nothing, do_nothing


    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if len(x.shape) == 2:
            x.unsqueeze_(0)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)
    


    return merge

def pitome(
    metric=None,
    class_token: bool = False,
    indices:torch.Tensor=None, 
    scores:torch.Tensor=None,
    protected_size:int=0,
    r:int=None
) -> Tuple[Callable, Callable]:
    B, H, T, T = scores.shape
    # seperate protected token and mergeable tokens 
    if protected_size > 0: 
        # Protected tokens are at the end of the sequence
        merge_idx = indices[..., :-protected_size]
        protected_idx = indices[..., -protected_size:]
    else:
        merge_idx = indices
        protected_idx = indices[..., :-protected_size]
    # a_idx, b_idx = merge_idx[..., :r], merge_idx[..., r:] 
    
    if protected_size > 0:
        steps = int(merge_idx.shape[-1] / (merge_idx.shape[-1] - r))
        # Sort the order of idx again
        merge_idx, _ = merge_idx.sort(dim=-1)
        idx_buffer = torch.zeros_like(merge_idx)
        idx_buffer[..., ::steps] = 1
        for h in range(0, idx_buffer.shape[1]):
            starting_pos = max(steps // idx_buffer.shape[1], 1)
            idx_buffer[:,h,starting_pos::steps] = 1
        sorted_01_idx = idx_buffer.argsort(dim=-1)
        # Evenly distribute the dst tokens 
        b_idx = merge_idx.gather(dim=-1, index=sorted_01_idx[...,r:])
        a_idx = merge_idx.gather(dim=-1, index=sorted_01_idx[...,:r])
        # breakpoint()

        # b_idx = torch.cat([b_idx, protected_idx], dim=-1)
        # protected_size = 0
    else:
        a_idx, b_idx = merge_idx[..., :r], merge_idx[..., r:] 
        # a_idx, b_idx = merge_idx[..., -r:], merge_idx[..., :-r]     

    # get similarity scores between mergeable tokens
    scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, H, T, b_idx.shape[-1])) 
    scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, H, a_idx.shape[-1], b_idx.shape[-1]))
    _, dst_idx = scores.max(dim=-1) 
    
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]

        B, H, T, C = x.shape
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        if protected_size > 0:
            protected = x.gather(dim=-2, index=protected_idx.unsqueeze(-1).expand(B, H, protected_idx.shape[-1], C))
            
        src = x.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, H, a_idx.shape[-1], C))
        dst = x.gather(dim=-2, index=b_idx.unsqueeze(-1).expand(B, H, b_idx.shape[-1], C))

        dst_values = dst
        
        if mode != "prune":
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(-1).expand(B, H, r, C), src, reduce=mode)

        if protected_size > 0:
            if class_token:
                return torch.cat([x_cls, protected, dst], dim=-2)
            return torch.cat([protected, dst], dim=-2)
        
        if class_token:
            return torch.cat([x_cls, dst], dim=-2)
        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        if class_token:
            x_cls=x[...,0,:].unsqueeze(1)
            x=x[...,1:,:]

        _, _, _, C = x.shape
        
        protected = x[..., :protected_size, :]
        dst = x[..., protected_size:, :]
        
        src = dst.gather(dim=-2, index=dst_idx[...,None].expand(B, H, dst_idx.shape[-1], C))
        out = torch.zeros(B, H,  T, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=b_idx[...,None].expand(B, H, b_idx.shape[-1], C), src=dst)
        
        if protected_size > 0:
            out.scatter_(dim=-2, index=protected_idx[...,None].expand(B, H, protected_size, C), src=protected)
            
        out.scatter_(dim=-2, index=a_idx[...,None].expand(B, H, a_idx.shape[-1], C), src=src)
        
        if class_token:
            return torch.cat([x_cls, out], dim=-2)
        return out

    return merge, unmerge


def pitome_bsm(
    metric=None,
    class_token: bool = False,
    indices:torch.Tensor=None,
    scores:torch.Tensor=None,
    r:int=None,
    protected_size:int=0
) -> Tuple[Callable, Callable]:

    with torch.no_grad():
        B, H, T, _ = scores.shape
        a_idx, b_idx = indices[..., ::2], indices[..., 1::2] 
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, H, T, b_idx.shape[-1])) 
        scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, H, a_idx.shape[-1], b_idx.shape[-1]))
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]

        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        if mode != "prune":
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if class_token:
            return torch.cat([x_cls, unm, dst], dim=1)
        return torch.cat([unm, dst], dim=1)
    return merge

def cal_attn_score(q:torch.Tensor, k:torch.Tensor, sigma=1):   
   B, H, T, C = k.shape
   q = q.mean(1)
   k = k.mean(1)
   q = F.normalize(q, p=2, dim=-1)
   k = F.normalize(k, p=2, dim=-1)
   sim = q.reshape(-1, q.shape[-1]) @ k.mean(1).transpose(-1, -2) 
   
   score = sim.max(-1)[0]
   entropy = score.reshape(q.shape[0], q.shape[1])
   entropy =  (entropy.mean(-1) - entropy.std(-1))[..., None].expand((q.shape[0], q.shape[1]))
   
   score = score.reshape(B, 1, T).expand(B, H, T)
      
   return  score, entropy

def pitome_vision(
    metric: torch.Tensor, 
    ratio:float=0.5,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
    alpha=1.0,
    use_bsm=False,
    use_bsm_pitome=False,
    protected_ratio:float=0
):
    if use_bsm:
        return bsm(metric=metric, ratio=ratio, class_token=class_token)
    with torch.no_grad():
        if class_token:
            metric=metric[:,1:,:]
        B,H,T,C = metric.shape
        
        r = int(T * ratio)
        protected_size = int(T * protected_ratio)

        # calculate energy score  
        metric = F.normalize(metric, p=2, dim=-1)
        metric = metric.mean(dim=1).unsqueeze(1).expand(-1, H, -1, -1)
        sim = metric@metric.transpose(-1,-2)
        energy_score = F.elu((sim - margin), alpha=alpha).mean(dim=-1)
        indices =  torch.argsort(energy_score, descending=True)
        if use_bsm_pitome:
            return pitome_bsm(metric=metric, class_token=class_token, indices=indices, scores=sim, r=r)
        else:
            return pitome(metric=metric, class_token=class_token, indices=indices, scores=sim, r=r, protected_size=protected_size)

def boundary_estimate(v, kernel_size=11):
    B, H, T, C = v.shape
    v = v.mean(dim=1, keepdim=True)
    v = F.normalize(v, dim=-1)
    v_seq = v.permute(1,-1,0,-2).reshape(-1, C, B*T)
    pooled_v = F.avg_pool1d(v_seq, kernel_size=kernel_size, stride=1, count_include_pad=False)
    
    left_boundary_pad = pooled_v[:, :, 0].unsqueeze(-1).expand(-1, C, kernel_size // 2)
    right_boundary_pad = pooled_v[:, :, -1].unsqueeze(-1).expand(-1, C, kernel_size // 2)
    pooled_v = torch.cat([left_boundary_pad, pooled_v, right_boundary_pad], dim=-1)
    
    pooled_v = pooled_v.reshape(-1, C, B, T).permute(2, 0, 3, 1)
    pooled_v = F.normalize(pooled_v, dim=-1)
    # pooled_v = pooled_v.mean(dim=1, keepdim=True).expand(-1, H, -1, -1)
    boundary_score = (pooled_v * v).sum(dim=-1)
    
    boundary_score = boundary_score.reshape(B, 1, T).expand(B, H, T)    
    
    # a, m = 1.0, 0.0
    # boundary_score = (torch.exp(boundary_score - m) - 1.0) * a
    
    return boundary_score

def pitome_boundary(
    metric: torch.Tensor, 
    ratio:float=0.5,
    class_token: bool = False,
    use_bsm=False,
    use_bsm_pitome=False,
    protected_ratio:float=0
):
    if use_bsm:
        return bsm(metric=metric, ratio=ratio, class_token=class_token)
    with torch.no_grad():
        if class_token:
            metric=metric[:,1:,:]
        B,H,T,C = metric.shape
        
        r = int(T * ratio)
        protected_size = int(T * protected_ratio)
        # Lowest score is the most important / boundary and put to the last
        boundary_score = boundary_estimate(metric, 11)
        # object_score, _ = cal_attn_score(metric, metric)
        # score = torch.maximum(object_score, boundary_score)
        score = boundary_score
        indices =  torch.argsort(score, descending=True)

        # calculate energy score  
        metric = F.normalize(metric, p=2, dim=-1) 
        sim = metric@metric.transpose(-1,-2)

        if use_bsm_pitome:
            return pitome_bsm(metric=metric, class_token=class_token, indices=indices, scores=sim, r=r)
        else:
            return pitome(metric=metric, class_token=class_token, indices=indices, scores=sim, r=r, protected_size=protected_size)

def merge_mean(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="mean")
    return x

def prune(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="prune")
    return x


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x*size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, size 


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

def merge_attention_mask(
    merge, attention_mask: torch.Tensor
): 
    attention_mask = merge(attention_mask, mode="amax")
    return attention_mask 

