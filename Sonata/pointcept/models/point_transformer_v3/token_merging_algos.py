import torch
import math
from typing import Callable, Tuple
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

def important_patch_based_matching(
    metric: torch.Tensor,
    r: int,
    stride: int,
    class_token: bool = False,
    distill_token: bool = False,
    no_rand: bool = False,
    margin: float = 0.9,
    alpha: float = 1.0,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        raise
    if distill_token:
        raise

    t = metric.shape[2]
    
    gather = torch.gather
    
    if r <= 0:
        raise
    
    with torch.no_grad():

        
        n_tokens = metric.shape[2]
        n_segments = metric.shape[0]
        n_heads = metric.shape[1]
        
        n_bins = n_tokens // stride
        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        # if no_rand:
        #     rand_idx = torch.zeros(n_segments, n_heads, n_bins, 1, device=metric.device, dtype=torch.int64)
        # else:
        #     rand_idx = torch.randint(stride, size=(n_segments, n_heads, n_bins, 1), device=metric.device)

        # sim = metric @ metric.transpose(-1,2)
        # energy_score = F.elu((sim - margin), alpha=alpha).mean(dim=-1)
        energy_score = boundary_estimate(metric, kernel_size=11)

        if (n_bins * stride) <= n_tokens:
            energy_score = energy_score[:, :, :n_bins * stride]
            energy_score = energy_score.reshape(n_segments, n_heads, n_bins, stride)
            # high_energy_index = energy_score.argmax(-1) # [n_segments, n_heads, n_bins]
            high_energy_index = energy_score.argmax(-1) # [n_segments, n_heads, n_bins]
            high_energy_index = high_energy_index[...,None] # [n_segments, n_heads, n_bins, 1]
            
            # if energy_score.max() - energy_score.min() < 100:
            #     high_energy_index = torch.zeros(n_segments, n_heads, n_bins, 1, device=metric.device, dtype=torch.int64)
                # high_energy_index = torch.randint(stride, size=(n_segments, n_heads, n_bins, 1), device=metric.device)
        else:
            raise

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(n_segments, n_heads, n_bins, stride, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=3, index=high_energy_index, src=-torch.ones_like(high_energy_index, dtype=high_energy_index.dtype, device=metric.device))
        idx_buffer_view = idx_buffer_view.reshape(n_segments, n_heads, n_bins * stride)
        
        if (n_bins * stride) < n_tokens:
            idx_buffer = torch.zeros(n_segments, n_heads, n_tokens, device=metric.device, dtype=torch.int64)
            idx_buffer[..., :(n_bins * stride)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view
        
        rand_idx = idx_buffer.reshape(n_segments, n_heads, -1, 1).argsort(dim=2)
        
        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = n_bins
        a_idx = rand_idx[:,:, num_dst:, :] # src
        b_idx = rand_idx[:,:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            N = x.shape[-2]
            src = gather(x, dim=2, index=a_idx.expand(n_segments, n_heads, N - num_dst, C))
            dst = gather(x, dim=2, index=b_idx.expand(n_segments, n_heads, num_dst, C))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True) # n_segments x 2 x n_tokens x n_heads
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)
        
        # Can't reduce more than the # tokens in src
        r = min(a.shape[2], r)
        
        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1) # Score max value / Score max index
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens | modulo of number of subsets
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        # Which token id within subset A (src) is merged to subset B (unm) [0-len(dst)]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        B, H, T, C = src.shape
        
        unm = src.gather(dim=-2, index=unm_idx.expand(B, H, T - r, C))
        src = src.gather(dim=-2, index=src_idx.expand(B, H, r, C))
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, H, r, C), src, reduce=mode)

        if distill_token:
            raise
        else:
            return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[-2]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n1, n2, _, c = unm.shape
        

        src = dst.gather(dim=-2, index=dst_idx.expand(n1, n2,  r, c))
        out = torch.zeros(n1, n2,  metric.shape[-2], c, device=x.device, dtype=x.dtype)

        out = torch.zeros(n1, n2, n_tokens, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(n1, n2, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=unm_idx).expand(n1, n2, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=src_idx).expand(n1, n2, r, c), src=src)
        return out

    return merge, unmerge

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[2]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # n_segments x 2 x n_tokens x C

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2) # n_segments x 2 x n_token_set_A x n_token_set_B
        
        # Random merge
        n_tokens = metric.shape[2]
        id_min = 0
        id_max = n_tokens - 1
        # End random merge

        if class_token:
            id_min += 1

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n1, n2, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n1, n2, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n1, n2, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n1, n2, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[-2]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n1, n2, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n1, n2,  r, c))
        out = torch.zeros(n1, n2,  metric.shape[-2], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n1, n2, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n1, n2, r, c), src=src)

        return out

    return merge, unmerge

def progressive_merging(
    metric: torch.Tensor,
    r: int,
    threshold: float,
) -> Tuple[Callable, Callable]:
    """_summary_

    Args:
        metric (torch.Tensor): _description_
        r (int): _description_
        threshold (float): _description_

    Returns:
        Tuple[Callable, Callable]: _description_
    """
    
    with torch.no_grad():

        n_segments, n_heads, n_tokens, C = metric.shape
    
        metric = metric / metric.norm(dim=-1, keepdim=True) # n_segments x 2 x n_tokens x n_heads
        edges_scores = (metric[:,:,:-1,:] * metric[:,:,1:,:]).sum(-1) # n_segments x 2 x (n_tokens - 1)
        
        sorted_scores, sorted_indices = torch.sort(edges_scores, dim=-1, descending=True)
        

        
        merge_indices = sorted_indices[...,:r]
        
        merge_mask = torch.zeros(n_segments, n_heads, n_tokens, dtype=torch.bool, device=metric.device)
        merge_mask.scatter_(-1, merge_indices, 1)
        merge_mask[...,1:] = merge_mask[...,1:] | merge_mask[...,:-1]
        

        n_remaining_tokens = n_tokens - r
    
        
        def find_src_and_dst_idx():
            # Set the right most index of each 1s consecutive segment to be the dst index
            idx = torch.arange(merge_mask.shape[-1], device=merge_mask.device).expand_as(merge_mask)
            
            merged_edge_mask = torch.zeros(n_segments, n_heads, n_tokens-1, dtype=torch.bool, device=metric.device)
            merged_edge_mask.scatter_(-1, merge_indices, 1)
            

            dst_mask = torch.zeros_like(merge_mask)
            src_mask = torch.zeros_like(merge_mask)
            src_mask[...,:-1] = merged_edge_mask
            src_mask[...,-1] = False
            
        
            # Destination nodes are nodes that are not merged at all or have the all previous nodes merged to it 
            # dst_mask[...,1:-1] = merged_edge_mask[...,:-1] & (merged_edge_mask[...,1:] == 0) # Previous edge is merged, but next edge is not merged
            # dst_mask[...,1:-1] |= (merged_edge_mask[...,:-1] == 0) & (merged_edge_mask[...,1:] == 0) # Previous edge and next edge are not merged
            dst_mask[...,1:-1] = merged_edge_mask[...,1:] == 0 # next edge is not merged
            dst_mask[...,-1] = True # last node is always a dst node
            dst_mask[...,0] = ~merged_edge_mask[...,0] # If first node is not merged
            
            dst_idx = dst_mask.long() * idx
            dst_idx[dst_idx == 0] = 10**9
            
            tree_depth = math.ceil(math.log2(min(n_tokens, r)))
            for i in range(tree_depth):
                delta = int(2 ** i)
                dst_idx[...,:-delta] = torch.minimum(dst_idx[...,:-delta], dst_idx[...,delta:])
            
            dst_idx = dst_idx * merge_mask
            dst_idx[~merge_mask] = idx[~merge_mask]
            
            src_idx = src_mask.long() * idx 
            
            src_idx = src_idx[src_mask].reshape(n_segments, n_heads, -1)
            src_to_dst_idx = torch.gather(dst_idx, dim=-1, index=src_idx)
            
            dst_idx = dst_idx[dst_mask].reshape(n_segments, n_heads, n_remaining_tokens)
            
            return src_idx, dst_idx, src_to_dst_idx
        
        src_idx, dst_idx, src_to_dst_idx = find_src_and_dst_idx()
        
        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            C = x.shape[-1]
            src = torch.gather(x, dim=-2, index=src_idx[...,None].expand(n_segments, n_heads, src_idx.shape[-1], C))
            dst = x.scatter_reduce(-2, src_to_dst_idx[...,None].expand(n_segments, n_heads, src_idx.shape[-1], C), src, reduce=mode)
            dst = torch.gather(dst, dim=-2, index=dst_idx[...,None].expand(n_segments, n_heads, n_remaining_tokens, C))
            return dst
    
        def unmerge(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(n_segments, n_heads, n_tokens, x.shape[-1], device=x.device, dtype=x.dtype)
            out = out.scatter_(-2, index=dst_idx[...,None].expand(n_segments, n_heads, n_remaining_tokens, x.shape[-1]), src=x)
            src = out.gather(dim=-2, index=src_to_dst_idx[...,None].expand(n_segments, n_heads, src_to_dst_idx.shape[-1], x.shape[-1]))
            out = out.scatter_(-2, index=src_idx[...,None].expand(n_segments, n_heads, src_to_dst_idx.shape[-1], x.shape[-1]), src=src)
    
            return out
    
        return merge, unmerge

def pitome_bsm(
    metric=None,
    class_token: bool = False,
    indices:torch.Tensor=None,
    scores:torch.Tensor=None,
    r:int=None
) -> Tuple[Callable, Callable]:

    with torch.no_grad():
        B, T, _ = scores.shape
        a_idx, b_idx = indices[..., ::2], indices[..., 1::2] 
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, b_idx.shape[-1])) 
        scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[-1], b_idx.shape[-1]))
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
    
def pool_reduction_sampling(x0, kernel_size, method="AvgPool1d"):
    '''
        x0: [B, H, L, C]
        kernel_size: int    
    '''
    B, H, L, C = x0.shape
    def pool(x):
        '''
            x: [B, H, L, C]
            kernel_size: int    
        '''
        if L % kernel_size != 0:
            x1 = x[..., :-(L % kernel_size), :].reshape(B, H, -1, kernel_size, C) # [B, H, L'//ks, ks, C]
            x1 = x1.mean(-2)
            x2 = x[..., -(L % kernel_size):, :].mean(-2).reshape(B, H, 1, C)
            x = torch.cat([x1, x2], -2) # [B, H, L'//ks + 1, ks, C]
        else:
            x = x.reshape(B, H, -1, kernel_size, C)
            x = x.mean(-2)
        return x

    def unpool(x):
        '''
            x: [B, H, L//ks, C]
            kernel_size: int    
        '''
        x = x.unsqueeze(-2) #[B, H, L//ks, 1, C]
        x = x.repeat(1, 1, 1, kernel_size, 1) #[B, H, L, KS, C]
        x = x.reshape(B, H, -1, C)
        x = x[:,:,:L,:]
        return x

    return pool, unpool
       

def patch_based_matching(
    metric: torch.Tensor,
    r: int,
    stride: int,
    class_token: bool = False,
    distill_token: bool = False,
    no_rand: bool = False,
    mean_head: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        raise
    if distill_token:
        raise

    t = metric.shape[2]
    
    gather = torch.gather
    
    if r <= 0:
        raise
    
    with torch.no_grad(): 
        n_tokens = metric.shape[2]
        n_segments = metric.shape[0]
        n_heads = metric.shape[1]
        
        n_bins = n_tokens // stride
        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(n_segments, n_heads, n_bins, 1, device=metric.device, dtype=torch.int64)
            for h in range(0, n_heads):
                rand_idx[:,h] = (max(stride // n_heads, 1) * h) % stride
        else:
            rand_idx = torch.randint(stride, size=(n_segments, n_heads, n_bins, 1), device=metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(n_segments, n_heads, n_bins, stride, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype, device=metric.device))
        idx_buffer_view = idx_buffer_view.reshape(n_segments, n_heads, n_bins * stride)
        
        if (n_bins * stride) < n_tokens:
            idx_buffer = torch.zeros(n_segments, n_heads, n_tokens, device=metric.device, dtype=torch.int64)
            idx_buffer[..., :(n_bins * stride)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view
        
        rand_idx = idx_buffer.reshape(n_segments, n_heads, -1, 1).argsort(dim=2)
        
        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = n_bins
        a_idx = rand_idx[:,:, num_dst:, :] # src
        b_idx = rand_idx[:,:, :num_dst, :] # dst

        def split(x):
            B, H, T, C = x.shape
            src = gather(x, dim=2, index=a_idx.expand(B, H, T - num_dst, C))
            dst = gather(x, dim=2, index=b_idx.expand(B, H, num_dst, C))
            return src, dst

        if mean_head:
            metric = metric.mean(dim=1, keepdim=True).expand(-1, n_heads, -1, -1)
        metric = metric / metric.norm(dim=-1, keepdim=True) # n_segments x 2 x n_tokens x n_heads
        a, b = split(metric)
        
        scores = a @ b.transpose(-1, -2)
        
        # Can't reduce more than the # tokens in src
        r = min(a.shape[2], r)
        
        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1) # Score max value / Score max index
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens | modulo of number of subsets
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        # Which token id within subset A (src) is merged to subset B (unm) [0-len(dst)]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n1, n2, t1, c = src.shape
        
        unm = src.gather(dim=-2, index=unm_idx.expand(n1, n2, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n1, n2, r, c))
        if mode != "prune":
            dst = dst.scatter_reduce(-2, dst_idx.expand(n1, n2, r, c), src, reduce=mode)

        if distill_token:
            raise
        else:
            return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[-2]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n1, n2, _, c = unm.shape
        

        src = dst.gather(dim=-2, index=dst_idx.expand(n1, n2,  r, c))
        out = torch.zeros(n1, n2,  metric.shape[-2], c, device=x.device, dtype=x.dtype)

        out = torch.zeros(n1, n2, n_tokens, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(n1, n2, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=unm_idx).expand(n1, n2, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=src_idx).expand(n1, n2, r, c), src=src)
        return out

    return merge, unmerge

def random_patch_based_matching(
    metric: torch.Tensor,
    r: int,
    stride: int,
    class_token: bool = False,
    distill_token: bool = False,
    no_rand: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        raise
    if distill_token:
        raise

    t = metric.shape[2]
    
    gather = torch.gather
    
    if r <= 0:
        raise
    
    with torch.no_grad():

        
        n_tokens = metric.shape[2]
        n_segments = metric.shape[0]
        n_heads = metric.shape[1]
        
        n_bins = n_tokens // stride
        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(n_segments, n_heads, n_bins, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(stride, size=(n_segments, n_heads, n_bins, 1), device=metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(n_segments, n_heads, n_bins, stride, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype, device=metric.device))
        idx_buffer_view = idx_buffer_view.reshape(n_segments, n_heads, n_bins * stride)
        
        if (n_bins * stride) < n_tokens:
            idx_buffer = torch.zeros(n_segments, n_heads, n_tokens, device=metric.device, dtype=torch.int64)
            idx_buffer[..., :(n_bins * stride)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view
        
        rand_idx = idx_buffer.reshape(n_segments, n_heads, -1, 1).argsort(dim=2)
        
        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = n_bins
        a_idx = rand_idx[:,:, num_dst:, :] # src
        b_idx = rand_idx[:,:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            N = x.shape[-2]
            src = gather(x, dim=2, index=a_idx.expand(n_segments, n_heads, N - num_dst, C))
            dst = gather(x, dim=2, index=b_idx.expand(n_segments, n_heads, num_dst, C))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True) # n_segments x 2 x n_tokens x n_heads
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)
        
        # Can't reduce more than the # tokens in src
        r = min(a.shape[2], r)
        
        # Find the most similar greedily
        scores = torch.rand_like(scores)
        node_max, node_idx = scores.max(dim=-1) # Score max value / Score max index
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens | modulo of number of subsets
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        # Which token id within subset A (src) is merged to subset B (unm) [0-len(dst)]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n1, n2, t1, c = src.shape
        
        unm = src.gather(dim=-2, index=unm_idx.expand(n1, n2, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n1, n2, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n1, n2, r, c), src, reduce=mode)

        if distill_token:
            raise
        else:
            return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[-2]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n1, n2, _, c = unm.shape
        

        src = dst.gather(dim=-2, index=dst_idx.expand(n1, n2,  r, c))
        out = torch.zeros(n1, n2,  metric.shape[-2], c, device=x.device, dtype=x.dtype)

        out = torch.zeros(n1, n2, n_tokens, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(n1, n2, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=unm_idx).expand(n1, n2, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=src_idx).expand(n1, n2, r, c), src=src)
        return out

    return merge, unmerge

def boundary_estimate(v, kernel_size=11):
    B, H, T, C = v.shape
    v = v.mean(dim=1, keepdim=True)
    v = F.normalize(v, dim=-1)
    v_seq = v.permute(1,-1,0,-2).reshape(-1, C, B*T) # --> H, C, B, T --> H, C, (B * T)
    pooled_v = F.avg_pool1d(v_seq, kernel_size=kernel_size, stride=1, count_include_pad=False)
    
    # Dupplicate left and right for boundary
    left_boundary_pad = pooled_v[:, :, 0].unsqueeze(-1).expand(-1, C, kernel_size // 2)
    right_boundary_pad = pooled_v[:, :, -1].unsqueeze(-1).expand(-1, C, kernel_size // 2)
    pooled_v = torch.cat([left_boundary_pad, pooled_v, right_boundary_pad], dim=-1)
    
    pooled_v = pooled_v.reshape(-1, C, B, T).permute(2, 0, 3, 1) # --> H, C, B, T --> B, H, T, C
    pooled_v = F.normalize(pooled_v, dim=-1)
    # pooled_v = pooled_v.mean(dim=1, keepdim=True).expand(-1, H, -1, -1)
    boundary_score = -(pooled_v * v).sum(dim=-1)
    
    boundary_score = boundary_score.reshape(B, 1, T).expand(B, H, T)
    # boundary_score = boundary_score.reshape(B, H, T)
    # a, m = 1.0, 0.0
    # boundary_score = (torch.exp(boundary_score - m) - 1.0) * a
    return boundary_score

def cal_density_score(q:torch.Tensor, k:torch.Tensor, sigma=1):   
   B, H, T, C = k.shape
   q = q.mean(1)
   k = k.mean(1)
   q = F.normalize(q, p=2, dim=-1)
   k = F.normalize(k, p=2, dim=-1)
   sim = q.reshape(-1, q.shape[-1]) @ k.mean(1).transpose(-1, -2) 
   
   score = sim.max(-1)[0]
   
   score = -score.reshape(B, 1, T).expand(B, H, T)
      
   return  score

def ALGM(
    metric: torch.Tensor,
    r: int,
    stride: int,
    class_token: bool = False,
    distill_token: bool = False,
    no_rand: bool = False,
    alpha: float = 1.0,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        raise
    if distill_token:
        raise

    t = metric.shape[2]
    
    gather = torch.gather
    
    if r <= 0:
        raise
    
    with torch.no_grad(): 
        n_tokens = metric.shape[2]
        n_segments = metric.shape[0]
        n_heads = metric.shape[1]
        
        B, H, T, C = metric.shape
        
        n_bins = n_tokens // stride
        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        token_idx = torch.arange(T, device=metric.device).expand_as(metric[:,:1,:,0]) # [B, 1, T]
        
        
        metric_binned = metric[:, :, :n_bins * stride, :].reshape(n_segments, n_heads, n_bins, stride, C)
        metric_binned = metric_binned.mean(1, keepdim=True)
        bin_sim = (metric_binned @ metric_binned.transpose(-1, -2)).mean(-1).mean(-1)
        bin_mean, bin_var = bin_sim.mean(), bin_sim.var()
        
        thresholed_bin_sim = bin_sim > (bin_mean + alpha * bin_var)
        thresholed_bin_sim = thresholed_bin_sim.float()
        num_bins_selected = thresholed_bin_sim.sum(dim=-1, keepdim=True)
        num_bins_selected = torch.clamp(num_bins_selected.mean(), 10, num_bins_selected.mean() + 1).long()
        
        sorted_bin_sim, sorted_bin_indices = torch.sort(bin_sim, dim=-1, descending=True)
        merge_mask = torch.zeros(n_segments, 1, n_bins, dtype=torch.bool, device=metric.device)
        merge_mask.scatter_(-1, sorted_bin_indices[..., :num_bins_selected], 1)
        
        # Use the first token of the bin as the dsttmp token
        token_idx_bin = token_idx.clone()[:, :, :n_bins*stride].reshape(n_segments, 1, n_bins, stride) # [B, 1, n_bins, stride]
        merge_mask = merge_mask[..., None].expand_as(token_idx_bin)
        token_idx_bin[merge_mask] = token_idx_bin[...,:1].expand_as(token_idx_bin)[merge_mask]
        dst_idx_merged = token_idx_bin[merge_mask].reshape(-1, stride)[:,0].reshape(B, -1)
        dst_idx_single = token_idx_bin[~merge_mask].reshape(-1, stride).reshape(B, -1)
        dst_idx_remaining = token_idx.clone()[:, :, n_bins*stride:].reshape(n_segments, -1) 
        dst_idx = torch.cat([dst_idx_merged, dst_idx_single, dst_idx_remaining], dim=-1)[:,None,:]
        n_remaining_tokens = dst_idx.shape[-1]
        
        src_to_dst_idx = token_idx.clone()
        src_to_dst_idx[:, :, :n_bins*stride] = token_idx_bin.reshape(B, 1, -1)
        src_to_dst_idx = src_to_dst_idx.expand(-1, H, -1)
        
        src_idx = token_idx.clone()
        
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        C = x.shape[-1]
        src = torch.gather(x, dim=-2, index=src_idx[...,None].expand(n_segments, n_heads, src_idx.shape[-1], C))
        dst = x.scatter_reduce(-2, src_to_dst_idx[...,None].expand(n_segments, n_heads, src_idx.shape[-1], C), src, reduce=mode)
        dst = torch.gather(dst, dim=-2, index=dst_idx[...,None].expand(n_segments, n_heads, n_remaining_tokens, C))
        return dst
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(n_segments, n_heads, n_tokens, x.shape[-1], device=x.device, dtype=x.dtype)
        out = out.scatter_(-2, index=dst_idx[...,None].expand(n_segments, n_heads, n_remaining_tokens, x.shape[-1]), src=x)
        src = out.gather(dim=-2, index=src_to_dst_idx[...,None].expand(n_segments, n_heads, src_to_dst_idx.shape[-1], x.shape[-1]))
        out = out.scatter_(-2, index=src_idx[...,None].expand(n_segments, n_heads, src_to_dst_idx.shape[-1], x.shape[-1]), src=src)

        return out

    return merge, unmerge

def important_sampling_matching(
    metric: torch.Tensor,
    kernel_size: int,
    r: int,
    stride: int,
    class_token: bool = False,
    distill_token: bool = False,
    no_rand: bool = False,
) -> Tuple[Callable, Callable]:
    if class_token:
        raise
    if distill_token:
        raise

    t = metric.shape[2]
    
    gather = torch.gather
    
    if r <= 0:
        raise
    
    with torch.no_grad():
        n_tokens = metric.shape[2]
        n_segments = metric.shape[0]
        n_heads = metric.shape[1]
        
        n_bins = n_tokens // stride
        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            # rand_idx = torch.zeros(n_segments, n_heads, n_bins, 1, device=metric.device, dtype=torch.int64)
            rand_idx = torch.zeros(n_segments, n_heads, n_bins, 1, device=metric.device, dtype=torch.int64)
            for h in range(1, n_heads):
                rand_idx[:,h] = (stride // n_heads * h) % stride
        else:
            rand_idx = torch.randint(stride, size=(n_segments, n_heads, n_bins, 1), device=metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(n_segments, n_heads, n_bins, stride, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype, device=metric.device))
        idx_buffer_view = idx_buffer_view.reshape(n_segments, n_heads, n_bins * stride)
        
        if (n_bins * stride) < n_tokens:
            idx_buffer = torch.zeros(n_segments, n_heads, n_tokens, device=metric.device, dtype=torch.int64)
            idx_buffer[..., :(n_bins * stride)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view
        
        # higher score tokens are more important
        num_dst = n_bins
        boundary_score = boundary_estimate(metric, kernel_size)
        # boundary_score = cal_density_score(metric, metric)
        # sorted_score, sorted_indices =  torch.sort(boundary_score, descending=True)
        # boundary_threshold = sorted_score[...,num_dst].unsqueeze(-1)
        # boundary_idx = -(boundary_score > boundary_threshold).long()
        
        mean = boundary_score.mean()
        var = boundary_score.var()
        boundary_mask = (boundary_score > mean + var).long()
        
        idx_buffer = boundary_mask * 1.1 + idx_buffer + torch.randn(*idx_buffer.shape, device=metric.device) * 0.1
        # idx_buffer = boundary_mask 
        # idx_buffer = boundary_idx
        
        rand_idx = idx_buffer.reshape(n_segments, n_heads, -1, 1).argsort(dim=2)
        
        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = n_bins
        a_idx = rand_idx[:,:, num_dst:, :] # src
        b_idx = rand_idx[:,:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            N = x.shape[-2]
            src = gather(x, dim=2, index=a_idx.expand(n_segments, n_heads, N - num_dst, C))
            dst = gather(x, dim=2, index=b_idx.expand(n_segments, n_heads, num_dst, C))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True) # n_segments x 2 x n_tokens x n_heads
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)
        
        # Can't reduce more than the # tokens in src
        r = min(a.shape[2], r)
        
        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1) # Score max value / Score max index
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens | modulo of number of subsets
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        # Which token id within subset A (src) is merged to subset B (unm) [0-len(dst)]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n1, n2, t1, c = src.shape
        
        unm = src.gather(dim=-2, index=unm_idx.expand(n1, n2, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n1, n2, r, c))
        if mode != "prune":
            dst = dst.scatter_reduce(-2, dst_idx.expand(n1, n2, r, c), src, reduce=mode)

        if distill_token:
            raise
        else:
            return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[-2]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n1, n2, _, c = unm.shape
        

        src = dst.gather(dim=-2, index=dst_idx.expand(n1, n2,  r, c))
        out = torch.zeros(n1, n2,  metric.shape[-2], c, device=x.device, dtype=x.dtype)

        out = torch.zeros(n1, n2, n_tokens, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(n1, n2, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=unm_idx).expand(n1, n2, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=src_idx).expand(n1, n2, r, c), src=src)
        return out

    return merge, unmerge