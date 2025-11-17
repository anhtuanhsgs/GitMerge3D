import torch


import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

def furthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: (N, 3) point cloud
    npoint: int, number of points to sample
    Returns: (npoint,) indices of selected centroids
    """
    N, _ = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (1,), device=xyz.device).item()

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=0)[1].item()
    return centroids


def fps_knn_downsample(points: torch.Tensor, features: torch.Tensor, grid_coords: torch.Tensor, ratio: float):
    """
    Args:
        points: (N, 3)
        features: (N, C)
        ratio: float in (0,1), ratio of points to sample

    Returns:
        downsampled_points: (M, 3)
        downsampled_features: (M, C)
        assignments: (N,) index of centroid for each point
    """
    N = points.shape[0]
    M = int(N * ratio)

    # Step 1: Furthest Point Sampling
    centroid_idx = furthest_point_sampling(points, M)  # (M,)
    centroids = points[centroid_idx]  # (M, 3)

    # 2. Assign using sklearn NearestNeighbors (CPU)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centroids.cpu().numpy())
    distances, indices = nbrs.kneighbors(points.cpu().numpy())  # (N, 1)
    assignments = torch.from_numpy(indices[:, 0]).to(device=features.device, dtype=torch.long)  # (N,)


    # Step 3: Aggregate features and point positions
    downsampled_points = torch.zeros((M, 3), device=points.device)
    downsampled_features = torch.zeros((M, features.shape[1]), device=features.device)
    downsampled_grid_coords = torch.zeros((M, 3), dtype=features.dtype, device=features.device)
    counts = torch.zeros(M, device=points.device).unsqueeze(1)

    downsampled_points.index_add_(0, assignments, points)
    downsampled_features.index_add_(0, assignments, features)
    downsampled_grid_coords.index_add_(0, assignments, grid_coords.float())
    
    counts.index_add_(0, assignments, torch.ones(N, 1, device=points.device))

    downsampled_points /= counts
    downsampled_features /= counts
    downsampled_grid_coords /= counts
    
    downsampled_grid_coords = downsampled_grid_coords.round().long()
    
    return downsampled_points, downsampled_features, downsampled_grid_coords, assignments

def map_to_original_from_centroids(downsampled_features: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
    """
    Args:
        downsampled_features: (M, C) features after processing (e.g., logits)
        assignments: (N,) centroid index for each original point

    Returns:
        mapped_features: (N, C)
    """
    return downsampled_features[assignments]


def voxel_downsample(points: torch.Tensor, features: torch.Tensor, grid_coords: torch.Tensor, voxel_size: float):
    """
    Args:
        points: Tensor of shape (N, 3), float - 3D coordinates
        features: Tensor of shape (N, C), float - associated features
        voxel_size: float - size of each voxel

    Returns:
        downsampled_points: (M, 3)
        downsampled_features: (M, C)
    """
    assert points.shape[0] == features.shape[0]
    N, C = features.shape

    # Step 1: Compute voxel coordinates (integer grid index)
    voxel_coords = torch.floor(points / voxel_size).int()  # (N, 3)

    # Step 2: Hash voxel coordinates to unique voxel IDs
    voxel_ids = voxel_coords[:, 0] * 1_000_000 + voxel_coords[:, 1] * 1_000 + voxel_coords[:, 2]
    unique_ids, inverse_indices = torch.unique(voxel_ids, return_inverse=True)

    num_voxels = unique_ids.shape[0]
    device = points.device

    # Step 3: Initialize output buffers
    downsampled_points = torch.zeros((num_voxels, 3), dtype=points.dtype, device=device)
    downsampled_features = torch.zeros((num_voxels, C), dtype=features.dtype, device=device)
    downsampled_grid_coords = torch.zeros((num_voxels, 3), dtype=features.dtype, device=device)
    counts = torch.zeros((num_voxels,), dtype=torch.long, device=device)

    # Step 4: Aggregate point positions and features
    downsampled_points.index_add_(0, inverse_indices, points)
    downsampled_features.index_add_(0, inverse_indices, features)
    downsampled_grid_coords.index_add_(0, inverse_indices, grid_coords.float())
    counts.index_add_(0, inverse_indices, torch.ones_like(inverse_indices))
    

    # Step 5: Average
    downsampled_points /= counts.unsqueeze(1)
    downsampled_features /= counts.unsqueeze(1)
    downsampled_grid_coords /= counts.unsqueeze(1)
    
    downsampled_grid_coords = downsampled_grid_coords.round().long()

    return downsampled_points, downsampled_features, downsampled_grid_coords

def voxel_downsample_map_logits_to_original(points: torch.Tensor, downsampled_points: torch.Tensor, logits: torch.Tensor, voxel_size: float):
    """
    Args:
        points: (N, 3) original point cloud
        downsampled_points: (M, 3) points after voxel downsampling
        logits: (M, C) logits predicted per downsampled point
        voxel_size: float - same voxel size used in downsampling

    Returns:
        original_logits: (N, C) logits assigned to each original point
    """
    # Step 1: Compute voxel coordinates
    original_voxels = torch.floor(points / voxel_size).int()
    down_voxels = torch.floor(downsampled_points / voxel_size).int()

    # Step 2: Hash voxel coordinates to match groups
    def hash_voxels(voxel_coords):
        return voxel_coords[:, 0] * 1_000_000 + voxel_coords[:, 1] * 1_000 + voxel_coords[:, 2]

    original_ids = hash_voxels(original_voxels)
    downsampled_ids = hash_voxels(down_voxels)

    # Step 3: Create map from voxel ID -> logits
    id_to_index = {vid.item(): idx for idx, vid in enumerate(downsampled_ids)}
    index_map = torch.tensor([id_to_index[vid.item()] for vid in original_ids], device=logits.device)

    # Step 4: Map logits
    original_logits = logits[index_map]

    return original_logits


def random_downsample(points: torch.Tensor, features: torch.Tensor, grid_coord: torch.Tensor, ratio: float):
    """
    Randomly downsample a point cloud.
    Args:
        points: (N, 3)
        features: (N, C)
        ratio: float, fraction of points to keep

    Returns:
        sampled_points: (M, 3)
        sampled_features: (M, C)
        sampled_indices: (M,) indices of sampled points
    """
    N = points.shape[0]
    M = int(N * ratio)
    indices = torch.randperm(N)[:M].to(points.device)
    return points[indices], features[indices], grid_coord[indices], indices

def knn_map_back(processed_features: torch.Tensor, sampled_points: torch.Tensor, original_points: torch.Tensor) -> torch.Tensor:
    """
    Map features from sampled points back to original points using 1-NN.
    Args:
        processed_features: (M, C), features or logits of downsampled points
        sampled_points: (M, 3), downsampled point positions
        original_points: (N, 3), full point cloud

    Returns:
        mapped_features: (N, C)
    """
    # Use CPU for sklearn
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(sampled_points.cpu().numpy())
    distances, indices = nbrs.kneighbors(original_points.cpu().numpy())  # (N, 1)
    nearest_indices = torch.from_numpy(indices[:, 0]).to(original_points.device)

    # Assign features
    return processed_features[nearest_indices]