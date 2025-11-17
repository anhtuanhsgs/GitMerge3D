import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from collections import OrderedDict
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from fvcore.nn import FlopCountAnalysis
import warnings
from tqdm.auto import tqdm
# from pointcept.datasets.transform import Compose
# from pointcept.utils.config import Config
# from pointcept.utils.visualization import save_point_cloud
# from pointcept.utils.comm import get_world_size
# from pointcept.datasets.utils import point_collate_fn
import pointcept.utils.comm as comm
# from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import VALID_TOME_MODES

from pointcept.utils.env import get_random_seed, set_seed

from pointcept.engines.pcd_downsampling_methods import *

from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_20,
    VALID_CLASS_IDS_20
)
from torchprofile.handlers import handlers
from torchprofile.utils.trace import trace
from copy import deepcopy


from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    # intersection_and_union_gpu,
    # make_dirs,
)

# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torchprofile import profile_macs

import pyviz3d.visualizer as viz
import random
import argparse

from pointcept.engines.defaults import (
    # default_argument_parser,
    default_config_parser,
    # default_setup,
)
from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    SerializedAttention,
    MLP,
    Point,
    Block
)

import math


# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"


def cal_energy_score(metric: torch.Tensor, sigma=0.1):
    metric = F.normalize(metric, p=2, dim=-1)
    sim = metric @ metric.transpose(-1, 2)
    energy_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))
                    ).mean(-1) * 1/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))
    return energy_score, sim


VALID_TOME_MODES = ["patch", "tome", "progressive",
                    "pitome", "random_patch", "base", 'important_patch']


def create_color_palette():
    return np.array([
        (0, 0, 0),
        (174, 199, 232),		# wall
        (152, 223, 138),		# floor
        (31, 119, 180), 		# cabinet
        (255, 187, 120),		# bed
        (188, 189, 34), 		# chair
        (140, 86, 75),  		# sofa
        (255, 152, 150),		# table
        (214, 39, 40),  		# door
        (197, 176, 213),		# window
        (148, 103, 189),		# bookshelf
        (196, 156, 148),		# picture
        (23, 190, 207), 		# counter
        (178, 76, 76),
        (247, 182, 210),		# desk
        (66, 188, 102),
        (219, 219, 141),		# curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14), 		# refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),		# shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  		# toilet
        (112, 128, 144),		# sink
        (96, 207, 209),
        (227, 119, 194),		# bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  		# otherfurn
        (100, 85, 144),
        (0, 0, 0),
    ], dtype=np.uint8)


class Engine:
    def __init__(
        self,
        model_config,
        data_config,
        checkpoint='./models/PointTransformerV3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth',
        keywords='backbone.',
        replacement='',
        config_name='',
        config_path='',
    ):

        # self.additional_info = {
        #     "replace_attn": None,
        #     "tome": "patch",
        #     "r": 0.1,
        #     "stride": 50,
        #     "tome_mlp": True,
        #     "tome_attention": True,
        #     "trace_back": False,
        #     "single_head_tome": False,
        #     "margin": 0.9,
        #     "alpha": 1.0,
        #     "threshold": 0.9
        # }
        
        if 'scannet200' in config_path:
            checkpoint = 'exp/sonata/semseg-sonata-v1m1-0-base-1a-scannet200-lin/model/model_best.pth'
        elif 'scannet' in config_path:
            checkpoint = 'exp/sonata/semseg-sonata-v1m1-0-base-0c-scannet-ft/model/model_best.pth'
        elif 's3dis' in config_path:
            checkpoint = 'exp/sonata/semseg-sonata-v1m1-0-base-3a-s3dis-lin/model/model_best.pth'    
        self.checkpoint = checkpoint
        self.keywords = keywords
        self.replacement = replacement
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.data_config = data_config
        self.get_model(self.model_config)
        self.dataset = build_dataset(self.data_config)

    def get_model(self, model_config):
        self.model = build_model(model_config).cuda()
        checkpoint = torch.load(self.checkpoint)
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                if comm.get_world_size() > 1:
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
            weight[key] = value
        self.model.load_state_dict(weight, strict=True)
        self.model.training = False

    def get_data(self,  idx):
        data = self.dataset[idx]
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda(non_blocking=True)
        return data

    def get_pca_color(self, feat):
        feat_centered = feat - feat.mean(dim=-2, keepdim=True)
        u, s, v = torch.pca_lowrank(feat_centered, center=False, q=3)
        X_pca = feat_centered @ v[:, :3]
        # Normalize PCA values to range [0, 255] for color
        X_pca_min = X_pca.min(dim=0)[0]  # Min per component
        X_pca_max = X_pca.max(dim=0)[0]  # Max per component
        X_pca_scaled = (X_pca - X_pca_min) / (X_pca_max -
                                              X_pca_min)  # Normalize to [0,1]
        # Scale to [0,255] and convert to uint8
        X_pca_uint8 = (X_pca_scaled * 255).to(torch.uint8).cpu()
        return X_pca_uint8.numpy()

    def register_hooks(self):
        gflops =   defaultdict(lambda: 0)
        hooks = []
        hooked_layer_names = []
        

        def hook_fn(module, point: Point, output):
            point = point[0].copy()

            shortcut = point.feat
            point = module.cpe(point)
            point.feat = shortcut + point.feat

            shortcut = point.feat
            if module.pre_norm:
                point = module.norm1(point)

            copied_point = {
                'feat': point.feat,
                'offset': point.offset,
                'serialized_order': point.serialized_order,
                'serialized_inverse': point.serialized_inverse,
            }
            flops = FlopCountAnalysis(module.attn, copied_point)
            flops.unsupported_ops_warnings(
                False).uncalled_modules_warnings(False)
            attn_flops = flops.total() / 1e9
            # print('flops attn: {:.4g} G'.format(attn_flops))
            gflops['attn flops'] += attn_flops
            
            point = module.drop_path(module.attn(point))
            point.feat = shortcut + point.feat
            if not module.pre_norm:
                point = module.norm1(point)

            shortcut = point.feat
            if module.pre_norm:
                point = module.norm2(point)

            # point = module.drop_path(module.mlp(point))
            # point.feat = shortcut + point.feat
            # if not module.pre_norm:
            # point = module.norm2(point)
            # point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        for name, module in self.model.named_modules():
            if isinstance(module, Block):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                hooked_layer_names.append(name)

        return hooks, gflops, hooked_layer_names

    def remove_get_feature_hook(self, hooks):
        for hook in hooks:
            hook.remove()


def cal_macs(args):

    set_seed(args.seed)
    attn_flops = 0
    cfg = default_config_parser(args.config, None)
    config_name = os.path.basename(args.config).split(".")[0]
    
    config_path = args.config
    # Extract relevant parts to construct log path
    base_name = os.path.basename(config_path)           # "tome-wpatch.py"
    name_no_ext = os.path.splitext(base_name)[0]        # "tome-wpatch"
    parent_dir = os.path.basename(os.path.dirname(config_path))  # "scannet"

    # Construct log file path
    log_path = f"exp/flops/{parent_dir}-{name_no_ext}.log"
    os.makedirs("exp/flops/", exist_ok=True)

    model_config = cfg.model
    data_config = cfg.data.val

    if args.merge_rates is None:
        if "additional_info" in model_config["backbone"].keys():
            merge_rates = [model_config["backbone"]["additional_info"]["r"]]
        else:
            merge_rates = [0.0]
    else:
        merge_rates = args.merge_rates
        
    if args.alphas is None:
        if "additional_info" in model_config["backbone"].keys() and "alpha" in model_config["backbone"]["additional_info"].keys():
            alphas = [model_config["backbone"]["additional_info"]["alpha"]]
        else:
            alphas = [0.0]
    else:
        alphas = args.alphas
    tmp = 0
        
    strides = [math.ceil(1.0 / (1.0 - r)) for r in merge_rates]
    
    with open(log_path, "w") as f:
        # Clear cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        f.write("r,stride, gflops, peak_memory\n")
        for r, stride in zip(merge_rates, strides):
            if "additional_info" in model_config["backbone"].keys():
                model_config["backbone"]["additional_info"]["r"] = r
                model_config["backbone"]["additional_info"]["stride"] = stride
                if alphas is not None:
                    model_config["backbone"]["additional_info"]["alpha"] = alphas[min(tmp, len(alphas)-1)] 
                
            engine = Engine(
                model_config=model_config,
                data_config=data_config,
                config_name=config_name,
                config_path=config_path,
            )

            hooks, gflops, hook_layer_names = engine.register_hooks()
            scene_count = 0
            for idx in tqdm(range(min(len(engine.dataset), args.max_scene_count)), desc='Calculating FLOPS'):
                data = engine.get_data(idx=idx)
                scene_count += 1
                with torch.no_grad():
                    processed_input_dict = data
                    if "additional_info" in model_config["backbone"].keys():
                        if "downsample" in model_config["backbone"]["additional_info"]:
                            downsample = True
                            if model_config["backbone"]["additional_info"]["downsample"] == "voxel":
                                coord, feat, grid_coord = voxel_downsample(data['coord'], data['feat'], data['grid_coord'], model_config["backbone"]["additional_info"]["voxel_size"])
                            elif model_config["backbone"]["additional_info"]["downsample"] == "fps":
                                coord, feat, grid_coord, assignments = fps_knn_downsample(data['coord'], data['feat'], data['grid_coord'], model_config["backbone"]["additional_info"]["downsample_ratio"])
                            elif model_config["backbone"]["additional_info"]["downsample"] == "random":
                                downsample_ratio = model_config["backbone"]["additional_info"]["downsample_ratio"]
                                coord, feat, grid_coord, indices = random_downsample(data['coord'], data['feat'], data['grid_coord'], downsample_ratio)
                            else:
                                raise NotImplementedError()
                            processed_input_dict = {
                                "coord": coord,
                                "feat": feat,
                                "grid_coord": grid_coord,
                                "offset": torch.tensor([coord.shape[0]]).cuda(),
                            }     
                            print(data['coord'].shape[0], coord.shape[0])
                    if scene_count % 10 == 0:
                        print(scene_count, "gflops", gflops['attn flops']/scene_count)
                    _ = engine.model(processed_input_dict)
            peak_memory = torch.cuda.max_memory_allocated()
            final_gflops = gflops['attn flops']/scene_count
            print(f"r\tstride\tfinal_gflops\tpeak_memory\n")
            
            print(f"{r}\t{stride}\t{final_gflops:.2f}\t{peak_memory / (1024 ** 3):.2f}\n")
            # Write to the log file
            f.write(f"{r},{stride},{final_gflops:.2f},{peak_memory / (1024 ** 3):.2f}\n")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict and visualize features")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index of the data sample to visualize")
    parser.add_argument(
        "--config", type=str, default="configs/scannet/semseg-pt-v3m1-0-base.py", help="Path to the config file")
    parser.add_argument(
        "--merge_rates", type=float, default=None, nargs="+", help="Merge rates for the model")
    parser.add_argument("--alphas", type=float, default=None, nargs="+", help="Alpha values for the model")
    parser.add_argument("--max_scene_count", type=int, default=1000,
                        help="Maximum number of scenes to process")
    
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()


    cal_macs(args)