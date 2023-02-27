# %%
import os
import sys

sys.path.insert(0, '../')
import torch
import numpy as np
import imageio
import glob
from einops import rearrange
from matplotlib import pyplot as plt

# try:
#     import piplite
#
#     await piplite.install(['ipywidgets'])
# except ImportError:
#     pass
# import ipywidgets as widgets
# %%
from engine.trainer import Trainer
from engine.eval import evaluation_path
from data import dataset_dict
from utils.opt import config_parser
from utils.vis import plot_palette_colors, visualize_depth_numpy, visualize_palette_components_numpy
from utils.color import rgb2hex, hex2rgb
from utils.ray import get_rays, ndc_rays_blender


# %% md
## Utils
# %%
def print_divider():
    print()


path_redirect = [
    # option name, path in the config, redirected path
    ('palette_path', '../data_palette', './data_palette')
]
# %%
run_dir = './logs/fern/'
ckpt_path = None
out_dir = os.path.join(run_dir, 'demo_out')

print('Run dir:', run_dir)
print('Demo output dir:', out_dir)
# %% md
## Load and Setup
# %%
# Read args
parser = config_parser()
# 对args.txt里的参数进行获取
config_path = os.path.join(run_dir, 'args.txt')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        args, remainings = parser.parse_known_args(args=[], config_file_contents=f.read())

        # override ckpt path
        if ckpt_path is not None:
            setattr(args, 'ckpt', ckpt_path)

        # redirect path
        for entry in path_redirect:
            setattr(args, entry[0], getattr(args, entry[0]).replace(entry[1], entry[2]))

        print('Args loaded:', args)
else:
    print(f'ERROR: cannot read args in {run_dir}.')
print_divider()

# Setup trainer
print('Initializing trainer and model...')
ckpt_dir = os.path.join(run_dir, 'checkpoints')
tb_dir = os.path.join(run_dir, 'tensorboard')
# 训练器
trainer = Trainer(args, run_dir, ckpt_dir, tb_dir)
# 模型
model = trainer.build_network()
model.eval()
print_divider()

# Create downsampled dataset
dataset = dataset_dict[args.dataset_name]
ds_test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train * 2., is_stack=True)
print('Downsampled dataset loaded')

# %% md
## Palette Editing
# %%
palette_prior = trainer.palette_prior.detach().cpu().numpy()
palette = model.renderModule.palette.get_palette_array().detach().cpu().numpy()
# %%
print('Initial palette prior:')
plot_palette_colors(palette_prior)
print(palette)
# %%

# %%
print('Optimized palette:')
new_palette = palette.clip(0., 1.)
# palette[1] = palette[1].clip(0.5,1.)
new_palette = new_palette.clip(0.5, 0.7)
print(new_palette)
plot_palette_colors(new_palette)

# %%

# %%