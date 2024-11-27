import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from mmengine.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine.runner import Runner
from mmengine.config import Config
import mmcv
import os.path as osp
import torch
import os
import shutil

# MMSeg imports
from mmseg.registry import MODELS, DATASETS
from mmseg.models.builder import BACKBONES, HEADS
from mmseg.models.segmentors import *
from mmseg.models.backbones import *
from mmseg.models.decode_heads import *
from mmseg.datasets import BaseSegDataset
from mmseg.models.segmentors import EncoderDecoder

from mmseg.registry import DATASETS
from torch.utils.data import DataLoader

@MODELS.register_module()
class DebugEncoderDecoder(EncoderDecoder):
    def extract_feat(self, inputs):
        # Debug statement to inspect input type and shape
        print(f"[DEBUG] Input to backbone: type={type(inputs)}, shape={inputs.shape if isinstance(inputs, torch.Tensor) else 'N/A'}")
        return super().extract_feat(inputs)


# Register components
MODELS.register_module(module=EncoderDecoder, force=True)
BACKBONES.register_module(module=SwinTransformer, force=True)
HEADS.register_module(module=UPerHead, force=True)

@DATASETS.register_module()
class IndianPinesDataset(BaseSegDataset):
   CLASSES = tuple(range(17))
   PALETTE = None
   
   def __init__(self, data_root, data_prefix, patch_size=7, max_samples=2000, **kwargs):
       self.patch_size = patch_size
       self.half_patch = patch_size // 2
       self.max_samples = max_samples
       
       # Load data
       self.data = np.load('/Users/mengshuli/Desktop/Thesis/hyperspectral-dl/mmsegmentation/custom_script/data/images/Indian_Pines_raw.npy')
       self.label = np.load('/Users/mengshuli/Desktop/Thesis/hyperspectral-dl/mmsegmentation/custom_script/data/annotations/Indian_Pines_gt.npy')
       self.height, self.width, self.channels = self.data.shape
       
       # Get valid positions before parent init
       valid_positions = np.argwhere(self.label > 0)
       if len(valid_positions) > max_samples:
           indices = np.random.choice(len(valid_positions), max_samples, replace=False)
           valid_positions = valid_positions[indices]
       self.valid_positions = valid_positions
       
       super().__init__(
           data_root=data_root,
           data_prefix=data_prefix,
           **kwargs
       )

   def load_data_list(self):
       return [{'index': i} for i in range(len(self.valid_positions))]

   def prepare_data(self, idx):
    i, j = self.valid_positions[idx]

    patch = self.data[
        max(0, i-self.half_patch):min(self.height, i+self.half_patch+1),
        max(0, j-self.half_patch):min(self.width, j+self.half_patch+1),
        :
    ]

    if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
        pad_height = max(0, self.patch_size - patch.shape[0])
        pad_width = max(0, self.patch_size - patch.shape[1])
        patch = np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')

    patch = torch.from_numpy(patch).float()
    patch = patch.permute(2, 0, 1)  # Channels first.

    # Debugging dataset output
    print(f"[DEBUG] Dataset output: patch type={type(patch)}, patch shape={patch.shape}, label={self.label[i, j]}")

    return dict(
    inputs=patch,
    data_samples=dict(
        gt_sem_seg=torch.tensor(self.label[i, j]).long()
    )
    )['inputs']  # Return only the 'inputs' tensor



DATASETS.register_module(module=IndianPinesDataset, force=True)

cfg = Config(dict(
    model=dict(
        # type='mmseg.EncoderDecoder',
        type = 'mmseg.DebugEncoderDecoder',
        backbone=dict(
            type='mmseg.SwinTransformer',
            in_channels=5,   # # of channels of the HSI image
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            pretrain_img_size=224),
        decode_head=dict(
            type='mmseg.UPerHead',
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=17,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))),
    dataset_type='mmseg.IndianPinesDataset',
    data_root='data/indian_pines',
    # train_pipeline=[dict(type='PackSegInputs')],
    train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ToTensor', keys=['img']),
    dict(type='PackSegInputs')
    ],
    train_dataloader=dict(
    batch_size=32,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='mmseg.IndianPinesDataset',
        data_root='data/indian_pines',
        data_prefix=dict(
            img_path='images',
            seg_map_path='annotations'
        ),
        max_samples=2000
    )
    ),
    val_dataloader=dict(
        batch_size=1,
        num_workers=2,
        dataset=dict(
            type='mmseg.IndianPinesDataset',
            data_root='data/indian_pines',
            data_prefix=dict(
                img_path='images',
                seg_map_path='annotations'
            ),
            max_samples=500
        )
    ),
    test_dataloader=dict(
        batch_size=1,
        num_workers=2,
        dataset=dict(
            type='mmseg.IndianPinesDataset',
            data_root='data/indian_pines',
            data_prefix=dict(
                img_path='images',
                seg_map_path='annotations'
            ),
            max_samples=500
        )
    ),
    optimizer=dict(type='AdamW', lr=0.001),
    optim_wrapper=dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.001)),
    param_scheduler=[dict(type='PolyLR', power=0.9, eta_min=0.0001, by_epoch=True)],
    train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1),
    val_cfg=dict(type='ValLoop'),
    test_cfg=dict(type='TestLoop'),
    val_evaluator=dict(
        type='mmseg.IoUMetric',
        iou_metrics=['mIoU'],
        nan_to_num=0,
        use_label_map=False
    ),
    test_evaluator=dict(type='mmseg.IoUMetric', iou_metrics=['mIoU']),
    default_hooks=dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=1),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True),
        sampler_seed=dict(type='DistSamplerSeedHook')
    ),
    work_dir='./work_dirs/indian_pines'
))


def debug_dataloader(cfg):
    from mmseg.registry import DATASETS
    from torch.utils.data import DataLoader

    dataset = DATASETS.build(cfg.train_dataloader['dataset'])
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_dataloader['batch_size'],
        shuffle=cfg.train_dataloader['sampler']['shuffle'],
        num_workers=cfg.train_dataloader['num_workers']
    )

    for i, batch in enumerate(dataloader):
        print(f"[DEBUG] Batch {i}: {batch}")
        if isinstance(batch, dict) and 'inputs' in batch:
            print(f"[DEBUG] inputs type: {type(batch['inputs'])}")
            print(f"[DEBUG] inputs shape: {batch['inputs'].shape if isinstance(batch['inputs'], torch.Tensor) else 'N/A'}")
        else:
            print(f"[DEBUG] Unexpected batch format: {type(batch)}")
        if i >= 1:  # Limit debugging to a couple of batches
            break



def main():
    print("Initializing the runner...")
    debug_dataloader(cfg)  # Debug the dataloader
    runner = Runner.from_cfg(cfg)
    print("Starting training...")
    runner.train()


if __name__ == '__main__':
    print("Initializing the script...")
    main()
