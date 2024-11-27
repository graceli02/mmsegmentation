# configs/indian_pines/unet_swin_transformer.py

_base_ = [
    '../_base_/models/swin_unet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# Dataset settings
dataset_type = 'IndianPinesDataset'
data_root = '/Users/mengshuli/Desktop/Thesis/hyperspectral-dl/Indian_Pines_Code/PCA5/output/'

# Normalization configuration
mean = [-1.24985552e-11, 1.35621774e-11, 5.08233426e-12, 2.65462146e-12,
        -2.76951410e-12, -1.01714167e-12, 1.21835521e-13, 3.36156163e-13,
        1.21511088e-13, -6.41080213e-14, 1.37765196e-12, -1.57403144e-12,
        3.33364414e-12, 6.29746676e-13, -6.63704029e-13, -2.11657585e-13,
        -3.76269097e-12, -1.43707733e-12, 9.29566309e-13, -1.55198079e-13]

std = [5176.46031545, 3034.10495249, 765.11440713, 566.9204123, 521.43942277,
       449.73732348, 395.4251346, 376.51083654, 346.63033059, 338.68671257,
       323.40730686, 312.25387872, 296.48157615, 272.19795933, 257.46061857,
       247.07578694, 244.78438237, 229.82389726, 197.92899662, 190.21673458]

img_norm_cfg = dict(
    mean=mean,
    std=std,
    to_rgb=False)

# Pipeline settings
train_pipeline = [
    dict(type='LoadNPYImageFromFile'),
    dict(type='LoadNPYAnnotations', reduce_zero_label=False),
    dict(type='HyperspectralResize', scale=(128, 128), keep_ratio=True),
    dict(type='HyperspectralRandomFlip', prob=0.5),
    dict(type='HyperspectralNormalize', **img_norm_cfg),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadNPYImageFromFile'),
    dict(type='HyperspectralResize', scale=(128, 128), keep_ratio=True),
    dict(type='HyperspectralNormalize', **img_norm_cfg),
    dict(type='PackSegInputs')
]
# Dataset configs
train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    # data_prefix=dict(
    #     img_path='Indian_Pines_raw.npy',
    #     seg_map_path='Indian_Pines_gt.npy'),
    data_prefix = None,
    pipeline=train_pipeline)

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    # data_prefix=dict(
    #     img_path='Indian_Pines_raw.npy',
    #     seg_map_path='Indian_Pines_gt.npy'),
    data_prefix = None,
    pipeline=test_pipeline)

test_dataset = val_dataset

# Dataloader configs
train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

test_dataloader = val_dataloader

# Model configuration
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SwinTransformer',
        in_channels=5,    # number of channels of the HSI image
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_size=4,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/swin_tiny_patch4_window7_224_22k.pth',
            prefix='backbone.',
            patch_conv_modify=True
        )
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels = [96, 192, 384, 768],  # Matches Swin Transformer's output dimensions
        in_index=[0, 1, 2, 3], 
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=16,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=16,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Optimizer config
optim_wrapper = dict(
    _delete_=True,  # This line is crucial
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        weight_decay=0.01),
    clip_grad=dict(max_norm=1.0))

# Learning rate config
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=20000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# Training settings
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# Evaluation configs
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Runtime settings
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

# Important: Reset the _base_ configuration
_base_ = []