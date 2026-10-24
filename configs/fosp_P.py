# FoSp (AAAI 2024) on Protocol P. Derived from the official config
#   third_party/FoSp/config/fosp_config.py  (github.com/LujianYao/FoSp @ 02b9fd1)
# Changes (all others identical):
#   1. data paths -> Protocol P train/val VOC view (data/views/fosp_P_trainval; SAM pseudo-labels 0/1)
#   2. logical batch 6 = 3 micro-batches x 2 images (GradientCumulativeOptimizerHook) to fit <= 5 GB VRAM;
#      iteration-based schedule scaled x3 so the number of optimizer updates (40k), warm-up (1.5k updates),
#      evaluation/checkpoint interval (1k updates) are unchanged. BatchNorm in the decode head sees 2 images.
#   3. workers_per_gpu 12 -> 2 (CPU budget), auto_resume True (crash recovery)
# Model selection: official train.py registers EvalHook(save_best='mf_beta') on the validation split.
CUM = 3
MICRO = 2
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoderFOSP',
    pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth',
    backbone=dict(
        type='MixVisionTransformer', in_channels=3, embed_dims=64, num_stages=4, num_layers=[3, 4, 18, 3],
        num_heads=[1, 2, 5, 8], patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3),
        mlp_ratio=4, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1),
    decode_head=dict(
        type='FOSP', in_channels=[64, 128, 320, 512], in_index=[0, 1, 2, 3], channels=256, dropout_ratio=0.1,
        num_classes=2, out_channels=1, threshold=0.4, norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False, loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'SmokeDataset'
data_root = 'D:/wildfire_research/data/views/fosp_P_trainval'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(512, 512), flip=False,
         transforms=[dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'),
                     dict(type='Normalize', **img_norm_cfg), dict(type='ImageToTensor', keys=['img']),
                     dict(type='Collect', keys=['img'])]),
]
data = dict(
    samples_per_gpu=MICRO,
    workers_per_gpu=2,
    train=dict(type=dataset_type, data_root=data_root, img_dir='JPEGImages', ann_dir='SegmentationClass',
               split='ImageSets/Segmentation/train.txt', pipeline=train_pipeline),
    val=dict(type=dataset_type, data_root=data_root, img_dir='JPEGImages', ann_dir='SegmentationClass',
             split='ImageSets/Segmentation/val.txt', pipeline=test_pipeline),
    test=dict(type=dataset_type, data_root=data_root, img_dir='JPEGImages', ann_dir='SegmentationClass',
              split='ImageSets/Segmentation/val.txt', pipeline=test_pipeline))  # no test split in this view
log_config = dict(interval=50 * CUM, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys=dict(pos_block=dict(decay_mult=0.0), norm=dict(decay_mult=0.0),
                                                     head=dict(lr_mult=10.0))))
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=CUM)
lr_config = dict(policy='poly', warmup='linear', warmup_iters=1500 * CUM, warmup_ratio=1e-06, power=1.0,
                 min_lr=0.0, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000 * CUM)
checkpoint_config = dict(by_epoch=False, interval=1000 * CUM, max_keep_ckpts=2)
# rule='greater' added: the official train.py registers EvalHook(save_best='mf_beta') but mmseg's EvalHook only
# knows greater_keys ['mIoU','mAcc','aAcc'], so the released code raises "Cannot infer the rule for key mf_beta"
# (verified 2026-10-18 with mmcv 1.6.2). F-beta is higher-is-better, so this preserves the authors' intent.
evaluation = dict(interval=1000 * CUM, metric='SmokeSeg', pre_eval=True, rule='greater')
work_dir = 'D:/wildfire_research/experiments/FoSp_P_seed2077'  # official default seed 2077
# max_keep_ckpts=2 only limits disk usage (the best checkpoint is saved separately by EvalHook)
gpu_ids = [0]
auto_resume = True
