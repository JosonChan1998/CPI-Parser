_base_ = [
    '../_base_/datasets/mhp_v2.py',
    '../_base_/models/deformable_detr_parser_r50.py',
    '../_base_/schedules/schedule_cihp_3x.py',
    '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
flip_map = ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33])
train_pipeline = [
    dict(type='LoadFourierImageFromFile',
         aug_type=['sunny', 'snow', 'oil', 'sketch', 'anime']),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_parse=True,
         with_seg=True),
    dict(type='Resize',
         img_scale=[(1400, 512), (1400, 640), (1400, 704),
                    (1400, 768), (1400, 800)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, flip_map=flip_map),
    dict(type='LoadParseAnnotations', num_parse_classes=59),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img',
                               'aug_img',
                               'gt_bboxes',
                               'gt_labels',
                               'gt_parsings',
                               'gt_parse_labels',
                               'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline))

custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in21k_20220124-13b83eec.pth'  # noqa
model=dict(
    type='CausalParser',
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='base',
        out_indices=[1, 2, 3],
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='CausalParserHead',
        use_aug=True,
        num_parse_classes=59,
        parse_logit_stride=8,
        parse_feat_stride=8,
        sem_feat_levels=[0, 1, 2],
        parse_start_level=0,
        parse_head_channels=12
    ),
    test_cfg=dict(
        score_thr=0.2,
        save_root='work_dirs/causal_parser_convnext_b_inter_8_135k_mhp'
    ))

evaluation = dict(interval=100, metric='bbox')

# optimizer
optimizer = dict(
    _delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
        },
        decay_rate=0.8,
        decay_type='layer_wise',
        num_layers=12))
