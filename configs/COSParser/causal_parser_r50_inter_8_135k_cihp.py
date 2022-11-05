_base_ = [
    '../_base_/datasets/cihp.py',
    '../_base_/models/deformable_detr_parser_r50.py',
    '../_base_/schedules/schedule_cihp_3x.py',
    '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
flip_map = ([14, 15], [16, 17], [18, 19])
train_pipeline = [
    dict(type='LoadFourierImageFromFile',
         aug_type=['sketch', 'anime']),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_parse=True,
         with_seg=True),
    dict(type='Resize',
         img_scale=[(1400, 512), (1400, 640), (1400, 704),
                    (1400, 768), (1400, 800), (1400, 864)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, flip_map=flip_map),
    dict(type='LoadParseAnnotations', num_parse_classes=20),
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
    workers_per_gpu=1,
    train=dict(pipeline=train_pipeline))

model=dict(
    type='CausalParser',
    backbone=dict(
        out_indices=(1, 2, 3)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='CausalParserHead',
        use_aug=True,
        parse_logit_stride=8,
        parse_feat_stride=8,
        sem_feat_levels=[0, 1, 2],
        parse_start_level=0
    ),
    test_cfg=dict(
        score_thr=0.01,
        save_root='work_dirs/causal_parser_r50_inter_8_135k_cihp'
    ))

evaluation = dict(interval=100, metric='bbox')