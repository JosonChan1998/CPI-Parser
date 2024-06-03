import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Linear, ConvModule
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid

from .deformable_detr_head import DeformableDETRHead
from ..builder import HEADS, build_loss
from ..utils import (ConvUpsample, relative_coordinate_maps,
                     parse_dynamic_params, dynamic_forward,
                     aligned_bilinear)

@HEADS.register_module()
class DeformableDETRParserHead(DeformableDETRHead):

    def __init__(self,
                 *args,
                 num_parse_classes,
                 parse_logit_stride=4,
                 parse_feat_stride=4,
                 sem_feat_levels=[0, 1, 2, 3],
                 parse_start_level=1,
                 parse_range=16.0,
                 sem_stack_convs=2,
                 sem_channels=256,
                 parse_br_stack_convs=2,
                 parse_br_channels=256,
                 parse_head_stack_convs=2,
                 parse_head_channels=32,
                 num_param_fcs=2,
                 use_rel_coord=True,
                 loss_seg=dict(
                    type='CrossEntropyLoss', loss_weight=1.0, ignore_index=255),
                 loss_dice=dict(
                    type='DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    reduction='mean',
                    naive_dice=True,
                    eps=1.0,
                    loss_weight=1.0),
                 loss_mask=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    reduction='mean',
                    loss_weight=20.0),
                 loss_part_cls=dict(
                    type='BCELossWithLogits', loss_weight=1.0),
                 **kwargs):
        # basic settings
        self.num_parse_classes = num_parse_classes
        self.parse_logit_stride = parse_logit_stride
        self.parse_feat_stride = parse_feat_stride
        self.sem_feat_levels = sem_feat_levels
        self.parse_start_level = parse_start_level
        self.parse_range = parse_range
        # networks
        self.sem_stack_convs = sem_stack_convs
        self.sem_channels = sem_channels
        self.parse_br_stack_convs=parse_br_stack_convs
        self.parse_br_channels=parse_br_channels
        self.parse_head_stack_convs=parse_head_stack_convs
        self.parse_head_channels=parse_head_channels
        self.num_param_fcs = num_param_fcs
        # settings
        self.use_rel_coord = use_rel_coord
        # init
        super(DeformableDETRParserHead, self).__init__(*args, **kwargs)
        # loss
        self.loss_dice = build_loss(loss_dice)
        self.loss_mask = build_loss(loss_mask)
        self.loss_part_cls = build_loss(loss_part_cls)
        self.loss_seg = build_loss(loss_seg)

    def _init_layers(self):
        super(DeformableDETRParserHead, self)._init_layers()

        # semantics FPN
        self.sem_conv = nn.ModuleList()
        for i, _ in enumerate(self.sem_feat_levels):
            self.sem_conv.append(
                ConvUpsample(
                    self.in_channels,
                    self.sem_channels,
                    num_layers=i if i > 0 else 1,
                    num_upsample=i if i > 0 else 0,
                    conv_cfg=None,
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)))
        # sem head
        sem_logit = []
        for _ in range(self.sem_stack_convs):
            sem_logit.append(
                ConvModule(
                    self.sem_channels,
                    self.sem_channels,
                    3,
                    padding=1,
                    stride=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    act_cfg=dict(type='ReLU'),
                    bias=False))
        sem_logit.append(
            nn.Conv2d(
                self.sem_channels,
                self.num_parse_classes,
                kernel_size=1,
                stride=1))
        self.sem_logit_head = nn.Sequential(*sem_logit)

        # parsing feat branch
        parse_branch = []
        for _ in range(self.parse_br_stack_convs):
            parse_branch.append(
                ConvModule(
                    self.parse_br_channels,
                    self.parse_br_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    act_cfg=dict(type='ReLU', inplace=True),
                    bias=False))
        
        parse_branch.append(
            nn.Conv2d(
                self.parse_br_channels,
                self.parse_head_channels,
                kernel_size=1,
                stride=1))
        self.parse_branch = nn.Sequential(*parse_branch)

        # parsing head
        self.weight_nums = []
        self.bias_nums = []
        for i in range(self.parse_head_stack_convs):
            if i == 0:
                # for rel_coords
                if self.use_rel_coord:
                    self.weight_nums.append(
                        (self.parse_head_channels + 2) * self.parse_head_channels)
                else:
                    self.weight_nums.append(
                        (self.parse_head_channels) * self.parse_head_channels)
                self.bias_nums.append(self.parse_head_channels)
            elif i == self.parse_head_stack_convs - 1:
                self.weight_nums.append(
                    self.parse_head_channels * self.num_parse_classes)
                self.bias_nums.append(self.num_parse_classes)
            else:
                self.weight_nums.append(
                    self.parse_head_channels * self.parse_head_channels)
                self.bias_nums.append(self.parse_head_channels)
        self.total_params = 0
        self.total_params += sum(self.weight_nums)
        self.total_params += sum(self.bias_nums)

        # add controller params
        num_pred = self.transformer.decoder.num_layers
        con_branch = []
        for _ in range(self.num_param_fcs):
            con_branch.append(Linear(self.embed_dims, self.embed_dims))
            con_branch.append(nn.ReLU())
        con_branch.append(Linear(self.embed_dims, self.total_params))
        con_branch = nn.Sequential(*con_branch)
        self.con_branches = nn.ModuleList(
            [con_branch for _ in range(num_pred)])
        
        # add part classification
        part_cls = []
        for _ in range(self.num_param_fcs):
            part_cls.append(Linear(self.embed_dims, self.embed_dims))
            part_cls.append(nn.ReLU())
        part_cls.append(Linear(self.embed_dims, self.num_parse_classes-1))
        part_cls = nn.Sequential(*part_cls)
        self.part_cls_branches = nn.ModuleList(
            [part_cls for _ in range(num_pred)])
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):

        outs = self(x, img_metas)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs,
                           gt_bboxes_ignore=gt_bboxes_ignore,
                           **kwargs)
        return losses

    def forward(self, feats, img_metas):
        # sem seg
        sem_logit, sem_feats = self.sem_head_forward(feats)
        # parsing branch
        parse_feats = self.parse_branch_forward(sem_feats)
        # query forward
        all_cls_scores, all_bbox_preds, \
        all_params, all_p_cls_scores, \
        enc_outputs_class, enc_outputs_coord = \
            self.query_forward(feats[self.parse_start_level:], img_metas)

        return all_cls_scores, all_bbox_preds, all_params, \
               all_p_cls_scores, enc_outputs_class, \
               enc_outputs_coord, sem_logit, parse_feats

    def sem_head_forward(self, x):
        feats = []
        for i, layer in enumerate(self.sem_conv):
            f = layer(x[self.sem_feat_levels[i]])
            feats.append(f)
        feats = torch.sum(torch.stack(feats, dim=0), dim=0)
        sem_logit = self.sem_logit_head(feats)

        return sem_logit, feats

    def parse_branch_forward(self, sem_feats):
        return self.parse_branch(sem_feats)

    def query_forward(self, mlvl_feats, img_metas):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_params = []
        outputs_p_classes = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_param = self.con_branches[lvl](hs[lvl])
            outputs_p_class = self.part_cls_branches[lvl](hs[lvl])

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_params.append(outputs_param)
            outputs_p_classes.append(outputs_p_class)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_params = torch.stack(outputs_params)
        outputs_p_classes = torch.stack(outputs_p_classes)

        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                outputs_params, outputs_p_classes, \
                enc_outputs_class, \
                enc_outputs_coord.sigmoid()
        else:
            return outputs_classes, outputs_coords, \
                outputs_params, outputs_p_classes, \
                None, None
    
    def parsing_head_forward(self, pos_params_list, parse_feats, centers=None, final_channel=20):

        if len(parse_feats.shape) == 3:
            parse_feats = parse_feats.unsqueeze(0)
        BS, D, H, W = parse_feats.size()
        parse_feats_list = []
        for i in range(BS):
            num_pos = pos_params_list[i].shape[0]
            parse_feats_per_img = parse_feats[i].repeat(num_pos, 1, 1).reshape(1, -1, H, W)
            parse_feats_list.append(parse_feats_per_img)
        parse_feats = torch.cat(parse_feats_list, 1)

        if self.use_rel_coord:
            rel_coords = relative_coordinate_maps(
                parse_feats.shape, centers, self.parse_feat_stride,
                stride=self.parse_feat_stride,
                range=self.parse_range)
            parse_feats = torch.cat([
                rel_coords.view(-1, 2, H, W),
                parse_feats.reshape(-1, D, H, W)], dim=1)
            parse_feats = parse_feats.view(1, -1, H, W)

        pos_params = torch.cat(pos_params_list, 0)
        pos_params = pos_params.reshape(-1, self.total_params)
        total_pos_num = pos_params.shape[0]
        weights, biases = parse_dynamic_params(
            pos_params,
            self.parse_head_channels,
            final_channel,
            self.weight_nums,
            self.bias_nums)
        parsing_logits = dynamic_forward(
            parse_feats,
            weights,
            biases,
            total_pos_num)
        parsing_logits = parsing_logits.reshape(total_pos_num, -1, H, W)
        return parsing_logits
    
    @force_fp32(apply_to=('all_cls_scores',
                          'all_bbox_preds',
                          'all_params',
                          'all_p_cls_scores',
                          'enc_outputs_class',
                          'enc_outputs_coord',
                          'seg_logits',
                          'parse_feats'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             all_params,
             all_p_cls_scores,
             enc_outputs_class,
             enc_outputs_coord,
             seg_logits,
             parse_feats,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             **kwargs):

        gt_semantic_seg = kwargs['gt_semantic_seg']
        gt_parsings = kwargs['gt_parsings']
        gt_parse_labels = kwargs['gt_parse_labels']

        loss_dict = dict()
        # semantic parsing loss
        start = int(self.parse_logit_stride // 2)
        gt_semantic_seg = gt_semantic_seg[:, :, start::self.parse_logit_stride,\
            start::self.parse_logit_stride]
        loss_seg = self.loss_seg(seg_logits, gt_semantic_seg.squeeze(1).long())
        loss_dict['loss_seg'] = loss_seg

        # DETR loss supervision at all dec layers
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels for _ in range(num_dec_layers)]
        all_gt_parsings_list = [gt_parsings for _ in range(num_dec_layers)]
        all_img_metas_list = [img_metas for _ in range(num_dec_layers)]
        all_gt_parse_labels_list = [gt_parse_labels for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou,\
        losses_part_cls, losses_mask, losses_dice =\
            multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_params, all_p_cls_scores, all_gt_bboxes_list,
            all_gt_labels_list, all_gt_parsings_list,
            all_gt_parse_labels_list, all_img_metas_list,
            parse_feats=parse_feats)

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_part_cls'] = losses_part_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i,\
            loss_iou_i, losses_mask_i,\
            losses_part_cls_i,loss_dice_i\
                             in zip(losses_cls[:-1],
                                    losses_bbox[:-1],
                                    losses_iou[:-1],
                                    losses_mask[:-1],
                                    losses_part_cls[:-1],
                                    losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = losses_mask_i
            loss_dict[f'd{num_dec_layer}.loss_part_cls'] = losses_part_cls_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1 
        
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    params,
                    p_cls_scores,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_parsings_list,
                    gt_parsings_labels_list,
                    img_metas,
                    parse_feats,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        params_list = [params[i] for i in range(num_imgs)]
        p_cls_scores_list = [p_cls_scores[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           params_list, p_cls_scores_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_parsings_list, gt_parsings_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg,
         pos_params_list, parsing_targets_list,
         pos_p_scores_list, parsing_labels_targets_list) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        
        # parsing labels loss
        pos_parsing_scores = torch.cat(pos_p_scores_list, 0)
        parsing_labels_targets = torch.cat(parsing_labels_targets_list, 0)
        valid_mask = parsing_labels_targets.sum(dim=-1).bool()
        num_pos = valid_mask.sum()
        if num_pos == 0:
            loss_part_cls = pos_parsing_scores.sum() * 0.
        else:
            loss_part_cls = self.loss_part_cls(
                pos_parsing_scores[valid_mask],
                parsing_labels_targets[valid_mask])

        # parsing forward
        with torch.no_grad():
            pos_inds = bbox_weights.max(1)[0].bool()
            centers = ((bboxes[:, 0] + bboxes[:, 2])/2, (bboxes[:, 1] + bboxes[:, 3])/2)
            centers = torch.stack(centers, 1)
            centers = centers[pos_inds]
        parsing_pred = self.parsing_head_forward(pos_params_list, parse_feats, centers)

        loss_mask, loss_dice = self.loss_parsing_forward(
            parsing_pred, parsing_targets_list, img_metas)

        return loss_cls, loss_bbox, loss_iou, loss_part_cls, loss_mask, loss_dice
    
    def loss_parsing_forward(self,
                             parsing_pred,
                             parsing_targets_list,
                             img_metas):
        # parsing targets
        start = int(self.parse_logit_stride // 2)
        stride = self.parse_logit_stride
        img_h, img_w = img_metas[0]['batch_input_shape']
        parsing_targets = []
        for i, parsing in enumerate(parsing_targets_list):
            h, w = parsing.size()[1:]
            parsing = F.pad(parsing, (0, img_w - w, 0, img_h - h), "constant", 0)
            parsing = parsing[:, start::stride, start::stride]
            parsing_targets.append(parsing.to(parsing_pred.device))
        parsing_targets = torch.cat(parsing_targets, dim=0).long()

        valid_inds = (parsing_targets.reshape(
                parsing_targets.shape[0], -1).max(dim=-1)[0] == 0)
        assert valid_inds.max() == 0

        parsing_bit_targets = []
        for parsing in parsing_targets:
            parsing_mask = []
            for class_id in range(self.num_parse_classes):
                parsing_mask.append(parsing == class_id)
            parsing_mask = torch.stack(parsing_mask, dim=0)
            parsing_bit_targets.append(parsing_mask)
        parsing_bit_targets = torch.stack(parsing_bit_targets, dim=0)
        B, C, H, W = parsing_pred.shape
        parsing_pred = parsing_pred.reshape(B*C, H*W)
        parsing_bit_targets = parsing_bit_targets.reshape(B*C, H*W)
        parsing_masks = parsing_bit_targets.sum(-1) > 0
        parsing_bit_targets = parsing_bit_targets[parsing_masks]

        loss_dice = self.loss_dice(
            parsing_pred[parsing_masks],
            parsing_bit_targets,
            avg_factor=parsing_bit_targets.shape[0])
        loss_mask = self.loss_mask(
            parsing_pred[parsing_masks].reshape(-1, 1),
            1 - parsing_bit_targets.reshape(-1).long(),
            avg_factor=parsing_bit_targets.shape[0]*H*W)

        return loss_mask, loss_dice

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    params_list,
                    p_cls_scores_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_parsings_list,
                    gt_parsings_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list,
         pos_params_list, parsing_targets_list,
         pos_p_scores_list, parsing_labels_targets_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             params_list, p_cls_scores_list, gt_bboxes_list, gt_labels_list,
             gt_parsings_list, gt_parsings_labels_list,
             img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg,
                pos_params_list, parsing_targets_list, pos_p_scores_list,
                parsing_labels_targets_list)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           params,
                           p_cls_score,
                           gt_bboxes,
                           gt_labels,
                           gt_parsings,
                           gt_parsings_labels,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # parsing target
        pos_params = params[pos_inds]
        parsing_targets = gt_parsings[pos_assigned_gt_inds]
        pos_p_scores = p_cls_score[pos_inds]
        parsing_labels_targets = gt_parsings_labels[pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, pos_params, parsing_targets, pos_p_scores,
                parsing_labels_targets)

    @force_fp32(apply_to=('all_cls_scores',
                          'all_bbox_preds',
                          'all_params',
                          'all_p_cls_scores',
                          'parse_feats'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   all_params,
                   all_p_cls_scores,
                   enc_cls_scores,
                   enc_bbox_preds,
                   seg_logits,
                   parse_feats,
                   img_metas,
                   rescale=False):
        # create dirs
        save_root = self.test_cfg.get('save_root', None)
        assert save_root is not None
        save_parse_path = os.path.join(save_root, 'val_parsing')
        save_seg_path = os.path.join(save_root, 'val_seg')
        if os.path.exists(save_parse_path) == False:
            os.makedirs(save_parse_path)
        if os.path.exists(save_seg_path) == False:
            os.makedirs(save_seg_path)

        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        params = all_params[-1]
        p_cls_scores = all_p_cls_scores[-1]

        results_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            param = params[img_id]
            p_cls_score = p_cls_scores[img_id]
            parse_feat = parse_feats[img_id]
            img_mate = img_metas[img_id]
            proposals = self._get_bboxes_single(
                cls_score, bbox_pred, param,
                p_cls_score, parse_feat, img_mate, rescale,
                save_parse_path, save_seg_path)
            results_list.append(proposals)

        return results_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           param,
                           p_cls_score,
                           parse_feat,
                           img_meta,
                           rescale=False,
                           save_parse_path=None,
                           save_seg_path=None):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        min_score = self.test_cfg.get('score_thr', 0.1)

        img_shape = img_meta['img_shape']
        scale_factor = img_meta['scale_factor']
        ori_shape = img_meta['ori_shape']

        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
        
        # add scores filter
        keep = scores > min_score
        scores = scores[keep]
        det_labels = det_labels[keep]
        bbox_pred = bbox_pred[bbox_index][keep]
        det_params = param[bbox_index][keep]
        det_part_scores = p_cls_score[bbox_index][keep].sigmoid()

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        det_center = bbox_xyxy_to_cxcywh(det_bboxes)[:, :2]
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        if det_bboxes.shape[0] > 0:
            # parsing forward
            parsing_logits = self.parsing_head_forward([det_params], parse_feat, det_center)
            if self.loss_mask.use_sigmoid:
                parsing_logits = parsing_logits.sigmoid()
            else:
                parsing_logits = F.softmax(parsing_logits, dim=1)
            
            bg_scores = det_part_scores.new_ones((det_part_scores.shape[0], 1))
            det_part_scores = torch.cat((bg_scores, det_part_scores), dim=-1)
            parsing_logits = torch.einsum("qc,qchw->qchw", det_part_scores, parsing_logits)

            if rescale:
                parsing_logits = aligned_bilinear(parsing_logits, self.parse_logit_stride)
                parsing_logits = parsing_logits[:, :, :img_shape[0], :img_shape[1]]
                parsing_logits = F.interpolate(
                    parsing_logits,
                    size=(ori_shape[0], ori_shape[1]),
                    mode='bilinear',
                    align_corners=True)
            parsing_maps = parsing_logits.argmax(dim=1)
            parsing_maps = parsing_maps.cpu().numpy()

            # save results
            filename = img_meta['ori_filename']
            for i in range(det_bboxes.shape[0]):
                parsing_name = filename.split('.jpg')[0] + '-' + str(i) + '.png'
                cv2.imwrite(os.path.join(save_parse_path, parsing_name), parsing_maps[i])

            seg = np.zeros(parsing_maps[0].shape, dtype=np.uint8)
            for i in range(det_bboxes.shape[0]-1, -1, -1):
                if float(det_bboxes[i][4]) > 0.2:
                    seg = cv2.bitwise_or(seg, parsing_maps[i].astype(np.uint8))
            seg_name = filename.replace('jpg', 'png')
            cv2.imwrite(os.path.join(save_seg_path, seg_name), seg)

        return det_bboxes, det_labels



@HEADS.register_module()
class DeformableDETRParserSoftmaxHead(DeformableDETRParserHead):

    def __init__(self,
                 *arg,
                 loss_parsing=dict(
                    type='CrossEntropyLoss', loss_weight=2.0),
                 **kwargs):
        super(DeformableDETRParserSoftmaxHead, self).__init__(*arg, **kwargs)
        self.loss_parsing = build_loss(loss_parsing)
    
    def _init_layers(self):
        super()._init_layers()
        self.part_cls_branches = None

    def forward(self, feats, img_metas):
        # sem seg
        sem_logit, sem_feats = self.sem_head_forward(feats)
        # parsing branch
        parse_feats = self.parse_branch_forward(sem_feats)
        # query forward
        all_cls_scores, all_bbox_preds, \
        all_params, enc_outputs_class, enc_outputs_coord = \
            self.query_forward(feats[self.parse_start_level:], img_metas)

        return all_cls_scores, all_bbox_preds, all_params, \
               enc_outputs_class, enc_outputs_coord, \
               sem_logit, parse_feats

    def query_forward(self, mlvl_feats, img_metas):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_params = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_param = self.con_branches[lvl](hs[lvl])

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_params.append(outputs_param)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_params = torch.stack(outputs_params)

        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                outputs_params, enc_outputs_class, \
                enc_outputs_coord.sigmoid()
        else:
            return outputs_classes, outputs_coords, \
                outputs_params, None, None

    @force_fp32(apply_to=('all_cls_scores',
                          'all_bbox_preds',
                          'all_params',
                          'enc_outputs_class',
                          'enc_outputs_coord',
                          'seg_logits',
                          'parse_feats'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             all_params,
             enc_outputs_class,
             enc_outputs_coord,
             seg_logits,
             parse_feats,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             **kwargs):

        gt_semantic_seg = kwargs['gt_semantic_seg']
        gt_parsings = kwargs['gt_parsings']

        loss_dict = dict()
        # semantic parsing loss
        start = int(self.parse_logit_stride // 2)
        gt_semantic_seg = gt_semantic_seg[:, :, start::self.parse_logit_stride,\
            start::self.parse_logit_stride]
        bs = gt_semantic_seg.shape[0]
        loss_seg = self.loss_seg(seg_logits[:bs], gt_semantic_seg.squeeze(1).long())
        loss_dict['loss_seg'] = loss_seg

        # DETR loss supervision at all dec layers
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels for _ in range(num_dec_layers)]
        all_gt_parsings_list = [gt_parsings for _ in range(num_dec_layers)]
        all_img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, losses_parsing, =\
            multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_params, all_gt_bboxes_list,
            all_gt_labels_list, all_gt_parsings_list,
            all_img_metas_list, parse_feats=parse_feats)

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_parsing'] = losses_parsing[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i,\
            loss_iou_i, loss_parsing_i \
                           in zip(losses_cls[:-1],
                                  losses_bbox[:-1],
                                  losses_iou[:-1],
                                  losses_parsing[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_parsing'] = loss_parsing_i
            num_dec_layer += 1 
        
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    params,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_parsings_list,
                    img_metas,
                    parse_feats,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        params_list = [params[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           params_list, gt_bboxes_list,
                                           gt_labels_list, gt_parsings_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg,
         pos_params_list, parsing_targets_list) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        # parsing forward
        with torch.no_grad():
            pos_inds = bbox_weights.max(1)[0].bool()
            centers = ((bboxes[:, 0] + bboxes[:, 2])/2, (bboxes[:, 1] + bboxes[:, 3])/2)
            centers = torch.stack(centers, 1)
            centers = centers[pos_inds]
        parsing_pred = self.parsing_head_forward(pos_params_list, parse_feats, centers)

        loss_parsing = self.loss_parsing_forward(
            parsing_pred, parsing_targets_list, img_metas)

        return loss_cls, loss_bbox, loss_iou, loss_parsing

    def loss_parsing_forward(self,
                             parsing_pred,
                             parsing_targets_list,
                             img_metas):
        # parsing targets
        start = int(self.parse_logit_stride // 2)
        stride = self.parse_logit_stride
        img_h, img_w = img_metas[0]['batch_input_shape']
        parsing_targets = []
        for i, parsing in enumerate(parsing_targets_list):
            h, w = parsing.size()[1:]
            parsing = F.pad(parsing, (0, img_w - w, 0, img_h - h), "constant", 0)
            parsing = parsing[:, start::stride, start::stride]
            parsing_targets.append(parsing.to(parsing_pred.device))
        parsing_targets = torch.cat(parsing_targets, dim=0).long()

        valid_inds = (parsing_targets.reshape(
            parsing_targets.shape[0], -1).max(dim=-1)[0] > 0)
        loss_parsing = self.loss_parsing(
            parsing_pred[valid_inds], parsing_targets[valid_inds])
        return loss_parsing

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    params_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_parsings_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list,
         pos_params_list, parsing_targets_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             params_list, gt_bboxes_list, gt_labels_list,
             gt_parsings_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg,
                pos_params_list, parsing_targets_list)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           params,
                           gt_bboxes,
                           gt_labels,
                           gt_parsings,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # parsing target
        pos_params = params[pos_inds]
        parsing_targets = gt_parsings[pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, pos_params, parsing_targets)

    @force_fp32(apply_to=('all_cls_scores',
                          'all_bbox_preds',
                          'all_params',
                          'all_p_cls_scores',
                          'parse_feats'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   all_params,
                   enc_cls_scores,
                   enc_bbox_preds,
                   seg_logits,
                   parse_feats,
                   img_metas,
                   rescale=False):
        # create dirs
        save_root = self.test_cfg.get('save_root', None)
        assert save_root is not None
        save_parse_path = os.path.join(save_root, 'val_parsing')
        save_seg_path = os.path.join(save_root, 'val_seg')
        if os.path.exists(save_parse_path) == False:
            os.makedirs(save_parse_path)
        if os.path.exists(save_seg_path) == False:
            os.makedirs(save_seg_path)

        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        params = all_params[-1]

        results_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            param = params[img_id]
            parse_feat = parse_feats[img_id]
            img_mate = img_metas[img_id]
            proposals = self._get_bboxes_single(
                cls_score, bbox_pred, param,
                parse_feat, img_mate, rescale,
                save_parse_path, save_seg_path)
            results_list.append(proposals)

        return results_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           param,
                           parse_feat,
                           img_meta,
                           rescale=False,
                           save_parse_path=None,
                           save_seg_path=None):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        min_score = self.test_cfg.get('score_thr', 0.1)

        img_shape = img_meta['img_shape']
        scale_factor = img_meta['scale_factor']
        ori_shape = img_meta['ori_shape']

        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
        
        # add scores filter
        keep = scores > min_score
        scores = scores[keep]
        det_labels = det_labels[keep]
        bbox_pred = bbox_pred[bbox_index][keep]
        det_params = param[bbox_index][keep]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        det_center = bbox_xyxy_to_cxcywh(det_bboxes)[:, :2]
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        if det_bboxes.shape[0] > 0:
            # parsing forward
            parsing_logits = self.parsing_head_forward([det_params], parse_feat, det_center)
            if self.loss_parsing.use_sigmoid:
                parsing_logits = parsing_logits.sigmoid()
            else:
                parsing_logits = F.softmax(parsing_logits, dim=1)

            if rescale:
                parsing_logits = aligned_bilinear(parsing_logits, self.parse_logit_stride)
                parsing_logits = parsing_logits[:, :, :img_shape[0], :img_shape[1]]
                if ori_shape[0] > 2500 or ori_shape[1] > 2500:
                    parsing_logits = parsing_logits.cpu()
                parsing_logits = F.interpolate(
                    parsing_logits,
                    size=(ori_shape[0], ori_shape[1]),
                    mode='bilinear',
                    align_corners=True)
            parsing_maps = parsing_logits.argmax(dim=1)
            parsing_maps = parsing_maps.cpu().numpy()

            # save results
            filename = img_meta['ori_filename']
            for i in range(det_bboxes.shape[0]):
                parsing_name = filename.split('.jpg')[0] + '-' + str(i) + '.png'
                cv2.imwrite(os.path.join(save_parse_path, parsing_name), parsing_maps[i])

            seg = np.zeros(parsing_maps[0].shape, dtype=np.uint8)
            for i in range(det_bboxes.shape[0]-1, -1, -1):
                if float(det_bboxes[i][4]) > 0.2:
                    parsing_map = parsing_maps[i].astype(np.uint8)
                    seg[parsing_map > 0] = parsing_map[parsing_map > 0]
            seg_name = filename.replace('jpg', 'png')
            cv2.imwrite(os.path.join(save_seg_path, seg_name), seg)

        return det_bboxes, det_labels
