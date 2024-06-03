import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        multi_apply, reduce_mean)

from .deformable_detr_parser_head import DeformableDETRParserHead
from ..builder import HEADS
from ..utils import (relative_coordinate_maps,
                     parse_dynamic_params, dynamic_forward,
                     aligned_bilinear)


@HEADS.register_module()
class CausalParserHead(DeformableDETRParserHead):

    def __init__(self,
                 *arg,
                 use_aug=False,
                 part_head_stack_convs=1,
                 **kwargs):
        self.use_aug = use_aug
        self.part_head_stack_convs = part_head_stack_convs
        super(CausalParserHead, self).__init__(*arg, **kwargs)

    def _init_layers(self):
        super(CausalParserHead, self)._init_layers()
        self.dec_num_layer = self.transformer.decoder.num_layers
        # part weight & bias
        self.part_weight = nn.Sequential(
            nn.Linear(self.embed_dims, self.parse_head_channels),
            nn.ReLU(),
            nn.Linear(self.parse_head_channels, self.parse_head_channels))
        self.part_bias = nn.Parameter(torch.Tensor(self.num_parse_classes-1))
    
        # cnt / cxt embedding
        self.cxt_attn = nn.Sequential(
            Linear(self.parse_head_channels, self.parse_head_channels),
            nn.ReLU(),
            Linear(self.parse_head_channels, self.num_parse_classes-1))

        # cnt / cxt kernels
        self.part_weight_nums = []
        self.part_bias_nums = []
        for _ in range(self.part_head_stack_convs):
            self.part_weight_nums.append(
                self.parse_head_channels * self.parse_head_channels)
            self.part_bias_nums.append(self.parse_head_channels)
        self.part_total_params = 0
        self.part_total_params += sum(self.part_weight_nums)
        self.part_total_params += sum(self.part_bias_nums)

        # cnt / cxt avg
        self.avg_pooler = nn.AdaptiveAvgPool2d((1, 1))

        self.cnt_controller = nn.Sequential(
            Linear(self.parse_head_channels, self.parse_head_channels),
            nn.ReLU(),
            Linear(self.parse_head_channels, self.part_total_params))

        self.cxt_controller = nn.Sequential(
            Linear(self.parse_head_channels, self.parse_head_channels),
            nn.ReLU(),
            Linear(self.parse_head_channels, self.part_total_params))

        # parser logit
        parser_channels = self.parse_head_channels * (self.num_parse_classes - 1)
        self.cnt_part_parser = nn.Sequential(
            nn.Conv2d(parser_channels,
                parser_channels, 3, 1, 1, groups=(self.num_parse_classes - 1)),
            nn.ReLU(),
            nn.Conv2d(parser_channels,
                self.num_parse_classes-1, 3, 1, 1, groups=(self.num_parse_classes - 1)))

        self.cxt_part_parser = nn.Sequential(
            nn.Conv2d(parser_channels,
                parser_channels, 3, 1, 1, groups=(self.num_parse_classes - 1)),
            nn.ReLU(),
            nn.Conv2d(parser_channels,
                self.num_parse_classes-1, 3, 1, 1, groups=(self.num_parse_classes - 1)))

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
        mask_parts = kwargs.get('mask_parts', None)
        if mask_parts is not None and mask_parts[0] == 1000:
            mask_parts = None

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
        all_gt_parse_labels_list = [gt_parse_labels for _ in range(num_dec_layers)]
        all_img_metas_list = [img_metas for _ in range(num_dec_layers)]
        all_num_layer_list = [i for i in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou,\
        losses_part_cls, losses_mask, losses_dice, \
        losses_cxt_mask, losses_cxt_dice, losses_factor =\
            multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_params, all_p_cls_scores, all_gt_bboxes_list,
            all_gt_labels_list, all_gt_parsings_list,
            all_gt_parse_labels_list, all_img_metas_list,
            all_num_layer_list, parse_feats=parse_feats,
            mask_parts=mask_parts)

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_part_cls'] = losses_part_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_cxt_mask'] = losses_cxt_mask[-1]
        loss_dict['loss_cxt_dice'] = losses_cxt_dice[-1]
        if isinstance(losses_factor[-1], torch.Tensor):
            loss_dict['loss_factor'] = losses_factor[-1]

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
            loss_dict[f'd{num_dec_layer}.loss_part_cls'] = losses_part_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = losses_mask_i
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
                    num_layer,
                    parse_feats,
                    mask_parts,
                    gt_bboxes_ignore_list=None):

        num_imgs = len(gt_bboxes_list)
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
         pos_p_scores_list, parsing_labels_targets_list,
         pos_inds_list) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores[:num_imgs].reshape(-1, self.cls_out_channels)
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
        bbox_preds = bbox_preds[:num_imgs]
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
        if num_layer == (self.dec_num_layer - 1):
            if self.use_aug:
                aug_parasm = params[num_imgs:]
                for i, pos_inds in enumerate(pos_inds_list):
                    pos_params_list.append(aug_parasm[i][pos_inds])

                parsing_labels_targets = parsing_labels_targets[None, ...].repeat(2, 1, 1).reshape(-1, self.num_parse_classes-1)
                if mask_parts is not None:
                    parsing_labels_targets[num_pos:, mask_parts] = 0

                centers = centers[None, ...].repeat(2, 1, 1).reshape(-1, 2)

            cnt_pred, cxt_pred, cnt_feats, cxt_feats = self.casual_head_forward(
                pos_params_list, parse_feats, parsing_labels_targets, centers)
            loss_mask, loss_dice, loss_cxt_mask, loss_cxt_dice, loss_factor = \
                self.loss_causal_forward(
                cnt_pred, cxt_pred, cnt_feats,
                cxt_feats, parsing_targets_list,
                mask_parts, img_metas)
        else:
            parsing_pred = self.parsing_head_forward(pos_params_list, parse_feats[:num_imgs], centers)
            loss_mask, loss_dice = self.loss_parsing_forward(
                parsing_pred, parsing_targets_list, img_metas)
            loss_cxt_mask = loss_cxt_dice = loss_factor = 0.

        return loss_cls, loss_bbox, loss_iou, loss_part_cls, \
               loss_mask, loss_dice, loss_cxt_mask, loss_cxt_dice, loss_factor

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
                parsing_labels_targets_list, pos_inds_list)

    def casual_head_forward(self, pos_params_list, parse_feats, parse_labels, centers=None):

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
            self.num_parse_classes,
            self.weight_nums,
            self.bias_nums)
        inst_weights = [weights[0]]
        inst_biases = [biases[0]]
        inst_feats = dynamic_forward(
            parse_feats,
            inst_weights,
            inst_biases,
            total_pos_num,
            extra_relu=True)
        inst_preds = F.conv2d(inst_feats,
            weights[1], biases[1], stride=1, padding=0, groups=total_pos_num)

        inst_preds = inst_preds.reshape(total_pos_num, -1, H, W)[:, :1, ...]
        inst_feats = inst_feats.reshape(total_pos_num, -1, H, W)

        # cnt embed
        part_weights = self.part_weight(self.part_cls_branches[-1][-1].weight)[:, :, None, None]
        cnt_masks = F.conv2d(
            inst_feats, part_weights, bias=self.part_bias, stride=1, padding=0).sigmoid()
        N, C, H, W = cnt_masks.shape
        cnt_embed = torch.bmm(
            cnt_masks.view(N, C, -1), inst_feats.view(N, D, -1).permute(0, 2, 1))
        normalizer = cnt_masks.view(N, C, -1).sum(-1).clamp(min=1e-6)
        cnt_embed = cnt_embed / normalizer[:, : ,None]
        # cxt embed
        parse_mask = ~(parse_labels[:, None, :].repeat(1, self.num_parse_classes-1, 1).bool())
        eye_mask = torch.eye(self.num_parse_classes-1, dtype=torch.bool, device=parse_labels.device)
        parse_mask = parse_mask + eye_mask[None, :, :]

        cxt_attn = self.cxt_attn(cnt_embed)
        valid_mask = parse_mask.sum(-1) != (self.num_parse_classes - 1)

        cxt_attn.masked_fill_(parse_mask, float('-inf'))
        cxt_attn[valid_mask] = cxt_attn[valid_mask].softmax(dim=-1)
        if ~valid_mask.sum() != 0:
            nan_mask = torch.zeros_like(parse_mask)
            nan_mask[~valid_mask] = True
            cxt_attn.masked_fill_(nan_mask, 0.)

        cxt_embed = cnt_embed[:, None, :, :].repeat(1, C, 1, 1)
        cxt_embed = torch.einsum('nkc,nkcd->nkd', cxt_attn, cxt_embed)

        # cnt/cxt feats
        cnt_kernels = self.cnt_controller(cnt_embed.reshape(-1, D))
        cxt_kernels = self.cxt_controller(cxt_embed.reshape(-1, D))
        cnt_weights, cnt_bias = parse_dynamic_params(
            cnt_kernels,
            self.parse_head_channels,
            self.parse_head_channels,
            self.part_weight_nums,
            self.part_bias_nums)
        cxt_weights, cxt_bias = parse_dynamic_params(
            cxt_kernels,
            self.parse_head_channels,
            self.parse_head_channels,
            self.part_weight_nums,
            self.part_bias_nums)

        inst_part_feats = inst_feats[:, None, ...].repeat(1, C, 1, 1, 1)
        inst_part_feats = inst_part_feats.reshape(1, -1, H, W)
        cnt_feats = dynamic_forward(
            inst_part_feats, cnt_weights, cnt_bias, total_pos_num * C, extra_relu=True)
        cxt_feats = dynamic_forward(
            inst_part_feats, cxt_weights, cxt_bias, total_pos_num * C, extra_relu=True)
        cnt_feats = cnt_feats.view(N, C, D, H, W)
        cxt_feats = cxt_feats.view(N, C, D, H, W)

        cnt_preds = self.cnt_part_parser(cnt_feats.view(N, -1, H, W))
        cxt_preds = self.cxt_part_parser(cxt_feats.view(N, -1, H, W))
        cnt_preds = torch.cat([inst_preds, cnt_preds], dim=1)
        cxt_preds = torch.cat([inst_preds, cxt_preds], dim=1)

        return cnt_preds, cxt_preds, cnt_feats, cxt_feats

    def loss_causal_forward(self,
                            cnt_pred,
                            cxt_pred,
                            cnt_feats,
                            cxt_feats,
                            parsing_targets_list,
                            mask_parts,
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
            parsing_targets.append(parsing.to(cnt_pred.device))
        parsing_targets = torch.cat(parsing_targets, dim=0).long()

        parsing_cnt_masks = []
        for parsing in parsing_targets:
            parsing_mask = []
            for class_id in range(self.num_parse_classes):
                parsing_mask.append(parsing == class_id)
            parsing_mask = torch.stack(parsing_mask, dim=0)
            parsing_cnt_masks.append(parsing_mask)
        parsing_cnt_masks = torch.stack(parsing_cnt_masks, dim=0)

        parsing_cxt_masks = []
        for parsing_target in parsing_cnt_masks:
            inst_mask = ~parsing_target[0]
            parsing_cxt_mask = []
            for class_id in range(1, self.num_parse_classes):
                parsing_cxt_mask.append(
                    (inst_mask.float() - parsing_target[class_id].float()).bool())
            parsing_cxt_mask = torch.stack(parsing_cxt_mask, dim=0)
            parsing_cxt_masks.append(parsing_cxt_mask)
        parsing_cxt_masks = torch.stack(parsing_cxt_masks, dim=0)

        B, C, H, W = cnt_pred.shape
        if self.use_aug:
            cnt_pred = cnt_pred.split(B//2 ,dim=0)[0]
            cxt_pred = cxt_pred.split(B//2 ,dim=0)[0]
        B, C, H, W = cnt_pred.shape
        cnt_pred = cnt_pred.reshape(B*C, H*W)
        cxt_pred = cxt_pred.reshape(B*C, H*W)
        parsing_targets = parsing_cnt_masks.reshape(B*C, H*W)
        parsing_masks = parsing_targets.sum(-1) > 0
        parsing_targets = parsing_targets[parsing_masks]

        loss_cnt_dice = self.loss_dice(
            cnt_pred[parsing_masks],
            parsing_targets,
            avg_factor=parsing_targets.shape[0])
        loss_cnt_mask = self.loss_mask(
            cnt_pred[parsing_masks].reshape(-1, 1),
            1 - parsing_targets.reshape(-1).long(),
            avg_factor=parsing_targets.shape[0]*H*W)
        
        loss_cxt_dice = self.loss_dice(
            cxt_pred[parsing_masks],
            parsing_targets,
            avg_factor=parsing_targets.shape[0])
        loss_cxt_mask = self.loss_mask(
            cxt_pred[parsing_masks].reshape(-1, 1),
            1 - parsing_targets.reshape(-1).long(),
            avg_factor=parsing_targets.shape[0]*H*W)

        # factor loss
        if self.use_aug:
            B, C, H, W = parsing_cnt_masks.shape
            parsing_cnt_masks = parsing_cnt_masks[:,1:C,:,:]

            valid_inds = parsing_cnt_masks.reshape(B, C-1, -1).sum(-1) > 0
            if mask_parts is not None:
                valid_inds[:, mask_parts] = False
            
            if valid_inds.sum() == 0:
                factor_loss = parsing_cnt_masks.sum() * 0.
            else:
                v_parsing_cnt_masks = parsing_cnt_masks[valid_inds]
                v_parsing_cxt_masks = parsing_cxt_masks[valid_inds]

                # repeat for aug img
                valid_inds = valid_inds[None, ...].repeat(2, 1, 1).reshape(-1, C-1)
                v_parsing_cnt_masks = v_parsing_cnt_masks[None, ...].repeat(2, 1, 1, 1).reshape(-1, H, W)
                v_parsing_cxt_masks = v_parsing_cxt_masks[None, ...].repeat(2, 1, 1, 1).reshape(-1, H, W)

                cnt_feats = cnt_feats[valid_inds] * v_parsing_cnt_masks.unsqueeze(1)
                cxt_feats = cxt_feats[valid_inds] * v_parsing_cxt_masks.unsqueeze(1)

                cnt_avg_feats = self.avg_pooler(cnt_feats).squeeze()
                cxt_avg_feats = self.avg_pooler(cxt_feats).squeeze()
                N, _ = cnt_avg_feats.shape
                cnt_avg_feats, aug_cnt_avg_feats = cnt_avg_feats.split(N//2, dim=0)
                cxt_avg_feats, aug_cxt_avg_feats = cxt_avg_feats.split(N//2, dim=0)

                ori_feats = torch.stack((cnt_avg_feats, cxt_avg_feats), dim=1)
                ori_feats = F.normalize(ori_feats, p=2, dim=-1)
                aug_feats = torch.stack((aug_cnt_avg_feats, aug_cxt_avg_feats), dim=1)
                aug_feats = F.normalize(aug_feats, p=2, dim=-1)

                cor = torch.bmm(ori_feats, aug_feats.permute(0, 2, 1))
                on_diag = torch.diagonal(cor, dim1=1, dim2=2).add_(-1).pow_(2).mean()
                off_diag = cor.flatten(1)[:, :-1].view(-1, 1, 3)[:, :, 1:].flatten().pow_(2).mean()
                factor_loss = (on_diag + off_diag) * 5
        else:
            factor_loss = 0.

        return loss_cnt_mask, loss_cnt_dice, loss_cxt_mask, loss_cxt_dice, factor_loss

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
            part_labels = det_part_scores > 0.4
            cnt_preds, cxt_preds, _, _ = self.casual_head_forward([det_params], parse_feat, part_labels, det_center)
            cnt_preds = cnt_preds.sigmoid()
            cxt_preds = cxt_preds.sigmoid()
            parsing_logits = cnt_preds * 0.5 + cxt_preds * 0.5
            
            bg_scores = det_part_scores.new_ones((det_part_scores.shape[0], 1))
            det_part_scores = torch.cat((bg_scores, det_part_scores), dim=-1)
            parsing_logits = torch.einsum("qc,qchw->qchw", det_part_scores, parsing_logits)

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
