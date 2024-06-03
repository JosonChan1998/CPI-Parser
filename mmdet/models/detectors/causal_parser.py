# Copyright (c) OpenMMLab. All rights reserved.
import torch
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .deformable_detr_parser import DeformableDETRParser


@DETECTORS.register_module()
class CausalParser(DeformableDETRParser):

    def __init__(self, *arg, **kwargs):
        super(CausalParser, self).__init__(*arg, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if 'aug_img' in kwargs.keys():
            img = torch.cat([img, kwargs['aug_img']], dim=0)
            bs = img.shape[0]
            img_metas = [img_metas[i % (bs//2)] for i in range(bs)]

        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore,
                                              None, **kwargs)
        return losses
