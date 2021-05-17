import numpy as np
import torch


class CenterNetGT(object):
    @staticmethod
    def generate(config, batched_input):
        box_scale = 1 / config.MODEL.CENTERNET.DOWN_SCALE
        num_classes = config.MODEL.CENTERNET.NUM_CLASSES
        output_size = config.MODEL.CENTERNET.OUTPUT_SIZE
        min_overlap = config.MODEL.CENTERNET.MIN_OVERLAP
        tensor_dim = config.MODEL.CENTERNET.TENSOR_DIM

        alpha = .54
        beta = .54
        wh_area_process = 'log'

        heatmap_list, box_target_list, reg_weight_list = [], [], []
        for data in batched_input:
            # img_size = (data['height'], data['width'])

            bbox_dict = data["instances"].get_fields()
            boxes, classes = bbox_dict["gt_boxes"], bbox_dict["gt_classes"]
            num_boxes = boxes.tensor.shape[0]

            # init gt tensors
            heatmap = torch.zeros(num_classes, *output_size)
            reg_weight = torch.zeros(1, *output_size)
            box_target = boxes.tensor.new_ones((4, *output_size)) * -1
            fake_heatmap = boxes.tensor.new_zeros(*output_size)

            if wh_area_process == 'log':
                boxes_areas_log = boxes.area().log()  # [num_gt,]
            elif wh_area_process == 'sqrt':
                boxes_areas_log = boxes.area().sqrt()
            else:
                boxes_areas_log = boxes.area()

            boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

            if wh_area_process == 'norm':
                boxes_area_topk_log[:] = 1.

            # sort by area
            boxes = boxes[boxes_ind]
            classes = classes[boxes_ind]
            # resize to feature map scale
            boxes.scale(box_scale, box_scale)

            # TTFNET
            feat_hs, feat_ws = boxes.tensor[:, 3] - boxes.tensor[:, 1], boxes.tensor[:, 2] - boxes.tensor[:, 0]

            h_radiuses_alpha = (feat_hs / 2. * alpha).int()
            w_radiuses_alpha = (feat_ws / 2. * alpha).int()
            if alpha != beta:
                h_radiuses_beta = (feat_hs / 2. * beta).int()
                w_radiuses_beta = (feat_ws / 2. * beta).int()

            for k in range(classes.shape[0]):
                cls_id = classes[k].int().item()

            centers = boxes.get_centers()
            centers_int = centers.to(torch.int32)
            for k in range(boxes_ind.shape[0]):
                cls_id = classes[k].int().item()
                fake_heatmap = fake_heatmap.zero_()
                CenterNetGT.draw_truncate_gaussian(fake_heatmap, centers_int[k],
                                                   h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
                heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

                if alpha != beta:
                    fake_heatmap = fake_heatmap.zero_()
                    CenterNetGT.draw_truncate_gaussian(fake_heatmap, centers_int[k],
                                                       h_radiuses_beta[k].item(),
                                                       w_radiuses_beta[k].item())

                box_target_inds = fake_heatmap > 0
                box_target[:, box_target_inds] = boxes.tensor[k][:, None]
                cls_id = 0
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div

            heatmap_list.append(heatmap)
            box_target_list.append(box_target)
            reg_weight_list.append(reg_weight)

        gt_dict = {
            "gt_heatmap": torch.stack(heatmap_list, dim=0).detach(),
            "gt_box": torch.stack(box_target_list, dim=0).detach(),
            "gt_reg_weight": torch.stack(reg_weight_list, dim=0).detach(),
        }
        return gt_dict

    @staticmethod
    def gaussian_2d(shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = CenterNetGT.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap
