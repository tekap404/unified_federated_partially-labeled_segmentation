# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from monai.metrics.utils import do_metric_reduction
from monai.metrics.utils import get_mask_edges, get_surface_distance
from monai.metrics import CumulativeIterationMetric
from medpy import metric

class HausdorffDistanceMetric(CumulativeIterationMetric):
    """
    Modify MONAI's HausdorffDistanceMetric for Kaggle UW-Madison GI Tract Image Segmentation

    """

    def __init__(
        self,
        reduction = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):
        return compute_hausdorff_distance(pred, gt)

    def aggregate(self):
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        """
        data = self.get_buffer()
        # do metric reduction
        f, _ = do_metric_reduction(data, self.reduction)
        return f

def compute_directed_hausdorff(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    if np.sum(pred) == 0:
        return 1.0
    if np.sum(gt) == 0:
        return 1.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()

    if dist > max_dist:
        return 1.0
    return dist / max_dist

def compute_hausdorff_score(pred, gt):
    y = gt.float().to("cpu").numpy()
    y_pred = pred.float().to("cpu").numpy()

    # hausdorff distance score
    batch_size, n_class = y_pred.shape[:2]
    spatial_size = y_pred.shape[2:]
    max_dist = np.sqrt(np.sum([l**2 for l in spatial_size]))
    hd_score = np.empty((batch_size, n_class))
    for b, c in np.ndindex(batch_size, n_class):
        hd_score[b, c] = 1 - compute_directed_hausdorff(y_pred[b, c], y[b, c], max_dist)

    return torch.from_numpy(hd_score)

def compute_hausdorff_distance(pred, gt):   # (bs*d,c,h,w)
    x = pred.float()
    y = gt.float()
    distance_matrix = torch.cdist(x,y,p=2) # p=2 means Euclidean Distance
    
    value1 = distance_matrix.min(3)[0].max(2, keepdim=True)[0]  # (bs*d,c,1)
    value2 = distance_matrix.min(2)[0].max(2, keepdim=True)[0]
    
    value = torch.cat((value1, value2), dim=2)
    
    hd = value.max(2)[0]    # (bs*d,c)
    return hd

class JaccardCoefficientMetric(CumulativeIterationMetric):
    def __init__(
        self,
        reduction = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):
        return compute_jaccard_coefficient(pred, gt)

    def aggregate(self):
        data = self.get_buffer()
        f, _ = do_metric_reduction(data, self.reduction)
        return f

def compute_jaccard_coefficient(pred, gt): 
    intersecion = torch.multiply(pred, gt)
    jc = intersecion.sum(dim=-1).sum(dim=-1) / (pred.sum(dim=-1).sum(dim=-1) + gt.sum(dim=-1).sum(dim=-1) - intersecion.sum(dim=-1).sum(dim=-1) + 1e-8)
    
    return jc

class AverageSurfaceDistanceMetric(CumulativeIterationMetric):
    def __init__(
        self,
        reduction = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):

        return compute_asd(pred, gt)

    def aggregate(self):
        data = self.get_buffer()
        f, _ = do_metric_reduction(data, self.reduction)
        return f

def compute_asd(pred, gt): 
    asd = metric.binary.asd(pred.cpu().numpy(), gt.cpu().numpy())
    asd_result = torch.Tensor([asd,1])
    return asd_result

class HD95Metric(CumulativeIterationMetric):
    def __init__(
        self,
        reduction = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):
        return compute_hd95(pred, gt)

    def aggregate(self):
        data = self.get_buffer()
        f, _ = do_metric_reduction(data, self.reduction)
        return f

def compute_hd95(pred, gt):
    hd95 = metric.binary.hd95(pred.cpu().numpy(), gt.cpu().numpy())
    hd95_result = torch.Tensor([hd95,1])

    return hd95_result

class IOUMetric(CumulativeIterationMetric):
    def __init__(
        self,
        reduction = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):
        return binary_iou(pred, gt)

    def aggregate(self):
        data = self.get_buffer()
        f, _ = do_metric_reduction(data, self.reduction)
        return f

def binary_iou(s, g):
    N, C, H, W = s.shape
    s = s.view(N,C,H*W)
    g = g.view(N,C,H*W)

    intersecion = torch.multiply(s, g)
    union = torch.tensor(s + g > 0)
    iou = intersecion.sum(dim=-1) / (union.sum(dim=-1) + 1e-10)

    return iou


class RelativeVolumeErrorMetric(CumulativeIterationMetric):
    def __init__(
        self,
        reduction = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):
        return binary_relative_volume_error(pred, gt)

    def aggregate(self):
        data = self.get_buffer()
        f, _ = do_metric_reduction(data, self.reduction)
        return f

def binary_relative_volume_error(s_volume, g_volume):
    s_v = float(torch.sum(s_volume))
    g_v = float(torch.sum(g_volume))
    rve = abs(s_v - g_v) / g_v
    rve_result = torch.Tensor([rve]).unsqueeze(1)

    return rve_result

class SensitivityMetric(CumulativeIterationMetric):
    def __init__(
        self,
        reduction = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):
        return compute_sensitivity(pred, gt)

    def aggregate(self):
        data = self.get_buffer()
        f, _ = do_metric_reduction(data, self.reduction)
        return f

def compute_sensitivity(pred, label):
    N, C, H, W = pred.shape
    pred = pred.view(N,C,H*W)
    label = label.view(N,C,H*W)
    
    tp = torch.sum((pred == 1) & (label == 1), dim=-1)
    tn = torch.sum((pred == 0) & (label == 0), dim=-1)
    fp = torch.sum((pred == 1) & (label == 0), dim=-1)
    fn = torch.sum((pred == 0) & (label == 1), dim=-1)
    sensitivity = tp / (tp + fn)

    return sensitivity


class SpecificityMetric(CumulativeIterationMetric):
    def __init__(
        self,
        reduction = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):
        return compute_specificity(pred, gt)

    def aggregate(self):
        data = self.get_buffer()
        f, _ = do_metric_reduction(data, self.reduction)
        return f

def compute_specificity(pred, label):
    N, C, H, W = pred.shape
    pred = pred.view(N,C,H*W)
    label = label.view(N,C,H*W)
    
    tp = torch.sum((pred == 1) & (label == 1), dim=-1)
    tn = torch.sum((pred == 0) & (label == 0), dim=-1)
    fp = torch.sum((pred == 1) & (label == 0), dim=-1)
    fn = torch.sum((pred == 0) & (label == 1), dim=-1)
    specificity = tn / (tn + fp)

    return specificity