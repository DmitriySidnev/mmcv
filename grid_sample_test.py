import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import time

from mmcv.ops.point_sample import bilinear_grid_sample
from mmcv.onnx.symbolic import register_extra_symbolics
from mmcv.ops import get_onnxruntime_op_path


class BilinearGridSample(nn.Module):
    def __init__(self):
        super(BilinearGridSample, self).__init__()

    def forward(self, img, points):
        return bilinear_grid_sample(img, points)


class PytorchGridSample(nn.Module):
    def __init__(self):
        super(PytorchGridSample, self).__init__()

    def forward(self, img, points):
        return F.grid_sample(img, points)


def measure(sess, input_img, points, N=100):
    start = time.time()
    for _ in range(N):
        x = sess.run(['output'], {'data': input_img.cpu().numpy(), 'points': points.cpu().numpy()})[0]
    end = time.time()
    avg = (end - start) / N
    return avg


e = 1e-3
cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True
np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed(999)
import random
random.seed(918)

n, c, h, w = random.randint(2, 32), random.randint(1, 32) * 2, random.randint(1, 32) * 2, random.randint(1, 32) * 2
gh, gw = random.randint(1, 32) * 2, random.randint(1, 32) * 2
input_img = torch.rand((n, c, h, w), dtype=torch.float32, device='cuda:0')
points = torch.rand((n, gh, gw, 2), dtype=torch.float32, device='cuda:0')


my_model = BilinearGridSample()
mmcv_model = PytorchGridSample()

opset_v = 11
register_extra_symbolics(opset_v)
p = get_onnxruntime_op_path()

blob = (input_img, points)
torch.onnx.export(my_model, blob, 'my_grid_sample.onnx',
                  verbose=True, export_params=True,
                  input_names=['data', 'points'], output_names=['output'],
                  opset_version=opset_v)

torch.onnx.export(mmcv_model, blob, 'mmcv_grid_sample.onnx',
                  verbose=True, export_params=True,
                  input_names=['data', 'points'], output_names=['output'],
                  opset_version=opset_v)

import onnxruntime as rt
sess_my = rt.InferenceSession('my_grid_sample.onnx')
sess_mmcv = rt.InferenceSession('mmcv_grid_sample.onnx')

N = 100
my_time = measure(sess_my, input_img, points, N)
mmcv_time = measure(sess_mmcv, input_img, points, N)
print(f'MY vs MMCV: {my_time} vs {mmcv_time}')
