import numpy as np
import os
import time
from tqdm import tqdm

import onnx
import onnxruntime as rt
from onnxoptimizer import optimize

import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn

from mmcv.onnx.symbolic import register_extra_symbolics
from mmcv.ops import get_onnxruntime_op_path
from mmcv.ops.point_sample import bilinear_grid_sample, SimpleRoIAlign


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
    total_time = end - start
    avg = total_time / N
    return total_time, avg


e = 1e-5
cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True
np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed(999)
import random
random.seed(918)

n, c, h, w = random.randint(2, 32), random.randint(1, 32) * 2, random.randint(1, 32) * 2, random.randint(1, 32) * 2
input_img = torch.rand((n, c, h, w), dtype=torch.float32, device='cuda:0')

objects_num = random.randint(1, 100)
rois = torch.rand((objects_num, 5), dtype=torch.float32, device='cuda:0')

simple_roi_align = SimpleRoIAlign(7, 1)

opset_v = 11
register_extra_symbolics(opset_v)
p = get_onnxruntime_op_path()

blob = (input_img, rois)
dynamic_axes = {
    'data': {0: 'batch', 2: 'height', 3: 'width'},
    'rois': {0: 'objects_num'},
    'output': {0: 'objects_num'}
}
torch.onnx.export(simple_roi_align, blob, 'simple_roi_align.onnx',
                  verbose=True, export_params=True,
                  input_names=['data', 'rois'], output_names=['output'],
                  opset_version=opset_v, dynamic_axes=dynamic_axes)

import onnxruntime as rt
sess_sra = rt.InferenceSession('simple_roi_align.onnx')

N = 100
print('SimpleRoIAlign test starts')
for i in range(N):
    n, h, w = random.randint(1, 32), random.randint(1, 224), random.randint(1, 224)
    input_img = torch.rand((n, c, h, w), dtype=torch.float32, device='cuda:0')
    objects_num = random.randint(1, 100)
    rois = torch.rand((objects_num, 5), dtype=torch.float32, device='cuda:0')
    x1 = sess_sra.run(['output'], {'data': input_img.cpu().numpy(), 'rois': rois.cpu().numpy()})[0]
    x2 = simple_roi_align(input_img, rois)
    if not torch.allclose(torch.from_numpy(x1).to(x2.device), x2, atol=e):
        print(f'{i}: failed')
    else:
        print(f'{i}: OK')

#####################################################################

n, c, h, w = 8, 256, 56, 56
gh, gw = 28, 28
input_img = torch.rand((n, c, h, w), dtype=torch.float32, device='cuda:0')
points = torch.rand((n, gh, gw, 2), dtype=torch.float32, device='cuda:0')

my_model = BilinearGridSample()
mmcv_model = PytorchGridSample()

opset_v = 11
register_extra_symbolics(opset_v)

ort_custom_op_path = get_onnxruntime_op_path()
assert os.path.exists(ort_custom_op_path)
session_options = rt.SessionOptions()
session_options.register_custom_ops_library(ort_custom_op_path)

blob = (input_img, points)
torch.onnx.export(my_model, blob, 'my_grid_sample.onnx',
                  verbose=False, export_params=True,
                  input_names=['data', 'points'], output_names=['output'],
                  opset_version=opset_v)

torch.onnx.export(mmcv_model, blob, 'mmcv_grid_sample.onnx',
                  verbose=False, export_params=True,
                  input_names=['data', 'points'], output_names=['output'],
                  opset_version=opset_v)

sess_my = rt.InferenceSession('my_grid_sample.onnx')
sess_mmcv = rt.InferenceSession('mmcv_grid_sample.onnx', session_options)

N = 1000
print('> MY test starts')
my_time, my_avg = measure(sess_my, input_img, points, N)
print('> MMCV test starts')
mmcv_time, mmcv_avg = measure(sess_mmcv, input_img, points, N)
print(f'MY vs MMCV ({N} iterations):\n'
      f'  total time: {my_time} vs {mmcv_time}\n'
      f'  average time: {my_avg} vs {mmcv_avg}')
