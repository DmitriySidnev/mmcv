import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import time

from mmcv.ops.point_sample import bilinear_grid_sample, SimpleRoIAlign
from mmcv.onnx.symbolic import register_extra_symbolics
from mmcv.ops import get_onnxruntime_op_path


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
