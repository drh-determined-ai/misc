# Shamelessly stolen from https://stackoverflow.com/questions/65739700/pytorch-cuda-error-no-kernel-image-is-available-for-execution-on-the-device-on

import torch
import sys
print('A', sys.version)
print('B', torch.__version__)
print('C', torch.cuda.is_available())
print('D', torch.backends.cudnn.enabled)
device = torch.device('cuda')
print('E', torch.cuda.get_device_properties(device))
print('F', torch.tensor([1.0, 2.0]).cuda())

