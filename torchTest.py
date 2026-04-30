import torch
from openfl.utils.printer import log

log("gpu_info", torch.cuda.is_available())     # True
log("gpu_info",torch.version.hip)             # e.g. '6.1.0'
log("gpu_info",torch.version.cuda)            # None
log("gpu_info",torch.cuda.get_device_name(0)) # AMD Radeon RX 7900 XTX