import torch
from openfl.utils.printer import log

log("setup_env", torch.cuda.is_available())     # True
log("setup_env",torch.version.hip)             # e.g. '6.1.0'
log("setup_env",torch.version.cuda)            # None
log("setup_env",torch.cuda.get_device_name(0)) # AMD Radeon RX 7900 XTX