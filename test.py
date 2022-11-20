import torch
import torch.distributed as dist
import dump_collectives

import os

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "25600"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# for k, v in dist.Backend._plugins.items():
    # print(k, v)

# torch.cuda.set_device(0)
# dist.init_process_group(backend="gloo")
dist.init_process_group(backend="dump")
