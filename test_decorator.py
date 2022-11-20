import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import inspect

from typing import *

try:
    import dump_collectives
except:
    pass

from ar_queue import _ar_queue as Q

def dump_args(func):
    argspec = inspect.getfullargspec(func)
    def warp(*args, **kwargs):
        dumped = {}
        for k, v in zip(argspec.args[-len(argspec.defaults):], argspec.defaults):
            dumped[k] = v
        for k, v in zip(argspec.args, args):
            dumped[k] = v
        if argspec.varargs is not None:
            dumped[argspec.varargs] = args[len(argspec.args):]
        dumped.update(kwargs)
            
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"dumped({rank}/{world_size}): {dumped}")
        func(*args, **kwargs)
    return warp

# dist.reduce = dump_args(dist.reduce)
# dist.all_reduce = dump_args(dist.all_reduce)
# dist.all_gather = dump_args(dist.all_gather)
# dist.scatter = dump_args(dist.scatter)
# dist.broadcast = dump_args(dist.broadcast)


class AllReduce(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
        ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_channels,
            out_channels,
        )
    
    def forward(self, x):
        x = self.linear(x)
        # dist.all_reduce(x)
        return x

def run(rank, world_size):
    net = AllReduce(2, 3)
    net = nn.parallel.DistributedDataParallel(net)
    
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    x = torch.randn(2)
    loss = net(x).mean()
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    print(Q.get())
    
    
def init_process(rank, world_size, fn, backend="dump"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    dist.init_process_group(backend)
    fn(rank, world_size)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    
    world_size = 3
    processes = []
    
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()