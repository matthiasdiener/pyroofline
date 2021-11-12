#!/usr/bin/env python

import numpy as np
import pyopencl as cl

from kernel import src

a_np = np.zeros(50000, dtype=np.float64)

params = np.zeros(2, dtype=np.int32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(
    ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
params_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=params)


# prg = cl.Program(ctx, src).build()

# knl = prg.ocl_kernel
# knl(queue, a_np.shape, None, np.uint64(50000), np.uint64(5), a_g, params_g )

# cl.enqueue_copy(queue, params, params_g)

# queue.finish()
# print("Finished.")

programs = {}

for bit in (16, 32, 64):
    for flops in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        options = tuple((f"-DFP{bit}", f"-DERT_FLOP={flops}"))
        print(options)
        programs[options] = cl.Program(ctx, src).build(options=options)

        knl = programs[options].ocl_kernel

        evt = knl(queue, a_np.shape, None, np.uint64(
            50000), np.uint64(5), a_g, params_g)

        queue.finish()
        time = evt.profile.end - evt.profile.start

        cl.enqueue_copy(queue, params, params_g)

        print(params, time)

