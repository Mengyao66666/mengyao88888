import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for out_h in nl.affine_range(out_pool_height):
            for out_w in nl.affine_range(out_pool_width):

                # Allocate a tensor in PSUM
                res_psum = nl.zeros((out_channels, 1), dtype=nl.float32, buffer=nl.psum)

                # 所有窗口都加载到SBUF，并且展开。每一行都是in_channels * filter_height * filter_width的长度，然后展开
                x_flat = nl.load(X[b, :, out_h:out_h + filter_height, out_w:out_w + filter_width]).reshape(-1)
                print(f"[DEBUG]     batch={b}, h={out_h}, w={out_w}: window loaded, shape={x_flat.shape}")

                for out_c in nl.affine_range(out_channels):
                    w_flat = W[out_c].reshape(in_channels * filter_height * filter_width)
                    res_psum[out_c] += nl.mutmul(w_flat, x_flat)
                print(
                    f"[DEBUG]     batch={b}, h={out_h}, w={out_w}: matmul done, res_psum shape={res_psum.shape}")

                res_psum[:, 0] = res_psum[:, 0] + bias[:]

                res_sb = nl.copy(res_psum[:, 0], dtype=X_out.dtype)
                print(f"[DEBUG]     batch={b}, h={out_h}, w={out_w}: copied to SBUF")
                nl.store(X_out[b, :, out_h, out_w], value=res_sb)
                print(f"[DEBUG]     batch={b}, h={out_h}, w={out_w}: stored to HBM ✓")

        print(f"[DEBUG] Batch {b} COMPLETED")

    print(f"[DEBUG] All batches COMPLETED, returning X_out")
    return X_out

# def kernel(X, W, bias):
#     batch_size, in_channels, input_height, input_width = X.shape
#     out_channels, in_channels_, filter_height, filter_width = W.shape
#     batch_size = 1
#     # out_channels_ = bias.shape[0]
#
#     # 定义遍历范围
#     out_height = input_height - filter_height + 1
#     out_width = input_width - filter_width + 1
#
#     # 定义输出
#     output = nl.ndarray([out_channels, out_height, out_width], dtype=X.dtype, buffer=nl.psum)
#
#     # 直接卷积
#     for out_h in nl.affine_range(out_height):
#         for out_w in nl.affine_range(out_width):
#
#             # Allocate a tensor in PSUM
#             res_psum = nl.zeros((out_channels,), dtype=nl.float32, buffer=nl.psum)
#
#             # 所有窗口都加载到SBUF，并且展开。每一行都是in_channels * filter_height * filter_width的长度，然后展开
#             x_flat = nl.load(X[:, :, out_h:out_h + filter_height, out_w:out_w + filter_width]).reshape(-1)
#
#             for out_c in nl.affine_range(out_channels):
#                 w_flat = W[out_c].reshape(in_channels * filter_height * filter_width)
#                 res_psum[out_c] += nl.mutmul(w_flat, x_flat)
#
#             res_sb = nl.copy(res_psum, dtype=output.dtype)
#             nl.store(output[:, :, out_h, out_w], value=res_sb)














