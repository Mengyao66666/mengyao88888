import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

import os
os.environ["NEURON_CC_FLAGS"] = "--verbose"


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

    # W_tile = nl.ndarray(
    #     (out_channels, in_channels, filter_height, filter_width),
    #     dtype=W.dtype,
    #     buffer=nl.sbuf
    # )
    # W_tile[...] = nl.load(W[:, :, :, :])
    W_tile = nl.load(W[:, :, :, :])

    # bias_tile = nl.ndarray(shape=bias.shape, dtype=bias.dtype, buffer=nl.sbuf)
    # bias_tile = nl.ndarray(shape=(out_channels, 1), dtype=bias.dtype, buffer=nl.sbuf)

    bias_tile = nl.load(bias.reshape(out_channels, 1))

    # bias_tile = nl.ndarray((out_channels,), dtype=bias.dtype, buffer=nl.sbuf)
    # bias_tile[...] = nl.load(bias[:])

    # Process the images in batches
    for b in nl.affine_range(batch_size):

        # 所有窗口都加载到SBUF，并且展开。每一行都是in_channels * filter_height * filter_width的长度，然后展开
        # x_tile = nl.ndarray(
        #     (in_channels, input_height, input_width),
        #     dtype=X.dtype,
        #     buffer=nl.sbuf
        # )
        # x_tile[...] = nl.load(X[b, :, :, :])

        x_tile = nl.load(X[b, :, :, :])

        out_tile = nl.ndarray(
            (nl.par_dim(out_channels), out_pool_height, out_pool_width),
            dtype=X_out.dtype,
            buffer=nl.sbuf
        )

        for out_h in nl.affine_range(out_pool_height):
            for out_w in nl.affine_range(out_pool_width):

                ps = nl.zeros(
                    (nl.par_dim(out_channels), 1),
                    dtype=np.float32,
                    buffer=nl.psum
                )

                # Accumulate over filter dimensions
                for fh in nl.affine_range(filter_height):
                    for fw in nl.affine_range(filter_width):
                        # 创建 mgrid
                        weight_mgrid = nl.mgrid[0:out_channels, 0:in_channels]
                        input_mgrid = nl.mgrid[0:in_channels, 0:1]

                        weight_slice = nl.ndarray(
                            (out_channels, in_channels),
                            dtype=W.dtype,
                            buffer=nl.sbuf
                        )
                        i_oc = nl.arange(out_channels)[:, None]
                        i_ic = nl.arange(in_channels)[None, :]
                        weight_slice[...] = W_tile[i_oc, i_ic, fh, fw]

                        input_slice = nl.ndarray(
                            (in_channels, 1),
                            dtype=X.dtype,
                            buffer=nl.sbuf
                        )
                        i_ic_col = nl.arange(in_channels)[:, None]
                        input_slice[...] = x_tile[i_ic_col, out_h + fh, out_w + fw]

                        # 然后用 mgrid 索引做 matmul
                        result = nisa.nc_matmul(
                            weight_slice[weight_mgrid.p, weight_mgrid.x],
                            input_slice[input_mgrid.p, input_mgrid.x]
                        )

                        result_mgrid = nl.mgrid[0:out_channels, 0:1]
                        ps[result_mgrid.p, result_mgrid.x] += result

                ps_mgrid = nl.mgrid[0:out_channels, 0:1]
                i_oc = nl.arange(out_channels)[:, None]
                out_tile[i_oc, out_h, out_w] = nl.copy(ps[ps_mgrid.p, 0])


        # Write back to HBM with bias added
        for oc in nl.affine_range(out_channels):
            channel_data = out_tile[oc:oc + 1, :, :]  # [1, out_pool_height, out_pool_width]
            bias_val = bias_tile[oc:oc + 1, 0]  # [1]

            # nl.add 现在可以工作了
            channel_with_bias = nl.add(channel_data, bias_val)

            nl.store(X_out[b, oc:oc + 1, :, :], value=channel_with_bias)


        print(f"[DEBUG] Batch {b} COMPLETED")

    print(f"[DEBUG] All batches COMPLETED, returning X_out")
    return X_out






















