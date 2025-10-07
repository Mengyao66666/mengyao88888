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

    W_tile = nl.ndarray(
        (out_channels, in_channels, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf
    )
    W_tile[...] = nl.load(W[:, :, :, :])

    bias_tile = nl.ndarray((out_channels, 1), dtype=bias.dtype, buffer=nl.sbuf)
    i_oc = nl.arange(out_channels)
    bias_tile[i_oc, 0] = nl.load(bias[i_oc])

    # bias_tile = nl.ndarray((out_channels,), dtype=bias.dtype, buffer=nl.sbuf)
    # bias_tile[...] = nl.load(bias[:])

    # Process the images in batches
    for b in nl.affine_range(batch_size):

        # 所有窗口都加载到SBUF，并且展开。每一行都是in_channels * filter_height * filter_width的长度，然后展开
        x_tile = nl.ndarray(
            (in_channels, input_height, input_width),
            dtype=X.dtype,
            buffer=nl.sbuf
        )
        x_tile[...] = nl.load(X[b, :, :, :])

        out_tile = nl.ndarray(
            (out_channels, out_pool_height, out_pool_width),
            dtype=nl.float32,
            buffer=nl.sbuf
        )

        for out_h in nl.affine_range(out_pool_height):
            for out_w in nl.affine_range(out_pool_width):
                ps = nl.zeros(
                    (nl.par_dim(out_channels), 1),
                    dtype=nl.float32,
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

                        # Accumulate to PSUM
                        result_mgrid = nl.mgrid[0:out_channels, 0:1]
                        ps[result_mgrid.p, result_mgrid.x] += result

                result_mgrid = nl.mgrid[0:out_channels, 0:1]
                i_oc = nl.arange(out_channels)

                out_tile[i_oc, out_h, out_w] = ps[result_mgrid.p, 0]

        # X_out[b, :, :, :] = out_tile
        i_oc = nl.arange(out_channels)[:, None, None]
        X_out[b, i_oc, :, :] = nl.copy(out_tile[i_oc, :, :]) + bias_tile[i_oc, 0]

        print(f"[DEBUG] Batch {b} COMPLETED")

        print(f"[DEBUG] All batches COMPLETED, returning X_out")
    return X_out

    # Various tiling dimensions (You may want to define more of them)
    # c_in_pmax = nl.tile_size.pmax
    # n_tiles_c_in = in_channels // c_in_pmax

    # 以下几步都是转换filter
    # w_reshape = nl.ndarray((in_channels_, filter_height, filter_width, out_channels), dtype=W.dtype, buffer=nl.sbuf)
    #
    # # 临时缓冲区，用于加载数据
    # w_temp = nl.ndarray((filter_height, filter_width, in_channels_), dtype=W.dtype, buffer=nl.sbuf)
    #
    # for out_c in nl.sequential_range(out_channels):
    #     # 加载当前 out_channel 的所有数据
    #     w_temp[...] = nl.load(W[out_c, :, :, :])
    #
    #     # 重排维度: (filter_height, filter_width, in_channels_) -> (in_channels_, filter_height, filter_width, out_c)
    #     for in_c in nl.affine_range(in_channels_):
    #         for fH in nl.affine_range(filter_height):
    #             for fW in nl.affine_range(filter_width):
    #                 w_reshape[in_c, fH, fW, out_c] = w_temp[fH, fW, in_c]

    # 计算各个tile。卷积核不切了。
    # out_height_size = 8
    # out_height_tile = (out_height + out_height_size - 1) // out_height_size
    # out_width_size = 8
    # out_width_tile = (out_width + out_width_size - 1) // out_width_size
    # out_channel_size = 128
    # out_channel_tile = (out_channels + out_channel_size - 1) // out_channel_size
    # for b in nl.affine_range(batch_size):

        # output = nl.zeros(
        #     shape=(nl.par_dim(out_channels_), out_height, out_width),
        #     dtype=W.dtype,
        #     buffer=nl.sbuf
        # )
        #
        # # 外层：遍历输出位置
        # for out_h in nl.affine_range(out_height):
        #     for out_w in nl.affine_range(out_width):
        #
        #         res_psum = nl.zeros((out_channels, 1), dtype=nl.float32, buffer=nl.psum)
        #
        #         for in_c in nl.affine_range(in_channels):
        #
        #             # Filter slice: (out_c, kh, kw)
        #             filter_slice = nl.ndarray((out_channels, filter_height * filter_width),
        #                                       dtype=W.dtype, buffer=nl.sbuf)
        #             input_slice = nl.ndarray((filter_height * filter_width, 1),
        #                                      dtype=X.dtype, buffer=nl.sbuf)
        #
        #             # 把 filter[out_c, in_c, kh, kw] reshape 成 (out_c, kh*kw)
        #             idx = 0
        #             for fh in nl.affine_range(filter_height):
        #                 for fw in nl.affine_range(filter_width):
        #                     filter_slice[:, idx] = nl.load(W[:, in_c, fh, fw])
        #                     input_slice[idx, 0] = nl.load(X[0, in_c, out_h_idx + fh, out_w_idx + fw])
        #                     idx += 1
        #
        #             # Matmul 累加到 PSUM
        #             res_psum += nl.matmul(filter_slice, input_slice)
        #
        #             # Copy 从 PSUM 到 SBUF
        #         res_sb = nl.copy(res_psum, dtype=output_hbm.dtype)
        #
        #         # Store 回 HBM
        #         nl.store(output_hbm[:, out_h_idx, out_w_idx], value=res_sb[:, 0])
        # 这个应该没问题。in channel在计算的时候没了，所以输出out C，out H，out W。最后再通过向量索引加上batch。
        # par_dim表示out channel分的块要并行处理
        # out_b = nl.zeros(
        #     shape=(out_channel_tile, out_height_tile, out_width_tile, par_dim(out_channel_size), out_height_size,
        #            out_width_size),
        #     dtype=W.dtype,
        #     buffer=nl.sbuf
        # )
        #
        # # 给input分配空间
        # # 形状存疑。in channels是需要的，拿来和filter对齐。
        # prefetch_x = nl.zeros(
        #     shape=(out_height_tile, out_width_tile, in_channels, out_height_size, out_width_size),
        #     dtype=W.dtype,
        #     buffer=nl.sbuf
        # )
        #
        # # 创建索引（？）
        # c_i = nl.arange(in_channels)[:, None, None]
        # oh_i = nl.arange(out_height_size)[None, :, None]
        # iw_i = nl.arange(out_width)[None, None, :]
        # k0 = ht * out_height_size + oh_i
        #
        # # 加载input到SBUF（？）
        # prefetch_x[c_i, oh_i, iw_i] = nl.load(X[b, c_i, k0, iw_i])
        #
        # # 给filter分配空间。我的filter H呢？
        # # prefetch_filter = nl.zeros(
        # #     shape=(par_dim(in_channels), filter_width, out_channels_),
        # #     dtype=W.dtype,
        # # )
        #
        # # 以下三行可以并行处理
        # i_cin = nl.arange(in_channels)[:, None, None]
        # i_w_f = nl.arange(filter_width)[None, :, None]
        # i_cout = nl.arange(out_channels_)[None, None, :]
        #
        # # 加载filter到SBUF
        # # prefetch_filter[i_cin, i_w_f, i_cout] = nl.load(w_reshape[i_cin, i_w_f, i_cout])
        #
        # # 开始计算
        # for k0_tile in nl.affine_range(out_height_tile):
        #     for k1_tile in nl.affine_range(out_width_tile):
        #         for c_out_tile in nl.affine_range(out_channel_tile):
        #             ps = nl.zeros(
        #                 shape=(par_dim(out_channel_size), out_height_size, out_width_size),
        #                 dtype=np.float32,
        #                 buffer=nl.psum
        #             )
        #
        #             # for w in nl.affine_range(filter_width):
        #                 # 创建并行索引
        #                 这个好像在遍历卷积核
        #                 i_cin = nl.arange(in_channels)[:, None, None, None]
        #                 i_cout = nl.arange(out_channel_size)[:, None, None]
        #                 i_k0 = nl.arange(out_height_size)[None, :, None]
        #                 i_k1 = nl.arange(out_width_size)[None, None, :]
        #
        #                 k1 = k1_tile * out_width_size + i_k1
        #
        #                 # 这个到底是啥。。到底有哪些维度
        #                 # prefetch_x[c_i, oh_i, iw_i] = nl.load(X[b, c_i, k0, iw_i])
        #                 img_local = prefetch_x[i_cin, k0_tile, i_k0]
        #
        #                 c_out = c_out_tile * out_channel_size + i_cout
        #                 filter_local = w_reshape[i_cin, w, c_out]
        #
        #                 ps[i_cout, i_k0, i_k1] += nisa.nc_matmul(
        #                     filter_local[c_out < out_channels],
        #                     img_local[k1 < out_height],
        #                 )
        #
        #             # i_cout_out, i_k0, i_k1, i_n = create_indices(COUT_TILE_SIZES, K0_COMP_TILE_SIZES, K1_TILE_SIZES,
        #             #                                              N_COMP_TILE_SIZES)
        #             i_cout = nl.arange(out_channel_size)[:, None, None]
        #             i_k0 = nl.arange(out_height_size)[None, :, None]
        #             i_k1 = nl.arange(out_width_size)[None, None, :]
        #             out_b[c_out_tile, k0_tile, k1_tile, i_cout, i_k0, i_k1] += ps[i_cout, i_k0, i_k1]
        # # 写回结果
        # for k0_tile in nl.affine_range(out_height_tile):
        #     for k1_tile in nl.affine_range(out_width_tile):
        #         for c_out_tile in nl.affine_range(out_channel_tile):
        #                 # i_cout, i_k0, i_k1, i_n = create_indices(COUT_TILE_SIZES, K0_COMP_TILE_SIZES, K1_TILE_SIZES,
        #                 #                                          N_COMP_TILE_SIZES)
        #
        #                 i_cout = nl.arange(out_channel_size)[:, None, None]
        #                 i_k0 = nl.arange(out_height_size)[None, :, None]
        #                 i_k1 = nl.arange(out_width_size)[None, None, :]
        #
        #                 c_out = c_out_tile * out_channel_size + i_cout
        #                 k0 = k0_tile * out_height_size + i_k0
        #                 k1 = k1_tile * out_width_size + i_k1
        #
        #                 nl.store(
        #                     X_out[b, c_out, k0, k1],
        #                     # out_channel_tile, out_height_tile, out_width_tile, par_dim(out_channel_size),
        #                     # out_height_size, out_width_size
        #                     out_b[c_out_tile, k0_tile, k1_tile, i_cout, i_k0, i_k1]
        #                 )


























