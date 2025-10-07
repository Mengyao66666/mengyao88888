import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here.

        # Precompute output size
        self.out_h = (H - kernel_size + 2 * padding) // stride + 1
        self.out_w = (W - kernel_size + 2 * padding) // stride + 1

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        

    def im2col_manual(self, x):
        N = x.shape[0]        # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))

        # rowNum: each row contains one patch (C*KH*KW elements)
        rowNum = out_h * out_w

        # patches shape: (N, out_h*out_w, C*KH*KW)
        patches = torch.zeros(N, rowNum, C * KH * KW)

        # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
        # Refer to Lecture 3 for implementing this operation.

        idx = 0
        for h in range(out_h):
            for w in range(out_w):
                h_start = h * S
                w_start = w * S
                # Extract patch: (N, C, KH, KW) -> flatten to (N, C*KH*KW)
                patch = x_pad[:, :, h_start:h_start + KH, w_start:w_start + KW]
                patches[:, idx, :] = patch.reshape(N, -1)
                idx += 1
        
        # patches = ...
        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        # KH = KW = self.kernel_size

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        weight_flat = self.weight.reshape(C_out, -1)

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        out = torch.matmul(cols, weight_flat.T)

        # TO DO: 4) Add bias.
        out = out + self.bias

        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        out = out.permute(0, 2, 1)  # (N, C_out, out_h*out_w)
        out = out.reshape(N, C_out, self.out_h, self.out_w)

        return out

    def forward(self, x):
        return self.conv2d_manual(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 4, 22, 22
    x = torch.randn(N, C, H, W)
    out_channels=8
    kernel_size=7
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1)
    out = model(x)

    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
