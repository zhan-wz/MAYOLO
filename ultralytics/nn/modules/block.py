# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
)

# 导入自定义的Mamba模块
from mamba_ssm import Mamba
#方式一 ：只有两条mamba路径

# class EfficientMambaAttention(nn.Module):
#     """
#     使用Mamba的注意力模块
#     """
#
#     def __init__(self, channel, kernel_size=7):
#         """
#         初始化函数
#
#         参数:
#             channel (int): 输入的通道数
#             kernel_size (int, optional): 核大小，默认为7
#         """
#         super(EfficientMambaAttention, self).__init__()
#         # 定义sigmoid激活函数
#         self.sigmoid_x = nn.Sigmoid()
#
#         # 沿着宽度方向进行自适应平均池化
#         self.avg_x = nn.AdaptiveAvgPool2d((None, 1))
#
#         # 沿着高度方向进行自适应平均池化
#         self.avg_y = nn.AdaptiveAvgPool2d((1, None))
#
#         # 定义另一个sigmoid激活函数
#         self.sigmoid_y = nn.Sigmoid()
#
#         # 初始化Mamba模块，用于处理x方向
#         self.mamba_x = Mamba(
#             d_model=channel,  # 模型维度
#             d_state=16,  # SSM状态扩展因子
#             d_conv=4,  # 局部卷积宽度
#             expand=2,  # 块扩展因子
#         )
#
#         # 初始化Mamba模块，用于处理y方向
#         self.mamba_y = Mamba(
#             d_model=channel,  # 模型维度
#             d_state=16,  # SSM状态扩展因子
#             d_conv=4,  # 局部卷积宽度
#             expand=2,  # 块扩展因子
#         )
#
#     def forward(self, x):
#         """
#         前向传播函数
#
#         参数:
#             x (torch.Tensor): 输入张量
#
#         返回:
#             torch.Tensor: 输出张量
#         """
#
#         # 获取输入张量的尺寸
#         b, c, h, w = x.size()
#
#         # 沿着宽度方向进行平均池化，并转置
#         x_x = self.avg_x(x).squeeze(3).transpose(-1, -2)
#
#         # 通过Mamba模块处理x_x
#         x_ma = self.mamba_x(x_x).transpose(-1, -2)
#
#         # 应用sigmoid激活函数并重塑为原始尺寸
#         x_x = self.sigmoid_x(x_ma).view(b, c, h, 1)
#
#         # 沿着高度方向进行平均池化，并转置
#         x_y = self.avg_y(x).squeeze(2).transpose(-1, -2)
#
#         # 通过Mamba模块处理x_y，应用sigmoid激活函数，并重塑为原始尺寸
#         x_y = self.sigmoid_y(self.mamba_y(x_y).transpose(-1, -2)).view(b, c, 1, w)
#
#         # 返回x与x_x和x_y的逐元素乘积
#         return x * x_x.expand_as(x) * x_y.expand_as(x)

# 方式二：多了一条卷积路，普通卷积～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
# class TiedBlockConv2d(nn.Module):
#     def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0, bias=False, dropout_tbc=0.0):
#         super(TiedBlockConv2d, self).__init__()
#
#         self.stride = stride
#         self.padding = padding
#         self.out_planes = planes
#         self.kernel_size = kernel_size
#         self.dropout_tbc = dropout_tbc
#
#         # 创建标准卷积层
#         self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride,
#                               padding=padding, bias=bias)
#
#         if self.dropout_tbc > 0.0:
#             self.drop_out = nn.Dropout(self.dropout_tbc)
#
#     def forward(self, x):
#         # 对输入进行卷积
#         x = self.conv(x)
#
#         if self.dropout_tbc > 0:
#             x = self.drop_out(x)
#
#         return x
#
# class EfficientMambaAttention(nn.Module):
#     def __init__(self, channel, kernel_size=7, local_kernel_size=3):
#         super(EfficientMambaAttention, self).__init__()
#
#         self.sigmoid_x = nn.Sigmoid()
#         self.avg_x = nn.AdaptiveAvgPool2d((None, 1))
#         self.avg_y = nn.AdaptiveAvgPool2d((1, None))
#         self.sigmoid_y = nn.Sigmoid()
#
#         # 定义Mamba模块
#         self.mamba_x = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#         self.mamba_y = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#         # 局部卷积模块：修改padding和stride使输出与输入匹配
#         self.local_conv = TiedBlockConv2d(
#             in_planes=channel,
#             planes=channel,  # 确保输出通道数一致
#             kernel_size=local_kernel_size,
#             stride=1,
#             padding=1,  # 增加padding，使输出大小与输入一致
#             dropout_tbc=0.0
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # 全局Mamba处理
#         x_x = self.avg_x(x).squeeze(3).transpose(-1, -2)
#         x_ma = self.mamba_x(x_x).transpose(-1, -2)
#         x_x = self.sigmoid_x(x_ma).view(b, c, h, 1)
#
#         x_y = self.avg_y(x).squeeze(2).transpose(-1, -2)
#         x_y = self.sigmoid_y(self.mamba_y(x_y).transpose(-1, -2)).view(b, c, 1, w)
#
#         # 扩展x_x和x_y
#         x_x = x_x.expand_as(x)
#         x_y = x_y.expand_as(x)
#
#         # 局部卷积输出
#         local_output = self.local_conv(x)
#
#         # 输出融合：此时local_output形状与x一致
#         output = x * x_x * x_y + local_output
#         return output
# 方式三：另外一条路换成TBC卷积
# class TiedBlockConv2d(nn.Module):
#     '''Tied Block Conv2d'''
#
#     # 初始化函数
#     def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0, bias=False,
#                  B=2, args=None, dropout_tbc=0.0, groups=1):
#         super(TiedBlockConv2d, self).__init__()  # 调用父类nn.Module的初始化函数
#
#         # 断言确保输入和输出的通道数可以被B整除
#         assert planes % B == 0
#         assert in_planes % B == 0
#
#         self.B = B  # 定义一个参数B，用于分组
#         self.stride = stride  # 卷积的步长
#         self.padding = padding  # 卷积的填充大小
#         self.out_planes = planes  # 输出的通道数
#         self.kernel_size = kernel_size  # 卷积核的大小
#         self.dropout_tbc = dropout_tbc  # Tied Block Conv2d中的dropout率
#
#         # 创建一个标准的二维卷积层，但输入的通道数和输出的通道数都被B整除
#         self.conv = nn.Conv2d(in_planes // self.B, planes // self.B, kernel_size=kernel_size, stride=stride, \
#                               padding=padding, bias=bias, groups=groups)
#
#         # 如果dropout_tbc大于0，则添加一个dropout层
#         if self.dropout_tbc > 0.0:
#             self.drop_out = nn.Dropout(self.dropout_tbc)
#
#             # 前向传播函数
#
#     def forward(self, x):
#         n, c, h, w = x.size()  # 获取输入张量的尺寸：批量大小、通道数、高度、宽度
#
#         # 将输入张量重新排列，使其通道数变为原来的B倍，而批量大小变为原来的1/B
#         x = x.contiguous().view(n * self.B, c // self.B, h, w)
#
#         # 计算输出张量的高度和宽度
#         h_o = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
#         w_o = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
#
#         # 对重新排列后的张量进行卷积操作
#         x = self.conv(x)
#
#         # 将卷积后的张量重新排列回原来的形状
#         x = x.view(n, self.out_planes, h_o, w_o)
#
#         # 如果在初始化时指定了dropout_tbc，则对卷积后的张量进行dropout操作
#         if self.dropout_tbc > 0:
#             x = self.drop_out(x)
#
#         return x  # 返回处理后的张量
#
#
# class EfficientMambaAttention(nn.Module):
#     def __init__(self, channel, kernel_size=7, local_kernel_size=3):
#         super(EfficientMambaAttention, self).__init__()
#
#         self.sigmoid_x = nn.Sigmoid()
#         self.avg_x = nn.AdaptiveAvgPool2d((None, 1))
#         self.avg_y = nn.AdaptiveAvgPool2d((1, None))
#         self.sigmoid_y = nn.Sigmoid()
#
#         # 定义Mamba模块
#         self.mamba_x = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#         self.mamba_y = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#         # 局部卷积模块：修改padding和stride使输出与输入匹配
#         self.local_conv = TiedBlockConv2d(
#             in_planes=channel,
#             planes=channel,  # 确保输出通道数一致
#             kernel_size=local_kernel_size,
#             stride=1,
#             padding=1,  # 增加padding，使输出大小与输入一致
#             dropout_tbc=0.0
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # 全局Mamba处理
#         x_x = self.avg_x(x).squeeze(3).transpose(-1, -2)
#         x_ma = self.mamba_x(x_x).transpose(-1, -2)
#         x_x = self.sigmoid_x(x_ma).view(b, c, h, 1)
#
#         x_y = self.avg_y(x).squeeze(2).transpose(-1, -2)
#         x_y = self.sigmoid_y(self.mamba_y(x_y).transpose(-1, -2)).view(b, c, 1, w)
#
#         # 扩展x_x和x_y
#         x_x = x_x.expand_as(x)
#         x_y = x_y.expand_as(x)
#
#         # 局部卷积输出
#         local_output = self.local_conv(x)
#
#         # 输出融合：此时local_output形状与x一致
#         output = x * x_x * x_y + local_output
#         return output
#方式四，用交叉注意力机制
#方式五，拼接改成乘法，不分割通道
#～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
# 加残差代码块
# class EfficientMambaAttention(nn.Module):
#     """
#     该类处理输入张量并使用 CoordAtt 和 EfficientMambaAttention 进行处理。
#     """
#
#     def __init__(self, channel, device='cuda:0'):
#         """
#         初始化函数，创建 CoordAtt 和 EfficientMambaAttention 模块实例。
#
#         参数:
#             channel (int): 输入通道数，用于初始化 CoordAtt 和 EfficientMambaAttention。
#             device (torch.device): 计算设备（CPU 或 GPU）。
#         """
#         super(EfficientMambaAttention, self).__init__()
#         # 确保设备有效，并选择设备
#         if torch.cuda.is_available():
#             device_count = torch.cuda.device_count()
#             print(f"可用 GPU 数量: {device_count}")
#             # 确保 device 是字符串类型并且包含了设备编号
#             if isinstance(device, int):
#                 device = f"cuda:{device}"  # 如果 device 是整数，转换为字符串形式
#             # 确保指定的设备在可用设备范围内
#             if int(device.split(":")[-1]) < device_count:
#                 self.device = torch.device(device)
#             else:
#                 print(f"设备 {device} 不存在, 将使用默认设备 cuda:0")
#                 self.device = torch.device('cuda:0')  # 使用可用的设备
#
#         else:
#             raise RuntimeError("CUDA 不可用，请确保 GPU 可用")
#         # 将输入的 channel 分为两部分，动态传递给每个模块
#         split_idx = channel // 2  # 分割索引
#
#         # 创建 CoordAtt 和 EfficientMambaAttention 模块
#         self.coord_att = CoordAtt(inp=split_idx).to(self.device)
#         self.mamba_attention = EfficientMambaAttention00(channel=split_idx).to(self.device)
#
#     def forward(self, x):
#         """
#         前向传播函数，处理输入张量。
#
#         参数:
#             input_tensor (torch.Tensor): 输入的 4D 张量，形状为 (batch_size, channels, height, width)
#
#         返回:
#             torch.Tensor: 处理后的 4D 张量，形状与输入相同
#         """
#
#         channels = x.shape[1]
#         split_idx = channels // 2  # 计算切分索引
#         #
#         # # 动态分割输入张量
#         input_tensor_part1 = x[:, :split_idx, :, :]  # 第一部分（前一半通道）
#         input_tensor_part2 = x[:, split_idx:, :, :]  # 第二部分（后一半通道）
#
#         # 通过 CoordAtt 处理第一部分
#         # a_h, a_w = self.coord_att(input_tensor_part1)
#         outs1 = self.coord_att(input_tensor_part1)
#         # 通过 EfficientMambaAttention 处理第二部分
#         # x_x, x_y = self.mamba_attention(input_tensor_part2)
#         outs2 = self.mamba_attention(input_tensor_part2)
#         # 交叉注意力计算
#         # result_h = a_h * x_y  # a_h 和 x_y 相乘
#         # result_w = a_w * x_x  # a_w 和 x_x 相乘
#         # result_h = torch.cat([a_h, x_x], dim=1)
#         # result_w = torch.cat([a_w, x_y], dim=1)
#         # # 拼接结果
#         # output1 = torch.cat([result_h, result_w], dim=1)  # 在维度1上拼接
#         # result_h = torch.sigmoid(result_h)
#         # result_w = torch.sigmoid(result_w)
#         # output = x * result_h * result_w
#         output = torch.cat([outs1, outs2], dim=1)
#         # output = x * output1
#         return output
# # 自定义 h_sigmoid 激活函数
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# # 自定义 h_swish 激活函数
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# # CoordAtt 注意力机制模块
# class CoordAtt(nn.Module):
#     def __init__(self, inp, reduction=32):
#         super(CoordAtt, self).__init__()
#         oup = inp  # 输出通道数和输入通道数相同
#
#         # 自适应池化处理高宽维度
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度方向的池化
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度方向的池化
#
#         mip = max(8, inp // reduction)  # 中间通道数，通过reduction调整
#
#         # 卷积层初始化
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#         n, c, h, w = x.size()
#
#         # 高度方向池化
#         x_h = self.pool_h(x)
#         # 宽度方向池化，并交换维度
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         # 拼接池化结果
#         y = torch.cat([x_h, x_w], dim=2)
#
#         # 通过卷积和激活处理
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#
#         # 切分结果
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)  # 宽度方向维度交换
#
#         # 生成水平和垂直注意力图
#         a_h = self.conv_h(x_h)
#         a_w = self.conv_w(x_w)
#
#         # 最后通过sigmoid激活
#         a_h = a_h.sigmoid()
#         a_w = a_w.sigmoid()
#
#         # 输出通过加权后与原输入相乘
#         outs1 = identity * a_w * a_h
#         return outs1
#
#
# # EfficientMambaAttention 使用 Mamba 模块
# class EfficientMambaAttention00(nn.Module):
#     def __init__(self, channel, kernel_size=7):
#         super(EfficientMambaAttention00, self).__init__()
#
#         self.sigmoid_x = nn.Sigmoid()
#
#         # 沿着宽度方向进行自适应平均池化
#         self.avg_x = nn.AdaptiveAvgPool2d((None, 1))
#
#         # 沿着高度方向进行自适应平均池化
#         self.avg_y = nn.AdaptiveAvgPool2d((1, None))
#
#         self.sigmoid_y = nn.Sigmoid()
#
#         # 初始化Mamba模块，用于处理x方向
#         self.mamba_x = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#         # 初始化Mamba模块，用于处理y方向
#         self.mamba_y = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # 沿着宽度方向进行平均池化，并转置
#         x_x = self.avg_x(x).squeeze(3).transpose(-1, -2)
#
#         # 通过Mamba模块处理x_x
#         x_ma = self.mamba_x(x_x).transpose(-1, -2)
#
#         # # 应用sigmoid激活函数并重塑为原始尺寸
#         x_x = self.sigmoid_x(x_ma).view(b, c, h, 1)
#
#         # 沿着高度方向进行平均池化，并转置
#         x_y = self.avg_y(x).squeeze(2).transpose(-1, -2)
#
#         # # 通过Mamba模块处理x_y，应用sigmoid激活函数，并重塑为原始尺寸
#         x_y = self.sigmoid_y(self.mamba_y(x_y).transpose(-1, -2)).view(b, c, 1, w)
#
#
#         # 返回x与x_x和x_y的逐元素乘积
#         outs2 = x * x_x.expand_as(x) * x_y.expand_as(x)
#         return outs2

#～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

# #
# class EfficientMambaAttention(nn.Module):
#     def __init__(self, channel, device):
#         super(EfficientMambaAttention, self).__init__()
#         self.efficient_mamba_attention = CombinedBlock(channel, device)
#         # 如果输入和输出通道数不同，可以通过卷积调整
#         self.adjust_channels = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn = nn.BatchNorm2d(channel)
#
#     def forward(self, x):
#         # 保存输入张量，用于残差连接
#         residual = x
#         # 通过 EfficientMambaAttention
#         x = self.efficient_mamba_attention(x)
#         # 如果需要调整通道数，取消注释下面两行
#         # residual = self.adjust_channels(residual)
#         # residual = self.bn(residual)
#         # 逐元素相加实现残差
#         out = x * residual
#         return out
# class CombinedBlock(nn.Module):
#     def __init__(self, channel, device):
#         """
#         组合 MB 和 ECA_block 的模块。
#
#         参数:
#             channel (int): 输入通道数。
#             device (torch.device): 计算设备。
#         """
#         super(CombinedBlock, self).__init__()
#         self.mb = MB(channel, device)
#         self.eca = ECA_block(channel)  # ECA_block 接受与 CA_MB 输出相同的通道数
#
#     def forward(self, x):
#
#         x = self.mb(x)  # 通过 MB 模块
#         x = self.eca(x)    # 通过 ECA_block 模块
#         #print("x:", x.shape)
#         return x
# class ECA_block(nn.Module):
#     def __init__(self, channel, b=1, gamma=2):
#         super(ECA_block, self).__init__()
#         kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.conv = self.conv.to(device)  # 将卷积层加载到指定设备
#
#         y = self.avg_pool(x)
#         # 确保输入张量和卷积层的权重在相同的设备上
#         device = y.device  # 获取输入张量所在的设备
#
#         # 如果输入张量在 CUDA 上，确保卷积层也在 CUDA 上
#         self.conv = self.conv.to(device)  # 将卷积层移动到输入张量所在的设备
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         # out = x * y.expand_as(x)
#         out = y.expand_as(x)
#
#         return out
# class TiedBlockConv2d(nn.Module):
#     '''Tied Block Conv2d'''
#
#     # 初始化函数
#     def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0, bias=False,
#                  B=2, args=None, dropout_tbc=0.0, groups=1):
#         super(TiedBlockConv2d, self).__init__()  # 调用父类nn.Module的初始化函数
#
#         # 断言确保输入和输出的通道数可以被B整除
#         assert planes % B == 0
#         assert in_planes % B == 0
#
#         self.B = B  # 定义一个参数B，用于分组
#         self.stride = stride  # 卷积的步长
#         self.padding = padding  # 卷积的填充大小
#         self.out_planes = planes  # 输出的通道数
#         self.kernel_size = kernel_size  # 卷积核的大小
#         self.dropout_tbc = dropout_tbc  # Tied Block Conv2d中的dropout率
#
#         # 创建一个标准的二维卷积层，但输入的通道数和输出的通道数都被B整除
#         self.conv = nn.Conv2d(in_planes // self.B, planes // self.B, kernel_size=kernel_size, stride=stride, \
#                               padding=padding, bias=bias, groups=groups)
#
#         # 如果dropout_tbc大于0，则添加一个dropout层
#         if self.dropout_tbc > 0.0:
#             self.drop_out = nn.Dropout(self.dropout_tbc)
#
#             # 前向传播函数
#
#     def forward(self, x):
#         n, c, h, w = x.size()  # 获取输入张量的尺寸：批量大小、通道数、高度、宽度
#
#         # 将输入张量重新排列，使其通道数变为原来的B倍，而批量大小变为原来的1/B
#         x = x.contiguous().view(n * self.B, c // self.B, h, w)
#
#         # 计算输出张量的高度和宽度
#         h_o = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
#         w_o = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
#
#         # 对重新排列后的张量进行卷积操作
#         x = self.conv(x)
#
#         # 将卷积后的张量重新排列回原来的形状
#         x = x.view(n, self.out_planes, h_o, w_o)
#
#         # 如果在初始化时指定了dropout_tbc，则对卷积后的张量进行dropout操作
#         if self.dropout_tbc > 0:
#             x = self.drop_out(x)
#
#         return x  # 返回处理后的张量
#
#
# class MB(nn.Module):
#     def __init__(self, channel, kernel_size=7, local_kernel_size=3):
#         super(MB, self).__init__()
#
#         self.sigmoid_x = nn.Sigmoid()
#         self.avg_x = nn.AdaptiveAvgPool2d((None, 1))
#         self.avg_y = nn.AdaptiveAvgPool2d((1, None))
#         self.sigmoid_y = nn.Sigmoid()
#
#         # 定义Mamba模块
#         self.mamba_x = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#         self.mamba_y = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#         # 局部卷积模块：修改padding和stride使输出与输入匹配
#         self.local_conv = TiedBlockConv2d(
#             in_planes=channel,
#             planes=channel,  # 确保输出通道数一致
#             kernel_size=local_kernel_size,
#             stride=1,
#             padding=1,  # 增加padding，使输出大小与输入一致
#             dropout_tbc=0.0
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # 全局Mamba处理
#         x_x = self.avg_x(x).squeeze(3).transpose(-1, -2)
#         x_ma = self.mamba_x(x_x).transpose(-1, -2)
#         x_x = self.sigmoid_x(x_ma).view(b, c, h, 1)
#
#         x_y = self.avg_y(x).squeeze(2).transpose(-1, -2)
#         x_y = self.sigmoid_y(self.mamba_y(x_y).transpose(-1, -2)).view(b, c, 1, w)
#
#         # 扩展x_x和x_y
#         x_x = x_x.expand_as(x)
#         x_y = x_y.expand_as(x)
#
#         # 局部卷积输出
#         local_output = self.local_conv(x)
#         # 输出融合：此时local_output形状与x一致
#         output = x * x_x * x_y + local_output
#         return output
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

#
# import torch.nn as nn
# import torch.nn.functional as F
# from mamba_ssm import Mamba  # 确保你已经正确导入了Mamba模块
# import torch
#
# print(f"Available GPUs: {torch.cuda.device_count()}")
# print(f"Current CUDA device: {torch.cuda.current_device()}")
# print(f"CUDA availability: {torch.cuda.is_available()}")
#
#
# # S2MLP 定义
# class S2MLP(nn.Module):
#     def __init__(self, in_channels, out_channels, d_state=8, *args, **kwargs):
#         super().__init__()
#         self.out_channels = out_channels
#
#         # 1x1 卷积确保输出通道正确
#         self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#         # Split Attention 分支（强制匹配 out_channels）
#         self.mlp_split = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
#             nn.Flatten(),  # [B, C]
#             nn.Linear(out_channels, out_channels // 4),  # 降维
#             nn.GELU(),
#             nn.Linear(out_channels // 4, out_channels),  # 恢复通道数
#             nn.Sigmoid()  # 输出 [B, out_channels]
#         )
#
#         self.act = nn.GELU()
#
#     def forward(self, x):
#         identity = x
#         x = self.act(self.proj(x))  # [B, out_channels, H, W]
#
#         # 生成注意力权重 [B, out_channels, 1, 1]
#         attn = self.mlp_split(x).view(-1, self.out_channels, 1, 1)
#
#         return x * attn  # 残差连接
#
#     def spatial_shift1(self, x):
#         b, w, h, c = x.size()
#         x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
#         x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
#         x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
#         x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
#         return x
#
#     def spatial_shift2(self, x):
#         b, w, h, c = x.size()
#         x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
#         x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
#         x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
#         x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
#         return x
#
#     def split_attention(self, x_all):
#         b, k, h, w, c = x_all.shape
#         x_all = x_all.reshape(b, k, -1, c)
#         a = torch.sum(torch.sum(x_all, 1), 1)
#         hat_a = self.mlp_split_2(self.gelu(self.mlp_split_1(a)))
#         hat_a = hat_a.reshape(b, 3, c)
#         bar_a = self.softmax(hat_a)
#         attention = bar_a.unsqueeze(-2)
#         out = attention * x_all
#         out = torch.sum(out, 1).reshape(b, h, w, c)
#         return out
#
#
# # S2MLP 加入到 CoordAtt 后
# class CoordAttWithS2MLP(nn.Module):
#     def __init__(self, inp, reduction=32, s2mlp_channels=512, device=None):
#         super(CoordAttWithS2MLP, self).__init__()
#         oup = inp
#         self.device = device  # 记录设备
#
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = nn.GELU()
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#         # 添加 S2MLP 模块
#         # self.s2mlp = S2MLP(channels=s2mlp_channels)
#
#         # 转移所有层到设备上
#         self.to(self.device)
#     def forward(self, x):
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)  # 卷积操作不需要手动转移设备，因为网络已被转移
#         y = self.bn1(y)
#         y = self.act(y)
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h)  # 直接执行卷积，设备已一致
#         a_w = self.conv_w(x_w)  # 直接执行卷积，设备已一致
#
#         # # 通过 S2MLP 增强注意力图
#         # a_h = self.s2mlp(a_h)
#         # a_w = self.s2mlp(a_w)
#         a_h_mamba = a_h
#         a_w_mamba = a_w
#         a_h = a_h.sigmoid()
#         a_w = a_w.sigmoid()
#
#         return a_h, a_w ,a_h_mamba,a_w_mamba
#
#
#
# # EfficientMambaAttentionWithS2MLP 结合了 S2MLP 和 Mamba
# class EfficientMambaAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, device='cuda:0'):
#         super(EfficientMambaAttention, self).__init__()
#         # 确保设备有效，并选择设备
#         if torch.cuda.is_available():
#             device_count = torch.cuda.device_count()
#             print(f"可用 GPU 数量: {device_count}")
#             # 确保 device 是字符串类型并且包含了设备编号
#             if isinstance(device, int):
#                 device = f"cuda:{device}"  # 如果 device 是整数，转换为字符串形式
#             # 确保指定的设备在可用设备范围内
#             if int(device.split(":")[-1]) < device_count:
#                 self.device = torch.device(device)
#             else:
#                 print(f"设备 {device} 不存在, 将使用默认设备 cuda:0")
#                 self.device = torch.device('cuda:0')  # 使用可用的设备
#
#         else:
#             raise RuntimeError("CUDA 不可用，请确保 GPU 可用")
#
#         #split_idx = channel // 2  # 分割索引
#         self.efficient_mamba_attention = EfficientMambaAttention00(channel = in_channels,s2mlp_channels= out_channels).to(self.device)
#         self.coord_att_with_s2mlp = CoordAttWithS2MLP(inp=in_channels).to(self.device)
#         self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn = nn.BatchNorm2d(in_channels)
#
#         # 新增卷积层，调整 output1 的通道数
#         self.adjust_output1_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#
#         # # 将模型转移到设备上
#         # self.to(device)
#
#     def forward(self, x):
#         residual = x
#         # 通过 CoordAtt 和 S2MLP 增强注意力
#         a_h, a_w, a_h_mamba, a_w_mamba = self.coord_att_with_s2mlp(x)
#
#         # 通过 EfficientMambaAttention
#         x_x, x_y = self.efficient_mamba_attention(x,a_h_mamba, a_w_mamba)
#
#         # 交叉注意力计算
#         result_h = a_h * x_y
#         result_w = a_w * x_x
#
#         # 拼接 result_h 和 result_w
#         output1 = torch.cat([result_h, result_w], dim=1)
#
#         # 通过卷积层调整 output1 的通道数与 x 匹配
#         output1 = self.adjust_output1_channels(output1)
#
#         # 逐元素相乘，确保通道数一致
#         output = x * output1
#         return output
#
#
#
# # EfficientMambaAttention00 模块，保留原样
#
# class EfficientMambaAttention00(nn.Module):
#     def __init__(self, channel, s2mlp_channels=512):
#         super(EfficientMambaAttention00, self).__init__()
#
#         self.sigmoid_x = nn.Sigmoid()
#         self.avg_x = nn.AdaptiveAvgPool2d((None, 1))
#         self.avg_y = nn.AdaptiveAvgPool2d((1, None))
#         self.sigmoid_y = nn.Sigmoid()
#
#         self.mamba_x = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#         self.mamba_y = Mamba(
#             d_model=channel,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )
#
#         # 添加 S2MLP 模块
#         self.s2mlp_x = S2MLP(in_channels=channel,out_channels=s2mlp_channels)
#         self.s2mlp_y = S2MLP(in_channels=channel,out_channels=s2mlp_channels)
#
#     def forward(self, x, a_h_mamba, a_w_mamba):
#         b, c, h, w = x.size()
#
#         # 对 x 进行自适应池化
#         x_x = self.avg_x(x).squeeze(3).transpose(-1, -2)
#         x_ma = self.mamba_x(x_x).transpose(-1, -2)
#         x_x = self.sigmoid_x(x_ma).view(b, c, h, 1)
#         x_x = x_x + a_h_mamba
#         # 通过 S2MLP 增强 x_x
#         x_x = self.s2mlp_x(x_x)
#
#         x_y = self.avg_y(x).squeeze(2).transpose(-1, -2)
#         x_y = self.mamba_y(x_y).transpose(-1, -2)
#         x_y = self.sigmoid_y(x_y).view(b, c, 1, w)
#         x_y= x_y + a_w_mamba
#         # 通过 S2MLP 增强 x_y
#         x_y = self.s2mlp_y(x_y)
#
#         return x_x, x_y

#～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
# 0509   s2mlp--->mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# MLPBlock 模块替代原 S2MLP
class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):

        x = self.mlp(x)

        return x


# CoordAttWithMLP 模块
class CoordAttWithMLP(nn.Module):
    def __init__(self, inp, reduction=32, mlp_channels=512, device=None):
        super(CoordAttWithMLP, self).__init__()
        oup = inp
        self.device = device

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.GELU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.to(self.device)

    def forward(self, x):

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)


        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))


        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)
        a_h_mamba = a_h
        a_w_mamba = a_w


        return a_h.sigmoid(), a_w.sigmoid(), a_h_mamba, a_w_mamba


# EfficientMambaAttention00 模块
class EfficientMambaAttention00(nn.Module):
    def __init__(self, channel, mlp_channels=512):
        super(EfficientMambaAttention00, self).__init__()
        self.avg_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_y = nn.AdaptiveAvgPool2d((1, None))
        self.sigmoid = nn.Sigmoid()

        self.mamba_x = Mamba(d_model=channel, d_state=16, d_conv=4, expand=2)
        self.mamba_y = Mamba(d_model=channel, d_state=16, d_conv=4, expand=2)

        # self.mlp_x = MLPBlock(in_channels=channel, out_channels=mlp_channels)
        # self.mlp_y = MLPBlock(in_channels=channel, out_channels=mlp_channels)
        self.mlp_x = MLPBlock(in_channels=channel, out_channels=channel)
        self.mlp_y = MLPBlock(in_channels=channel, out_channels=channel)

    def forward(self, x, a_h_mamba, a_w_mamba):

        b, c, h, w = x.shape

        x_x = self.avg_x(x).squeeze(3).transpose(-1, -2)
        x_x = self.mamba_x(x_x).transpose(-1, -2).view(b, c, h, 1)
        x_x = self.sigmoid(x_x + a_h_mamba)

        x_x = self.mlp_x(x_x)

        x_y = self.avg_y(x).squeeze(2).transpose(-1, -2)
        x_y = self.mamba_y(x_y).transpose(-1, -2).view(b, c, 1, w)
        x_y = self.sigmoid(x_y + a_w_mamba)

        x_y = self.mlp_y(x_y)

        return x_x, x_y


# 主模块
class EfficientMambaAttention(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda:0'):
        super(EfficientMambaAttention, self).__init__()
        if torch.cuda.is_available():
            device = f"cuda:{device}" if isinstance(device, int) else device
            self.device = torch.device(device)
        else:
            raise RuntimeError("CUDA 不可用")

        self.coord_att_with_mlp = CoordAttWithMLP(inp=in_channels, device=self.device)
        # self.efficient_mamba_attention = EfficientMambaAttention00(
        #     channel=in_channels, mlp_channels=out_channels
        # ).to(self.device)
        self.efficient_mamba_attention = EfficientMambaAttention00(
            channel=in_channels, mlp_channels=in_channels  # 保持通道一致
        ).to(self.device)

        # self.adjust_output1_channels = nn.Conv2d(
        #     in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        # )
        # 主模块初始化中替换以下代码：
        # self.adjust_output1_channels = nn.Conv2d(
        #     in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        # )
        self.adjust_output1_channels = nn.Conv2d(
            in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):

        a_h, a_w, a_h_mamba, a_w_mamba = self.coord_att_with_mlp(x)

        x_x, x_y = self.efficient_mamba_attention(x, a_h_mamba, a_w_mamba)

        result_h = a_h * x_y
        result_w = a_w * x_x
        output1 = torch.cat([result_h, result_w], dim=1)

        output1 = self.adjust_output1_channels(output1)

        output = x * output1
        return output


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~fengexian~~~~~~~~~~~~~~
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k ** 2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc ** 0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(nn.Module):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass through RepBottleneck layer."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepCSP(nn.Module):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through RepCSP layer."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class DownSimper(nn.Module):
    """DownSimper."""

    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, self.c, 3, 2, d=3)
        self.cv2 = Conv(c1, self.c, 1, 1, 0)

    def forward(self, x):
        x1 = self.cv1(x)
        x = self.cv2(x)
        x2, x3 = x.chunk(2, 1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x3 = torch.nn.functional.avg_pool2d(x3, 3, 2, 1)

        return torch.cat((x1, x2, x3), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out
