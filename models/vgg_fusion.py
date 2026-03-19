

from typing import List, Tuple, Dict
import pandas as pd
import torch
import torch.nn as nn

from .utils import program_conv_filters
from .activation import get_activation_class
import torch.nn.functional  as F

vgg_layer_cfgs: Dict[str, List[Dict[str, int]]] = {
    "2D-VGG-11": [
        {"layers": 1, "channel_mul": 1},
        {"layers": 1, "channel_mul": 2},
        {"layers": 2, "channel_mul": 4},
        {"layers": 2, "channel_mul": 8},
        {"layers": 2, "channel_mul": 8},
    ],
    "2D-VGG-13": [
        {"layers": 2, "channel_mul": 1},
        {"layers": 2, "channel_mul": 2},
        {"layers": 2, "channel_mul": 4},
        {"layers": 2, "channel_mul": 8},
        {"layers": 2, "channel_mul": 8},
    ],
    "2D-VGG-16": [
        {"layers": 2, "channel_mul": 1},
        {"layers": 2, "channel_mul": 2},
        {"layers": 3, "channel_mul": 4},
        {"layers": 3, "channel_mul": 8},
        {"layers": 3, "channel_mul": 8},
    ],
    "2D-VGG-19": [
        {"layers": 2, "channel_mul": 1},
        {"layers": 2, "channel_mul": 2},
        {"layers": 4, "channel_mul": 4},
        {"layers": 4, "channel_mul": 8},
        {"layers": 4, "channel_mul": 8},
    ],
}


class TripletDataVerifier(nn.Module):
    """三重数据一致性验证器"""

    def forward(self, x_1d, x_2d, age, ids=None):
        batch_size = x_1d.size(0)

        # 维度验证
        assert x_2d.size(0) == batch_size, f"2D batch mismatch: {x_2d.size(0)}  vs {batch_size}"
        if age is not None:
            assert age.size(0) == batch_size, f"Age batch mismatch: {age.size(0)}  vs {batch_size}"

        # ID验证（如果有）
        if ids is not None:
            if isinstance(ids, dict):
                assert torch.all(ids["1d"] == ids["2d"]), "1D/2D ID mismatch"
                if "age" in ids:
                    assert torch.all(ids["1d"] == ids["age"]), "Age ID mismatch"
            else:
                assert len(ids) == batch_size, "ID length mismatch"
        return x_1d, x_2d, age


class AgeProjector(nn.Module):
    """年龄特征投影器"""

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )

    def forward(self, age):
        return self.net(age.unsqueeze(1).float())
class VGG1D2DFusion(nn.Module):
    def __init__(
            self,
            model: str,
            total_channels: int,  # 修改为接收总通道数

            out_dims: int,
            seq_len_1d: int,
            seq_len_2d: Tuple[int],
            use_age: str = "fc",
            base_channels: int = 64,
            dropout: float = 0.7,
            batch_norm: bool = True,
            fc_stages: int = 2,
            base_pool_1d: str = "max",
            base_pool_2d: str = "max",
            final_pool: str = "average",
            activation: str = "relu",
            fusion_method: str = "concat",
            age_embedding_dim: int = 32,
            enforce_triplet_pairing: bool = True,
            **kwargs,
    ):
        super().__init__()

        if seq_len_2d[0] % 2 != 0 or seq_len_2d[1] % 2 != 0:
            print(f"Warning: Input image dimensions {seq_len_2d} contain odd numbers. "
                  f"Padding to make them even.")
            self.pad2d = nn.ZeroPad2d((0, seq_len_2d[1] % 2, 0, seq_len_2d[0] % 2))
            seq_len_2d = (seq_len_2d[0] + seq_len_2d[0] % 2,
                          seq_len_2d[1] + seq_len_2d[1] % 2)
        else:
            self.pad2d = nn.Identity()
        # 保存初始输入通道数
        self.enforce_triplet_pairing = enforce_triplet_pairing
            self.in_channels_1d = int(total_channels * 0.4)  # 1D分支占40%
        self.in_channels_2d = total_channels - self.in_channels_1d  # 剩余给2D分支


        # 处理奇数通道情况
        if total_channels % 2 != 0:
            self.in_channels_1d += 1
            print(f"注意：通道数为奇数，1D分支多分配1个通道（{self.in_channels_1d}:{self.in_channels_2d} ）")
            # 验证分配结果
        if self.in_channels_1d <= 0 or self.in_channels_2d <= 0:
                raise ValueError(f"无效的通道分配: 1D={self.in_channels_1d},  2D={self.in_channels_2d}")

        print(f"通道分配比例 - 1D:{self.in_channels_1d}  ({(self.in_channels_1d / total_channels) * 100:.1f}%)  : "
                  f"2D:{self.in_channels_2d}  ({(self.in_channels_2d / total_channels) * 100:.1f}%)")

            # 参数验证
        if use_age not in ["fc", "conv", "no"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) receives one of ['fc', 'conv', 'no'].")
        if final_pool not in ["average", "max"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(final_pool) receives one of ['average', 'max'].")
        if fc_stages < 1:
            raise ValueError(f"{self.__class__.__name__}.__init__(fc_stages) receives an integer >= 1.")
        if fusion_method not in ["concat", "add"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(fusion_method) receives 'concat' or 'add'.")

        self.use_age = use_age
        self.age_embedding_dim = age_embedding_dim
        self.fusion_method = fusion_method
        self.fc_stages = fc_stages
        self.batch_norm = batch_norm
        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)

        # 年龄处理模块
        if self.use_age == "conv":
            self.in_channels_1d += 1
        elif self.use_age == "fc":
            self.age_projector = AgeProjector(age_embedding_dim)
        else:
            self.age_projector = None

        # 初始化池化层
        if final_pool == "average":
            self.final_pool_1d = nn.AdaptiveAvgPool1d(1)
            self.final_pool_2d = nn.AdaptiveAvgPool2d(1)
        else:
            self.final_pool_1d = nn.AdaptiveMaxPool1d(1)
            self.final_pool_2d = nn.AdaptiveMaxPool2d(1)

        # 初始化分支
        self._init_1d_branch(model, self.in_channels_1d, seq_len_1d, base_pool_1d, base_channels)
        self._init_2d_branch(model, self.in_channels_2d, seq_len_2d, base_pool_2d, base_channels)

        # 计算输出长度
        self._calculate_output_length(seq_len_1d, seq_len_2d)

        self.fusion_norm = nn.BatchNorm1d(self.output_length) if self.batch_norm else nn.Identity()
        self.fusion_dropout = nn.Dropout(p=0.5)
        self.fusion_dim = self.output_length + (self.age_embedding_dim if self.use_age == "fc" else 0)

        # 初始化全连接层
        self._init_fc_layers(out_dims, dropout, age_embedding_dim if use_age == "fc" else 0)

        self.reset_weights()
        self.triplet_verifier = TripletDataVerifier() if enforce_triplet_pairing else None

    def _calculate_output_length(self, seq_len_1d, seq_len_2d):
        """计算1D和2D分支的输出长度"""
        # 1D分支输出长度计算
        self.output_length_1d = self._calculate_1d_output_length(seq_len_1d)

        # 2D分支输出长度计算
        self.output_length_2d = self._calculate_2d_output_length(seq_len_2d)

        # 总输出长度
        if self.fusion_method == "concat":
            self.output_length = self.output_length_1d + self.output_length_2d
        else:
            self.output_length = max(self.output_length_1d, self.output_length_2d)

    def _calculate_1d_output_length(self, seq_len):
        """计算1D分支的输出长度"""
        with torch.no_grad():
            dummy_input = torch.randn(1, self.in_channels_1d, seq_len)
            x = self.conv_stage1_1d(dummy_input)
            x = self.conv_stage2_1d(x)
            x = self.conv_stage3_1d(x)
            x = self.conv_stage4_1d(x)
            x = self.conv_stage5_1d(x)
            x = self.final_pool_1d(x)  # 现在final_pool_1d已经被初始化
            return torch.flatten(x, 1).shape[1]

    def _calculate_2d_output_length(self, img_size):
        """计算2D分支的输出长度"""
        with torch.no_grad():
            print(f"Input image size: {img_size}")
            dummy_input = torch.randn(1, self.in_channels_2d, *img_size)

            def print_shape(name, x):
                print(f"{name} shape: {x.shape}")
                return x

            x = print_shape("input", dummy_input)
            x = print_shape("after stage1", self.conv_stage1_2d(x))
            x = print_shape("after stage2", self.conv_stage2_2d(x))
            x = print_shape("after stage3", self.conv_stage3_2d(x))
            x = print_shape("after stage4", self.conv_stage4_2d(x))
            x = print_shape("after stage5", self.conv_stage5_2d(x))
            x = print_shape("after final pool", self.final_pool_2d(x))

            return torch.flatten(x, 1).shape[1]




    def _init_fc_layers(self, out_dims, dropout, age_embedding_dim=0):
        """初始化全连接层，考虑年龄embedding"""
        input_dim = self.output_length
        if self.use_age == "fc":
            input_dim += age_embedding_dim

        current_dim = input_dim
        fc_stage = []
        for _ in range(self.fc_stages - 1):
            out_features = max(current_dim // 2, out_dims)
            layer = [
                nn.Linear(current_dim, out_features, bias=not self.batch_norm),
                nn.Dropout(p=dropout)
            ]

            if self.batch_norm:
                layer.append(nn.BatchNorm1d(out_features))

            layer.append(self.nn_act())
            fc_stage.append(nn.Sequential(*layer))
            current_dim = out_features

        fc_stage.append(nn.Linear(current_dim, out_dims, bias=True))
        self.fc_layers = nn.ModuleList(fc_stage)
        self.fc_stage = nn.Sequential(*self.fc_layers)

    def _standardize_input(self, x, age, sample_ids):
        """将不同输入格式标准化为四元组输出"""
        if isinstance(x, dict):
            if 'signal' in x:  # 原始信号格式
                signal = x['signal']
                # 确保信号是4D的 [batch, channels, height, width]
                if signal.dim() == 3:  # 如果是3D [batch, channels, time]
                    # 需要将时间维度reshape为空间维度
                    # 这里假设原始数据可以reshape为 [batch, channels, h, w]
                    # 需要根据实际数据情况调整
                    h = int(signal.shape[2] ** 0.5)
                    w = signal.shape[2] // h
                    if h * w != signal.shape[2]:
                        w += 1  # 补零填充
                        signal = F.pad(signal, (0, h * w - signal.shape[2]))
                    signal = signal.view(signal.shape[0], signal.shape[1], h, w)

                split_point = int(signal.shape[1] * 0.4)  # 80-20分割
                return (
                    signal[:, :split_point].flatten(start_dim=2),  # 1d [batch, channels_1d, time]
                    signal[:, split_point:],  # 2d [batch, channels_2d, height, width]
                    x.get('age', age),
                    x.get('ids', sample_ids)
                )
            elif all(k in x for k in ["1d", "2d"]):  # 预分割格式
                # 确保2d部分是4D
                x_2d = x["2d"]
                if (x_2d.dim()

                        == 3):
                    h = int(x_2d.shape[2] ** 0.5)
                    w = x_2d.shape[2] // h
                    if h * w != x_2d.shape[2]:
                        w += 1
                        x_2d = F.pad(x_2d, (0, h * w - x_2d.shape[2]))
                    x_2d = x_2d.view(x_2d.shape[0], x_2d.shape[1], h, w)
                return (
                    x["1d"],  # 1d
                    x_2d,  # 2d
                    x.get("age", age),
                    x.get("ids", sample_ids)
                )
            else:
                raise ValueError(
                    "字典输入必须包含'signal'或'1d'和'2d'键\n"
                    f"当前字典键: {list(x.keys())}"
                )
        elif isinstance(x, (tuple, list)):  # 元组/列表格式
            if len(x) not in [2, 3]:
                raise ValueError(
                    "元组/列表输入必须包含2或3个元素\n"
                    "(1d_tensor, 2d_tensor[, age_tensor])\n"
                    f"实际长度: {len(x)}"
                )
            # 处理2d部分
            x_2d = x[1]
            if x_2d.dim() == 3:
                h = int(x_2d.shape[2] ** 0.5)
                w = x_2d.shape[2] // h
                if h * w != x_2d.shape[2]:
                    w += 1
                    x_2d = F.pad(x_2d, (0, h * w - x_2d.shape[2]))
                x_2d = x_2d.view(x_2d.shape[0], x_2d.shape[1], h, w)
            return (
                x[0],  # 1d
                x_2d,  # 2d
                x[2] if len(x) == 3 else age,
                sample_ids
            )
        elif torch.is_tensor(x):  # 单一tensor输入
            if x.dim() == 3:
                h = int(x.shape[2] ** 0.5)
                w = x.shape[2] // h
                if h * w != x.shape[2]:
                    w += 1
                    x = F.pad(x, (0, h * w - x.shape[2]))
                x = x.view(x.shape[0], x.shape[1], h, w)
            split_point = int(x.shape[1] * 0.4)
            return (
                x[:, :split_point].flatten(start_dim=2),  # 1d
                x[:, split_point:],  # 2d
                age,
                sample_ids
            )
        else:
            raise TypeError(
                "输入必须是以下类型之一:\n"
                "- 包含'signal'键的字典\n"
                "- 包含'1d'和'2d'键的字典\n"
                "- (1d_tensor, 2d_tensor[, age_tensor])元组/列表\n"
                "- 单一tensor(将自动分割)\n"
                f"实际类型: {type(x)}"
            )

    def forward_embeddings(self, x, age=None, sample_ids=None, apply_dropout=True):
        try:
            x_1d, x_2d, age, sample_ids = self._standardize_input(x, age, sample_ids)
        except Exception as e:
            input_info = f"输入类型: {type(x)}"
            if hasattr(x, 'shape'):
                input_info += f", 形状: {x.shape}"
            elif isinstance(x, dict):
                input_info += f", 键: {list(x.keys())}"
            raise ValueError(
                f"输入标准化失败 - {str(e)}\n"
                f"{input_info}\n"
                f"模型期望输入格式说明:\n"
                f"- 1D部分: 3D [batch, channels, time]\n"
                f"- 2D部分: 4D [batch, channels, height, width]\n"
            ) from e

        assert x_1d.dim() == 3, f"1D输入应为3D [batch, channels, time], 实际得到: {x_1d.shape}"
        assert x_2d.dim() == 4, f"2D输入应为4D [batch, channels, height, width], 实际得到: {x_2d.shape}"

        self._validate_inputs(x_1d, x_2d, age)

        if self.triplet_verifier is not None:
            self.triplet_verifier(x_1d, x_2d, age, sample_ids)

        age_feature = None
        if self.use_age != "no" and age is not None:
            if self.use_age == "fc":
                age_feature = self.age_projector(age)
            elif self.use_age == "conv":
                N, _, L = x_1d.size()
                age = age.view(N, 1, 1).expand(N, 1, L)
                x_1d = torch.cat([x_1d, age], dim=1)

        x_1d, x_2d = self._encode_modalities(x_1d, x_2d)
        x = self._fuse_modalities(x_1d, x_2d)
        x = self.fusion_norm(x)
        x = self.fusion_dropout(x) if apply_dropout else x

        if self.use_age == "fc" and age_feature is not None:
            x = torch.cat([x, age_feature], dim=1)

        return x

    def forward(self, x, age=None, sample_ids=None):
        embeddings = self.forward_embeddings(x, age, sample_ids)
        return self.fc_stage(embeddings)

    def forward_branch_features(self, x, age=None, sample_ids=None):
        """
        返回三路分支特征（已经过各自 encoder + pooling + flatten）
        - feat_1d: [N, D1]
        - feat_2d: [N, D2]
        - feat_age: [N, Da] or None（仅 use_age == 'fc'）
        """
        x_1d, x_2d, age, sample_ids = self._standardize_input(x, age, sample_ids)
        self._validate_inputs(x_1d, x_2d, age)

        if self.triplet_verifier is not None:
            self.triplet_verifier(x_1d, x_2d, age, sample_ids)

        feat_age = None
        if self.use_age != "no" and age is not None:
            if self.use_age == "fc":
                feat_age = self.age_projector(age)
            elif self.use_age == "conv":
                N, _, L = x_1d.size()
                age_ = age.view(N, 1, 1).expand(N, 1, L)
                x_1d = torch.cat([x_1d, age_], dim=1)

        feat_1d, feat_2d = self._encode_modalities(x_1d, x_2d)
        return feat_1d, feat_2d, feat_age

    def logits_from_branch_features(self, feat_1d, feat_2d, feat_age=None, apply_dropout=False):
        """
        给定分支特征，走融合+分类头输出 logits。
        """
        x = self._fuse_modalities(feat_1d, feat_2d)
        x = self.fusion_norm(x)
        x = self.fusion_dropout(x) if apply_dropout else x
        if self.use_age == "fc" and feat_age is not None:
            x = torch.cat([x, feat_age], dim=1)
        return self.fc_stage(x)

    def _encode_modalities(self, x_1d, x_2d):
        x_1d = self.conv_stage1_1d(x_1d)
        x_1d = self.conv_stage2_1d(x_1d)
        x_1d = self.conv_stage3_1d(x_1d)
        x_1d = self.conv_stage4_1d(x_1d)
        x_1d = self.conv_stage5_1d(x_1d)
        x_1d = self.final_pool_1d(x_1d)
        x_1d = torch.flatten(x_1d, 1)

        x_2d = self.pad2d(x_2d)
        x_2d = self.conv_stage1_2d(x_2d)
        x_2d = self.conv_stage2_2d(x_2d)
        x_2d = self.conv_stage3_2d(x_2d)
        x_2d = self.conv_stage4_2d(x_2d)
        x_2d = self.conv_stage5_2d(x_2d)
        x_2d = self.final_pool_2d(x_2d)
        x_2d = torch.flatten(x_2d, 1)
        return x_1d, x_2d

    def _fuse_modalities(self, x_1d, x_2d):

        if self.fusion_method == "concat":
            parts = [x_1d, x_2d]  # 关键顺序/比例策略轻度隐藏：****
            return torch.cat(parts, dim=1)
        assert x_1d.shape == x_2d.shape, "For 'add' fusion, features must have same shape"
        return x_1d + x_2d




    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear,)):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _init_2d_branch(self, model, in_channels, img_size, base_pool, base_channels):
        """Initialize the 2D branch for image data"""
        self.base_pool_2d = nn.MaxPool2d if base_pool == "max" else nn.AvgPool2d
        self.current_channels_2d = in_channels  # 初始化当前通道数

        # 硬编码的2D卷积参数
        # 修改2D卷积参数，确保尺寸匹配
        conv_filter_list = [
            {"kernel_size": 3, "stride": 1, "pool": 2},  # 第一层
            {"kernel_size": 3, "stride": 1, "pool": 2},  # 第二层
            {"kernel_size": 3, "stride": 1, "pool": 2},  # 第三层
            {"kernel_size": 3, "stride": 1, "pool": 2},  # 第四层
            {"kernel_size": 3, "stride": 1, "pool": 2}  # 第五层
        ]


        self.current_channels_2d = in_channels
        self.conv_stage1_2d = self._make_conv_stage(conv_filter_list[0], vgg_layer_cfgs[model][0], base_channels)
        self.conv_stage2_2d = self._make_conv_stage(conv_filter_list[1], vgg_layer_cfgs[model][1], base_channels)
        self.conv_stage3_2d = self._make_conv_stage(conv_filter_list[2], vgg_layer_cfgs[model][2], base_channels)
        self.conv_stage4_2d = self._make_conv_stage(conv_filter_list[3], vgg_layer_cfgs[model][3], base_channels)
        self.conv_stage5_2d = self._make_conv_stage(conv_filter_list[4], vgg_layer_cfgs[model][4], base_channels)

    class ResidualBlock(nn.Module):
        def __init__(self, conv_layers, shortcut, activation):
            super().__init__()
            self.conv_layers = nn.Sequential(*conv_layers)
            self.shortcut = shortcut
            self.activation = activation

        def forward(self, x):
            return self.activation(self.conv_layers(x) + self.shortcut(x))

    def _make_conv_stage(self, conv_filter, cfg, base_channels):
        conv_layers = []
        in_channels = self.current_channels_2d
        out_channels = cfg["channel_mul"] * base_channels

        # 计算主路径的输出尺寸
        for k in range(cfg["layers"]):
            stride = conv_filter["stride"] if k == 0 else 1
            if k == 0 and conv_filter["pool"] > 1:
                # 将池化层合并到第一个卷积的stride中
                stride *= conv_filter["pool"]

            conv_layers.extend([
                nn.Conv2d(
                    in_channels if k == 0 else out_channels,
                    out_channels,
                    kernel_size=conv_filter["kernel_size"],
                    padding=conv_filter["kernel_size"] // 2,
                    stride=stride,
                    bias=not self.batch_norm,
                ),
                nn.BatchNorm2d(out_channels) if self.batch_norm else nn.Identity(),
                self.nn_act(),
            ])

        # shortcut路径 - 确保与主路径相同的下采样
        if in_channels != out_channels:
            shortcut_stride = conv_filter["stride"] * conv_filter["pool"] if conv_filter["pool"] > 1 else conv_filter[
                "stride"]
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=shortcut_stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 如果通道数相同，但仍需要下采样
            if conv_filter["pool"] > 1:
                shortcut = nn.Sequential(
                    self.base_pool_2d(conv_filter["pool"]),
                    nn.Identity()
                )
            else:
                shortcut = nn.Identity()

        # 更新当前通道数
        self.current_channels_2d = out_channels

        print(f"Stage config - in: {in_channels}, out: {out_channels}, "
              f"stride: {conv_filter['stride']}, pool: {conv_filter['pool']}")

        return self.ResidualBlock(nn.Sequential(*conv_layers), shortcut, self.nn_act())


    def _init_1d_branch(self, model, in_channels, seq_len, base_pool, base_channels):
        """Initialize the 1D branch for EEG data"""
        self.base_pool_1d = nn.MaxPool1d if base_pool == "max" else nn.AvgPool1d

        # 确保 conv_filter_list 是正确的结构
        conv_filter_list = [{"kernel_size": 9} for _ in vgg_layer_cfgs[model]]

        # 调用 program_conv_filters 并确保它返回列表
        program_conv_filters(
            sequence_length=seq_len,
            conv_filter_list=conv_filter_list,
            output_lower_bound=4,
            output_upper_bound=8,
            class_name=self.__class__.__name__
        )

        # 直接使用预定义的 conv_filter_list，而不是返回值
        self.current_channels_1d = in_channels
        self.conv_stage1_1d = self._make_conv_stage_1d(conv_filter_list[0], vgg_layer_cfgs[model][0], base_channels)
        self.conv_stage2_1d = self._make_conv_stage_1d(conv_filter_list[1], vgg_layer_cfgs[model][1], base_channels)
        self.conv_stage3_1d = self._make_conv_stage_1d(conv_filter_list[2], vgg_layer_cfgs[model][2], base_channels)
        self.conv_stage4_1d = self._make_conv_stage_1d(conv_filter_list[3], vgg_layer_cfgs[model][3], base_channels)
        self.conv_stage5_1d = self._make_conv_stage_1d(conv_filter_list[4], vgg_layer_cfgs[model][4], base_channels)

    def _make_conv_stage_1d(self, conv_filter, cfg, base_channels):
        conv_layers: List[nn.Module] = []

        if conv_filter["pool"] > 1:
            conv_layers += [self.base_pool_1d(conv_filter["pool"])]

        for k in range(cfg["layers"]):
            if k == 0:
                stride = conv_filter["stride"]
            else:
                stride = 1

            if self.batch_norm:
                conv_layers += [
                    nn.Conv1d(
                        in_channels=self.current_channels_1d,  # 使用 current_channels_1d
                        out_channels=cfg["channel_mul"] * base_channels,
                        kernel_size=conv_filter["kernel_size"],
                        padding=conv_filter["kernel_size"] // 2,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(cfg["channel_mul"] * base_channels),
                    self.nn_act(),
                ]
            else:
                conv_layers += [
                    nn.Conv1d(
                        in_channels=self.current_channels_1d,  # 使用 current_channels_1d
                        out_channels=cfg["channel_mul"] * base_channels,
                        kernel_size=conv_filter["kernel_size"],
                        padding=conv_filter["kernel_size"] // 2,
                        stride=stride,
                        bias=True,
                    ),
                    self.nn_act(),
                ]

            self.current_channels_1d = cfg["channel_mul"] * base_channels  # 更新 current_channels_1d
        return nn.Sequential(*conv_layers)




    def get_output_length(self):
        return self.output_length

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        x = self.forward_embeddings(x, age, apply_dropout=False)

        if target_from_last == 0:
            return self.fc_stage(x)

        if target_from_last > len(self.fc_layers):
            raise ValueError(
                f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                f"an integer equal to or smaller than fc_stages={len(self.fc_layers)}."
            )

        for layer in list(self.fc_layers)[: len(self.fc_layers) - target_from_last]:
            x = layer(x)
        return x

    def get_fusion_embedding_dim(self):
        return self.fusion_dim

