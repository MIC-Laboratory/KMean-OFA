# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from ofa.utils.layers import (
    MBConvLayer,
    ConvLayer,
    IdentityLayer,
    set_layer_from_config,
)
from ofa.utils.layers import ResNetBottleneckBlock, LinearLayer
from ofa.utils import (
    MyModule,
    val2list,
    get_net_device,
    build_activation,
    make_divisible,
    SEModule,
    MyNetwork,
)
from .dynamic_op import (
    DynamicSeparableConv2d,
    DynamicConv2d,
    DynamicBatchNorm2d,
    DynamicSE,
    DynamicGroupNorm,
)
from .dynamic_op import DynamicLinear

__all__ = [
    "adjust_bn_according_to_idx",
    "copy_bn",
    "DynamicMBConvLayer",
    "DynamicConvLayer",
    "DynamicLinearLayer",
    "DynamicResNetBottleneckBlock",
]


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def copy_bn(target_bn, src_bn):
    feature_dim = (
        target_bn.num_channels
        if isinstance(target_bn, nn.GroupNorm)
        else target_bn.num_features
    )

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


class DynamicLinearLayer(MyModule):
    def __init__(self, in_features_list, out_features, bias=True, dropout_rate=0):
        super(DynamicLinearLayer, self).__init__()

        self.in_features_list = in_features_list
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list),
            max_out_features=self.out_features,
            bias=self.bias,
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return "DyLinear(%d, %d)" % (max(self.in_features_list), self.out_features)

    @property
    def config(self):
        return {
            "name": DynamicLinear.__name__,
            "in_features_list": self.in_features_list,
            "out_features": self.out_features,
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        sub_layer = LinearLayer(
            in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate
        )
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.weight.data.copy_(
            self.linear.get_active_weight(self.out_features, in_features).data
        )
        if self.bias:
            sub_layer.linear.bias.data.copy_(
                self.linear.get_active_bias(self.out_features).data
            )
        return sub_layer

    def get_active_subnet_config(self, in_features):
        return {
            "name": LinearLayer.__name__,
            "in_features": in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
        }


class DynamicMBConvLayer(MyModule):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size_list=3,
        expand_ratio_list=1,
        stride=1,
        act_func="relu6",
        use_se=False,
        expansion=1
    ):
        super(DynamicMBConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = val2list(kernel_size_list)
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.in_channel_list) * max(self.expand_ratio_list) * expansion),
            MyNetwork.CHANNEL_DIVISIBLE,
        )
        # Display this if, always go to else
        if False and max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            DynamicConv2d(
                                max(self.in_channel_list), max_middle_channel
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max_middle_channel)),
                        ("act", build_activation(self.act_func)),
                    ]
                )
            )

        self.depth_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicSeparableConv2d(
                            max_middle_channel, self.kernel_size_list, self.stride
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func)),
                ]
            )
        )
        if self.use_se:
            self.depth_conv.add_module("se", DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(max_middle_channel, max(self.out_channel_list)),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                ]
            )
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.shortcut = nn.Sequential()
        if stride == 1 and max(self.in_channel_list) != max(self.out_channel_list):
            self.shortcut = nn.Sequential(
                nn.Conv2d(max(self.in_channel_list), max(self.out_channel_list), kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(max(self.out_channel_list)),
            )
    def forward(self, x):
        in_channel = x.size(1)
        self.inverted_bottleneck.conv.active_out_channel = make_divisible(
            round(self.inverted_bottleneck.conv.active_out_channel * self.active_expand_ratio),
            MyNetwork.CHANNEL_DIVISIBLE,
        )

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        out = self.inverted_bottleneck(x)
        out = self.depth_conv(out)
        out = self.point_linear(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

    @property
    def module_str(self):
        if self.use_se:
            return "SE(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )
        else:
            return "(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )

    @property
    def config(self):
        return {
            "name": DynamicMBConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "expand_ratio_list": self.expand_ratio_list,
            "stride": self.stride,
            "act_func": self.act_func,
            "use_se": self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMBConvLayer(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    def active_middle_channel(self, in_channel):
        return make_divisible(
            round(in_channel * self.active_expand_ratio), MyNetwork.CHANNEL_DIVISIBLE
        )

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        middle_channel = self.active_middle_channel(in_channel)
        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.get_active_filter(
                    middle_channel, in_channel
                ).data,
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(
                middle_channel, self.active_kernel_size
            ).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(
                middle_channel // SEModule.REDUCTION,
                divisor=MyNetwork.CHANNEL_DIVISIBLE,
            )
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.get_active_reduce_weight(se_mid, middle_channel).data
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
                self.depth_conv.se.get_active_reduce_bias(se_mid).data
            )

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.get_active_expand_weight(se_mid, middle_channel).data
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
                self.depth_conv.se.get_active_expand_bias(middle_channel).data
            )

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.get_active_filter(
                self.active_out_channel, middle_channel
            ).data
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            "name": MBConvLayer.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.active_kernel_size,
            "stride": self.stride,
            "expand_ratio": self.active_expand_ratio,
            "mid_channels": self.active_middle_channel(in_channel),
            "act_func": self.act_func,
            "use_se": self.use_se,
        }
    def Kmean(self,weight,sort_index,k,output_channel):
        """
        Apply K-means clustering to perform filter pruning based on similarity in weight vectors.

        Args:
            weight: The weight tensor of the layer to be pruned.
            sort_index: The sorted indices of the weights.
            k: The number of clusters for K-means.
            output_channel: The number of output channels.

        Return:
            pruning_index_group: The indices of filters to be pruned.

        Logic:
            1. Determine the number of filters to be removed based on the output channel size.
            2. Reshape the weight tensor into a 2D matrix.
            3. Perform dimensionality reduction using PCA to reduce the dimensionality of weight vectors.
            4. Apply K-means clustering to the reduced weight vectors.
            5. Group the filters based on the K-means labels obtained.
            6. Prune filters from each group based on their importance and the required pruning amount.
                - Iterate over each group and calculate the pruning amount 
                  by multiplying the removal ratio with the total number of filters in the group.
                - Sort the indices of each group based on the specified sorted order, 
                  ensuring the original indices are preserved.
                - Select filters for pruning by popping from the end of the sorted indices until the 
                  desired pruning amount is reached.
            7. Return the indices of the pruned filters.

        
        """
        
        import time
        start = time.time()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_filter = weight.shape[0]
        remove_filter = num_filter - output_channel
        if k == 1:
            return sort_index[output_channel:]
            
        
        m_weight_vector = weight.reshape(num_filter, -1)
        
        
        n_clusters = k

        kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init='auto').fit(m_weight_vector)
        
        # print("K:",n_clusters)
        labels = kmeans.labels_
        group = [[] for _ in range(n_clusters)]
        for idx in range(num_filter):
            group[labels[idx]].append(idx)
        lock_group_index = []
        copy_group = copy.deepcopy(group)
        for filter_index_group in copy_group:
            if len(filter_index_group) == 1:
                group.remove(filter_index_group)

        # The reminding item in group can be pruned by some crition
        pruning_index_group = []
        pruning_left_index_group = [[] for _ in range(len(group))]
        total_left_filter = sum(len(filter_index_group)
                                for filter_index_group in group)
        percentage_group = [int(
            100*(len(filter_index_group)/total_left_filter)) for filter_index_group in group]
        pruning_amount_group = [
            int(remove_filter*(percentage/100)) for percentage in percentage_group]
        sorted_idx_origin = copy.deepcopy(sort_index)
        for counter, filter_index_group in enumerate(group, 0):
            temp = copy.deepcopy(filter_index_group)
            temp.sort(key=lambda e: (list(sorted_idx_origin).index(e),e) if e in list(sorted_idx_origin)  else (len(list(sorted_idx_origin)),e))
            sorted_idx = torch.tensor(temp,device=device)
            filetr_index_group_temp = copy.deepcopy(list(sorted_idx))
            
            for sub_index in sorted_idx[len(sorted_idx)-pruning_amount_group[counter]:]:
                if len(filetr_index_group_temp) == 1:
                    continue
                pruning_index_group.append(filetr_index_group_temp.pop(filetr_index_group_temp.index(sub_index)))
            for left_index in filetr_index_group_temp:
                pruning_left_index_group[counter].append(left_index)
        # first one is the least important weight and the last one is the most important weight
        while (len(pruning_index_group) < remove_filter):
            pruning_amount = len(pruning_index_group)
            for left_index in pruning_left_index_group:
                if (len(left_index) <= 1):
                    continue
                if (len(pruning_index_group) >= remove_filter):
                    break
                pruning_index_group.append(left_index.pop(-1))
            if (pruning_amount >= len(pruning_index_group)):
                raise ValueError('infinity loop')
        return torch.tensor(pruning_index_group).to(device)

    def L1norm_pruning(self,weight):
        """
        Apply L1 norm pruning to the given layer.

        Args:
            layer: The layer to be pruned.

        Return:
            sorted_indices: The sorted indices of important filters.

        Logic:
        1. Clone the weight data of the layer to avoid modifying the original weights.
        2. Check the shape of the weight tensor to determine if it is a convolutional layer.
        3. Calculate the importance of each filter by summing the absolute values of the weights along the appropriate dimensions.
           - For a 4-dimensional weight tensor, the sum is calculated along dimensions (1, 2, 3).
           - For a 2-dimensional weight tensor, the sum is calculated along dimension 0 (channels).
        4. Sort the importance values and obtain the corresponding indices in descending order.
        5. Return the sorted indices

    Note:
        The importance values indicate the amount of contribution of each filter to the overall model's performance.
        Filters with higher importance values are considered more significant and likely to be kept during pruning.
        """
        # weight = layer.weight.data.clone()
        weight = torch.tensor(weight)
        
        if len(weight.shape) == 4:
            importance = torch.sum(torch.abs(weight),dim=(1,2,3))
        else:
            importance = torch.sum(torch.abs(weight),dim=0)
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        return sorted_idx

    def Kmean_L1norm(self,weight,k):
        

        """
        Apply K-L1norm pruning to the given layer.

        Args:
            layer: The layer to be pruned.

        Return:
            sorted_idx: The sorted indices of the most important filters.

        Logic:
            1. Clone the weight data of the layer to avoid modifying the original weights.
            2. Sort layer filter by the L1 norm and obtain the corresponding indices.
            3. Return the sorted indices, representing the most important filters based on K-L1norm.
        """
        # weight = layer.weight.data.clone()
        sort_index = self.L1norm_pruning(weight)
        output_channel = int(weight.shape[0] * 0.5)
        print(output_channel)
        weight = weight.reshape(weight.shape[0],-1)
        
        pca = PCA(n_components=0.8).fit(weight)
        
        weight = pca.fit_transform(weight)
        
        
        
        
        pruning_index =  self.Kmean(weight,sort_index,k,output_channel)
        """
        using l1norm to sort the pruning index, and put them to the end of sorted_idx
        indicate they are not important
        However, experiment find out it doesn't help
        so I comment out them
        """
        # pruning_weight = weight[pruning_index,:,:,:]
        # important = torch.sum(torch.abs(pruning_weight),dim=(1,2,3))
        # pruning_weight,pruning_index = torch.sort(important)
        keep_index = [i.item() for i in sort_index if i not in pruning_index]
        keep_index = torch.as_tensor(keep_index,device=self.device)
        pruning_index = torch.as_tensor(pruning_index,device=self.device)
        return torch.cat((keep_index,pruning_index)).type(torch.IntTensor).to(self.device)
    def re_organize_middle_weights(self, expand_ratio_stage=0,k=1):
        # importance = torch.sum(
        #     torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3)
        # )
        # if isinstance(self.depth_conv.bn, DynamicGroupNorm):
        #     channel_per_group = self.depth_conv.bn.channel_per_group
        #     importance_chunks = torch.split(importance, channel_per_group)
        #     for chunk in importance_chunks:
        #         chunk.data.fill_(torch.mean(chunk))
        #     importance = torch.cat(importance_chunks, dim=0)
        # if expand_ratio_stage > 0:
        #     sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
        #     sorted_expand_list.sort(reverse=True)
        #     target_width_list = [
        #         make_divisible(
        #             round(max(self.in_channel_list) * expand),
        #             MyNetwork.CHANNEL_DIVISIBLE,
        #         )
        #         for expand in sorted_expand_list
        #     ]

        #     right = len(importance)
        #     base = -len(target_width_list) * 1e5
        #     for i in range(expand_ratio_stage + 1):
        #         left = target_width_list[i]
        #         importance[left:right] += base
        #         base += 1e5
        #         right = left
        
        # sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

        sorted_idx = self.Kmean_L1norm(self.conv2.conv.conv.weight.data.clone().cpu().detach().numpy(),k)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )

        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.use_se:
            # se expand: output dim 0 reorganize
            se_expand = self.depth_conv.se.fc.expand
            se_expand.weight.data = torch.index_select(
                se_expand.weight.data, 0, sorted_idx
            )
            se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
            # se reduce: input dim 1 reorganize
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce.weight.data = torch.index_select(
                se_reduce.weight.data, 1, sorted_idx
            )
            # middle weight reorganize
            se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
            se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

            se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
            se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)

        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            return sorted_idx


class DynamicConvLayer(MyModule):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bn=True,
        act_func="relu6",
    ):
        super(DynamicConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act = build_activation(self.act_func)

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    @property
    def module_str(self):
        return "DyConv(O%d, K%d, S%d)" % (
            self.active_out_channel,
            self.kernel_size,
            self.stride,
        )

    @property
    def config(self):
        return {
            "name": DynamicConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(
            self.conv.get_active_filter(self.active_out_channel, in_channel).data
        )
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            "name": ConvLayer.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }


class DynamicResNetBottleneckBlock(MyModule):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        expand_ratio_list=0.25,
        kernel_size=3,
        stride=1,
        act_func="relu",
        downsample_mode="avgpool_conv",
    ):
        super(DynamicResNetBottleneckBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)),
            MyNetwork.CHANNEL_DIVISIBLE,
        )
        # max_middle_channel = max(self.out_channel_list)
        # self.max_middle_channel = val2list(max_middle_channel)
        # self.out_channel_list = [make_divisible(
        #     round(max(self.out_channel_list) * max(self.expand_ratio_list)),
        #     MyNetwork.CHANNEL_DIVISIBLE,
        # )]

        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(max(self.in_channel_list), max_middle_channel),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(
                            max_middle_channel, max_middle_channel, kernel_size, stride
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )

        self.conv3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicConv2d(max_middle_channel, max(self.out_channel_list)*4),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list)*4)),
                ]
            )
        )

        if self.stride == 1 and max(self.in_channel_list) == max(self.out_channel_list)*4:
            self.downsample = IdentityLayer(
                max(self.in_channel_list), max(self.out_channel_list)*4
            )
        elif self.downsample_mode == "conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            DynamicConv2d(
                                max(self.in_channel_list),
                                max(self.out_channel_list)*4,
                                stride=stride,
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max(self.out_channel_list)*4)),
                    ]
                )
            )
        elif self.downsample_mode == "avgpool_conv":
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "avg_pool",
                            nn.AvgPool2d(
                                kernel_size=stride,
                                stride=stride,
                                padding=0,
                                ceil_mode=True,
                            ),
                        ),
                        (
                            "conv",
                            DynamicConv2d(
                                max(self.in_channel_list), max(self.out_channel_list)
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                    ]
                )
            )
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def forward(self, x):
        feature_dim = self.active_middle_channels

        self.conv1.conv.active_out_channel = self.active_out_channel
        self.conv2.conv.active_out_channel = feature_dim
        self.conv3.conv.active_out_channel = max(self.out_channel_list)*4
        if not isinstance(self.downsample, IdentityLayer):
            self.downsample.conv.active_out_channel = max(self.out_channel_list)*4

        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return "(%s, %s)" % (
            "%dx%d_BottleneckConv_in->%d->%d_S%d"
            % (
                self.kernel_size,
                self.kernel_size,
                self.active_middle_channels,
                self.active_out_channel,
                self.stride,
            ),
            "Identity"
            if isinstance(self.downsample, IdentityLayer)
            else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            "name": DynamicResNetBottleneckBlock.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "expand_ratio_list": self.expand_ratio_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "act_func": self.act_func,
            "downsample_mode": self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicResNetBottleneckBlock(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        return feature_dim

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        sub_layer.conv1.conv.weight.data.copy_(
            self.conv1.conv.get_active_filter(
                self.active_out_channel, in_channel
            ).data
        )
        copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

        sub_layer.conv2.conv.weight.data.copy_(
            self.conv2.conv.get_active_filter(
                self.active_middle_channels, self.active_out_channel
            ).data
        )
        copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        sub_layer.conv3.conv.weight.data.copy_(
            self.conv3.conv.get_active_filter(
                self.active_out_channel*4, self.active_middle_channels
            ).data
        )
        copy_bn(sub_layer.conv3.bn, self.conv3.bn.bn)

        if not isinstance(self.downsample, IdentityLayer):
            sub_layer.downsample.conv.weight.data.copy_(
                self.downsample.conv.get_active_filter(
                    self.active_out_channel*4, in_channel
                ).data
            )
            copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            "name": ResNetBottleneckBlock.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.active_expand_ratio,
            "mid_channels": self.active_out_channel,
            "act_func": self.act_func,
            "groups": 1,
            "downsample_mode": self.downsample_mode,
        }

    def Kmean(self,weight,sort_index,k,output_channel):
        """
        Apply K-means clustering to perform filter pruning based on similarity in weight vectors.

        Args:
            weight: The weight tensor of the layer to be pruned.
            sort_index: The sorted indices of the weights.
            k: The number of clusters for K-means.
            output_channel: The number of output channels.

        Return:
            pruning_index_group: The indices of filters to be pruned.

        Logic:
            1. Determine the number of filters to be removed based on the output channel size.
            2. Reshape the weight tensor into a 2D matrix.
            3. Perform dimensionality reduction using PCA to reduce the dimensionality of weight vectors.
            4. Apply K-means clustering to the reduced weight vectors.
            5. Group the filters based on the K-means labels obtained.
            6. Prune filters from each group based on their importance and the required pruning amount.
                - Iterate over each group and calculate the pruning amount 
                  by multiplying the removal ratio with the total number of filters in the group.
                - Sort the indices of each group based on the specified sorted order, 
                  ensuring the original indices are preserved.
                - Select filters for pruning by popping from the end of the sorted indices until the 
                  desired pruning amount is reached.
            7. Return the indices of the pruned filters.

        
        """
        
        import time
        start = time.time()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_filter = weight.shape[0]
        remove_filter = num_filter - output_channel
        if k == 1:
            return sort_index[output_channel:]
            
        
        m_weight_vector = weight.reshape(num_filter, -1)
        
        
        n_clusters = k

        kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init='auto').fit(m_weight_vector)
        
        # print("K:",n_clusters)
        labels = kmeans.labels_
        group = [[] for _ in range(n_clusters)]
        for idx in range(num_filter):
            group[labels[idx]].append(idx)
        lock_group_index = []
        copy_group = copy.deepcopy(group)
        for filter_index_group in copy_group:
            if len(filter_index_group) == 1:
                group.remove(filter_index_group)

        # The reminding item in group can be pruned by some crition
        pruning_index_group = []
        pruning_left_index_group = [[] for _ in range(len(group))]
        total_left_filter = sum(len(filter_index_group)
                                for filter_index_group in group)
        percentage_group = [int(
            100*(len(filter_index_group)/total_left_filter)) for filter_index_group in group]
        pruning_amount_group = [
            int(remove_filter*(percentage/100)) for percentage in percentage_group]
        sorted_idx_origin = copy.deepcopy(sort_index)
        for counter, filter_index_group in enumerate(group, 0):
            temp = copy.deepcopy(filter_index_group)
            temp.sort(key=lambda e: (list(sorted_idx_origin).index(e),e) if e in list(sorted_idx_origin)  else (len(list(sorted_idx_origin)),e))
            sorted_idx = torch.tensor(temp,device=device)
            filetr_index_group_temp = copy.deepcopy(list(sorted_idx))
            
            for sub_index in sorted_idx[len(sorted_idx)-pruning_amount_group[counter]:]:
                if len(filetr_index_group_temp) == 1:
                    continue
                pruning_index_group.append(filetr_index_group_temp.pop(filetr_index_group_temp.index(sub_index)))
            for left_index in filetr_index_group_temp:
                pruning_left_index_group[counter].append(left_index)
        # first one is the least important weight and the last one is the most important weight
        while (len(pruning_index_group) < remove_filter):
            pruning_amount = len(pruning_index_group)
            for left_index in pruning_left_index_group:
                if (len(left_index) <= 1):
                    continue
                if (len(pruning_index_group) >= remove_filter):
                    break
                pruning_index_group.append(left_index.pop(-1))
            if (pruning_amount >= len(pruning_index_group)):
                raise ValueError('infinity loop')
        return torch.tensor(pruning_index_group).to(device)

    def L1norm_pruning(self,weight):
        """
        Apply L1 norm pruning to the given layer.

        Args:
            layer: The layer to be pruned.

        Return:
            sorted_indices: The sorted indices of important filters.

        Logic:
        1. Clone the weight data of the layer to avoid modifying the original weights.
        2. Check the shape of the weight tensor to determine if it is a convolutional layer.
        3. Calculate the importance of each filter by summing the absolute values of the weights along the appropriate dimensions.
           - For a 4-dimensional weight tensor, the sum is calculated along dimensions (1, 2, 3).
           - For a 2-dimensional weight tensor, the sum is calculated along dimension 0 (channels).
        4. Sort the importance values and obtain the corresponding indices in descending order.
        5. Return the sorted indices

    Note:
        The importance values indicate the amount of contribution of each filter to the overall model's performance.
        Filters with higher importance values are considered more significant and likely to be kept during pruning.
        """
        # weight = layer.weight.data.clone()
        weight = torch.tensor(weight)
        
        if len(weight.shape) == 4:
            importance = torch.sum(torch.abs(weight),dim=(1,2,3))
        else:
            importance = torch.sum(torch.abs(weight),dim=0)
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        return sorted_idx

    def Kmean_L1norm(self,weight,k):
        

        """
        Apply K-L1norm pruning to the given layer.

        Args:
            layer: The layer to be pruned.

        Return:
            sorted_idx: The sorted indices of the most important filters.

        Logic:
            1. Clone the weight data of the layer to avoid modifying the original weights.
            2. Sort layer filter by the L1 norm and obtain the corresponding indices.
            3. Return the sorted indices, representing the most important filters based on K-L1norm.
        """
        # weight = layer.weight.data.clone()
        sort_index = self.L1norm_pruning(weight)
        output_channel = int(weight.shape[0] * 0.5)
        print(output_channel)
        weight = weight.reshape(weight.shape[0],-1)
        
        pca = PCA(n_components=0.8).fit(weight)
        
        weight = pca.fit_transform(weight)
        
        
        
        
        pruning_index =  self.Kmean(weight,sort_index,k,output_channel)
        """
        using l1norm to sort the pruning index, and put them to the end of sorted_idx
        indicate they are not important
        However, experiment find out it doesn't help
        so I comment out them
        """
        # pruning_weight = weight[pruning_index,:,:,:]
        # important = torch.sum(torch.abs(pruning_weight),dim=(1,2,3))
        # pruning_weight,pruning_index = torch.sort(important)
        keep_index = [i.item() for i in sort_index if i not in pruning_index]
        keep_index = torch.as_tensor(keep_index,device=self.device)
        pruning_index = torch.as_tensor(pruning_index,device=self.device)
        return torch.cat((keep_index,pruning_index)).type(torch.IntTensor).to(self.device)



    def re_organize_middle_weights(self, expand_ratio_stage=0,k=2):
        # conv3 -> conv2
        # importance = torch.sum(
        #     torch.abs(self.conv3.conv.conv.weight.data), dim=(0, 2, 3)
        # )
        # if isinstance(self.conv2.bn, DynamicGroupNorm):
        #     channel_per_group = self.conv2.bn.channel_per_group
        #     importance_chunks = torch.split(importance, channel_per_group)
        #     for chunk in importance_chunks:
        #         chunk.data.fill_(torch.mean(chunk))
        #     importance = torch.cat(importance_chunks, dim=0)
        # if expand_ratio_stage > 0:
        #     sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
        #     sorted_expand_list.sort(reverse=True)
        #     target_width_list = [
        #         make_divisible(
        #             round(max(self.out_channel_list) * expand),
        #             MyNetwork.CHANNEL_DIVISIBLE,
        #         )
        #         for expand in sorted_expand_list
        #     ]
        #     right = len(importance)
        #     base = -len(target_width_list) * 1e5
        #     for i in range(expand_ratio_stage + 1):
        #         left = target_width_list[i]
        #         importance[left:right] += base
        #         base += 1e5
        #         right = left
        # sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        sorted_idx = self.Kmean_L1norm(self.conv2.conv.conv.weight.data.clone().cpu().detach().numpy(),k)
        
        self.conv3.conv.conv.weight.data = torch.index_select(
            self.conv3.conv.conv.weight.data, 1, sorted_idx
        )
        adjust_bn_according_to_idx(self.conv2.bn.bn, sorted_idx)
        self.conv2.conv.conv.weight.data = torch.index_select(
            self.conv2.conv.conv.weight.data, 0, sorted_idx
        )

        # conv2 -> conv1
        # importance = torch.sum(
        #     torch.abs(self.conv2.conv.conv.weight.data), dim=(0, 2, 3)
        # )
        # if isinstance(self.conv1.bn, DynamicGroupNorm):
        #     channel_per_group = self.conv1.bn.channel_per_group
        #     importance_chunks = torch.split(importance, channel_per_group)
        #     for chunk in importance_chunks:
        #         chunk.data.fill_(torch.mean(chunk))
        #     importance = torch.cat(importance_chunks, dim=0)
        # if expand_ratio_stage > 0:
        #     sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
        #     sorted_expand_list.sort(reverse=True)
        #     target_width_list = [
        #         make_divisible(
        #             round(max(self.out_channel_list) * expand),
        #             MyNetwork.CHANNEL_DIVISIBLE,
        #         )
        #         for expand in sorted_expand_list
        #     ]
        #     right = len(importance)
        #     base = -len(target_width_list) * 1e5
        #     for i in range(expand_ratio_stage + 1):
        #         left = target_width_list[i]
        #         importance[left:right] += base
        #         base += 1e5
        #         right = left
        # sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        # sorted_idx = self.Kmean_L1norm(self.conv2.conv.conv.weight.data.clone().cpu().detach().numpy(),k)
        # self.conv2.conv.conv.weight.data = torch.index_select(
        #     self.conv2.conv.conv.weight.data, 1, sorted_idx
        # )
        # adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        # self.conv1.conv.conv.weight.data = torch.index_select(
        #     self.conv1.conv.conv.weight.data, 0, sorted_idx
        # )

        return None
