import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0,2,1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def relative_pos(B,S,sample_xyz,knn_xyz):
    fps_xyz=sample_xyz.view(B, S, 1, -1).repeat(1, 1, 32, 1)
    xyz_relative=fps_xyz-knn_xyz
    return xyz_relative

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz,npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):

    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class AttenInter(nn.Module):
    def __init__(self, d_points, d_model, kneighbors):
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = kneighbors

    def forward(self,xyz,points):

        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]
        knn_xyz = index_points(xyz, knn_idx)

        pre = points
        x = self.fc1(points)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)


        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res


class newLocalGrouper(nn.Module):
    def __init__(self, channel, groups, nsample_list, mlp_list, use_xyz=True, normalize="anchor", **kwargs):
        super(newLocalGrouper, self).__init__()
        self.groups = groups
        self.nsample_list = nsample_list
        self.use_xyz = use_xyz
        self.npoints=groups
        self.mlp_list=mlp_list
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            self.affine_alpha = nn.Parameter(torch.ones(
                [1, 1, 1, channel + 3]))
            self.affine_beta = nn.Parameter(torch.zeros(
                [1, 1, 1, channel + 3]))

        self.fc1qq = nn.Linear(channel, channel+3)
        self.fc2 = nn.Linear(channel+3, channel+3)

        self.w_qsqq = nn.Linear(channel+3, channel+3, bias=False)
        self.w_ksqq = nn.Linear(channel+3, channel+3, bias=False)
        self.w_vsqq = nn.Linear(channel+3, channel+3, bias=False)

        self.fc_deltaqq = nn.Sequential(
            nn.Linear(3, channel + 3),
            nn.ReLU(),
            nn.Linear(channel + 3, channel + 3)
        )

        self.fc_gammaqq = nn.Sequential(
            nn.Linear(channel + 3, channel + 3),
            nn.ReLU(),
            nn.Linear(channel + 3, channel + 3)
        )
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        mlp_list_array=np.array(mlp_list)
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = channel + 6
            for out_channel in mlp_list_array[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        self.fc1=nn.ModuleList()
        self.w_qs=nn.ModuleList()
        self.w_ks=nn.ModuleList()
        self.w_vs=nn.ModuleList()
        self.fc_delta=nn.ModuleList()
        self.fc_gamma=nn.ModuleList()
        for i in range(len(mlp_list)):
            d_model=mlp_list_array[i][-1]
            self.fc1.append(nn.Linear(channel, d_model))
            self.w_qs.append(nn.Linear(d_model, d_model, bias=False))
            self.w_ks.append(nn.Linear(d_model, d_model, bias=False))
            self.w_vs.append(nn.Linear(d_model, d_model, bias=False))
            self.fc_delta.append(nn.Sequential(
                nn.Linear(3, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ))
            self.fc_gamma.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ))

    def forward(self,xyz,points):
        B,N,C=xyz.shape
        S=self.groups
        xyz = xyz.contiguous()

        fps_idx = farthest_point_sample(xyz, self.npoints).long()

        new_xyz = index_points(xyz, fps_idx)

        new_points = index_points(points, fps_idx)


        new_points_list = []

        nsample = np.array(self.nsample_list)
        i=0
        for K in nsample:
            group_idx = knn_point(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            new_points_x = self.fc1[i](new_points)
            q = self.w_qs[i](new_points_x)

            relative_xyz = grouped_xyz - new_xyz.view(B, S, 1, C)
            pos_enc = self.fc_delta[i](relative_xyz)
            if points is not None:
                grouped_points = index_points(points, group_idx)
            else:
                grouped_points = grouped_xyz

            if self.use_xyz:
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

            if self.normalize is not None:
                if self.normalize == "center":
                    mean = torch.mean(grouped_points, dim=2, keepdim=True)
                if self.normalize == "anchor":
                    mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                    mean = mean.unsqueeze(dim=2)

                std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(
                    dim=-1).unsqueeze(dim=-1)

                grouped_points = (grouped_points - mean) / (std + 1e-5)

                grouped_points = self.affine_alpha * grouped_points + self.affine_beta

            grouped_points = torch.cat([grouped_points, relative_xyz], dim=-1)

            grouped_points = grouped_points.permute(0, 3, 2, 1)

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            grouped_points = grouped_points.permute(0, 3, 2, 1)

            k = self.w_ks[i](grouped_points)
            v = self.w_vs[i](grouped_points)
            q_new=q.unsqueeze(2)

            attn = self.fc_gamma[i](q_new - k + pos_enc)
            scale = torch.tensor(1.0 / math.sqrt(k.size(-1)), dtype=attn.dtype, device=attn.device)
            attn = F.softmax(attn * scale, dim=2)

            res = torch.einsum('bsnf,bsnf->bsf', attn, v+pos_enc)
            new_points_list.append(res)
            i = i + 1

        new_points_concat = torch.cat(new_points_list, dim=2)
        new_points = new_points_concat.view(B, S, 1, -1).repeat(1, 1, nsample[-1], 1)
        return new_xyz, new_points




class ConvBNReLU1D(nn.Module):
    def  __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)




class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)



class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,activation='relu', use_xyz=True):
        super(PreExtraction, self).__init__()
        self.transfer = ConvBNReLU1D(out_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self,x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)

        x = x.reshape(-1, d, s)

        x = self.transfer(x)
        batch_size, _, _ = x.size()

        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)


        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self,x):
        return self.operation(x)


class PointNetFeaturePropagation1(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation1, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2):
        points2=points2.permute(0,2,1)
        B, N, C = xyz1.shape
        _,S,_=xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)

            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)

            new_points = torch.cat([points1, interpolated_points], dim=-1)

        else:
            new_points = interpolated_points

        new_points=new_points.permute(0,2,1)
        new_points=self.fuse(new_points)
        new_points=self.extraction(new_points)

        return new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):


        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))


        return new_points


class TransMLP(nn.Module):

    def __init__(self, num_classes=50, points=2048, embed_dim=16, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 64, 128, 128], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[2, 2, 2, 2],
                 att_dims=[32, 128, 512, 1024], att_inter_dims=[512, 256, 128, 128],
                 gmp_dim=64, cls_dim=64, **kwargs):
        super(TransMLP,self).__init__()
        self.stages = len(pre_blocks)  # 4
        self.class_num = num_classes
        self.points = points  # 2048
        self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)  # 6:xyz和法线坐标  6：输入通道数  embed_dim：输出通道数
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel= embed_dim
        anchor_points=self.points
        en_dims=[last_channel]

        mlp_list = [[[32, 48, 64], [32, 48, 64]],
                    [[64, 64, 128], [64, 96, 128]],
                    [[128, 128, 256], [128, 196, 256]],
                    [[512, 768, 1024]]]

        nsample_list = [[64, 128],
                        [64, 128],
                        [32, 64],
                        [32]]

        j=0
        out_channel_list=[128,256,512,1024]

        for i in range(len(pre_blocks)):

            out_channel=out_channel_list[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]

            att_dim=att_dims[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            local_grouper=newLocalGrouper(last_channel, anchor_points, nsample_list[i], mlp_list[i], use_xyz, normalize )

            self.local_grouper_list.append(local_grouper)

            pre_block_module = PreExtraction(out_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)

            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

            en_dims.append(last_channel)


        self.decode_list = nn.ModuleList()
        self.atteninter=nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0, en_dims[0])
        assert len(en_dims) == len(de_dims) == len(de_blocks) + 1
        for i in range(len(en_dims) - 1):
            att_inter_dims_num=att_inter_dims[i]
            kneighbor = k_neighbors[i]
            self.atteninter.append(
                AttenInter(att_inter_dims_num,att_inter_dims_num,kneighbor)
            )

        self.act = get_activation(activation)


        self.cls_map = nn.Sequential(
            ConvBNReLU1D(16, cls_dim, bias=bias, activation=activation),
            ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation)
        )

        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:

            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(en_dims), gmp_dim, bias=bias, activation=activation)

        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim + cls_dim + de_dims[-1], 64, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias)
        )
        self.en_dims = en_dims

        self.fp4 = PointNetFeaturePropagation(in_channel=1536, mlp=[512, 512])
        self.fp3 = PointNetFeaturePropagation(in_channel=768, mlp=[512, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+16+3, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)


    def forward(self,x, norm_plt,cls_label):
        xyz = x.permute(0, 2, 1)
        x = torch.cat([x, norm_plt], dim=1)
        B, C, N = x.shape
        x = self.embedding(x)

        xyz_list = [xyz]
        x_list = [x]


        for i in range(self.stages):

            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x=self.pre_blocks_list[i](x)
            x=self.pos_blocks_list[i](x)
            xyz_list.append(xyz)
            x_list.append(x)

        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        l3_points = self.fp4(xyz_list[1], xyz_list[0], x_list[1], x_list[0])
        l3_points = (self.atteninter[0](xyz_list[1], l3_points.permute(0, 2, 1))).permute(0, 2, 1)
        l2_points = self.fp3(xyz_list[2], xyz_list[1], x_list[2], l3_points)
        l2_points = (self.atteninter[1](xyz_list[2], l2_points.permute(0, 2, 1))).permute(0, 2, 1)
        l1_points = self.fp2(xyz_list[3], xyz_list[2], x_list[3], l2_points)
        l1_points = (self.atteninter[2](xyz_list[3], l1_points.permute(0, 2, 1))).permute(0, 2, 1)
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        xyz_4=xyz_list[4].permute(0,2,1)
        cat=torch.cat([cls_label_one_hot, xyz_4, x_list[4]], 1)
        l0_points = self.fp1(xyz_list[4], xyz_list[3], cat, l1_points)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x



