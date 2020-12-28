import torch
import torch.nn as nn

from lib.sa.modules import Subtraction, Subtraction2, Aggregation


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 画像の各ピクセルごとの、-1 ~ 1の間での相対座標位置を返す関数
def position(H, W, is_cuda=True):
    if is_cuda:
        # unsqueeze(0)にて次元を増やす (shape: 1 x W)
        # repeat(H, 1)にて、1次元目の軸で、H分だけ繰り返す (shape: H x W)
        loc_w = torch.linspace(-1.0, 1.0, W).cuda() # -1から1の間でWidthの分だけステップ変化させる (loc_w: W)
        loc_w = loc_w.unsqueeze(0) # loc_w: 1 x W
        loc_w = loc_w.repeat(H, 1) # loc_w: H x W

        loc_h = torch.linspace(-1.0, 1.0, H).cuda() # loc_h: H
        loc_h = loc_h.unsqueeze(1) # loc_h: H x 1
        loc_h = loc_h.repeat(1, W) # loc_h: H x W
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc_w = loc_w.unsqueeze(0) # loc_w: 1 x H x W
    loc_h = loc_h.unsqueeze(0) # loc_h: 1 x H x W

    # loc[0, :, :]にはWidth方向の座標の情報、loc[1, :, :]にはHeight方向の座標の情報が格納されている
    loc = torch.cat([loc_w, loc_h], 0) # loc: 2 x H x W
    loc = loc.unsqueeze(0) # loc: 1 x 2 x H x W
    return loc


class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        # sa_type == 0はペアごとのSelf-Attention
        if sa_type == 0:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        # 入力xをそれぞれ別々の畳み込み処理に通す
        # x3は論文中のβを表し、x1とx2はAttentionを計算する関数αの引数になる値となる。
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x) # x1, x2, x3: bs x C x H x W
        if self.sa_type == 0:  # pairwise

            # 位置エンコーディングの計算処理
            # 画像の各ピクセルごとの、-1 ~ 1の間での相対座標位置を取得
            # 高さと幅の2つの相対座標を表すため、2つめの軸の次元は2となる
            pos_data = position(x.shape[2], x.shape[3], x.is_cuda) # pos_data: 1 x 2 x H x W
            # 相対座標位置を学習可能な畳み込み演算に通す
            pos_data = self.conv_p(pos_data) # pos_data: 1 x 2 x H x W
            # あるピクセル(p1)を中心とした、自分自身を含む9つのピクセル(P2)との相対位置(p2 -p1)を計算
            # これを全ピクセル分行うため、最後の軸の次元は、H*Wとなる。
            # 例えば、H:112、W:112の場合、112*112 = 12544となる。
            pos_data = self.subtraction(pos_data)  # pos_data: 1 x 2 x 9 x (H W)
            # 上記の入力データのバッチサイズ分repeatさせる
            pos_data = pos_data.repeat(x.shape[0], 1, 1, 1) # pos_data: bs x 2 x 9 x (H W)

            # あるピクセルの特徴ベクトルxaとその周囲9つのピクセル(xb)の特徴ベクトルの差分(xa - xb)を計算し、Attentionスコアとする
            # つまり、これは全ピクセルにおける9つの周囲ピクセルとの共起関係の値を表すtensorである
            # これは論文のEq3におけるδを表す
            attention = self.subtraction2(x1, x2) # attention: bs x C x 9 x (H W)
            # チャンネル方向に位置エンコーディングの
            position_wised_attention = torch.cat([attention, pos_data], 1) # position_wised_attention: bs x (C+2) x 9 x (H W)
            # 全ピクセルの9つの周囲ピクセルのAttentionの値を畳み込み層に渡して、チャンネル軸(C+2)で次元を圧縮する
            # これは論文中の Eq3におけるγ関数を表す
            embedded_attention = self.conv_w(position_wised_attention) # w: bs x C' x 9 x (H W)
            # 9つの周囲ピクセルの共起関係の値が合計で1になるよう、確率値で表す(いわゆるnormalized処理のようなもの)
            normalized_attention = self.softmax(embedded_attention) # w: bs x C' x 9 x (H W)
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            normalized_attention = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        # あるピクセル(i)における、9つ周囲ピクセル(j)におけるAttention値（α）と、
        # 各ピクセルにおける特徴量(β_j)のそれぞれの要素積（アダマール積）を取って、足し合わせる計算を行い、これをiの特徴量y_iとする
        # この処理を全ピクセル分実施する
        x = self.aggregation(x3, normalized_attention) # x: bs x C x H x W (x3と階数、次元ともに同じ)
        return x


class Bottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # このSAM (SA Block)がこのモデルのキモ
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out


class SAN(nn.Module):
    def __init__(self, sa_type, block, layers, kernels, num_classes):
        super(SAN, self).__init__()
        c = 64
        self.conv_in, self.bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
        self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        # SA Blockが呼び出される箇所（その1）
        # block: Bottleneckクラス
        self.layer0 = self._make_layer(sa_type, block, c, layers[0], kernels[0])

        c *= 4
        self.conv1, self.bn1 = conv1x1(c // 4, c), nn.BatchNorm2d(c)
        # SA Blockが呼び出される箇所（その2）
        self.layer1 = self._make_layer(sa_type, block, c, layers[1], kernels[1])

        c *= 2
        self.conv2, self.bn2 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        # SA Blockが呼び出される箇所（その3）
        self.layer2 = self._make_layer(sa_type, block, c, layers[2], kernels[2])

        c *= 2
        self.conv3, self.bn3 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        # SA Blockが呼び出される箇所（その4）
        self.layer3 = self._make_layer(sa_type, block, c, layers[3], kernels[3])

        c *= 2
        self.conv4, self.bn4 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        # SA Blockが呼び出される箇所（その5）
        self.layer4 = self._make_layer(sa_type, block, c, layers[4], kernels[4])

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, num_classes)

    def _make_layer(self, sa_type, block, planes, blocks, kernel_size=7, stride=1):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(sa_type, planes, planes // 16, planes // 4, planes, 8, kernel_size, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def san(sa_type, layers, kernels, num_classes):
    model = SAN(sa_type, Bottleneck, layers, kernels, num_classes)
    return model


if __name__ == '__main__':
    net = san(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=1000).cuda().eval()
    print(net)
    y = net(torch.randn(4, 3, 224, 224).cuda())
    print(y.size())
