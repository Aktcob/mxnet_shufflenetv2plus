import mxnet
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import mxnet.ndarray as F

def channel_shuffle(x):
    batchsize, num_channels, height, width = x.shape
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = F.transpose(x, axes=(1, 0, 2))
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

class SELayer(HybridBlock):
    def __init__(self, inplanes, isTensor=True):
        super(SELayer, self).__init__()
        if isTensor:
            self.SE_opr = nn.HybridSequential()
            self.SE_opr.add(nn.Conv2D(in_channels=inplanes, channels=inplanes // 4,
                                        kernel_size=1, strides=1, use_bias=False))
            self.SE_opr.add(nn.Activation('relu'))     
            self.SE_opr.add(nn.Conv2D(in_channels=inplanes // 4, channels=inplanes,
                                        kernel_size=1, strides=1, use_bias=False))                 

    def hybrid_forward(self, F, x):
        atten = F.contrib.AdaptiveAvgPooling2D(x,output_size = 1)
        atten = self.SE_opr(atten)
        atten = F.clip(atten + 3, 0, 6) / 6.0
        return x * atten

class HS(HybridBlock):
    def __init__(self):
        super(HS, self).__init__()

    def hybrid_forward(self, F, inputs):
        clip = F.clip(inputs + 3, 0, 6) / 6.0
        return inputs * clip

class Shufflenet(HybridBlock):
    def __init__(self, inp, oup, base_mid_channels, ksize, stride, activation, useSE):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        outputs = oup - inp
        branch_main = [
            # pw
            nn.Conv2D(in_channels=inp, channels=base_mid_channels,
                                        kernel_size=1, strides=1, padding=0, use_bias=False),
            nn.BatchNorm(),
            None,
            # dw
            nn.Conv2D(in_channels=base_mid_channels, channels=base_mid_channels,
                                        kernel_size=ksize, strides=stride, padding=pad, groups=base_mid_channels, use_bias=False),
            nn.BatchNorm(),
            # pw-linear
            nn.Conv2D(in_channels=base_mid_channels, channels=outputs,
                                        kernel_size=1, strides=1, padding=0, use_bias=False),
            nn.BatchNorm(),
            None,
        ]
        if activation == 'ReLU':
            assert useSE == False
            '''This model should not have SE with ReLU'''
            branch_main[2] = nn.Activation('relu')
            branch_main[-1] = nn.Activation('relu')
        else:
            branch_main[2] = HS()
            branch_main[-1] = HS()
            if useSE:
                branch_main.append(SELayer(outputs))
        self.branch_main = nn.HybridSequential()
        for layer in branch_main:
            self.branch_main.add(layer)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2D(in_channels=inp, channels=inp,
                                            kernel_size=ksize, strides=stride, padding=pad, groups=inp, use_bias=False),
                nn.BatchNorm(),
                nn.Conv2D(in_channels=inp, channels=inp,
                                            kernel_size=1, strides=1, padding=0, use_bias=False),
                nn.BatchNorm(),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.Activation('relu')
            else:
                branch_proj[-1] = HS()     
            self.branch_proj = nn.HybridSequential()
            for layer in branch_proj:
                self.branch_proj.add(layer) 
        else:
            self.branch_proj = None

    def hybrid_forward(self, F, old_x):
        if self.stride==1:
            x_proj, x = channel_shuffle(old_x)
            # print x_proj.shape, x.shape, self.branch_main(x).shape
            return F.concat(x_proj, self.branch_main(x), dim=1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            # print (self.branch_proj(x_proj).shape, self.branch_main(x).shape)
            return F.concat(self.branch_proj(x_proj), self.branch_main(x), dim=1)

class Shuffle_Xception(HybridBlock):
    def __init__(self, inp, oup, base_mid_channels, stride, activation, useSE):
        super(Shuffle_Xception, self).__init__()
        assert stride in [1, 2]
        assert base_mid_channels == oup//2
        self.base_mid_channel = base_mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp
        branch_main = [
            # dw
            nn.Conv2D(in_channels=inp, channels=inp,
                            kernel_size=3, strides=stride, padding=1, groups=inp, use_bias=False),
            nn.BatchNorm(),
            # pw
            nn.Conv2D(in_channels=inp, channels=base_mid_channels,
                            kernel_size=1, strides=1, padding=0, use_bias=False),
            nn.BatchNorm(),
            None,
            # dw
            nn.Conv2D(in_channels=base_mid_channels, channels=base_mid_channels,
                            kernel_size=3, strides=stride, padding=1, groups=base_mid_channels, use_bias=False),
            nn.BatchNorm(),
            # pw
            nn.Conv2D(in_channels=base_mid_channels, channels=base_mid_channels,
                            kernel_size=1, strides=1, padding=0, use_bias=False),
            nn.BatchNorm(),
            None,
            # dw
            nn.Conv2D(in_channels=base_mid_channels, channels=base_mid_channels,
                            kernel_size=3, strides=stride, padding=1, groups=base_mid_channels, use_bias=False),
            nn.BatchNorm(),
            # pw
            nn.Conv2D(in_channels=base_mid_channels, channels=outputs,
                            kernel_size=1, strides=1, padding=0, use_bias=False),
            nn.BatchNorm(),
            None,
        ]

        if activation == 'ReLU':
            branch_main[4] = nn.Activation('relu')
            branch_main[9] = nn.Activation('relu')
            branch_main[14] = nn.Activation('relu')
        else:
            branch_main[4] = HS()
            branch_main[9] = HS()
            branch_main[14] = HS()
        assert None not in branch_main

        if useSE:
            assert activation != 'ReLU'
            branch_main.append(SELayer(outputs))
        self.branch_main = nn.HybridSequential()
        for layer in branch_main:
            self.branch_main.add(layer)

        if self.stride == 2:
            branch_proj = [
                # dw
                nn.Conv2D(in_channels=inp, channels=inp,
                                kernel_size=3, strides=stride, padding=1, groups=inp, use_bias=False),
                nn.BatchNorm(),
                # pw-linear
                nn.Conv2D(in_channels=inp, channels=inp,
                                kernel_size=1, strides=1, padding=0, use_bias=False),
                nn.BatchNorm(),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.Activation('relu')
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.HybridSequential()
            for layer in branch_proj:
                self.branch_proj.add(layer) 

    def hybrid_forward(self, F, old_x):
        if self.stride==1:
            x_proj, x = channel_shuffle(old_x)
            return F.concat(x_proj, self.branch_main(x), dim=1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return F.concat(self.branch_proj(x_proj), self.branch_main(x), dim=1)

