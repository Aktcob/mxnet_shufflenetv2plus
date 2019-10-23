import mxnet
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import mxnet.ndarray as F
from blocks import Shufflenet, Shuffle_Xception, HS, SELayer

class ShuffleNetV2_Plus(HybridBlock):
    def __init__(self, input_size=224, n_class=1000, architecture=None, model_size='Large'):
        super(ShuffleNetV2_Plus, self).__init__()
        print('model size is ', model_size)
        assert input_size % 32 == 0
        assert architecture is not None
        self.stage_repeats = [4, 4, 8, 4]
        if model_size == 'Large':
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
        elif model_size == 'Medium':
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
        elif model_size == 'Small':
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
        else:
            raise NotImplementedError
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.HybridSequential()
        self.first_conv.add(nn.Conv2D(in_channels=3, channels=input_channel,
                                        kernel_size=3, strides=2, padding=1, use_bias=False))
        self.first_conv.add(nn.BatchNorm()) 
        self.first_conv.add(HS()) 

        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            activation = 'HS' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False
            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture[archIndex]
                archIndex += 1
                if blockIndex == 0:
                    print('Shuffle3x3')
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 1:
                    print('Shuffle5x5')
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=5, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 2:
                    print('Shuffle7x7')
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=7, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 3:
                    print('Xception',stride)
                    self.features.append(Shuffle_Xception(inp, outp, base_mid_channels=outp // 2, stride=stride,
                                    activation=activation, useSE=useSE))
                else:
                    raise NotImplementedError
                input_channel = output_channel
        assert archIndex == len(architecture)

        self.feature = nn.HybridSequential()
        for layer in self.features:
            self.feature.add(layer)
        self.conv_last = nn.HybridSequential()
        self.conv_last.add(nn.Conv2D(in_channels=input_channel, channels=1280,
                                       kernel_size=1, strides=1, padding=0, use_bias=False))
        self.conv_last.add(nn.BatchNorm()) 
        self.conv_last.add(HS()) 
        self.globalpool = nn.GlobalAvgPool2D()
        self.LastSE = SELayer(1280)
        self.fc = nn.HybridSequential()
        self.fc.add(nn.Dense(1280, use_bias=False))
        self.fc.add(HS())
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Dense(n_class, use_bias=False)
        # self._initialize_weights()

    def hybrid_forward(self, F, x):
        x = self.first_conv(x)
        x = self.feature(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = self.LastSE(x)
        x = x.reshape(-1, 1280)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = F.max(x)
        return x

if __name__ == "__main__":
    ctx = mxnet.gpu(0)
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    net = ShuffleNetV2_Plus(architecture=architecture, model_size='Small')
    net.initialize(ctx=ctx)
    
    import time
    for i in range(300):
        start = time.time()
        test_data = nd.ones((1, 3, 640, 640),ctx=ctx)
        test_outputs = net(test_data)
        print(test_outputs)
        print (time.time()-start)
