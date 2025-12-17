from audioop import bias
import torch
import torch.nn as nn
from itertools import cycle


class CSCNet(nn.Module):
    def __init__(self, num_input_channels=1, num_output_channels=1, kc=64, ks=7, ista_iters=3):
        super(CSCNet, self).__init__()
        self._ista_iters = ista_iters
        self._layers = ista_iters

        self.first_softthrsh = nn.Softshrink(0.1)
        self.rest_softthrsh = nn.ModuleList([nn.Softshrink(0.1) for _ in range(self._layers)])

        def build_conv_layers(in_ch, out_ch, count):
            """Conv layer wrapper
            """
            return nn.ModuleList([
                nn.Conv2d(in_ch, out_ch, kernel_size=(ks, ks), stride=(1, 1), padding=(ks//2, ks//2))
                for _ in range(count)
            ])

        self.fist_encoder = build_conv_layers(num_input_channels, kc, 1)[0]
        self.unfold_encoder = build_conv_layers(num_input_channels, kc, self._layers)
        self.unfold_decoder = build_conv_layers(kc, num_input_channels, self._layers)
        self.last_decoder = build_conv_layers(kc, num_output_channels, 1)[0]

    def forward_unfolding(self, inputs, csc):
        for _itr, lyr in zip(range(self._ista_iters), cycle(range(self._layers))):
            res = inputs - self.unfold_decoder[lyr](csc)
            sc_residual = self.unfold_encoder[lyr](res)
            csc = self.rest_softthrsh[lyr](csc + sc_residual)
        return csc

    def forward(self, inputs):
        csc = self.first_softthrsh(self.fist_encoder(inputs))
        csc = self.forward_unfolding(inputs, csc)
        outputs = self.last_decoder(csc)
        return csc, outputs

#def __init__(self, num_input_channels=1, num_output_channels=1, kc=128, ks=7, ista_iters=9):
class RCSCNet(nn.Module):
    def __init__(self, num_input_channels=1, num_output_channels=1, kc=64, ks=7, ista_iters=3):
        super(RCSCNet, self).__init__()
        self._ista_iters = ista_iters
        self._layers = ista_iters
        self.cscnet = CSCNet(kc=kc, ista_iters=ista_iters, ks=ks, num_input_channels=num_input_channels, num_output_channels=num_output_channels)

        def build_conv_layers(in_ch, out_ch, count):
            """Conv layer wrapper
            """
            return nn.ModuleList([
                nn.Conv2d(in_ch, out_ch, kernel_size=(ks, ks), stride=(1, 1), padding=(ks // 2, ks // 2))
                for _ in range(count)
            ])
        self.sigmoid = nn.Sigmoid()
        self.zNET = build_conv_layers(kc, kc, self._layers)

        # def build_conv3d_layers(in_ch, out_ch, count, l = 3):
        def build_conv3d_layers(in_ch, out_ch, count, l = 3):
            """Conv layer wrapper
            """
            return nn.ModuleList([
                nn.Conv3d(in_ch, out_ch, kernel_size=(l, 1, 1), stride=(1, 1, 1), padding=(l // 2, 0 // 2, 0 // 2))
                for _ in range(count)
            ])

        # self.decoder = build_conv3d_layers(kc * 2, kc, self._layers)
        self.batchnorm = nn.ModuleList([nn.BatchNorm2d(kc, affine=True) for _ in range(self._layers)])

    def forward(self, inputs):
        hsi_shape = inputs.shape[1]
        for i in range(hsi_shape):
            if i == 0:
                input_now = inputs[:, i:i + 1, :, :]
                csc = self.cscnet.first_softthrsh(self.cscnet.fist_encoder(input_now))
                csc = csc.unsqueeze(0)
            else:
                input_now = inputs[:, i:i + 1, :, :]
                csc_ = self.cscnet.first_softthrsh(self.cscnet.fist_encoder(input_now))
                csc_ = csc_.unsqueeze(0)
                csc = torch.cat((csc, csc_), 0)
        # csc (31, N, 128, 64, 64) (128, 512, 512)
        for _itr, lyr in zip(range(self._ista_iters), cycle(range(self._layers))):
            csc_ = []
            if _itr % 2 == 0:
                for i in range(hsi_shape):
                    if i == 0:
                        input_now = inputs[:, i:i + 1, :, :]
                        res = input_now - self.cscnet.unfold_decoder[lyr](csc[i, :, :, :, :])
                        sc_residual = self.cscnet.unfold_encoder[lyr](res)
                        tmp_csc = self.cscnet.rest_softthrsh[lyr](csc[i, :, :, :, :] + sc_residual)
                        csc_.append(tmp_csc)
                    else:
                        input_now = inputs[:, i:i + 1, :, :]
                        res = input_now - self.cscnet.unfold_decoder[lyr](csc[i, :, :, :, :])
                        sc_residual = self.cscnet.unfold_encoder[lyr](res)
                        tmp_csc = self.cscnet.rest_softthrsh[lyr](csc[i, :, :, :, :] + sc_residual)
                        zNET = self.zNET[lyr](tmp_csc)
                        zNET = self.sigmoid(self.batchnorm[lyr](zNET))
                        csc_.append(zNET * csc_[i-1] + (1 - zNET) * tmp_csc)

            else:
                for i in range(hsi_shape - 1, -1, -1):
                    if i == hsi_shape - 1:
                        input_now = inputs[:, i:i + 1, :, :]
                        res = input_now - self.cscnet.unfold_decoder[lyr](csc[i, :, :, :, :])
                        sc_residual = self.cscnet.unfold_encoder[lyr](res)
                        tmp_csc = self.cscnet.rest_softthrsh[lyr](csc[i, :, :, :, :] + sc_residual)
                        csc_.insert(0, tmp_csc)
                    else:
                        input_now = inputs[:, i:i + 1, :, :]
                        res = input_now - self.cscnet.unfold_decoder[lyr](csc[i, :, :, :, :])
                        sc_residual = self.cscnet.unfold_encoder[lyr](res)
                        tmp_csc = self.cscnet.rest_softthrsh[lyr](csc[i, :, :, :, :] + sc_residual)
                        zNET = self.zNET[lyr](tmp_csc)
                        zNET = self.sigmoid(self.batchnorm[lyr](zNET))
                        csc_.insert(0, zNET * csc_[0] + (1 - zNET) * tmp_csc)

            csc_ = torch.stack(csc_, dim=0)

            csc = csc_

        for i in range(hsi_shape):
            csc_now = csc[i, :, :, :, :]
            if i == 0:
                outputs = self.cscnet.last_decoder(csc_now)
            else:
                output_ = self.cscnet.last_decoder(csc_now)
                outputs = torch.cat((outputs, output_), 1)
        return csc, outputs


if __name__ == '__main__':
    x = torch.randn((4, 28, 64, 64))
    model = RCSCNet()
    y1, y2 = model(x)
    print(y1.shape)
    print(y2.shape)
    '''
    for name, param in model.named_parameters():
        print(name)
    model1 = CSCNet()
    for name, param in model1.named_parameters():
        print(name)
    '''
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))

    from thop import profile

    input = torch.randn((4, 28, 64, 64))
    flops, params = profile(model, inputs=(input,))
    print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2),
                                                      round(params / (10 ** 6), 2)))  # 4111514624.0 25557032.0 res50