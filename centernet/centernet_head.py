import torch
import torch.nn as nn


class SingleHead(nn.Module):
    def __init__(self, in_channel, inner_channel, out_channel, num_convs, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        head_convs=[]
        for i in range(num_convs):
            inc = in_channel if i==0 else inner_channel
            head_convs.append(nn.Conv2d(inc, inner_channel, kernel_size=3, padding=1))
            head_convs.append(nn.ReLU())
        head_convs.append(nn.Conv2d(inner_channel, out_channel, kernel_size=1))

        self.head_convs=nn.Sequential(*head_convs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if bias_fill:
            self.head_convs[-1].bias.data.fill_(bias_value)

    def forward(self, x):
        return self.head_convs(x)


class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, cfg):
        super(CenternetHead, self).__init__()
        self.cls_head = SingleHead(
            64,
            inner_channel=128,
            out_channel=cfg.MODEL.CENTERNET.NUM_CLASSES,
            num_convs=2,
            bias_fill=True,
            bias_value=cfg.MODEL.CENTERNET.BIAS_VALUE,
        )
        self.wh_head = SingleHead(
            64,
            inner_channel=64,
            out_channel=4,
            num_convs=2,
        )

    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)
        wh = self.wh_head(x)
        pred = {"cls": cls, "wh": wh}
        return pred
