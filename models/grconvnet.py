import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard 2-conv residual block (BN + ReLU), identity shortcut."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = self.bn2(self.conv2(x))
        return x + x_in


class GRConvNet(nn.Module):
    """
    Generative Residual Convolutional Network for antipodal grasp detection
    (Kumra et al., 2020 -- arXiv:1909.04810), upsampling variant.

    Same input/output contract as GGCNN in this repo: takes an
    (input_channels, 300, 300) image and returns (pos, cos, sin, width) maps at
    300x300, plus a compute_loss() with the same dict shape, so it is a drop-in
    for --network in train_ggcnn.py / eval_ggcnn.py / predict_grasp.py.

    Downsamples x4 (two strided convs), 5 residual blocks at the bottleneck,
    upsamples x4 with bilinear interpolation (avoids transpose-conv checkerboarding
    and keeps the 300 -> 75 -> 300 sizing exact). 1x1 output heads like GGCNN2.
    """

    def __init__(self, input_channels=1, channel_size=32, dropout=False, prob=0.0):
        super().__init__()

        c = channel_size
        self.conv1 = nn.Conv2d(input_channels, c, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c * 4)

        self.res1 = ResidualBlock(c * 4, c * 4)
        self.res2 = ResidualBlock(c * 4, c * 4)
        self.res3 = ResidualBlock(c * 4, c * 4)
        self.res4 = ResidualBlock(c * 4, c * 4)
        self.res5 = ResidualBlock(c * 4, c * 4)

        self.conv4 = nn.Conv2d(c * 4, c * 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(c * 2)
        self.conv5 = nn.Conv2d(c * 2, c, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(c)
        self.conv6 = nn.Conv2d(c, c, kernel_size=9, stride=1, padding=4)
        self.bn6 = nn.BatchNorm2d(c)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        self.pos_output = nn.Conv2d(c, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(c, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(c, 1, kernel_size=1)
        self.width_output = nn.Conv2d(c, 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
