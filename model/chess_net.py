import torch
import torch.nn as nn

class ChessNet(nn.Module):

    def __init__(
        self,
        input_channels=18,
        channels=128,
        num_blocks=8,
        policy_channels=32,
        action_space_size=4544,
        value_channels=32
    ):
        super().__init__()

        self.backbone = ChessBackbone(
            input_channels=input_channels,
            channels=channels,
            num_blocks=num_blocks
        )

        self.policy_head = PolicyHead(
            channels=channels,
            policy_channels=policy_channels,
            action_space_size=action_space_size
        )

        self.value_head = ValueHead(
            channels=channels,
            value_channels=value_channels
        )

    def forward(self, x):

        features = self.backbone(x)

        policy_logits = self.policy_head(features)

        value = self.value_head(features)

        return policy_logits, value



class ResidualBlock(nn.Module):

    def __init__(self, channels=128):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = self.relu(out)

        return out


class ChessBackbone(nn.Module):

    def __init__(self, input_channels=18, channels=128, num_blocks=8):
        super().__init__()

        self.input_conv = nn.Conv2d(
            input_channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.input_bn = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):

        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.relu(x)

        x = self.residual_blocks(x)

        return x


class PolicyHead(nn.Module):

    def __init__(self, channels=128, policy_channels=32, action_space_size=1):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            policy_channels,
            kernel_size=3,
            padding=1
        )

        self.bn = nn.BatchNorm2d(policy_channels)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(
            policy_channels * 8 * 8,
            action_space_size
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)

        logits = self.fc(x)

        return logits


class ValueHead(nn.Module):

    def __init__(self, channels=128, value_channels=32):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            value_channels,
            kernel_size=1
        )

        self.bn = nn.BatchNorm2d(value_channels)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(
            value_channels * 8 * 8,
            256
        )

        self.fc2 = nn.Linear(
            256,
            1
        )

        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        value = self.tanh(x)

        return value