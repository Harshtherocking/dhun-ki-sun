import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, is_first_layer=False):
        super().__init__()
        self.is_first_layer = is_first_layer
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        if not self.is_first_layer:
            x = self.maxpool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class BlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, is_last_layer=False):
        super().__init__()
        self.is_last_layer = is_last_layer
        if self.is_last_layer:
            final_out = out_channels
            out_channels = in_channels
        
        self.conv1 = nn.Conv1d(2 * in_channels, 2 * out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(2 * out_channels, 2 * out_channels, kernel_size=3, padding=1)
        
        if not is_last_layer:
            self.convT = nn.ConvTranspose1d(2 * out_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.conv3 = nn.Conv1d(2 * out_channels, final_out, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x) if self.is_last_layer else self.convT(x)

class UNet1D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.block_1_down = BlockDown(input_channels, 64, is_first_layer=True)
        self.block_2_down = BlockDown(64, 128)
        self.block_3_down = BlockDown(128, 256)
        self.block_4_down = BlockDown(256, 512)

        self.block = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        )

        self.block_4_up = BlockUp(512, 256)
        self.block_3_up = BlockUp(256, 128)
        self.block_2_up = BlockUp(128, 64)
        self.block_1_up = BlockUp(64, output_channels, is_last_layer=True)

    def forward(self, x):
        assert x.shape[1] == self.input_channels, "Input channels mismatch"
        
        b1_out = self.block_1_down(x)
        b2_out = self.block_2_down(b1_out)
        b3_out = self.block_3_down(b2_out)
        b4_out = self.block_4_down(b3_out)

        x = self.block(b4_out)
        x = torch.cat([b4_out, x], dim=1)
        x = self.block_4_up(x)
        x = torch.cat([b3_out, x], dim=1)
        x = self.block_3_up(x)
        x = torch.cat([b2_out, x], dim=1)
        x = self.block_2_up(x)
        x = torch.cat([b1_out, x], dim=1)
        x = self.block_1_up(x)
        
        return x

if __name__ == "__main__":
    batch_size = 2
    channels = 1  # Assuming mono audio
    sequence_length = 220500  # 5 sec at 44.1kHz
    
    model = UNet1D(channels, channels).cuda()
    
    sample_input = torch.rand(batch_size, channels, sequence_length).cuda()
    output = model(sample_input)
    print(output.shape)  # Should match input shape
