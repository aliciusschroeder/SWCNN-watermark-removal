import torch
import torch.nn as nn

class DnCNN(nn.Module):
    """
    DnCNN: A Deep Convolutional Neural Network for Image Denoising.

    This network is designed to remove noise from images by learning the residual noise.
    It consists of multiple convolutional layers with ReLU activations and batch normalization.
    
    Args:
        channels (int): Number of input and output channels (e.g., 1 for grayscale, 3 for RGB).
        num_of_layers (int, optional): Total number of convolutional layers. Defaults to 17.
    """
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        
        # Initial convolution layer without batch normalization
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, 
                                padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Intermediate layers with batch normalization
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, 
                          padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # Final convolution layer to map back to the original number of channels
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, 
                                padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the DnCNN.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        out = self.dncnn(x)
        return out


class DnCNN_RL(nn.Module):
    """
    DnCNN_RL: DnCNN with Residual Learning for Image Denoising.

    This variant of DnCNN adds the input to the network's output, enabling the network to learn the residual noise.
    
    Args:
        channels (int): Number of input and output channels.
        num_of_layers (int, optional): Total number of convolutional layers. Defaults to 17.
    """
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_RL, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        
        # Initial convolution layer without batch normalization
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, 
                                padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Intermediate layers with batch normalization
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, 
                          padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # Final convolution layer to map back to the original number of channels
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, 
                                padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the DnCNN_RL.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor representing the denoised image.
        """
        out = self.dncnn(x)
        return x + out  # Residual connection to add the learned noise back to the input


class VGG16(nn.Module):
    """
    VGG16 Feature Extractor.

    This class implements the convolutional layers of the VGG16 network up to the second max-pooling layer.
    It is typically used for feature extraction purposes.
    """
    def __init__(self):
        super(VGG16, self).__init__()
        layers = []
        
        # Block 1: Conv -> ReLU -> Conv -> ReLU -> MaxPool
        layers.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 2: Conv -> ReLU -> Conv -> ReLU -> MaxPool
        layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the VGG16 feature extractor.

        Args:
            x (torch.Tensor): Input tensor with shape (N, 3, H, W).

        Returns:
            torch.Tensor: Extracted feature maps.
        """
        x = self.features(x)
        return x


class HN(nn.Module):
    """
    HN: Hierarchical Network for Image Processing.

    This network follows a U-Net-like architecture with encoding and decoding blocks.
    It is designed for tasks such as image segmentation or image restoration.
    
    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 3.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(HN, self).__init__()

        # Encoding Block 1: Two convolutions followed by max pooling
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Encoding Blocks 2-5: Each block has a convolution and max pooling
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Decoding Block 5: Convolution and transpose convolution for upsampling
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        # Decoding Block 4: Processes concatenated features and upsamples
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        # Decoding Blocks 3-2: Further processing and upsampling with concatenation
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        # Final Block: Combines all features and produces the output
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )

        # Initialize weights using He initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of convolutional and transpose convolutional layers 
        using He (Kaiming) normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the HN network.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        # Encoder pathway with successive pooling
        pool1 = self._block1(x)  # Downsample by factor of 2
        pool2 = self._block2(pool1)  # Downsample by factor of 2
        pool3 = self._block2(pool2)  # Downsample by factor of 2
        pool4 = self._block2(pool3)  # Downsample by factor of 2
        pool5 = self._block2(pool4)  # Bottom of the U-Net

        # Decoder pathway with upsampling and skip connections
        upsample5 = self._block3(pool5)  # Upsample by factor of 2
        concat5 = torch.cat((upsample5, pool4), dim=1)  # Concatenate skip connection
        upsample4 = self._block4(concat5)  # Upsample by factor of 2
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)  # Upsample by factor of 2
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)  # Upsample by factor of 2
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)  # Upsample by factor of 2
        concat1 = torch.cat((upsample1, x), dim=1)  # Final concatenation with input

        # Final processing to generate output
        return self._block6(concat1)
