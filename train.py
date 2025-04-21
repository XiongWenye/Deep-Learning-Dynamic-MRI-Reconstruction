import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from bme1312 import lab1 as lab
from bme1312.utils import image2kspace, kspace2image, pseudo2real, pseudo2complex, complex2pseudo
from torch.utils.tensorboard import SummaryWriter
from torch.fft import ifft2, fft2
from CS_mask import cartesian_mask

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import argparse
import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import math
from functools import partial

from bme1312.utils  import lr_scheduler,compute_psnr, compute_ssim

def variable_density_mask(shape, acceleration=5, center_lines=11, seed=42):
    """
    Generate a variable density random undersampling pattern
    
    Args:
        shape: Shape of the mask (batch, frames, height, width)
        acceleration: Acceleration factor (default: 5)
        center_lines: Number of central k-space lines to sample (default: 11)
        seed: Random seed for reproducibility
        
    Returns:
        Variable density undersampling mask with 1s in sampled positions and 0s elsewhere
    """
    np.random.seed(seed)
    mask = np.zeros(shape, dtype=np.float32)
    
    # Calculate how many lines to sample for each frame
    total_lines = shape[2]  # k-space height
    sampled_lines = total_lines // acceleration  # Number of lines to sample per frame
    
    # Ensure we have enough lines to sample
    if sampled_lines < center_lines:
        sampled_lines = center_lines
        print(f"Warning: Acceleration factor too high, setting to {total_lines / center_lines}")
    
    for b in range(shape[0]):
        for f in range(shape[1]):
            # Sample different lines for each frame for incoherent artifacts
            # Always include central k-space lines
            center_offset = total_lines // 2 - center_lines // 2
            mask[b, f, center_offset:center_offset+center_lines, :] = 1.0
            
            # Calculate number of remaining lines to sample
            remaining_lines = sampled_lines - center_lines
            
            # Create a probability density that decreases away from the center
            prob_density = np.zeros(total_lines)
            for i in range(total_lines):
                # Quadratic variable density
                dist_from_center = abs(i - total_lines//2) / (total_lines//2)
                prob_density[i] = (1.0 - dist_from_center)**2
                
            # Zero out the central lines that are already sampled
            prob_density[center_offset:center_offset+center_lines] = 0
            
            # Normalize to create a probability distribution
            prob_density = prob_density / np.sum(prob_density)
            
            # Randomly select remaining lines with higher probability for center
            random_lines = np.random.choice(
                np.arange(total_lines), 
                size=remaining_lines, 
                replace=False, 
                p=prob_density
            )
            
            # Fill in the selected lines
            mask[b, f, random_lines, :] = 1.0
    
    # Add visualization code to plot the mask
    if shape[0] > 0 and shape[1] > 0:
        plt.figure(figsize=(12, 5))
        
        # Plot mask for one dynamic frame
        plt.subplot(1, 2, 1)
        plt.imshow(mask[0, 0], cmap='gray')
        plt.title('Undersampling Mask (Single Frame)')
        plt.xlabel('kx')
        plt.ylabel('ky')
        
        # Plot ky-t dimension (central slice in kx)
        plt.subplot(1, 2, 2)
        ky_t_view = mask[0, :, :, mask.shape[3]//2]
        plt.imshow(ky_t_view, cmap='gray', aspect='auto')
        plt.title('ky-t Undersampling Pattern')
        plt.xlabel('t (frame)')
        plt.ylabel('ky')
        
        plt.savefig('undersampling_mask.png')
        plt.close()
    
    return mask


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        
        # Improved dropout strategy with slightly lower rate
        self.dropout = nn.Dropout(p=0.2)
        
        # Add Leaky ReLU instead of ReLU for better gradient flow
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # Add instance normalization option for better training stability
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Add attention mechanism to bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        self.attention = self._attention_block(features * 16)

        # Upsampling path with skip connections
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")

        # Add a deeper output layer
        self.pre_conv = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
        # Initialize weights for better training
        self._initialize_weights()

    def forward(self, x):
        # Add print statements to debug sizes
        batch_size = x.shape[0]

        # Encoder path
        enc1 = self.encoder1(x)
        enc1_shape = enc1.shape  # Debug shape
        enc1 = self.dropout(enc1)

        # Apply pooling and get encoder features
        pool1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        enc2_shape = enc2.shape  # Debug shape
        enc2 = self.dropout(enc2)

        pool2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        enc3_shape = enc3.shape  # Debug shape

        pool3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3)
        enc4_shape = enc4.shape  # Debug shape
        enc4 = self.dropout(enc4)

        # Bottleneck
        pool4 = self.pool4(enc4)
        bottleneck = self.bottleneck(pool4)
        bottleneck_shape = bottleneck.shape  # Debug shape
        bottleneck = self.attention(bottleneck)

        # Decoder path - make sure shapes are compatible for skip connections
        dec4 = self.upconv4(bottleneck)
        dec4_shape = dec4.shape  # Debug shape

        # Check for dimension mismatch and fix if necessary
        if dec4.shape[2:] != enc4.shape[2:]:
            # Resize dec4 to match spatial dimensions of enc4
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=True)

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        if dec3.shape[2:] != enc3.shape[2:]:
            # Resize dec3 to match spatial dimensions of enc3
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=True)
            
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            # Resize dec2 to match spatial dimensions of enc2
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
            
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            # Resize dec1 to match spatial dimensions of enc1
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
            
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Apply final convolutions
        out = self.activation(self.pre_conv(dec1))
        return torch.sigmoid(self.conv(out))

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict([
                (name + "conv1", nn.Conv2d(in_channels=in_channels, out_channels=features, 
                                         kernel_size=3, padding=1, bias=False)),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
                (name + "conv2", nn.Conv2d(in_channels=features, out_channels=features, 
                                         kernel_size=3, padding=1, bias=False)),
                (name + "norm2", nn.BatchNorm2d(num_features=features)),
                (name + "relu2", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ])
        )
        
    def _attention_block(self, features):
        # Simple channel attention mechanism
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(features, features // 8, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(features // 8, features, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)





# 3D ResNet implementation

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=20,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()

        if isinstance(block_inplanes, int):
            block_inplanes = [block_inplanes, block_inplanes*2, block_inplanes*4, block_inplanes*8]
            
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        # Full 3D convolution
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(3, 7, 7),
                               stride=(1, 2, 2),
                               padding=(1, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.conv2 = nn.Conv3d(self.in_planes, 20, kernel_size=1)  
        
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out
        
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True) # Upsample spatial dims
        x = self.conv2(x)
        return torch.sigmoid(x)
    
def resnet18(**kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], [32, 64, 128, 256], **kwargs)
    return model


def imsshow(imgs, flag, titles=None, num_col=5, dpi=100, cmap=None, is_colorbar=False, is_ticks=False, save_path=None):
    num_imgs = len(imgs)
    num_row = math.ceil(num_imgs / num_col)
    fig_width = num_col * 3
    if is_colorbar:
        fig_width += num_col * 1.5
    fig_height = num_row * 3
    fig = plt.figure(dpi=dpi, figsize=(fig_width, fig_height))
    for i in range(num_imgs):
        ax = plt.subplot(num_row, num_col, i + 1)
        im = ax.imshow(imgs[i], cmap=cmap)
        if titles:
            plt.title(titles[i])
        if is_colorbar:
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
            plt.colorbar(im, cax=cax)
        if not is_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    # plt.show()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.close('all')

def process_data():
    dataset = np.load('./cine.npz')['dataset']
    labels = torch.Tensor(dataset)
    
    # Create variable density mask with acceleration factor 5
    mask = variable_density_mask(shape=(1, 20, 192, 192), acceleration=5, center_lines=11)
    mask = torch.Tensor(mask)
    
    # Apply mask to k-space data
    inputs_k = lab.image2kspace(labels)
    inputs_k = inputs_k * mask
    inputs = lab.kspace2image(inputs_k)
    inputs = lab.complex2pseudo(inputs)
    
    inputs2 = lab.pseudo2real(inputs).unsqueeze(2)
    inputs = torch.cat((inputs, inputs2), dim=2)
    
    # Visualize original and undersampled images
    for i in range(min(3, labels.shape[0])):  # Show first 3 images
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(labels[i, 0].numpy(), cmap='gray')
        plt.title('Fully Sampled Image')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(inputs[i, 0, 0].numpy(), cmap='gray')
        plt.title('Undersampled Image (Real)')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(mask[0, 0].numpy()), cmap='gray')
        plt.title('Sampling Mask')
        plt.colorbar()
        
        plt.savefig(f'comparison_image_{i}.png')
        plt.close()
    
    return (inputs, labels, mask) 

def train(in_channels, 
          out_channels,
          init_features,
          num_epochs,
          weight_decay,
          batch_size,
          initial_lr,
          loss_tpe
          ):
    inputs, labels,mask = process_data()
    model = UNet(in_channels= in_channels, out_channels = out_channels, init_features = init_features)
    model2 = UNet(in_channels= in_channels, out_channels = out_channels, init_features = init_features)
    model3=resnet18(n_input_channels=20)

    parser = argparse.ArgumentParser(description='Save images to specified folder structure.')
    parser.add_argument('folder_name', type=str, help='Name of the folder to save images.')
    args = parser.parse_args()
    
    base_dir = os.path.join('images', args.folder_name)
    sub_dirs = ['under_sampling', 'full_sampling', 'reconstruction']

    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

    
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        model = model.to('cuda')
        model2 = model2.to('cuda')
        model3 = model3.to('cuda')

    writer = SummaryWriter()

    if loss_tpe == 'L2':
        criterion = nn.MSELoss()
    elif loss_tpe == 'L1':
        criterion = nn.L1Loss()
    
    param_list = list(model.parameters()) + list(model2.parameters()) + list(model3.parameters())
    optimizer = optim.Adam(param_list, lr=initial_lr, weight_decay=weight_decay)
    # optimizer = optim.Adam(param_list, lr=initial_lr)


    dataset = TensorDataset(inputs, labels)

    train_size = 114
    val_size = 29
    test_size = 57
    
    warmup_epochs = 0.05*num_epochs
    warmup_lr = 1e-3*initial_lr

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    dataloader_test =  DataLoader(test_set, batch_size=batch_size, shuffle=True)


    for epoch in range(num_epochs):
        lr = lr_scheduler(epoch, warmup_epochs, warmup_lr, initial_lr, num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        train_loss = 0
        for x, y in dataloader_train:
            model.train()
            model2.train()
            model3.train()
        
            outputs1 = model(x[:, :, 0])
            outputs2 = model2(x[:, :, 1])

            tmp = torch.stack((outputs1, outputs2), dim=2)

            outputs = model3(lab.pseudo2real(tmp).unsqueeze(2))
            
            outputs=outputs.squeeze(2)
            
            loss = criterion(outputs, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(dataloader_train)
        
        model.eval()
        model2.eval()
        model3.eval()
        
        val_loss = 0
        with torch.no_grad():
            for x, y in dataloader_val:
                outputs1 = model(x[:, :, 0])
                outputs2 = model2(x[:, :, 1])

            tmp = torch.stack((outputs1, outputs2), dim=2)

            outputs = model3(lab.pseudo2real(tmp).unsqueeze(2))
            
            outputs=outputs.squeeze(2)
                
            val_loss += criterion(outputs, y).item()
        val_loss /= len(dataloader_val)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    model.eval()
    model2.eval()
    model3.eval()
    a = []
    b = []
    c = []
    index=0
    with torch.no_grad():
        for x, y in dataloader_test:
            
            outputs1 = model(x[:, :, 0])
            outputs2 = model2(x[:, :, 1])

            tmp = torch.stack((outputs1, outputs2), dim=2)

            outputs = model3(lab.pseudo2real(tmp).unsqueeze(2))
            
            outputs=outputs.squeeze(2)
            
            under_sampling_filename = os.path.join(base_dir, 'under_sampling', f'under_sampling_{index}.png')
            full_sampling_filename = os.path.join(base_dir, 'full_sampling', f'full_sampling_{index}.png')
            reconstruction_filename = os.path.join(base_dir, 'reconstruction', f'reconstruction_{index}.png')
            
            imsshow(x[0, :, 2].to('cpu').numpy(), num_col=5, cmap='gray', flag="under_sampling", is_colorbar=True,  save_path=under_sampling_filename)
            imsshow(y[0].to('cpu').numpy(), num_col=5, cmap='gray', flag="full_sampling", is_colorbar=True, save_path=full_sampling_filename)
            imsshow(outputs[0].to('cpu').numpy(), num_col=5, cmap='gray', flag="reconstruction", is_colorbar=True,  save_path=reconstruction_filename)
            for i in range(x.size(0)):
                a.append( criterion(outputs[i], y[i]).item() )
                b.append(compute_psnr(outputs[i].to('cpu').numpy(), y[i].to('cpu').numpy()))
                c.append(compute_ssim(outputs[i].to('cpu').numpy(), y[i].to('cpu').numpy()))
            index+=1
        
        a = torch.Tensor(a)
        b = torch.Tensor(b)
        c = torch.Tensor(c)
        a_mean = torch.mean(a)
        a_std = torch.std(a, dim = 0)
        b_mean = torch.mean(b)
        b_std = torch.std(b, dim = 0)
        c_mean = torch.mean(c)
        c_std = torch.std(c, dim = 0)
        
        print(f'loss: mean = {a_mean}, std = {a_std}')
        print(f'PSNR: mean = {b_mean}, std = {b_std}')
        print(f'SSIM: mean = {c_mean}, std = {c_std}')
    writer.close()
    filename = f'./saved_model'
    i = 1
    while os.path.exists(filename):
        filename = f'./saved_model_{i}'
        i += 1
    torch.save(model.state_dict(), filename)
    
    output_dir = "output"
    output_file = "output.txt"
    output_path = os.path.join(output_dir, output_file)

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "a") as file:
        epoch_output = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}'
        print(epoch_output)
        file.write(epoch_output + "\n")
        
        loss_output = f'loss: mean = {a_mean}, std = {a_std}'
        psnr_output = f'PSNR: mean = {b_mean}, std = {b_std}'
        ssim_output = f'SSIM: mean = {c_mean}, std = {c_std}'
        
        print(loss_output)
        print(psnr_output)
        print(ssim_output)
        
        file.write(loss_output + "\n")
        file.write(psnr_output + "\n")
        file.write(ssim_output + "\n")
    

train(in_channels=20,
      out_channels=20,
      init_features=64,
      num_epochs=300,
      weight_decay=1e-4,
      batch_size=10,
      initial_lr=1e-4,
      loss_tpe='L2'
    )