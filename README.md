# Deep-Learning-Dynamic-MRI-Reconstruction

<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/pipeline.png" alt="Pipeline">
    <p class="caption">Fig: Overall architecture of our proposed reconstruction network with dual UNet branches for real and imaginary components and 3D ResNet for temporal fusion</p>
</div>

This is a repository for the project "Deep Learning for Dynamic MRI Reconstruction" as part of the course BME1312 Artificial Intelligence in Biomedical Imaging at ShanghaiTech University. The project focuses on using deep learning techniques to reconstruct dynamic MRI images from undersampled data.

This project uses deep learning to reconstruct high-quality dynamic MRI images from undersampled data. We propose a deep-learning-based denoising framework combining two independent UNet modules and a 3D ResNet to explore the temporal correlation.
We generate variable density undersampling patterns with acceleration factor 5 and 11 central k-space lines, analyze the resulting aliasing artifacts, and evaluate reconstruction performance with PSNR and SSIM metrics. Additionally, we investigate the effects of dropout, dynamic learning rate schedules and compare L1 versus L2 losses.

## TO START
1. Clone the repository and download the dataset from [here](https://drive.google.com/file/d/1bhTKXgJm4aL1C5ollUoRh1JLarJO9Yxu/view?usp=sharing)

Our dataset cine. npz is a fully sampled cardiac cine MR image with the size of [nsamples, nt, nx, ny]
where nsamples is the number of samples, nt is the number of frames, nx and ny are the dimensions of the image. 

```bash

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python train.py output
```

After that, you can find the undersampled images, reconstructed images in the image folder, and the training log in the output folder. We also provide the full sampling images and both real and imaginary parts of the UNet-reconstructed images in the image folder for reference.

4. Analyze the results by comparing the reconstructed images with the original images. You can use metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) to evaluate the quality of the reconstructions, both of them are provided in the output.txt file.

## Variable Density Random Undersampling Pattern Generation

We generate a variable density random undersampling pattern (U) with the size of 
the given cine images for acceleration factor of 5. Eleven central k-space lines are sampled 
for each frame. Each sampling pattern must be a matrix with 1s in the sampled positions 
and  0s  in  the  remaining  ones. 

We also plot the undersampling mask for one dynamic frame and 
undersampling masks in the ky-t dimension.
![Undersampling Mask](https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/undersampling_mask.png)

We also obtained the aliased images as a result of the undersampling process with the generated patterns. For this we use the formula:  

$$
b = F^{-1} \cdot U \cdot F \cdot m
$$  

where $b$ is the aliased image, $F$ is the Fourier transform, $U$ is the undersampling mask, and $m$ is the original image. The aliased images are then used as input to the deep learning model for reconstruction.

Below are some examples of the aliased images generated from the original images.
<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/under_sampling_1.png" alt="Aliased Image 1">
</div>

<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/under_sampling_5.png" alt="Aliased Image 2">
    <p class="caption">Fig: Aliased image resulting from 5x undersampling of the cardiac MRI data</p>
</div>

And here are the comparison of the aliased images with the original images. We also show the sampling masks for some frames. It is noticeable that different frames have different sampling masks, which is a key feature of our approach to Deep Learning based reconstruction.

<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/comparison_image_0.png" alt="Undersampling Patterns Frame 0">
    <p class="caption">Fig: Comparison between fully sampled (left), undersampled (middle), and corresponding sampling mask (right) for frame 0</p>
</div>

<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/comparison_image_1.png" alt="Undersampling Patterns Frame 1">
    <p class="caption">Fig: Comparison between fully sampled (left), undersampled (middle), and corresponding sampling mask (right) for frame 1</p>
</div>

<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/comparison_image_2.png" alt="Undersampling Patterns Frame 2">
    <p class="caption">Fig: Comparison between fully sampled (left), undersampled (middle), and corresponding sampling mask (right) for frame 2</p>
</div>

<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/mask.png" alt="Multiple Sampling Masks">
    <p class="caption">Fig: Multiple sampling masks showing the variable density patterns across different temporal frames</p>
</div>

It is also clear to see that, for different dynamic frames, the undersampling masks are different.

## Reconstruction Network

All the details of the network are in the train.py file. 

To explore the temporal correlation, we chose to stack the dynamic images along the channel dimension. However, this brought out a problem as the input image is pseudo-complex, and the real and imaginary parts are not aligned. To solve this, we split the input into two branches, one for the real part and one for the imaginary part. The two branches are then concatenated at the end of the UNet structure. We added attention mechanisms to the bottleneck layer of the UNet structure to better capture the spatial correlation and channel correlation.   

However, UNet is a 2D structure, and the temporal correlation is not well captured. 
To solve this, we added a 3D ResNet structure after the UNet structure to better achieve this goal.

So in general, the reconstruction network consists of three components:

### Dual 2D UNet Architecture (Real & Imaginary Components)
Purpose: Process the real and imaginary parts of the complex MRI data  

Features:
- Encoder-decoder structure with skip connections
- Attention mechanism in the bottleneck layer
- Dropout (p=0.3) for regularization
- LeakyReLU activation (negative_slope=0.1)
- Weight Regularization for better training stability
- Channel and spatial attention modules

### 3D ResNet (Temporal Fusion)
Purpose: Integrate temporal information across the MRI sequence  

Features:
- 3D convolutions to process the temporal dimension
- Residual connections for better gradient flow
- Lightweight design with one residual block per layer
- Final 1×1×1 convolution to map features to output channels

The whole structure is shown in the figure below.
<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/pipeline.png" alt="Reconstruction Network">
    <p class="caption">Fig: Detailed architecture of our reconstruction network showing dual UNet branches for processing real and imaginary components separately, followed by a 3D ResNet for temporal fusion across frames</p>
</div>

### Training and Evaluation
Below are the details of the training parameters:
``` python
train(in_channels=20,
      out_channels=20,
      init_features=64,
      num_epochs=800,
      weight_decay=1e-4,
      batch_size=10,
      initial_lr=1e-4,
      loss_tpe='L2'
    )
```

Using the above parameters, we achieved a PSNR of 29.08446121 and SSIM of 0.84434632, which is a remarkable improvement over the aliased images. The whole training process took about 2 hours on a single NVIDIA RTX 2080 Ti GPU. More detailed results can be found in the output.txt file. We are also happy to show you some of the reconstructed images compared to the original images.
<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/reconstruction_1.png" alt="Reconstructed Image1">
    <figcaption>Fig 1: Reconstructed cardiac MRI image using our deep learning model</figcaption>
</div>

<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/full_sampling_1.png" alt="Full Sampling Image1">
    <figcaption>Fig 2: Fully sampled reference cardiac MRI image (ground truth)</figcaption>
</div>

<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/reconstruction_1.png" alt="Reconstructed Image2">
    <figcaption>Fig 3: Another view of the reconstructed cardiac MRI image</figcaption>
</div>

<div class="figure">
    <img src="https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/full_sampling_1.png" alt="Full Sampling Image2">
    <figcaption>Fig 4: Corresponding fully sampled reference image for comparison</figcaption>
</div>

## Discussion on the Effect of Dropout, Dynamic Learning Rate Schedules, and Loss Functions

## Unrolled Denoising Network with Data Consistency Layer

TODO: Finsh the README file based on the Headings above
