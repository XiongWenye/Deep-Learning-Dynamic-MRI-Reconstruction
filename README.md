# Deep-Learning-Dynamic-MRI-Reconstruction

![Pipeline](https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/pipeline.png)

This is a repository for the project "Deep Learning for Dynamic MRI Reconstruction" as part of the course BME1312 Artificial Intelligence in Biomedical Imaging at ShanghaiTech University. The project focuses on using deep learning techniques to reconstruct dynamic MRI images from undersampled data.

This project uses deep learning to reconstruct high-quality dynamic MRI images from undersampled data. We propose a deep-learning-based denoising framework combining two independent UNet modules and a 3D ResNet to explore the temporal correlation.
We generate variable density undersampling patterns with acceleration factor 5 and 11 central k-space lines, analyze the resulting aliasing artifacts, and evaluate reconstruction performance with PSNR and SSIM metrics. Additionally, we investigate the effects of dropout, dynamic learning rate schedules and compare L1 versus L2 losses.

## TO START
1. Clone the repository and download the dataset from [here](https://drive.google.com/file/d/1bhTKXgJm4aL1C5ollUoRh1JLarJO9Yxu/view?usp=sharing)

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
![Undersampling Patterns](https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/comparison_image_0.png)
![Undersampling Patterns](https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/comparison_image_1.png)
![Undersampling Patterns](https://github.com/XiongWenye/xiongwenye.github.io/blob/master/files/Deep%20Learning%20Dynamic%20MRI%20Reconstruction/comparison_image_2.png)


It is also clear to see that, for different dynamic frames, the undersampling masks are different.

## Reconstruction Network

## Discussion on the Effect of Dropout, Dynamic Learning Rate Schedules, and Loss Functions

## Unrolled Denoising Network with Data Consistency Layer

TODO: Finsh the README file based on the Headings above
