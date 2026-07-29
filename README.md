# KernelGANplus
## Step 1. Train demosaicer with QuadPhase Kernels, and generate demosaiced real images.
Modify the test config to the following:
```bash
color_pipeline_pre: &color_pipeline_pre ["bwlv_norm_no_lower_clip"]
color_pipeline_post: ["clip"]
```

## Step 2. Slide demosaiced real images to 1000x1000.
Feel free to explore different slide size.
```bash
python3 slide.py --img_dir [demosaiced_image_dir] --img_dir [slided_images_dir]
```

## Step 3. Modify get_kernel_1x_20250609.py to the following and run KernelGAN:
```bash
input_dir = [slided_images_dir]
output_dir = [estimated_kernels_dir]
```
```bash
python3 get_kernel_1x_20250609.py
```

## Step 4. Visualize generated kernels and delete failed kernels from [estimated_kernels_dir].
```bash
python3 visualize_1x.py --kernel_dir [estimated_kernels_dir] --separate_RGB --save_dir [visualized_kernels_dir]
```
example:
```bash
python3 visualize_1x.py --kernel_dir /group-volume/Fengjia-Contents/HexaDecaZoom/kernelGAN/binary_202504_burstHDR-vtt0424-12MP_evneg_v4_slide_outs1x_RGBv3 --separate_RGB --save_dir /group-volume/Fengjia-Contents/HexaDecaZoom/kernelGAN/binary_202504_burstHDR-vtt0424-12MP_evneg_v4_slide_outs1x_RGBv3_visualize
```

## Step 5. Save filtered kernels to [filtered_kernels_binary].
```bash
python3 save_kernel_np_1x.py --kernel_dir [estimated_kernels_dir] --save_np [filtered_kernels_binary]
```
example:
```bash
python3 save_kernel_np_1x.py --kernel_dir /group-volume/Fengjia-Contents/HexaDecaZoom/kernelGAN/binary_202504_burstHDR-vtt0424-12MP_evneg_v4_slide_outs1x_RGBv3 --save_np VTT0424_evneg_1x_RGB_v3.npy
```

## Step 6. Retrain demosaicer with [filtered_kernels_binary].
Add npy file [filtered_kernels_binary] to ISP_tools/unprocessing/degradation_tools/kernel/S25wideHexadeca/KernelGAN/, and change the train config to the following:
```bash
datasets:
  train:
    UNP:
      kernel:
        model: KernelGANQuadPhaseocl
        calib_kernel_path: [filtered_kernels_binary]
```
