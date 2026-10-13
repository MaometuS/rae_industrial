# RAE reconstruction fidelity on MVTec AD

## Mean over categories

| model               |   normal_psnr |   normal_ssim |   normal_err_p99 |   anom_psnr_in_mask |   passthrough_ratio |   defect_px_above_normal_p99 |   pixel_auroc_raw |   image_auroc_raw |
|:--------------------|--------------:|--------------:|-----------------:|--------------------:|--------------------:|-----------------------------:|------------------:|------------------:|
| RAE-DINOv2-B        |        19.882 |         0.559 |            0.183 |              17.487 |               1.416 |                        0.077 |             0.598 |             0.592 |
| RAE-DINOv2-B_512    |        20.819 |         0.665 |            0.164 |              19.282 |               1.274 |                        0.069 |             0.582 |             0.488 |
| RAE-MAE             |        33.299 |         0.915 |            0.037 |              29.476 |               1.556 |                        0.182 |             0.69  |             0.617 |
| RAE-SigLIP2         |        23.803 |         0.612 |            0.126 |              20.472 |               1.635 |                        0.122 |             0.663 |             0.595 |
| SDVAE-sd-vae-ft-ema |        30.933 |         0.827 |            0.067 |              26.448 |               2.227 |                        0.176 |             0.725 |             0.551 |

## normal_psnr by category

| category   |   RAE-DINOv2-B |   RAE-DINOv2-B_512 |   RAE-MAE |   RAE-SigLIP2 |   SDVAE-sd-vae-ft-ema |
|:-----------|---------------:|-------------------:|----------:|--------------:|----------------------:|
| bottle     |         22.092 |             22.399 |    36.273 |        24.727 |                33.924 |
| cable      |         18.137 |             19.443 |    29.166 |        19.502 |                25.366 |
| capsule    |         17.648 |             18.729 |    36.241 |        26.039 |                40.176 |
| carpet     |         14.748 |             15.283 |    24.637 |        15.058 |                19.793 |
| grid       |         14.815 |             19     |    30.674 |        20.456 |                22.48  |
| hazelnut   |         21.841 |             22.49  |    37.149 |        28.305 |                37.366 |
| leather    |         24.166 |             24.091 |    31.677 |        26.685 |                28.128 |
| metal_nut  |         20.14  |             20.362 |    33.109 |        23.81  |                30.655 |
| pill       |         22.467 |             21.96  |    39.51  |        29.7   |                41.486 |
| screw      |         24.017 |             18.971 |    35.875 |        28.139 |                36.778 |
| tile       |         16.122 |             18.797 |    29.574 |        17.057 |                24.549 |
| toothbrush |         19.533 |             23.932 |    34.051 |        24.153 |                31.226 |
| transistor |         19.54  |             24.214 |    35.297 |        23.647 |                33.404 |
| wood       |         19.586 |             19.627 |    35.352 |        25.432 |                31.858 |
| zipper     |         23.377 |             22.992 |    30.898 |        24.343 |                26.814 |

## normal_ssim by category

| category   |   RAE-DINOv2-B |   RAE-DINOv2-B_512 |   RAE-MAE |   RAE-SigLIP2 |   SDVAE-sd-vae-ft-ema |
|:-----------|---------------:|-------------------:|----------:|--------------:|----------------------:|
| bottle     |          0.735 |              0.821 |     0.966 |         0.79  |                 0.942 |
| cable      |          0.547 |              0.648 |     0.917 |         0.563 |                 0.824 |
| capsule    |          0.844 |              0.87  |     0.952 |         0.868 |                 0.94  |
| carpet     |          0.046 |              0.195 |     0.859 |         0.043 |                 0.69  |
| grid       |          0.553 |              0.778 |     0.944 |         0.63  |                 0.796 |
| hazelnut   |          0.666 |              0.729 |     0.952 |         0.83  |                 0.925 |
| leather    |          0.285 |              0.476 |     0.772 |         0.354 |                 0.524 |
| metal_nut  |          0.542 |              0.614 |     0.917 |         0.702 |                 0.858 |
| pill       |          0.808 |              0.795 |     0.967 |         0.856 |                 0.964 |
| screw      |          0.867 |              0.858 |     0.969 |         0.882 |                 0.95  |
| tile       |          0.195 |              0.523 |     0.906 |         0.25  |                 0.75  |
| toothbrush |          0.74  |              0.738 |     0.951 |         0.729 |                 0.916 |
| transistor |          0.737 |              0.832 |     0.958 |         0.774 |                 0.941 |
| wood       |          0.39  |              0.586 |     0.902 |         0.46  |                 0.784 |
| zipper     |          0.433 |              0.508 |     0.796 |         0.452 |                 0.605 |

## passthrough_ratio by category

| category   |   RAE-DINOv2-B |   RAE-DINOv2-B_512 |   RAE-MAE |   RAE-SigLIP2 |   SDVAE-sd-vae-ft-ema |
|:-----------|---------------:|-------------------:|----------:|--------------:|----------------------:|
| bottle     |          1.703 |              1.763 |     1.936 |         1.774 |                 2.713 |
| cable      |          1.112 |              1.064 |     1.056 |         1.083 |                 1.009 |
| capsule    |          0.814 |              0.78  |     1.138 |         0.919 |                 1.979 |
| carpet     |          0.897 |              0.837 |     0.984 |         0.849 |                 1.069 |
| grid       |          1.12  |              1.251 |     1.352 |         1.347 |                 1.34  |
| hazelnut   |          1.618 |              1.36  |     2.211 |         3.145 |                 5.109 |
| leather    |          1.224 |              1.226 |     1.247 |         1.346 |                 1.246 |
| metal_nut  |          1.142 |              1.157 |     1.507 |         1.611 |                 1.907 |
| pill       |          1.641 |              1.368 |     1.713 |         1.972 |                 2.426 |
| screw      |          2.552 |              1.099 |     1.858 |         3.261 |                 4.558 |
| tile       |          0.892 |              1.148 |     1.043 |         0.853 |                 0.994 |
| toothbrush |          2.757 |              2.389 |     2.812 |         2.199 |                 4.088 |
| transistor |          0.874 |              0.939 |     1.329 |         1.09  |                 1.639 |
| wood       |          1.129 |              1.079 |     1.58  |         1.241 |                 1.604 |
| zipper     |          1.771 |              1.648 |     1.573 |         1.838 |                 1.719 |

## How to read this

- normal_psnr / normal_ssim: fidelity on defect-free images. Low values = the frozen decoder loses industrial detail -> false positives later. Compare RAE rows against the SD-VAE row.
- normal_err_p99: the reconstruction noise floor on normal images (99th pct of smoothed |x-r|).
- anom_psnr_in_mask / passthrough_ratio: how faithfully defects themselves are reproduced. Ratio ~1 = defects reconstructed as well as normal texture (expected for a plain RAE; the restorer has to learn removal).
- defect_px_above_normal_p99: share of defect pixels whose raw error already exceeds the normal noise floor.
- pixel/image_auroc_raw: raw reconstruction error as detector. Not the target metric; sanity check only.