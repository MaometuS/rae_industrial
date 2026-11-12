# RAE-DiT IAD results (G_image_l1_adaptor_decfinetune)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5425 | 0.4895 | 0.4658 | 0.7167 | 0.6581 | 0.6454 | 0.5034 | 0.0171 | 0.0579 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3872 | 0.4296 | 0.3900 | 0.6029 | 0.6326 | 0.6151 | 0.4118 | 0.0176 | 0.1012 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4542 | 0.4904 | 0.4367 | 0.6650 | 0.6989 | 0.6764 | 0.6843 | 0.1135 | 0.4437 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.2729 | 0.3604 | 0.3335 | 0.6557 | 0.7153 | 0.7091 | 0.3411 | 0.0171 | 0.1008 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.1754 | 0.4783 | 0.3535 | 0.6143 | 0.7937 | 0.7379 | 0.5554 | 0.0171 | 0.1191 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.4117 | 0.4254 | 0.3921 | 0.7576 | 0.7304 | 0.7231 | 0.6267 | 0.0194 | 0.0403 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.2381 | 0.5029 | 0.4603 | 0.5965 | 0.7881 | 0.7405 | 0.7303 | 0.0364 | 0.4643 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5288 | 0.3417 | 0.3358 | 0.7431 | 0.6468 | 0.6392 | 0.5543 | 0.0074 | 0.1699 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.4695 | 0.5372 | 0.4946 | 0.7183 | 0.7435 | 0.7016 | 0.4525 | 0.0057 | 0.0830 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.7246 | 0.8746 | 0.8061 | 0.8295 | 0.9344 | 0.8942 | 0.9369 | 0.4278 | 0.8672 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4064 | 0.4400 | 0.3904 | 0.5678 | 0.6038 | 0.5956 | 0.2501 | 0.0131 | 0.0275 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7539 | 0.3879 | 0.3700 | 0.8381 | 0.5745 | 0.5746 | 0.6193 | 0.0256 | 0.0584 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.5058 | 0.2546 | 0.1609 | 0.7725 | 0.6258 | 0.5862 | 0.8251 | 0.0124 | 0.5366 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4310 | 0.4894 | 0.4509 | 0.7370 | 0.7776 | 0.7526 | 0.1811 | 0.0014 | 0.0031 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.5438 | 0.6083 | 0.6270 | 0.8010 | 0.8241 | 0.8417 | 0.4842 | 0.0022 | 0.0549 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.5296 | 0.4596 | 0.4229 | 0.4709 | 0.4310 | 0.4521 | 0.5880 | 0.0740 | 0.2497 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3942 | 0.3988 | 0.4192 | 0.3524 | 0.3413 | 0.3503 | 0.5180 | 0.0492 | 0.1864 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5338 | 0.4888 | 0.4454 | 0.4684 | 0.4179 | 0.3858 | 0.3342 | 0.0325 | 0.0528 |
