# RAE-DiT IAD results (F_image_l1_adaptor)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.4840 | 0.6456 | 0.6447 | 0.6730 | 0.7665 | 0.7716 | 0.7213 | 0.1472 | 0.4137 |
| single | 0.1 | 1 | best | latent_cos | 0.3421 | 0.3877 | 0.3752 | 0.5911 | 0.6162 | 0.6091 | 0.4854 | 0.0276 | 0.2899 |
| single | 0.1 | 1 | best | pixel | 0.4776 | 0.6634 | 0.6193 | 0.7060 | 0.8115 | 0.7842 | 0.8130 | 0.2338 | 0.6111 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.3527 | 0.4823 | 0.4402 | 0.6988 | 0.8126 | 0.7856 | 0.5805 | 0.1312 | 0.2909 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.0770 | 0.1148 | 0.1120 | 0.5706 | 0.5948 | 0.5875 | 0.3244 | 0.0106 | 0.0454 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.3804 | 0.4149 | 0.4234 | 0.7081 | 0.7304 | 0.7349 | 0.4795 | 0.0154 | 0.1805 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.7218 | 0.9006 | 0.9323 | 0.8594 | 0.9597 | 0.9761 | 0.9433 | 0.2771 | 0.8584 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.2264 | 0.2907 | 0.2648 | 0.5883 | 0.6192 | 0.6044 | 0.0914 | 0.0036 | 0.0031 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.6809 | 0.8363 | 0.8622 | 0.8737 | 0.9312 | 0.9460 | 0.8079 | 0.1100 | 0.6215 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.5082 | 0.8857 | 0.8311 | 0.6956 | 0.9444 | 0.9169 | 0.9438 | 0.4838 | 0.8537 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.5479 | 0.5432 | 0.5464 | 0.6797 | 0.7074 | 0.6964 | 0.7822 | 0.0493 | 0.5808 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.5536 | 0.8157 | 0.7961 | 0.7206 | 0.8851 | 0.8801 | 0.9633 | 0.4889 | 0.7834 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0023 | 0.3825 | 0.2273 | 0.5341 | 0.7326 | 0.6124 | 0.8870 | 0.0806 | 0.6057 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.4802 | 0.5188 | 0.5138 | 0.7176 | 0.7580 | 0.7472 | 0.9435 | 0.0444 | 0.7958 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.3038 | 0.6573 | 0.6355 | 0.6345 | 0.8320 | 0.8244 | 0.6354 | 0.0031 | 0.1224 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.8029 | 0.6658 | 0.6654 | 0.7420 | 0.6082 | 0.6301 | 0.7105 | 0.1963 | 0.4469 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.3792 | 0.4713 | 0.4392 | 0.3992 | 0.4017 | 0.4098 | 0.2856 | 0.0303 | 0.0243 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.5012 | 0.5038 | 0.5067 | 0.4281 | 0.4539 | 0.4724 | 0.7203 | 0.1187 | 0.3607 |
