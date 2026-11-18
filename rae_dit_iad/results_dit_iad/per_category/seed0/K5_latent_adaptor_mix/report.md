# RAE-DiT IAD results (K5_latent_adaptor_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.5596 | 0.8678 | 0.8818 | 0.8332 | 0.9529 | 0.9577 | 0.9396 | 0.4420 | 0.8168 |
| single | 1.0 | 1 | best | latent_cos | 0.8405 | 0.9495 | 0.9507 | 0.9329 | 0.9828 | 0.9830 | 0.9794 | 0.3231 | 0.9049 |
| single | 1.0 | 1 | best | pixel | 0.4603 | 0.6592 | 0.6466 | 0.7440 | 0.8616 | 0.8510 | 0.7729 | 0.0585 | 0.4346 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6806 | 0.6802 | 0.7203 | 0.8938 | 0.8764 | 0.9009 | 0.6845 | 0.0956 | 0.3223 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9813 | 0.3809 | 0.9302 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.3006 | 0.9767 | 0.9727 | 0.7522 | 0.9932 | 0.9921 | 0.9879 | 0.7342 | 0.9526 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2400 | 0.6382 | 0.5729 | 0.5943 | 0.8467 | 0.8010 | 0.8613 | 0.0214 | 0.5469 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.6811 | 0.8990 | 0.9014 | 0.8658 | 0.9655 | 0.9661 | 0.9775 | 0.2653 | 0.8795 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.8186 | 0.7590 | 0.7909 | 0.9143 | 0.9127 | 0.9233 | 0.8912 | 0.1498 | 0.6810 |
