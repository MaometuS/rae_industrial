# RAE-DiT IAD results (C_image_l1_decfinetune)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.7125 | 0.8553 | 0.8399 | 0.8477 | 0.9484 | 0.9433 | 0.7192 | 0.0515 | 0.3870 |
| single | 0.1 | 1 | best | latent_cos | 0.2206 | 0.3830 | 0.3644 | 0.6212 | 0.7278 | 0.7234 | 0.2819 | 0.0071 | 0.0521 |
| single | 0.1 | 1 | best | pixel | 0.6373 | 0.6756 | 0.7195 | 0.8179 | 0.8531 | 0.8606 | 0.7825 | 0.1719 | 0.5518 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.4326 | 0.4599 | 0.4607 | 0.7098 | 0.7565 | 0.7290 | 0.5964 | 0.0220 | 0.1843 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.0919 | 0.2155 | 0.1673 | 0.5752 | 0.6396 | 0.6274 | 0.2339 | 0.0095 | 0.0355 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9029 | 0.8660 | 0.8535 | 0.9700 | 0.9606 | 0.9576 | 0.6439 | 0.0236 | 0.2272 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.8421 | 0.8914 | 0.9783 | 0.9260 | 0.9497 | 0.9921 | 0.9687 | 0.3218 | 0.9194 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.3492 | 0.5505 | 0.5614 | 0.6671 | 0.8161 | 0.8194 | 0.3299 | 0.0048 | 0.0686 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.5221 | 0.8446 | 0.8262 | 0.7254 | 0.9362 | 0.9290 | 0.7945 | 0.0794 | 0.5468 |
