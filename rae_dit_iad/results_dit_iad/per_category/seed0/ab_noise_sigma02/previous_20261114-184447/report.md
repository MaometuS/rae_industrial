# RAE-DiT IAD results (ab_noise_sigma02)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.2 | 1 | best | disc | 0.6904 | 0.7547 | 0.7600 | 0.8412 | 0.8975 | 0.9004 | 0.6085 | 0.0293 | 0.2510 |
| single | 0.2 | 1 | best | latent_cos | 0.1266 | 0.3759 | 0.3171 | 0.5713 | 0.7127 | 0.6828 | 0.0200 | 0.0058 | 0.0003 |
| single | 0.2 | 1 | best | pixel | 0.5938 | 0.6182 | 0.6744 | 0.7895 | 0.8380 | 0.8529 | 0.7373 | 0.1274 | 0.5158 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.2 | 1 | carpet_best.pt | pixel | 0.4157 | 0.4402 | 0.4482 | 0.6953 | 0.7660 | 0.7461 | 0.5283 | 0.0219 | 0.1594 |
| carpet | single | 0.2 | 1 | carpet_best.pt | latent_cos | 0.1112 | 0.4402 | 0.3760 | 0.5845 | 0.7774 | 0.7368 | 0.0255 | 0.0081 | 0.0005 |
| carpet | single | 0.2 | 1 | carpet_best.pt | disc | 0.8796 | 0.7600 | 0.7881 | 0.9627 | 0.9037 | 0.9203 | 0.4548 | 0.0136 | 0.0183 |
| grid | single | 0.2 | 1 | grid_best.pt | pixel | 0.7719 | 0.7962 | 0.9006 | 0.8836 | 0.9099 | 0.9597 | 0.9464 | 0.2328 | 0.8722 |
| grid | single | 0.2 | 1 | grid_best.pt | latent_cos | 0.1420 | 0.3116 | 0.2581 | 0.5581 | 0.6481 | 0.6288 | 0.0146 | 0.0035 | 0.0000 |
| grid | single | 0.2 | 1 | grid_best.pt | disc | 0.5013 | 0.7494 | 0.7318 | 0.7197 | 0.8914 | 0.8805 | 0.7622 | 0.0451 | 0.4837 |
