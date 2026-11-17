# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.9478 | 0.9881 | 0.9956 | 0.9775 | 0.9926 | 0.9977 | 0.9858 | 0.4757 | 0.9338 |
| single | 0.5 | 1 | best | pixel | 0.4606 | 0.6398 | 0.6147 | 0.7087 | 0.8465 | 0.8212 | 0.6385 | 0.1608 | 0.4377 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.4342 | 0.4819 | 0.5124 | 0.7706 | 0.7931 | 0.8016 | 0.4158 | 0.0136 | 0.1114 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9494 | 0.9924 | 0.9960 | 0.9858 | 0.9977 | 0.9988 | 0.9814 | 0.4584 | 0.9052 |
| grid | single | 0.5 | 1 | grid_best.pt | pixel | 0.4444 | 0.5322 | 0.4804 | 0.7094 | 0.7983 | 0.7444 | 0.6525 | 0.0347 | 0.3600 |
| grid | single | 0.5 | 1 | grid_best.pt | latent_cos | 0.9858 | 0.9975 | 1.0000 | 0.9951 | 0.9991 | 1.0000 | 0.9828 | 0.2124 | 0.9379 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | pixel | 0.5032 | 0.9054 | 0.8514 | 0.6461 | 0.9482 | 0.9175 | 0.8471 | 0.4342 | 0.8417 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | latent_cos | 0.9082 | 0.9743 | 0.9907 | 0.9516 | 0.9809 | 0.9943 | 0.9934 | 0.7563 | 0.9583 |
