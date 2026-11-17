# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.9477 | 0.9858 | 0.9946 | 0.9781 | 0.9917 | 0.9973 | 0.9861 | 0.4953 | 0.9352 |
| single | 0.5 | 1 | best | pixel | 0.4608 | 0.6409 | 0.6177 | 0.7115 | 0.8375 | 0.8287 | 0.6420 | 0.1694 | 0.4369 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.4627 | 0.4683 | 0.5104 | 0.7967 | 0.7627 | 0.8132 | 0.4193 | 0.0138 | 0.1200 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9486 | 0.9908 | 0.9956 | 0.9856 | 0.9972 | 0.9987 | 0.9815 | 0.5073 | 0.9109 |
| grid | single | 0.5 | 1 | grid_best.pt | pixel | 0.4495 | 0.5238 | 0.4837 | 0.7117 | 0.7878 | 0.7498 | 0.6481 | 0.0337 | 0.3533 |
| grid | single | 0.5 | 1 | grid_best.pt | latent_cos | 0.9875 | 0.9975 | 0.9992 | 0.9956 | 0.9991 | 0.9997 | 0.9833 | 0.2096 | 0.9389 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | pixel | 0.4704 | 0.9307 | 0.8589 | 0.6262 | 0.9620 | 0.9230 | 0.8585 | 0.4606 | 0.8373 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | latent_cos | 0.9071 | 0.9693 | 0.9889 | 0.9529 | 0.9788 | 0.9936 | 0.9934 | 0.7689 | 0.9556 |
