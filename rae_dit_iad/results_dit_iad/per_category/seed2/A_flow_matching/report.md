# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.9741 | 0.9982 | 0.9988 | 0.9924 | 0.9994 | 0.9996 | 0.9818 | 0.3064 | 0.9248 |
| single | 0.5 | 1 | best | pixel | 0.4416 | 0.4879 | 0.4943 | 0.7423 | 0.7724 | 0.7714 | 0.5716 | 0.0349 | 0.2741 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.4238 | 0.4270 | 0.4908 | 0.7653 | 0.7519 | 0.7865 | 0.4478 | 0.0147 | 0.1256 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9607 | 0.9964 | 0.9976 | 0.9890 | 0.9989 | 0.9993 | 0.9810 | 0.4087 | 0.9117 |
| grid | single | 0.5 | 1 | grid_best.pt | pixel | 0.4595 | 0.5489 | 0.4979 | 0.7193 | 0.7928 | 0.7563 | 0.6954 | 0.0551 | 0.4226 |
| grid | single | 0.5 | 1 | grid_best.pt | latent_cos | 0.9875 | 1.0000 | 1.0000 | 0.9958 | 1.0000 | 1.0000 | 0.9826 | 0.2041 | 0.9379 |
