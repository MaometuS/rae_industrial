# RAE-DiT IAD results (K2_latent_adaptor_sigma05_15)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.9743 | 0.6762 | 0.7484 | 0.9870 | 0.8635 | 0.8996 | 0.8582 | 0.1210 | 0.7679 |
| single | 1.0 | 1 | best | latent_cos | 0.9956 | 1.0000 | 1.0000 | 0.9986 | 1.0000 | 1.0000 | 0.9554 | 0.2628 | 0.8125 |
| single | 1.0 | 1 | best | pixel | 0.5128 | 0.5165 | 0.5566 | 0.8252 | 0.8106 | 0.8422 | 0.3638 | 0.0119 | 0.0768 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.5128 | 0.5165 | 0.5566 | 0.8252 | 0.8106 | 0.8422 | 0.3638 | 0.0119 | 0.0768 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9956 | 1.0000 | 1.0000 | 0.9986 | 1.0000 | 1.0000 | 0.9554 | 0.2628 | 0.8125 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9743 | 0.6762 | 0.7484 | 0.9870 | 0.8635 | 0.8996 | 0.8582 | 0.1210 | 0.7679 |
