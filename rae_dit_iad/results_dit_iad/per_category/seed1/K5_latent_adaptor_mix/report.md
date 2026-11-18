# RAE-DiT IAD results (K5_latent_adaptor_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.2203 | 0.9077 | 0.8820 | 0.7059 | 0.9734 | 0.9670 | 0.9853 | 0.7060 | 0.9361 |
| single | 1.0 | 1 | best | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9804 | 0.3868 | 0.9229 |
| single | 1.0 | 1 | best | pixel | 0.6392 | 0.6437 | 0.6754 | 0.8761 | 0.8631 | 0.8821 | 0.6779 | 0.1018 | 0.3350 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6392 | 0.6437 | 0.6754 | 0.8761 | 0.8631 | 0.8821 | 0.6779 | 0.1018 | 0.3350 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9804 | 0.3868 | 0.9229 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2203 | 0.9077 | 0.8820 | 0.7059 | 0.9734 | 0.9670 | 0.9853 | 0.7060 | 0.9361 |
