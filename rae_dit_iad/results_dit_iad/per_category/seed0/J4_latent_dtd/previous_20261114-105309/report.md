# RAE-DiT IAD results (J4_latent_dtd)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.9543 | 0.9978 | 0.9978 | 0.9855 | 0.9989 | 0.9988 | 0.9915 | 0.6405 | 0.9568 |
| single | 1.0 | 1 | best | latent_cos | 0.9993 | 0.9990 | 0.9992 | 0.9996 | 0.9995 | 0.9996 | 0.9866 | 0.4140 | 0.9499 |
| single | 1.0 | 1 | best | pixel | 0.6031 | 0.7688 | 0.7589 | 0.7722 | 0.9043 | 0.8928 | 0.7695 | 0.2577 | 0.5579 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.7845 | 0.6898 | 0.7733 | 0.9206 | 0.8713 | 0.9144 | 0.6582 | 0.1061 | 0.3097 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 0.9996 | 0.9996 | 1.0000 | 0.9999 | 0.9999 | 0.9853 | 0.4036 | 0.9457 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.8864 | 0.9972 | 0.9980 | 0.9688 | 0.9991 | 0.9994 | 0.9961 | 0.8346 | 0.9787 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4712 | 0.6274 | 0.5414 | 0.6963 | 0.8480 | 0.7863 | 0.7435 | 0.1106 | 0.4860 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9878 | 0.2975 | 0.9559 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.9958 | 1.0000 | 1.0000 | 0.9985 | 1.0000 | 1.0000 | 0.9866 | 0.2624 | 0.9457 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | pixel | 0.5536 | 0.9893 | 0.9621 | 0.6997 | 0.9938 | 0.9778 | 0.9068 | 0.5564 | 0.8782 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | latent_cos | 0.9979 | 0.9975 | 0.9979 | 0.9988 | 0.9986 | 0.9988 | 0.9868 | 0.5410 | 0.9480 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | disc | 0.9807 | 0.9961 | 0.9954 | 0.9893 | 0.9977 | 0.9972 | 0.9918 | 0.8243 | 0.9459 |
