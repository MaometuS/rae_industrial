# RAE-DiT IAD results (K2_latent_adaptor_sigma05_15)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.9292 | 0.9310 | 0.9405 | 0.9610 | 0.9738 | 0.9770 | 0.9250 | 0.2268 | 0.8433 |
| single | 1.0 | 1 | best | latent_cos | 0.9160 | 0.9184 | 0.9280 | 0.9617 | 0.9609 | 0.9670 | 0.9687 | 0.3115 | 0.8164 |
| single | 1.0 | 1 | best | pixel | 0.4142 | 0.6158 | 0.5831 | 0.6917 | 0.8299 | 0.8070 | 0.6450 | 0.0848 | 0.4048 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.5201 | 0.5161 | 0.5622 | 0.8285 | 0.8075 | 0.8440 | 0.3700 | 0.0120 | 0.0742 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9976 | 0.9960 | 0.9976 | 0.9993 | 0.9987 | 0.9993 | 0.9743 | 0.3832 | 0.8710 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9960 | 0.9189 | 0.9442 | 0.9987 | 0.9716 | 0.9803 | 0.9537 | 0.2112 | 0.8911 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4018 | 0.4311 | 0.4077 | 0.6928 | 0.7324 | 0.6939 | 0.5543 | 0.0079 | 0.2312 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 1.0000 | 0.9983 | 0.9992 | 1.0000 | 0.9994 | 0.9997 | 0.9641 | 0.1813 | 0.8705 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 1.0000 | 0.9875 | 0.9958 | 1.0000 | 0.9955 | 0.9985 | 0.9461 | 0.1863 | 0.8567 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | pixel | 0.4882 | 0.8936 | 0.8011 | 0.6494 | 0.9484 | 0.8965 | 0.8048 | 0.3089 | 0.7898 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | latent_cos | 0.9443 | 0.9332 | 0.9461 | 0.9709 | 0.9527 | 0.9643 | 0.9792 | 0.4839 | 0.8198 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | disc | 0.9818 | 0.9646 | 0.9696 | 0.9902 | 0.9814 | 0.9839 | 0.8819 | 0.2581 | 0.8726 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2466 | 0.6225 | 0.5616 | 0.5960 | 0.8312 | 0.7935 | 0.8511 | 0.0103 | 0.5241 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7223 | 0.7461 | 0.7690 | 0.8765 | 0.8927 | 0.9048 | 0.9572 | 0.1975 | 0.7042 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.7389 | 0.8530 | 0.8524 | 0.8551 | 0.9466 | 0.9453 | 0.9182 | 0.2514 | 0.7528 |
