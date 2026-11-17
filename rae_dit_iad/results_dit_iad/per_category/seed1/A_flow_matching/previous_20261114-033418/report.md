# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.8859 | 0.9652 | 0.9749 | 0.9280 | 0.9736 | 0.9832 | 0.9685 | 0.4454 | 0.8873 |
| single | 0.5 | 1 | best | pixel | 0.4273 | 0.6092 | 0.5988 | 0.6437 | 0.7613 | 0.7599 | 0.6383 | 0.1170 | 0.4190 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.4462 | 0.4631 | 0.4956 | 0.7874 | 0.7623 | 0.8044 | 0.4152 | 0.0137 | 0.1174 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9502 | 0.9908 | 0.9956 | 0.9862 | 0.9972 | 0.9987 | 0.9809 | 0.4898 | 0.9069 |
| grid | single | 0.5 | 1 | grid_best.pt | pixel | 0.4386 | 0.5129 | 0.4770 | 0.7043 | 0.7765 | 0.7468 | 0.6486 | 0.0322 | 0.3525 |
| grid | single | 0.5 | 1 | grid_best.pt | latent_cos | 0.9875 | 0.9992 | 1.0000 | 0.9955 | 0.9997 | 1.0000 | 0.9834 | 0.2107 | 0.9389 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | pixel | 0.4664 | 0.9321 | 0.8568 | 0.6208 | 0.9633 | 0.9228 | 0.8562 | 0.4559 | 0.8361 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | latent_cos | 0.9057 | 0.9675 | 0.9868 | 0.9518 | 0.9774 | 0.9925 | 0.9934 | 0.7671 | 0.9555 |
| screw | single | 0.5 | 1 | screw_best.pt | pixel | 0.3620 | 0.7012 | 0.6995 | 0.6771 | 0.8653 | 0.8709 | 0.8983 | 0.0380 | 0.6533 |
| screw | single | 0.5 | 1 | screw_best.pt | latent_cos | 0.7053 | 0.9373 | 0.9328 | 0.8425 | 0.9784 | 0.9745 | 0.9721 | 0.3442 | 0.8913 |
| transistor | single | 0.5 | 1 | transistor_best.pt | pixel | 0.4233 | 0.4367 | 0.4650 | 0.4291 | 0.4392 | 0.4546 | 0.3731 | 0.0453 | 0.1359 |
| transistor | single | 0.5 | 1 | transistor_best.pt | latent_cos | 0.8808 | 0.9313 | 0.9592 | 0.8640 | 0.9152 | 0.9502 | 0.9130 | 0.4151 | 0.7438 |
