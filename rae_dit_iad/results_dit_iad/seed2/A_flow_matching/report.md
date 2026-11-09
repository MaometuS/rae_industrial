# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8601 | 0.9629 | 0.9725 | 0.9094 | 0.9707 | 0.9783 | 0.9607 | 0.4268 | 0.8754 |
| single | 0.5 | 1 | joint_best.pt | pixel | 0.4456 | 0.6201 | 0.6046 | 0.6475 | 0.7649 | 0.7587 | 0.6102 | 0.1110 | 0.4052 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | joint_best.pt | pixel | 0.5108 | 0.5482 | 0.5630 | 0.8189 | 0.8177 | 0.8333 | 0.3712 | 0.0122 | 0.0961 |
| carpet | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.9330 | 0.9892 | 0.9948 | 0.9808 | 0.9966 | 0.9984 | 0.9778 | 0.4711 | 0.9047 |
| grid | single | 0.5 | 1 | joint_best.pt | pixel | 0.4344 | 0.5221 | 0.5046 | 0.7081 | 0.7684 | 0.7524 | 0.6051 | 0.0167 | 0.3210 |
| grid | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.9774 | 0.9950 | 1.0000 | 0.9922 | 0.9982 | 1.0000 | 0.9819 | 0.2133 | 0.9407 |
| hazelnut | single | 0.5 | 1 | joint_best.pt | pixel | 0.4218 | 0.9186 | 0.8386 | 0.5853 | 0.9587 | 0.9198 | 0.8007 | 0.4532 | 0.8166 |
| hazelnut | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8939 | 0.9725 | 0.9900 | 0.9445 | 0.9779 | 0.9938 | 0.9937 | 0.7738 | 0.9608 |
| screw | single | 0.5 | 1 | joint_best.pt | pixel | 0.3546 | 0.6995 | 0.6606 | 0.6623 | 0.8640 | 0.8430 | 0.8936 | 0.0289 | 0.6444 |
| screw | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.6626 | 0.9377 | 0.9358 | 0.8236 | 0.9792 | 0.9777 | 0.9659 | 0.3106 | 0.8665 |
| transistor | single | 0.5 | 1 | joint_best.pt | pixel | 0.5063 | 0.4121 | 0.4562 | 0.4628 | 0.4155 | 0.4448 | 0.3806 | 0.0437 | 0.1478 |
| transistor | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8333 | 0.9200 | 0.9421 | 0.8060 | 0.9017 | 0.9214 | 0.8840 | 0.3651 | 0.7042 |
