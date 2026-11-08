# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8639 | 0.9641 | 0.9749 | 0.9121 | 0.9714 | 0.9810 | 0.9629 | 0.4273 | 0.8787 |
| single | 0.5 | 1 | joint_best.pt | pixel | 0.4563 | 0.6194 | 0.6112 | 0.6557 | 0.7687 | 0.7605 | 0.6126 | 0.1103 | 0.4051 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | joint_best.pt | pixel | 0.4972 | 0.5100 | 0.5638 | 0.8098 | 0.8073 | 0.8307 | 0.3703 | 0.0120 | 0.0893 |
| carpet | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.9330 | 0.9876 | 0.9908 | 0.9809 | 0.9963 | 0.9972 | 0.9799 | 0.4867 | 0.9099 |
| grid | single | 0.5 | 1 | joint_best.pt | pixel | 0.4236 | 0.5464 | 0.4921 | 0.7006 | 0.7912 | 0.7433 | 0.6113 | 0.0165 | 0.3056 |
| grid | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.9758 | 0.9967 | 1.0000 | 0.9917 | 0.9988 | 1.0000 | 0.9814 | 0.2058 | 0.9385 |
| hazelnut | single | 0.5 | 1 | joint_best.pt | pixel | 0.4650 | 0.8886 | 0.8432 | 0.6187 | 0.9424 | 0.9142 | 0.8196 | 0.4423 | 0.8200 |
| hazelnut | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8921 | 0.9696 | 0.9889 | 0.9434 | 0.9760 | 0.9931 | 0.9934 | 0.7675 | 0.9589 |
| screw | single | 0.5 | 1 | joint_best.pt | pixel | 0.4075 | 0.6946 | 0.6551 | 0.6918 | 0.8711 | 0.8541 | 0.9043 | 0.0382 | 0.6670 |
| screw | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.6635 | 0.9233 | 0.9317 | 0.8216 | 0.9747 | 0.9761 | 0.9703 | 0.3118 | 0.8772 |
| transistor | single | 0.5 | 1 | joint_best.pt | pixel | 0.4883 | 0.4575 | 0.5017 | 0.4577 | 0.4315 | 0.4602 | 0.3575 | 0.0423 | 0.1436 |
| transistor | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8550 | 0.9433 | 0.9629 | 0.8229 | 0.9114 | 0.9383 | 0.8897 | 0.3647 | 0.7092 |
