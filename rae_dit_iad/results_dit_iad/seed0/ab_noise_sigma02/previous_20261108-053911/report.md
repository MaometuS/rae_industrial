# RAE-DiT IAD results (ab_noise_sigma02)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.2 | 1 | joint_best.pt | disc | 0.6092 | 0.4829 | 0.4891 | 0.7516 | 0.6582 | 0.6647 | 0.4440 | 0.0172 | 0.0655 |
| single | 0.2 | 1 | joint_best.pt | latent_cos | 0.4851 | 0.4828 | 0.5094 | 0.6833 | 0.6626 | 0.6873 | 0.4864 | 0.0353 | 0.1973 |
| single | 0.2 | 1 | joint_best.pt | pixel | 0.5548 | 0.6351 | 0.6308 | 0.7149 | 0.7564 | 0.7545 | 0.5539 | 0.0964 | 0.2786 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.2 | 1 | joint_best.pt | pixel | 0.5016 | 0.5421 | 0.5433 | 0.7919 | 0.8114 | 0.7983 | 0.3369 | 0.0114 | 0.0665 |
| carpet | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.1954 | 0.4053 | 0.3800 | 0.6379 | 0.7430 | 0.7459 | 0.6680 | 0.0644 | 0.2557 |
| carpet | single | 0.2 | 1 | joint_best.pt | disc | 0.7440 | 0.4338 | 0.4783 | 0.9139 | 0.7316 | 0.7530 | 0.6866 | 0.0230 | 0.0834 |
| grid | single | 0.2 | 1 | joint_best.pt | pixel | 0.3567 | 0.4428 | 0.4219 | 0.6684 | 0.7203 | 0.6980 | 0.6314 | 0.0100 | 0.2927 |
| grid | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.6917 | 0.4403 | 0.5464 | 0.8803 | 0.7375 | 0.7975 | 0.7266 | 0.0374 | 0.3302 |
| grid | single | 0.2 | 1 | joint_best.pt | disc | 0.4528 | 0.5004 | 0.4946 | 0.6914 | 0.7296 | 0.7057 | 0.4609 | 0.0058 | 0.0889 |
| hazelnut | single | 0.2 | 1 | joint_best.pt | pixel | 0.5139 | 0.8961 | 0.8393 | 0.6749 | 0.9459 | 0.9076 | 0.9433 | 0.4241 | 0.8967 |
| hazelnut | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.5657 | 0.5221 | 0.5975 | 0.7070 | 0.6383 | 0.7076 | 0.4321 | 0.0333 | 0.0489 |
| hazelnut | single | 0.2 | 1 | joint_best.pt | disc | 0.7464 | 0.3739 | 0.3489 | 0.8383 | 0.5684 | 0.5581 | 0.4806 | 0.0185 | 0.0025 |
| screw | single | 0.2 | 1 | joint_best.pt | pixel | 0.9541 | 0.9274 | 0.9563 | 0.9852 | 0.9744 | 0.9860 | 0.5025 | 0.0023 | 0.0824 |
| screw | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.6007 | 0.5831 | 0.5720 | 0.8078 | 0.8052 | 0.7963 | 0.2782 | 0.0051 | 0.0476 |
| screw | single | 0.2 | 1 | joint_best.pt | disc | 0.5329 | 0.5360 | 0.5251 | 0.7892 | 0.7863 | 0.7723 | 0.1738 | 0.0014 | 0.0232 |
| transistor | single | 0.2 | 1 | joint_best.pt | pixel | 0.4475 | 0.3671 | 0.3929 | 0.4543 | 0.3301 | 0.3827 | 0.3554 | 0.0344 | 0.0546 |
| transistor | single | 0.2 | 1 | joint_best.pt | latent_cos | 0.3721 | 0.4633 | 0.4508 | 0.3834 | 0.3891 | 0.3895 | 0.3272 | 0.0362 | 0.3040 |
| transistor | single | 0.2 | 1 | joint_best.pt | disc | 0.5700 | 0.5704 | 0.5988 | 0.5251 | 0.4751 | 0.5344 | 0.4181 | 0.0375 | 0.1298 |
