# RAE-DiT IAD results (J_latent_simplenet_nodecoder)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5251 | 0.5032 | 0.5211 | 0.7070 | 0.7079 | 0.7221 | 0.7150 | 0.0668 | 0.2388 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7846 | 0.7644 | 0.7746 | 0.8753 | 0.8584 | 0.8670 | 0.7579 | 0.2150 | 0.3392 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4174 | 0.5687 | 0.5454 | 0.6384 | 0.7303 | 0.7265 | 0.5699 | 0.0738 | 0.3321 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.5052 | 0.5096 | 0.5542 | 0.8200 | 0.8044 | 0.8378 | 0.3400 | 0.0113 | 0.0686 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.8888 | 0.9001 | 0.9093 | 0.9673 | 0.9712 | 0.9745 | 0.8022 | 0.0565 | 0.3151 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.7095 | 0.4575 | 0.4908 | 0.9114 | 0.7908 | 0.8121 | 0.7486 | 0.0366 | 0.1925 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.3952 | 0.4311 | 0.3943 | 0.6874 | 0.7267 | 0.6898 | 0.5254 | 0.0072 | 0.2103 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7736 | 0.7953 | 0.8045 | 0.9145 | 0.8917 | 0.9069 | 0.5677 | 0.0111 | 0.1620 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.3993 | 0.6366 | 0.6650 | 0.7306 | 0.8190 | 0.8562 | 0.5262 | 0.0066 | 0.0280 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.4804 | 0.8914 | 0.7971 | 0.6456 | 0.9483 | 0.8971 | 0.8079 | 0.3070 | 0.7855 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.8061 | 0.7193 | 0.7325 | 0.8894 | 0.8522 | 0.8573 | 0.9240 | 0.5572 | 0.5906 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7250 | 0.5436 | 0.5671 | 0.8381 | 0.7121 | 0.7291 | 0.9007 | 0.1833 | 0.5971 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.2810 | 0.6380 | 0.5872 | 0.6090 | 0.8412 | 0.8100 | 0.8507 | 0.0105 | 0.5242 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.6548 | 0.6046 | 0.6050 | 0.8202 | 0.7936 | 0.7863 | 0.6151 | 0.0035 | 0.1835 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.3390 | 0.4778 | 0.4765 | 0.6526 | 0.7736 | 0.7809 | 0.6657 | 0.0034 | 0.1905 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.4254 | 0.3733 | 0.3942 | 0.4301 | 0.3309 | 0.3977 | 0.3256 | 0.0330 | 0.0722 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7996 | 0.8025 | 0.8217 | 0.7850 | 0.7831 | 0.8100 | 0.8805 | 0.4465 | 0.4449 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.4525 | 0.4004 | 0.4058 | 0.4025 | 0.4439 | 0.4323 | 0.7339 | 0.1039 | 0.1861 |
