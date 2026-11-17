# RAE-DiT IAD results (B_image_l1)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.5555 | 0.6172 | 0.6120 | 0.7404 | 0.7545 | 0.7565 | 0.7089 | 0.0697 | 0.2824 |
| single | 0.1 | 1 | best | latent_cos | 0.4093 | 0.4400 | 0.4452 | 0.6199 | 0.6335 | 0.6416 | 0.5645 | 0.0423 | 0.2948 |
| single | 0.1 | 1 | best | pixel | 0.4581 | 0.7162 | 0.7121 | 0.6822 | 0.8286 | 0.8354 | 0.7998 | 0.2335 | 0.6213 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.3070 | 0.3896 | 0.3636 | 0.6722 | 0.7545 | 0.7432 | 0.5075 | 0.0762 | 0.2211 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.3648 | 0.5843 | 0.5959 | 0.7094 | 0.8409 | 0.8539 | 0.8402 | 0.1160 | 0.6474 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.5120 | 0.6164 | 0.5999 | 0.8004 | 0.8215 | 0.8233 | 0.5022 | 0.0145 | 0.0464 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.7544 | 0.8755 | 0.9390 | 0.8706 | 0.9471 | 0.9782 | 0.9517 | 0.2994 | 0.8925 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.0952 | 0.1278 | 0.1153 | 0.5443 | 0.5528 | 0.5501 | 0.0455 | 0.0035 | 0.0000 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.3283 | 0.7005 | 0.6683 | 0.6247 | 0.8696 | 0.8627 | 0.7605 | 0.0816 | 0.4702 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.6279 | 0.9471 | 0.9114 | 0.7526 | 0.9724 | 0.9527 | 0.9658 | 0.5633 | 0.8962 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.4225 | 0.3586 | 0.3446 | 0.5923 | 0.5481 | 0.5414 | 0.7399 | 0.0359 | 0.4264 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.6079 | 0.6036 | 0.5829 | 0.7682 | 0.7591 | 0.7420 | 0.9050 | 0.1271 | 0.4887 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0437 | 0.7393 | 0.6647 | 0.5407 | 0.8993 | 0.8669 | 0.8910 | 0.0702 | 0.6455 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.9201 | 0.8186 | 0.8494 | 0.9714 | 0.9111 | 0.9321 | 0.7058 | 0.0039 | 0.1960 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5241 | 0.5763 | 0.5839 | 0.7488 | 0.7746 | 0.7798 | 0.6244 | 0.0030 | 0.1350 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.5575 | 0.6296 | 0.6817 | 0.5751 | 0.5695 | 0.6362 | 0.6829 | 0.1585 | 0.4511 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.2437 | 0.3108 | 0.3208 | 0.2823 | 0.3148 | 0.3306 | 0.4911 | 0.0520 | 0.2040 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.8054 | 0.5890 | 0.6250 | 0.7598 | 0.5476 | 0.5746 | 0.7524 | 0.1226 | 0.2715 |
