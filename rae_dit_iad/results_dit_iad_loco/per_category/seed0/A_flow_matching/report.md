# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | img_auroc_max_logical | img_auroc_max_structural | img_auroc_mean_logical | img_auroc_mean_structural | spro | spro_logical | spro_structural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.7182 | 0.6748 | 0.7037 | 0.8042 | 0.7780 | 0.7987 | 0.7578 | 0.2077 | 0.5690 | 0.6290 | 0.7367 | 0.7133 | 0.7287 | 0.4942 | 0.4364 | 0.5696 |
| single | 0.5 | 1 | best | pixel | 0.5514 | 0.5562 | 0.5535 | 0.7026 | 0.6908 | 0.6976 | 0.5459 | 0.0826 | 0.3663 | 0.5362 | 0.5770 | 0.4984 | 0.6128 | 0.2033 | 0.2260 | 0.1435 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | img_auroc_max_logical | img_auroc_max_structural | img_auroc_mean_logical | img_auroc_mean_structural | spro | spro_logical | spro_structural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breakfast_box | single | 0.5 | 1 | breakfast_box_best.pt | pixel | 0.3722 | 0.5342 | 0.4883 | 0.5741 | 0.6895 | 0.6435 | 0.4574 | 0.0626 | 0.1609 | 0.4940 | 0.5714 | 0.2747 | 0.4621 | 0.0487 | 0.0201 | 0.0752 |
| breakfast_box | single | 0.5 | 1 | breakfast_box_best.pt | latent_cos | 0.7378 | 0.6588 | 0.7289 | 0.8571 | 0.7711 | 0.8383 | 0.8283 | 0.2900 | 0.6820 | 0.6323 | 0.6832 | 0.7654 | 0.7123 | 0.4139 | 0.2605 | 0.5555 |
| juice_bottle | single | 0.5 | 1 | juice_bottle_best.pt | pixel | 0.5508 | 0.5568 | 0.5627 | 0.7925 | 0.7824 | 0.7930 | 0.8232 | 0.1640 | 0.5826 | 0.5482 | 0.5698 | 0.5554 | 0.5439 | 0.2568 | 0.2341 | 0.2937 |
| juice_bottle | single | 0.5 | 1 | juice_bottle_best.pt | latent_cos | 0.9062 | 0.8452 | 0.8783 | 0.9650 | 0.9370 | 0.9522 | 0.8633 | 0.3392 | 0.7682 | 0.7855 | 0.9353 | 0.8974 | 0.9195 | 0.7572 | 0.7149 | 0.8258 |
| pushpins | single | 0.5 | 1 | pushpins_best.pt | pixel | 0.6616 | 0.6186 | 0.6344 | 0.7187 | 0.6703 | 0.6999 | 0.4045 | 0.0347 | 0.4647 | 0.5651 | 0.6787 | 0.5594 | 0.7764 | 0.3368 | 0.3831 | 0.1613 |
| pushpins | single | 0.5 | 1 | pushpins_best.pt | latent_cos | 0.5975 | 0.5576 | 0.5597 | 0.6622 | 0.6365 | 0.6154 | 0.7375 | 0.1039 | 0.4397 | 0.4984 | 0.6241 | 0.5599 | 0.6398 | 0.3479 | 0.3472 | 0.3507 |
| screw_bag | single | 0.5 | 1 | screw_bag_best.pt | pixel | 0.5606 | 0.5464 | 0.5226 | 0.7030 | 0.6852 | 0.6816 | 0.5296 | 0.0755 | 0.4691 | 0.5546 | 0.5328 | 0.5111 | 0.6432 | 0.2477 | 0.3008 | 0.1434 |
| screw_bag | single | 0.5 | 1 | screw_bag_best.pt | latent_cos | 0.6326 | 0.5750 | 0.5807 | 0.7292 | 0.7172 | 0.7198 | 0.6976 | 0.1375 | 0.3904 | 0.5176 | 0.6707 | 0.5957 | 0.6942 | 0.3346 | 0.2684 | 0.4647 |
| splicing_connectors | single | 0.5 | 1 | splicing_connectors_best.pt | pixel | 0.6118 | 0.5251 | 0.5595 | 0.7248 | 0.6264 | 0.6700 | 0.5146 | 0.0762 | 0.1541 | 0.5193 | 0.5325 | 0.5911 | 0.6382 | 0.1266 | 0.1918 | 0.0439 |
| splicing_connectors | single | 0.5 | 1 | splicing_connectors_best.pt | latent_cos | 0.7169 | 0.7374 | 0.7711 | 0.8076 | 0.8283 | 0.8676 | 0.6621 | 0.1677 | 0.5646 | 0.7113 | 0.7704 | 0.7481 | 0.6774 | 0.6175 | 0.5909 | 0.6514 |
