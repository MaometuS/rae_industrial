# RAE-DiT IAD results (J3_latent_swap)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | img_auroc_max_logical | img_auroc_max_structural | img_auroc_mean_logical | img_auroc_mean_structural | spro | spro_logical | spro_structural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.7128 | 0.6875 | 0.6934 | 0.8117 | 0.7887 | 0.7964 | 0.6402 | 0.2077 | 0.4016 | 0.6872 | 0.6947 | 0.7236 | 0.7069 | 0.3975 | 0.4292 | 0.3319 |
| single | 1.0 | 1 | best | latent_cos | 0.7118 | 0.7227 | 0.7463 | 0.7953 | 0.8122 | 0.8332 | 0.6654 | 0.1867 | 0.5562 | 0.6837 | 0.7782 | 0.7213 | 0.7018 | 0.5105 | 0.4609 | 0.5926 |
| single | 1.0 | 1 | best | pixel | 0.5419 | 0.5637 | 0.5409 | 0.6861 | 0.6955 | 0.6889 | 0.5403 | 0.0756 | 0.3432 | 0.5528 | 0.5762 | 0.4908 | 0.5989 | 0.1761 | 0.2062 | 0.0999 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | img_auroc_max_logical | img_auroc_max_structural | img_auroc_mean_logical | img_auroc_mean_structural | spro | spro_logical | spro_structural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breakfast_box | single | 1.0 | 1 | breakfast_box_best.pt | pixel | 0.4102 | 0.6077 | 0.5269 | 0.5774 | 0.7417 | 0.6727 | 0.4288 | 0.0571 | 0.1254 | 0.5915 | 0.6225 | 0.2970 | 0.5147 | 0.0327 | 0.0117 | 0.0521 |
| breakfast_box | single | 1.0 | 1 | breakfast_box_best.pt | latent_cos | 0.8282 | 0.7262 | 0.7866 | 0.9030 | 0.8062 | 0.8592 | 0.8043 | 0.3129 | 0.6121 | 0.7206 | 0.7313 | 0.8339 | 0.8229 | 0.5148 | 0.3627 | 0.6551 |
| breakfast_box | single | 1.0 | 1 | breakfast_box_best.pt | disc | 0.8408 | 0.7814 | 0.7832 | 0.9086 | 0.8684 | 0.8734 | 0.7542 | 0.3463 | 0.5347 | 0.8005 | 0.7638 | 0.8866 | 0.7986 | 0.3839 | 0.3333 | 0.4307 |
| juice_bottle | single | 1.0 | 1 | juice_bottle_best.pt | pixel | 0.5210 | 0.5188 | 0.4975 | 0.7726 | 0.7485 | 0.7528 | 0.8116 | 0.1412 | 0.5554 | 0.5140 | 0.5259 | 0.5149 | 0.5302 | 0.2196 | 0.2092 | 0.2363 |
| juice_bottle | single | 1.0 | 1 | juice_bottle_best.pt | latent_cos | 0.8345 | 0.8314 | 0.8572 | 0.9326 | 0.9263 | 0.9419 | 0.8626 | 0.3600 | 0.7669 | 0.7907 | 0.8928 | 0.8553 | 0.8031 | 0.7392 | 0.7388 | 0.7398 |
| juice_bottle | single | 1.0 | 1 | juice_bottle_best.pt | disc | 0.8714 | 0.8325 | 0.8419 | 0.9516 | 0.9385 | 0.9421 | 0.7539 | 0.3706 | 0.5972 | 0.8117 | 0.8639 | 0.8729 | 0.8693 | 0.6054 | 0.6661 | 0.5070 |
| pushpins | single | 1.0 | 1 | pushpins_best.pt | pixel | 0.6130 | 0.5993 | 0.6173 | 0.6794 | 0.6385 | 0.6716 | 0.4517 | 0.0335 | 0.4590 | 0.5710 | 0.6312 | 0.5244 | 0.7126 | 0.3148 | 0.3753 | 0.0852 |
| pushpins | single | 1.0 | 1 | pushpins_best.pt | latent_cos | 0.5762 | 0.6323 | 0.6525 | 0.6119 | 0.6982 | 0.7117 | 0.5479 | 0.0345 | 0.3030 | 0.5877 | 0.6825 | 0.6012 | 0.5480 | 0.2318 | 0.1896 | 0.3915 |
| pushpins | single | 1.0 | 1 | pushpins_best.pt | disc | 0.5281 | 0.5282 | 0.5189 | 0.6284 | 0.5907 | 0.5992 | 0.5836 | 0.0683 | 0.2576 | 0.5483 | 0.5056 | 0.5500 | 0.5034 | 0.2502 | 0.2802 | 0.1366 |
| screw_bag | single | 1.0 | 1 | screw_bag_best.pt | pixel | 0.5280 | 0.5622 | 0.5075 | 0.6662 | 0.7091 | 0.6663 | 0.5063 | 0.0719 | 0.4408 | 0.5552 | 0.5739 | 0.4985 | 0.5772 | 0.1963 | 0.2459 | 0.0990 |
| screw_bag | single | 1.0 | 1 | screw_bag_best.pt | latent_cos | 0.5874 | 0.6247 | 0.6341 | 0.7001 | 0.7472 | 0.7625 | 0.4837 | 0.0826 | 0.5631 | 0.5518 | 0.7465 | 0.5641 | 0.6263 | 0.4380 | 0.3766 | 0.5585 |
| screw_bag | single | 1.0 | 1 | screw_bag_best.pt | disc | 0.6813 | 0.5966 | 0.6120 | 0.8005 | 0.7380 | 0.7472 | 0.5228 | 0.0914 | 0.3343 | 0.5470 | 0.6794 | 0.6312 | 0.7651 | 0.3308 | 0.3483 | 0.2966 |
| splicing_connectors | single | 1.0 | 1 | splicing_connectors_best.pt | pixel | 0.6373 | 0.5303 | 0.5552 | 0.7348 | 0.6396 | 0.6813 | 0.5032 | 0.0742 | 0.1353 | 0.5324 | 0.5276 | 0.6195 | 0.6599 | 0.1173 | 0.1886 | 0.0268 |
| splicing_connectors | single | 1.0 | 1 | splicing_connectors_best.pt | latent_cos | 0.7329 | 0.7988 | 0.8013 | 0.8287 | 0.8834 | 0.8907 | 0.6286 | 0.1433 | 0.5358 | 0.7679 | 0.8380 | 0.7519 | 0.7087 | 0.6288 | 0.6370 | 0.6184 |
| splicing_connectors | single | 1.0 | 1 | splicing_connectors_best.pt | disc | 0.6424 | 0.6986 | 0.7109 | 0.7694 | 0.8079 | 0.8204 | 0.5866 | 0.1620 | 0.2842 | 0.7284 | 0.6606 | 0.6772 | 0.5980 | 0.4170 | 0.5181 | 0.2884 |
