# RAE-DiT IAD results (knn_frozen)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | img_auroc_max_logical | img_auroc_max_structural | img_auroc_mean_logical | img_auroc_mean_structural | spro | spro_logical | spro_structural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn | 0.0 | 1 | none | knn | 0.7562 | 0.7459 | 0.7560 | 0.8364 | 0.8322 | 0.8382 | 0.8090 | 0.3750 | 0.5750 | 0.7041 | 0.8057 | 0.7495 | 0.7714 | 0.5831 | 0.5566 | 0.6078 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | img_auroc_max_logical | img_auroc_max_structural | img_auroc_mean_logical | img_auroc_mean_structural | spro | spro_logical | spro_structural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breakfast_box | knn | 0.0 | 1 | none | knn | 0.8041 | 0.8681 | 0.8654 | 0.8960 | 0.9286 | 0.9306 | 0.8966 | 0.5514 | 0.7809 | 0.8436 | 0.8907 | 0.8311 | 0.7793 | 0.6681 | 0.5539 | 0.7734 |
| juice_bottle | knn | 0.0 | 1 | none | knn | 0.9357 | 0.8697 | 0.9018 | 0.9766 | 0.9485 | 0.9619 | 0.8646 | 0.4491 | 0.7602 | 0.8269 | 0.9342 | 0.9223 | 0.9560 | 0.8698 | 0.8742 | 0.8628 |
| pushpins | knn | 0.0 | 1 | none | knn | 0.6136 | 0.5622 | 0.5717 | 0.6800 | 0.6299 | 0.6361 | 0.8014 | 0.4437 | 0.4861 | 0.5170 | 0.6129 | 0.5889 | 0.6413 | 0.3673 | 0.3969 | 0.2549 |
| screw_bag | knn | 0.0 | 1 | none | knn | 0.6639 | 0.6161 | 0.6172 | 0.7719 | 0.7560 | 0.7589 | 0.7265 | 0.1549 | 0.4065 | 0.5209 | 0.7752 | 0.6021 | 0.7672 | 0.3636 | 0.2598 | 0.5674 |
| splicing_connectors | knn | 0.0 | 1 | none | knn | 0.7634 | 0.8136 | 0.8240 | 0.8577 | 0.8977 | 0.9036 | 0.7558 | 0.2757 | 0.4414 | 0.8121 | 0.8155 | 0.8031 | 0.7130 | 0.6466 | 0.6984 | 0.5807 |
