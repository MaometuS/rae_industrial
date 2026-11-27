# RAE-DiT IAD results (knn_frozen)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | aupro_005 | px_f1max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn | 0.0 | 1 | none | knn | 0.5862 | 0.6302 | 0.6266 | 0.7252 | 0.7701 | 0.7631 | 0.8532 | 0.1744 | 0.4052 | 0.1800 | 0.2234 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | aupro_005 | px_f1max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| can | knn | 0.0 | 1 | none | knn | 0.4485 | 0.5392 | 0.4750 | 0.5041 | 0.6115 | 0.5441 | 0.6255 | 0.0001 | 0.2219 | 0.0601 | 0.0005 |
| fabric | knn | 0.0 | 1 | none | knn | 0.6082 | 0.5513 | 0.5949 | 0.6917 | 0.6755 | 0.6965 | 0.9732 | 0.2980 | 0.1804 | 0.0726 | 0.3519 |
| fruit_jelly | knn | 0.0 | 1 | none | knn | 0.7333 | 0.6275 | 0.5992 | 0.9033 | 0.8727 | 0.8608 | 0.9302 | 0.2201 | 0.6202 | 0.2880 | 0.3058 |
| rice | knn | 0.0 | 1 | none | knn | 0.6566 | 0.7651 | 0.7471 | 0.8288 | 0.8774 | 0.8663 | 0.9152 | 0.1531 | 0.6549 | 0.4114 | 0.2124 |
| sheet_metal | knn | 0.0 | 1 | none | knn | 0.5968 | 0.5556 | 0.5713 | 0.8438 | 0.8291 | 0.8267 | 0.8296 | 0.2059 | 0.1840 | 0.0688 | 0.2692 |
| vial | knn | 0.0 | 1 | none | knn | 0.6699 | 0.8574 | 0.8814 | 0.8382 | 0.9388 | 0.9462 | 0.9039 | 0.1772 | 0.8057 | 0.4234 | 0.2211 |
| wallplugs | knn | 0.0 | 1 | none | knn | 0.3846 | 0.4265 | 0.4207 | 0.5097 | 0.5217 | 0.5202 | 0.7458 | 0.0035 | 0.1742 | 0.0063 | 0.0086 |
| walnuts | knn | 0.0 | 1 | none | knn | 0.5919 | 0.7191 | 0.7231 | 0.6820 | 0.8342 | 0.8438 | 0.9018 | 0.3371 | 0.4007 | 0.1095 | 0.4181 |
