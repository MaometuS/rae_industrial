# RAE-DiT IAD results (J_latent_simplenet_nodecoder)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6774 | 0.7966 | 0.7962 | 0.8221 | 0.8996 | 0.8992 | 0.7971 | 0.1555 | 0.5225 |
| single | 0.1 | 1 | best | latent_cos | 0.8871 | 0.8339 | 0.8433 | 0.9497 | 0.9251 | 0.9275 | 0.8290 | 0.1941 | 0.4955 |
| single | 0.1 | 1 | best | pixel | 0.4211 | 0.6195 | 0.5872 | 0.6941 | 0.8317 | 0.8117 | 0.6322 | 0.0847 | 0.3990 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.5036 | 0.5084 | 0.5526 | 0.8195 | 0.8036 | 0.8367 | 0.3391 | 0.0113 | 0.0678 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.9234 | 0.9346 | 0.9342 | 0.9780 | 0.9811 | 0.9814 | 0.7777 | 0.0596 | 0.3922 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9013 | 0.9470 | 0.9514 | 0.9683 | 0.9850 | 0.9870 | 0.7945 | 0.0760 | 0.5940 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.3968 | 0.4294 | 0.3935 | 0.6893 | 0.7269 | 0.6892 | 0.5278 | 0.0073 | 0.2131 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.9891 | 0.9449 | 0.9632 | 0.9963 | 0.9816 | 0.9874 | 0.9033 | 0.1153 | 0.7288 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.5196 | 0.8881 | 0.8814 | 0.7215 | 0.9605 | 0.9570 | 0.8021 | 0.1032 | 0.5854 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.4818 | 0.8904 | 0.7971 | 0.6464 | 0.9479 | 0.8965 | 0.8099 | 0.3091 | 0.7861 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.9329 | 0.8407 | 0.8543 | 0.9650 | 0.9223 | 0.9301 | 0.9230 | 0.5766 | 0.5940 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.7362 | 0.7882 | 0.7768 | 0.8364 | 0.8859 | 0.8773 | 0.9606 | 0.4397 | 0.7609 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.3023 | 0.6497 | 0.6055 | 0.6211 | 0.8485 | 0.8245 | 0.8523 | 0.0111 | 0.5291 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.7030 | 0.6153 | 0.6214 | 0.8595 | 0.8152 | 0.8110 | 0.7118 | 0.0251 | 0.2670 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5526 | 0.5630 | 0.5753 | 0.7621 | 0.7670 | 0.7754 | 0.6313 | 0.0032 | 0.1497 |
