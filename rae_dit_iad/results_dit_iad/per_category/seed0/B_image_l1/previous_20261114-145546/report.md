# RAE-DiT IAD results (B_image_l1)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6374 | 0.6778 | 0.6899 | 0.7575 | 0.7943 | 0.8018 | 0.6692 | 0.0617 | 0.2336 |
| single | 0.1 | 1 | best | latent_cos | 0.2470 | 0.3711 | 0.3336 | 0.5607 | 0.6116 | 0.5924 | 0.2591 | 0.0133 | 0.1506 |
| single | 0.1 | 1 | best | pixel | 0.5041 | 0.7092 | 0.6942 | 0.7154 | 0.8327 | 0.8216 | 0.7767 | 0.2083 | 0.6070 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.3832 | 0.4884 | 0.4860 | 0.7201 | 0.8012 | 0.7884 | 0.3721 | 0.0165 | 0.1171 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.0506 | 0.2580 | 0.2067 | 0.5638 | 0.6680 | 0.6189 | 0.0326 | 0.0081 | 0.0011 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.8904 | 0.6834 | 0.7107 | 0.9669 | 0.8983 | 0.9112 | 0.5751 | 0.0172 | 0.1387 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.7769 | 0.8212 | 0.9165 | 0.8918 | 0.9203 | 0.9676 | 0.9442 | 0.2283 | 0.8775 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.1111 | 0.2941 | 0.2548 | 0.5481 | 0.6210 | 0.6035 | 0.0171 | 0.0035 | 0.0000 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.4971 | 0.8229 | 0.8195 | 0.7236 | 0.9244 | 0.9228 | 0.7740 | 0.0837 | 0.5245 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.6046 | 0.9357 | 0.8929 | 0.7805 | 0.9664 | 0.9431 | 0.9702 | 0.5857 | 0.9229 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.3339 | 0.5168 | 0.4689 | 0.5839 | 0.7112 | 0.6859 | 0.0568 | 0.0110 | 0.0000 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.7136 | 0.6582 | 0.7325 | 0.8264 | 0.8041 | 0.8526 | 0.6130 | 0.0244 | 0.0987 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0164 | 0.6878 | 0.5739 | 0.5367 | 0.8748 | 0.8147 | 0.8795 | 0.0649 | 0.6068 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.6421 | 0.5702 | 0.5606 | 0.8529 | 0.7841 | 0.7900 | 0.9034 | 0.0134 | 0.6804 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5665 | 0.5981 | 0.5934 | 0.7680 | 0.7878 | 0.7925 | 0.6324 | 0.0031 | 0.1463 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.7392 | 0.6129 | 0.6017 | 0.6477 | 0.6009 | 0.5940 | 0.7176 | 0.1460 | 0.5107 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.0971 | 0.2167 | 0.1771 | 0.2548 | 0.2736 | 0.2639 | 0.2856 | 0.0304 | 0.0714 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.5196 | 0.6262 | 0.5933 | 0.5026 | 0.5569 | 0.5301 | 0.7515 | 0.1802 | 0.2598 |
