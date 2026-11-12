# RAE-DiT IAD results (D_image_l1_latent)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6486 | 0.7629 | 0.7632 | 0.7633 | 0.8315 | 0.8331 | 0.7863 | 0.1574 | 0.4704 |
| single | 0.1 | 1 | best | latent_cos | 0.3652 | 0.4053 | 0.4079 | 0.5940 | 0.6310 | 0.6397 | 0.6918 | 0.0288 | 0.3767 |
| single | 0.1 | 1 | best | pixel | 0.5294 | 0.6725 | 0.6651 | 0.7174 | 0.7999 | 0.7964 | 0.7184 | 0.1425 | 0.4976 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.5237 | 0.5730 | 0.5618 | 0.7831 | 0.8014 | 0.7927 | 0.3676 | 0.0132 | 0.1052 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.3098 | 0.2099 | 0.2135 | 0.6826 | 0.6311 | 0.6274 | 0.7329 | 0.0317 | 0.4614 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9021 | 0.9406 | 0.9458 | 0.9688 | 0.9828 | 0.9852 | 0.7777 | 0.0691 | 0.5596 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.5865 | 0.7009 | 0.7569 | 0.7806 | 0.8674 | 0.8837 | 0.8130 | 0.0473 | 0.6173 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.3960 | 0.4570 | 0.5322 | 0.6648 | 0.6905 | 0.7459 | 0.5780 | 0.0077 | 0.1317 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.5246 | 0.8881 | 0.8864 | 0.7227 | 0.9598 | 0.9586 | 0.8075 | 0.1060 | 0.5831 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.7350 | 0.9246 | 0.8882 | 0.8409 | 0.9618 | 0.9417 | 0.9637 | 0.5371 | 0.8979 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.4004 | 0.6007 | 0.5854 | 0.5736 | 0.7842 | 0.7671 | 0.7800 | 0.0474 | 0.4493 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.7289 | 0.7818 | 0.7657 | 0.8329 | 0.8816 | 0.8688 | 0.9575 | 0.4232 | 0.7400 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.1638 | 0.6374 | 0.5860 | 0.5877 | 0.8442 | 0.8050 | 0.8290 | 0.0208 | 0.4745 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.5897 | 0.4837 | 0.4944 | 0.7933 | 0.7414 | 0.7473 | 0.9311 | 0.0182 | 0.7718 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5444 | 0.5626 | 0.5795 | 0.7595 | 0.7676 | 0.7776 | 0.6342 | 0.0032 | 0.1533 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.6383 | 0.5263 | 0.5325 | 0.5948 | 0.5247 | 0.5587 | 0.6185 | 0.0939 | 0.3931 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.1300 | 0.2754 | 0.2142 | 0.2559 | 0.3078 | 0.3109 | 0.4369 | 0.0389 | 0.0691 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.5429 | 0.6417 | 0.6388 | 0.5327 | 0.5658 | 0.5755 | 0.7546 | 0.1854 | 0.3159 |
