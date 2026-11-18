# RAE-DiT IAD results (K5_latent_adaptor_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.6081 | 0.8544 | 0.8539 | 0.8293 | 0.9349 | 0.9359 | 0.9246 | 0.4460 | 0.8462 |
| single | 1.0 | 1 | best | latent_cos | 0.8867 | 0.9100 | 0.9069 | 0.9484 | 0.9580 | 0.9574 | 0.9389 | 0.3321 | 0.7849 |
| single | 1.0 | 1 | best | pixel | 0.4949 | 0.7013 | 0.6784 | 0.6700 | 0.8094 | 0.8015 | 0.7118 | 0.1643 | 0.4726 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.8383 | 0.8222 | 0.8892 | 0.9497 | 0.9326 | 0.9628 | 0.6761 | 0.0977 | 0.3181 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9904 | 0.9936 | 0.9932 | 0.9976 | 0.9983 | 0.9982 | 0.9846 | 0.4161 | 0.9481 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.1296 | 0.6437 | 0.6172 | 0.6330 | 0.8880 | 0.8793 | 0.9884 | 0.7377 | 0.9351 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4937 | 0.6349 | 0.5547 | 0.7121 | 0.8539 | 0.7939 | 0.7603 | 0.1182 | 0.4870 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9872 | 0.3028 | 0.9529 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.3342 | 0.9699 | 0.9674 | 0.7399 | 0.9894 | 0.9879 | 0.9834 | 0.2302 | 0.9305 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | pixel | 0.4846 | 0.9618 | 0.9246 | 0.6562 | 0.9823 | 0.9609 | 0.8836 | 0.5237 | 0.8692 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | latent_cos | 0.9918 | 0.9868 | 0.9868 | 0.9951 | 0.9917 | 0.9917 | 0.9743 | 0.4438 | 0.8278 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | disc | 0.9064 | 0.9857 | 0.9868 | 0.9534 | 0.9920 | 0.9925 | 0.9818 | 0.6899 | 0.9431 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.1797 | 0.6061 | 0.5282 | 0.5766 | 0.8307 | 0.7705 | 0.8567 | 0.0353 | 0.5230 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.4915 | 0.6298 | 0.6102 | 0.7968 | 0.8708 | 0.8633 | 0.8753 | 0.1983 | 0.5749 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.7504 | 0.8004 | 0.8018 | 0.9103 | 0.9341 | 0.9334 | 0.9202 | 0.3106 | 0.7378 |
| transistor | single | 1.0 | 1 | transistor_best.pt | pixel | 0.4783 | 0.4817 | 0.4954 | 0.4551 | 0.4473 | 0.5192 | 0.3824 | 0.0467 | 0.1659 |
| transistor | single | 1.0 | 1 | transistor_best.pt | latent_cos | 0.9600 | 0.9396 | 0.9446 | 0.9527 | 0.9292 | 0.9339 | 0.8733 | 0.2994 | 0.6209 |
| transistor | single | 1.0 | 1 | transistor_best.pt | disc | 0.9200 | 0.8725 | 0.8962 | 0.9098 | 0.8709 | 0.8866 | 0.7490 | 0.2617 | 0.6847 |
