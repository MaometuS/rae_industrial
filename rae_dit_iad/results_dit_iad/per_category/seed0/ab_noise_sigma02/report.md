# RAE-DiT IAD results (ab_noise_sigma02)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.2 | 1 | best | disc | 0.6346 | 0.6991 | 0.6969 | 0.7580 | 0.8007 | 0.8009 | 0.7144 | 0.1407 | 0.3417 |
| single | 0.2 | 1 | best | latent_cos | 0.2840 | 0.4145 | 0.3821 | 0.5736 | 0.6445 | 0.6291 | 0.3156 | 0.0168 | 0.1730 |
| single | 0.2 | 1 | best | pixel | 0.5107 | 0.6985 | 0.6887 | 0.7053 | 0.8180 | 0.8087 | 0.8015 | 0.2072 | 0.6142 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.2 | 1 | carpet_best.pt | pixel | 0.4157 | 0.4402 | 0.4482 | 0.6953 | 0.7660 | 0.7461 | 0.5283 | 0.0219 | 0.1594 |
| carpet | single | 0.2 | 1 | carpet_best.pt | latent_cos | 0.1112 | 0.4402 | 0.3760 | 0.5845 | 0.7774 | 0.7368 | 0.0255 | 0.0081 | 0.0005 |
| carpet | single | 0.2 | 1 | carpet_best.pt | disc | 0.8796 | 0.7600 | 0.7881 | 0.9627 | 0.9037 | 0.9203 | 0.4548 | 0.0136 | 0.0183 |
| grid | single | 0.2 | 1 | grid_best.pt | pixel | 0.7719 | 0.7962 | 0.9006 | 0.8836 | 0.9099 | 0.9597 | 0.9464 | 0.2328 | 0.8722 |
| grid | single | 0.2 | 1 | grid_best.pt | latent_cos | 0.1420 | 0.3116 | 0.2581 | 0.5581 | 0.6481 | 0.6288 | 0.0146 | 0.0035 | 0.0000 |
| grid | single | 0.2 | 1 | grid_best.pt | disc | 0.5013 | 0.7494 | 0.7318 | 0.7197 | 0.8914 | 0.8805 | 0.7622 | 0.0451 | 0.4837 |
| hazelnut | single | 0.2 | 1 | hazelnut_best.pt | pixel | 0.6071 | 0.9546 | 0.9232 | 0.7506 | 0.9735 | 0.9551 | 0.9714 | 0.5752 | 0.9250 |
| hazelnut | single | 0.2 | 1 | hazelnut_best.pt | latent_cos | 0.3364 | 0.5518 | 0.5736 | 0.5949 | 0.7004 | 0.7297 | 0.1173 | 0.0114 | 0.0000 |
| hazelnut | single | 0.2 | 1 | hazelnut_best.pt | disc | 0.6921 | 0.7821 | 0.7564 | 0.8106 | 0.8704 | 0.8600 | 0.9617 | 0.4567 | 0.7368 |
| screw | single | 0.2 | 1 | screw_best.pt | pixel | 0.0115 | 0.7069 | 0.5809 | 0.5353 | 0.8766 | 0.8101 | 0.8840 | 0.0720 | 0.6129 |
| screw | single | 0.2 | 1 | screw_best.pt | latent_cos | 0.6856 | 0.5946 | 0.5694 | 0.8714 | 0.8066 | 0.7880 | 0.9140 | 0.0153 | 0.7093 |
| screw | single | 0.2 | 1 | screw_best.pt | disc | 0.5444 | 0.5606 | 0.5515 | 0.7579 | 0.7723 | 0.7649 | 0.6382 | 0.0032 | 0.1578 |
| transistor | single | 0.2 | 1 | transistor_best.pt | pixel | 0.7471 | 0.5946 | 0.5904 | 0.6615 | 0.5641 | 0.5726 | 0.6775 | 0.1339 | 0.5015 |
| transistor | single | 0.2 | 1 | transistor_best.pt | latent_cos | 0.1446 | 0.1742 | 0.1333 | 0.2592 | 0.2900 | 0.2625 | 0.5064 | 0.0459 | 0.1552 |
| transistor | single | 0.2 | 1 | transistor_best.pt | disc | 0.5554 | 0.6433 | 0.6567 | 0.5391 | 0.5659 | 0.5787 | 0.7551 | 0.1851 | 0.3120 |
