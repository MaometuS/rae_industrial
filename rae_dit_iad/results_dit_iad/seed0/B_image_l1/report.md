# RAE-DiT IAD results (B_image_l1)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5567 | 0.5404 | 0.5600 | 0.7228 | 0.7158 | 0.7281 | 0.7248 | 0.0536 | 0.2523 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3686 | 0.4704 | 0.4641 | 0.6045 | 0.6516 | 0.6440 | 0.2332 | 0.0135 | 0.0349 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4766 | 0.6580 | 0.6631 | 0.6862 | 0.7940 | 0.7973 | 0.7519 | 0.1673 | 0.5634 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.3768 | 0.3856 | 0.4222 | 0.6983 | 0.7160 | 0.7286 | 0.3462 | 0.0143 | 0.1004 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.0313 | 0.2978 | 0.2685 | 0.5591 | 0.6533 | 0.6422 | 0.0672 | 0.0082 | 0.0005 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.8086 | 0.5987 | 0.6356 | 0.9433 | 0.8573 | 0.8821 | 0.7660 | 0.0382 | 0.2437 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.6976 | 0.7987 | 0.8680 | 0.8451 | 0.8994 | 0.9379 | 0.8879 | 0.0977 | 0.7654 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2581 | 0.3993 | 0.3492 | 0.5941 | 0.6833 | 0.6268 | 0.0274 | 0.0035 | 0.0001 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.5113 | 0.6358 | 0.6332 | 0.7789 | 0.8245 | 0.8259 | 0.6121 | 0.0089 | 0.1260 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.5764 | 0.9343 | 0.9004 | 0.7252 | 0.9641 | 0.9447 | 0.9656 | 0.5396 | 0.9228 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4450 | 0.5143 | 0.5821 | 0.6244 | 0.6618 | 0.7048 | 0.2467 | 0.0129 | 0.0012 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7321 | 0.5921 | 0.6300 | 0.8462 | 0.7407 | 0.7580 | 0.8942 | 0.1303 | 0.6225 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.0152 | 0.6510 | 0.5727 | 0.5357 | 0.8633 | 0.8154 | 0.8730 | 0.0537 | 0.5722 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.9307 | 0.8629 | 0.9043 | 0.9745 | 0.9442 | 0.9645 | 0.3765 | 0.0018 | 0.0002 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.3320 | 0.5306 | 0.5499 | 0.6482 | 0.7983 | 0.7972 | 0.6255 | 0.0030 | 0.1072 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.7171 | 0.5204 | 0.5521 | 0.6269 | 0.5269 | 0.5601 | 0.6871 | 0.1315 | 0.4561 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.1779 | 0.2779 | 0.2163 | 0.2703 | 0.3156 | 0.2815 | 0.4484 | 0.0409 | 0.1725 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.3996 | 0.3446 | 0.3513 | 0.3974 | 0.3582 | 0.3776 | 0.7262 | 0.0874 | 0.1622 |
