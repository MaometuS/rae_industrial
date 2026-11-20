# RAE-DiT IAD results (J5_latent_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | img_auroc_max_logical | img_auroc_max_structural | img_auroc_mean_logical | img_auroc_mean_structural | spro | spro_logical | spro_structural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.6816 | 0.7168 | 0.7231 | 0.7867 | 0.8235 | 0.8246 | 0.6957 | 0.2565 | 0.4934 | 0.6937 | 0.7506 | 0.7052 | 0.6550 | 0.4933 | 0.4687 | 0.5322 |
| single | 1.0 | 1 | best | latent_cos | 0.7838 | 0.7659 | 0.7814 | 0.8699 | 0.8580 | 0.8696 | 0.7139 | 0.1855 | 0.4245 | 0.6892 | 0.8658 | 0.7393 | 0.8457 | 0.6237 | 0.5249 | 0.7698 |
| single | 1.0 | 1 | best | pixel | 0.5485 | 0.5696 | 0.5491 | 0.6913 | 0.7056 | 0.6998 | 0.5444 | 0.0779 | 0.3557 | 0.5493 | 0.5936 | 0.4918 | 0.6133 | 0.1922 | 0.2071 | 0.1402 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro | img_auroc_max_logical | img_auroc_max_structural | img_auroc_mean_logical | img_auroc_mean_structural | spro | spro_logical | spro_structural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breakfast_box | single | 1.0 | 1 | breakfast_box_best.pt | pixel | 0.4249 | 0.6168 | 0.5491 | 0.5973 | 0.7552 | 0.6944 | 0.4329 | 0.0595 | 0.1545 | 0.5837 | 0.6473 | 0.3017 | 0.5385 | 0.0565 | 0.0122 | 0.0973 |
| breakfast_box | single | 1.0 | 1 | breakfast_box_best.pt | latent_cos | 0.8544 | 0.7700 | 0.7993 | 0.9238 | 0.8643 | 0.8834 | 0.8430 | 0.3237 | 0.5299 | 0.6838 | 0.8495 | 0.8318 | 0.8752 | 0.5927 | 0.3968 | 0.7734 |
| breakfast_box | single | 1.0 | 1 | breakfast_box_best.pt | disc | 0.7771 | 0.8287 | 0.8363 | 0.8819 | 0.9072 | 0.9111 | 0.8537 | 0.5033 | 0.7024 | 0.8062 | 0.8496 | 0.8155 | 0.7417 | 0.5855 | 0.4721 | 0.6901 |
| juice_bottle | single | 1.0 | 1 | juice_bottle_best.pt | pixel | 0.5303 | 0.5280 | 0.5085 | 0.7764 | 0.7691 | 0.7698 | 0.8112 | 0.1472 | 0.5614 | 0.5088 | 0.5570 | 0.5082 | 0.5638 | 0.2363 | 0.2071 | 0.2836 |
| juice_bottle | single | 1.0 | 1 | juice_bottle_best.pt | latent_cos | 0.8972 | 0.8429 | 0.8663 | 0.9629 | 0.9387 | 0.9490 | 0.8112 | 0.2668 | 0.5750 | 0.7717 | 0.9505 | 0.8547 | 0.9615 | 0.7379 | 0.6665 | 0.8536 |
| juice_bottle | single | 1.0 | 1 | juice_bottle_best.pt | disc | 0.8559 | 0.8214 | 0.8339 | 0.9470 | 0.9294 | 0.9347 | 0.8153 | 0.3736 | 0.6844 | 0.8298 | 0.8089 | 0.8816 | 0.8172 | 0.7263 | 0.7872 | 0.6277 |
| pushpins | single | 1.0 | 1 | pushpins_best.pt | pixel | 0.6055 | 0.6020 | 0.6122 | 0.6731 | 0.6450 | 0.6701 | 0.4602 | 0.0347 | 0.4648 | 0.5677 | 0.6405 | 0.5205 | 0.7008 | 0.3208 | 0.3739 | 0.1195 |
| pushpins | single | 1.0 | 1 | pushpins_best.pt | latent_cos | 0.6728 | 0.6785 | 0.6914 | 0.7603 | 0.7597 | 0.7751 | 0.7028 | 0.1055 | 0.3768 | 0.6044 | 0.7617 | 0.6140 | 0.7389 | 0.5212 | 0.5122 | 0.5554 |
| pushpins | single | 1.0 | 1 | pushpins_best.pt | disc | 0.5436 | 0.5875 | 0.5942 | 0.6082 | 0.6727 | 0.6706 | 0.5858 | 0.0734 | 0.3231 | 0.5669 | 0.6106 | 0.5648 | 0.5197 | 0.2875 | 0.2757 | 0.3320 |
| screw_bag | single | 1.0 | 1 | screw_bag_best.pt | pixel | 0.5450 | 0.5657 | 0.5131 | 0.6762 | 0.7068 | 0.6762 | 0.5119 | 0.0728 | 0.4565 | 0.5580 | 0.5786 | 0.5095 | 0.6044 | 0.2116 | 0.2486 | 0.1391 |
| screw_bag | single | 1.0 | 1 | screw_bag_best.pt | latent_cos | 0.6986 | 0.7266 | 0.7248 | 0.8153 | 0.8280 | 0.8320 | 0.5541 | 0.0902 | 0.3635 | 0.6367 | 0.8767 | 0.6149 | 0.8383 | 0.5160 | 0.3384 | 0.8647 |
| screw_bag | single | 1.0 | 1 | screw_bag_best.pt | disc | 0.5978 | 0.5918 | 0.5904 | 0.7110 | 0.7497 | 0.7438 | 0.5998 | 0.1034 | 0.3543 | 0.5052 | 0.7366 | 0.5796 | 0.6282 | 0.3399 | 0.2399 | 0.5361 |
| splicing_connectors | single | 1.0 | 1 | splicing_connectors_best.pt | pixel | 0.6367 | 0.5354 | 0.5625 | 0.7337 | 0.6519 | 0.6883 | 0.5059 | 0.0753 | 0.1414 | 0.5281 | 0.5447 | 0.6194 | 0.6588 | 0.1356 | 0.1939 | 0.0616 |
| splicing_connectors | single | 1.0 | 1 | splicing_connectors_best.pt | latent_cos | 0.7959 | 0.8117 | 0.8250 | 0.8872 | 0.8994 | 0.9082 | 0.6585 | 0.1412 | 0.2772 | 0.7497 | 0.8905 | 0.7810 | 0.8148 | 0.7507 | 0.7103 | 0.8021 |
| splicing_connectors | single | 1.0 | 1 | splicing_connectors_best.pt | disc | 0.6334 | 0.7547 | 0.7607 | 0.7853 | 0.8586 | 0.8625 | 0.6241 | 0.2285 | 0.4028 | 0.7603 | 0.7476 | 0.6848 | 0.5681 | 0.5275 | 0.5687 | 0.4750 |
