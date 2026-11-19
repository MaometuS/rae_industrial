# RAE-DiT IAD results (K5_latent_adaptor_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.7420 | 0.9091 | 0.9071 | 0.8970 | 0.9643 | 0.9641 | 0.9354 | 0.5369 | 0.8174 |
| single | 1.0 | 1 | best | latent_cos | 0.9569 | 0.9710 | 0.9742 | 0.9799 | 0.9862 | 0.9880 | 0.9389 | 0.3514 | 0.8488 |
| single | 1.0 | 1 | best | pixel | 0.5711 | 0.6863 | 0.6612 | 0.7865 | 0.8431 | 0.8356 | 0.6892 | 0.1517 | 0.4492 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bottle | single | 1.0 | 1 | bottle_best.pt | pixel | 0.8444 | 0.9135 | 0.9413 | 0.9474 | 0.9740 | 0.9837 | 0.8415 | 0.4594 | 0.6469 |
| bottle | single | 1.0 | 1 | bottle_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9499 | 0.5807 | 0.8374 |
| bottle | single | 1.0 | 1 | bottle_best.pt | disc | 0.9944 | 1.0000 | 1.0000 | 0.9984 | 1.0000 | 1.0000 | 0.9847 | 0.8151 | 0.9359 |
| cable | single | 1.0 | 1 | cable_best.pt | pixel | 0.5617 | 0.4933 | 0.5290 | 0.7333 | 0.6335 | 0.6570 | 0.5174 | 0.0354 | 0.2968 |
| cable | single | 1.0 | 1 | cable_best.pt | latent_cos | 0.8913 | 0.9136 | 0.9277 | 0.9446 | 0.9535 | 0.9617 | 0.8984 | 0.2855 | 0.7313 |
| cable | single | 1.0 | 1 | cable_best.pt | disc | 0.9110 | 0.9145 | 0.9190 | 0.9522 | 0.9576 | 0.9600 | 0.9303 | 0.5252 | 0.7788 |
| capsule | single | 1.0 | 1 | capsule_best.pt | pixel | 0.7112 | 0.7228 | 0.7192 | 0.9161 | 0.9223 | 0.9203 | 0.3014 | 0.0291 | 0.0446 |
| capsule | single | 1.0 | 1 | capsule_best.pt | latent_cos | 0.9410 | 0.9358 | 0.9418 | 0.9868 | 0.9863 | 0.9870 | 0.9732 | 0.2883 | 0.8370 |
| capsule | single | 1.0 | 1 | capsule_best.pt | disc | 0.8564 | 0.8528 | 0.8584 | 0.9679 | 0.9660 | 0.9672 | 0.9688 | 0.3733 | 0.8299 |
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6806 | 0.6802 | 0.7203 | 0.8938 | 0.8764 | 0.9009 | 0.6845 | 0.0956 | 0.3223 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9813 | 0.3809 | 0.9302 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.3006 | 0.9767 | 0.9727 | 0.7522 | 0.9932 | 0.9921 | 0.9879 | 0.7342 | 0.9526 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4762 | 0.6316 | 0.5547 | 0.7151 | 0.8442 | 0.7964 | 0.7573 | 0.1114 | 0.4610 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9860 | 0.2948 | 0.9458 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.3258 | 0.9557 | 0.9373 | 0.7113 | 0.9855 | 0.9791 | 0.9809 | 0.2179 | 0.9140 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | pixel | 0.5186 | 0.9514 | 0.8789 | 0.6860 | 0.9745 | 0.9386 | 0.8655 | 0.4644 | 0.8514 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | latent_cos | 0.9850 | 0.9732 | 0.9764 | 0.9904 | 0.9793 | 0.9829 | 0.9789 | 0.3974 | 0.9184 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | disc | 0.7564 | 0.8500 | 0.8400 | 0.8697 | 0.9249 | 0.9218 | 0.9750 | 0.6957 | 0.8329 |
| leather | single | 1.0 | 1 | leather_best.pt | pixel | 0.4973 | 0.8380 | 0.7422 | 0.8068 | 0.9451 | 0.9096 | 0.7402 | 0.1205 | 0.5535 |
| leather | single | 1.0 | 1 | leather_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9195 | 0.1896 | 0.8335 |
| leather | single | 1.0 | 1 | leather_best.pt | disc | 0.6345 | 0.9103 | 0.8906 | 0.8718 | 0.9712 | 0.9639 | 0.9030 | 0.3320 | 0.7353 |
| metal_nut | single | 1.0 | 1 | metal_nut_best.pt | pixel | 0.4589 | 0.5836 | 0.5523 | 0.7609 | 0.8666 | 0.8429 | 0.6126 | 0.1833 | 0.4140 |
| metal_nut | single | 1.0 | 1 | metal_nut_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8812 | 0.3955 | 0.8491 |
| metal_nut | single | 1.0 | 1 | metal_nut_best.pt | disc | 0.9643 | 0.9922 | 0.9936 | 0.9924 | 0.9983 | 0.9986 | 0.9310 | 0.7151 | 0.9166 |
| pill | single | 1.0 | 1 | pill_best.pt | pixel | 0.5917 | 0.5897 | 0.5709 | 0.8782 | 0.8830 | 0.8673 | 0.7759 | 0.0765 | 0.5512 |
| pill | single | 1.0 | 1 | pill_best.pt | latent_cos | 0.9610 | 0.9534 | 0.9654 | 0.9929 | 0.9901 | 0.9930 | 0.9575 | 0.3816 | 0.9217 |
| pill | single | 1.0 | 1 | pill_best.pt | disc | 0.7916 | 0.8751 | 0.8726 | 0.9593 | 0.9750 | 0.9750 | 0.8887 | 0.5503 | 0.7620 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2400 | 0.6382 | 0.5729 | 0.5943 | 0.8467 | 0.8010 | 0.8613 | 0.0214 | 0.5469 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.6811 | 0.8990 | 0.9014 | 0.8658 | 0.9655 | 0.9661 | 0.9775 | 0.2653 | 0.8795 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.8186 | 0.7590 | 0.7909 | 0.9143 | 0.9127 | 0.9233 | 0.8912 | 0.1498 | 0.6810 |
| tile | single | 1.0 | 1 | tile_best.pt | pixel | 0.5797 | 0.7237 | 0.7002 | 0.8082 | 0.8732 | 0.8703 | 0.6550 | 0.2136 | 0.3682 |
| tile | single | 1.0 | 1 | tile_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8738 | 0.4163 | 0.7348 |
| tile | single | 1.0 | 1 | tile_best.pt | disc | 0.9845 | 1.0000 | 1.0000 | 0.9952 | 1.0000 | 1.0000 | 0.9832 | 0.8747 | 0.9246 |
| toothbrush | single | 1.0 | 1 | toothbrush_best.pt | pixel | 0.5667 | 0.5389 | 0.6056 | 0.8149 | 0.7177 | 0.7616 | 0.8726 | 0.0846 | 0.6479 |
| toothbrush | single | 1.0 | 1 | toothbrush_best.pt | latent_cos | 0.9472 | 0.9417 | 0.9389 | 0.9765 | 0.9750 | 0.9733 | 0.9708 | 0.2184 | 0.8551 |
| toothbrush | single | 1.0 | 1 | toothbrush_best.pt | disc | 0.8833 | 0.9444 | 0.9500 | 0.9628 | 0.9809 | 0.9826 | 0.9849 | 0.5251 | 0.8079 |
| transistor | single | 1.0 | 1 | transistor_best.pt | pixel | 0.4779 | 0.4571 | 0.4642 | 0.4619 | 0.4290 | 0.4893 | 0.3800 | 0.0430 | 0.1423 |
| transistor | single | 1.0 | 1 | transistor_best.pt | latent_cos | 0.9579 | 0.9550 | 0.9688 | 0.9450 | 0.9459 | 0.9582 | 0.8941 | 0.3754 | 0.7329 |
| transistor | single | 1.0 | 1 | transistor_best.pt | disc | 0.8775 | 0.9025 | 0.9054 | 0.8477 | 0.8934 | 0.8992 | 0.7751 | 0.4050 | 0.6324 |
| wood | single | 1.0 | 1 | wood_best.pt | pixel | 0.6167 | 0.8702 | 0.8254 | 0.8561 | 0.9591 | 0.9433 | 0.6493 | 0.2142 | 0.3457 |
| wood | single | 1.0 | 1 | wood_best.pt | latent_cos | 0.9921 | 0.9939 | 0.9939 | 0.9975 | 0.9980 | 0.9980 | 0.8807 | 0.4777 | 0.8533 |
| wood | single | 1.0 | 1 | wood_best.pt | disc | 0.3105 | 0.7421 | 0.7140 | 0.7401 | 0.9158 | 0.9088 | 0.9129 | 0.6091 | 0.7121 |
| zipper | single | 1.0 | 1 | zipper_best.pt | pixel | 0.7450 | 0.6626 | 0.5402 | 0.9240 | 0.9015 | 0.8520 | 0.8232 | 0.1235 | 0.5452 |
| zipper | single | 1.0 | 1 | zipper_best.pt | latent_cos | 0.9974 | 0.9989 | 0.9989 | 0.9993 | 0.9997 | 0.9997 | 0.9605 | 0.3235 | 0.8714 |
| zipper | single | 1.0 | 1 | zipper_best.pt | disc | 0.7209 | 0.9617 | 0.9624 | 0.9196 | 0.9897 | 0.9895 | 0.9329 | 0.5306 | 0.8460 |
