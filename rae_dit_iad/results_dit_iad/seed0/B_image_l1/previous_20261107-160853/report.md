# RAE-DiT IAD results (B_image_l1)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.6037 | 0.4971 | 0.5029 | 0.7467 | 0.6670 | 0.6812 | 0.4644 | 0.0179 | 0.0759 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4398 | 0.4403 | 0.4306 | 0.6338 | 0.6332 | 0.6249 | 0.4318 | 0.0181 | 0.1427 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4814 | 0.6009 | 0.5895 | 0.6819 | 0.7458 | 0.7476 | 0.5964 | 0.0917 | 0.3428 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.5165 | 0.5333 | 0.5594 | 0.8141 | 0.8065 | 0.8269 | 0.3423 | 0.0114 | 0.0669 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2576 | 0.4968 | 0.4013 | 0.6770 | 0.7846 | 0.7407 | 0.6151 | 0.0204 | 0.2107 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.6657 | 0.4398 | 0.4567 | 0.8791 | 0.7447 | 0.7521 | 0.6762 | 0.0223 | 0.0824 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.3810 | 0.4495 | 0.4002 | 0.6856 | 0.7393 | 0.6901 | 0.5683 | 0.0081 | 0.2353 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7510 | 0.3617 | 0.4194 | 0.8703 | 0.6694 | 0.6948 | 0.7130 | 0.0133 | 0.3216 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.4854 | 0.5113 | 0.5079 | 0.7184 | 0.7333 | 0.7235 | 0.4591 | 0.0058 | 0.0926 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.5039 | 0.9161 | 0.8457 | 0.6627 | 0.9590 | 0.9187 | 0.8979 | 0.3941 | 0.8555 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3957 | 0.4354 | 0.4093 | 0.5503 | 0.5953 | 0.5845 | 0.2123 | 0.0125 | 0.0382 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7575 | 0.4150 | 0.3746 | 0.8511 | 0.6016 | 0.5853 | 0.5295 | 0.0204 | 0.0048 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.5829 | 0.7379 | 0.7502 | 0.8283 | 0.8971 | 0.9065 | 0.8355 | 0.0111 | 0.4904 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4046 | 0.5112 | 0.4786 | 0.7073 | 0.7739 | 0.7459 | 0.1587 | 0.0014 | 0.0005 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.5739 | 0.5425 | 0.5706 | 0.8005 | 0.7821 | 0.8046 | 0.2174 | 0.0014 | 0.0398 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.4229 | 0.3679 | 0.3921 | 0.4190 | 0.3272 | 0.3956 | 0.3381 | 0.0337 | 0.0661 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3900 | 0.3967 | 0.4446 | 0.3642 | 0.3426 | 0.3587 | 0.4597 | 0.0429 | 0.1425 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5358 | 0.5767 | 0.6046 | 0.4846 | 0.4732 | 0.5406 | 0.4400 | 0.0397 | 0.1598 |
