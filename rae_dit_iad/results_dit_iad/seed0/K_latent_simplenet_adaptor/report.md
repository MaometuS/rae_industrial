# RAE-DiT IAD results (K_latent_simplenet_adaptor)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.4816 | 0.4941 | 0.5011 | 0.6559 | 0.6989 | 0.7041 | 0.6754 | 0.0681 | 0.2091 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7211 | 0.6935 | 0.7061 | 0.8320 | 0.8232 | 0.8289 | 0.6634 | 0.1919 | 0.2399 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4172 | 0.5684 | 0.5456 | 0.6383 | 0.7304 | 0.7252 | 0.5710 | 0.0743 | 0.3334 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.5044 | 0.5092 | 0.5530 | 0.8193 | 0.8046 | 0.8369 | 0.3415 | 0.0114 | 0.0693 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.8387 | 0.8467 | 0.8640 | 0.9493 | 0.9539 | 0.9601 | 0.5927 | 0.0190 | 0.0992 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.3162 | 0.2877 | 0.3154 | 0.6878 | 0.6634 | 0.6763 | 0.5917 | 0.0178 | 0.0194 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.3985 | 0.4302 | 0.3943 | 0.6887 | 0.7266 | 0.6900 | 0.5263 | 0.0073 | 0.2115 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5104 | 0.5322 | 0.5180 | 0.7263 | 0.7614 | 0.7481 | 0.3976 | 0.0051 | 0.0326 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.5129 | 0.4737 | 0.4837 | 0.7013 | 0.7636 | 0.7784 | 0.5303 | 0.0067 | 0.0450 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.4821 | 0.8932 | 0.7993 | 0.6465 | 0.9493 | 0.8977 | 0.8088 | 0.3093 | 0.7858 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7932 | 0.7371 | 0.7721 | 0.8735 | 0.8477 | 0.8632 | 0.8819 | 0.5038 | 0.4700 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.5064 | 0.5625 | 0.5468 | 0.6193 | 0.7080 | 0.6964 | 0.9009 | 0.1270 | 0.5818 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.2763 | 0.6368 | 0.5868 | 0.6072 | 0.8410 | 0.8109 | 0.8513 | 0.0106 | 0.5266 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.6323 | 0.5661 | 0.5669 | 0.8051 | 0.7717 | 0.7768 | 0.5685 | 0.0030 | 0.1827 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.4687 | 0.4497 | 0.4605 | 0.7148 | 0.7125 | 0.7158 | 0.5782 | 0.0027 | 0.0396 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.4246 | 0.3725 | 0.3946 | 0.4300 | 0.3305 | 0.3904 | 0.3269 | 0.0332 | 0.0737 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.8308 | 0.7854 | 0.8096 | 0.8055 | 0.7813 | 0.7963 | 0.8763 | 0.4287 | 0.4150 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.6038 | 0.6967 | 0.6992 | 0.5565 | 0.6471 | 0.6538 | 0.7760 | 0.1865 | 0.3596 |
