# RAE-DiT IAD results (ab_noise_sigma005)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.05 | 1 | joint_best.pt | disc | 0.5642 | 0.4932 | 0.4948 | 0.7272 | 0.6674 | 0.6781 | 0.4991 | 0.0191 | 0.0853 |
| single | 0.05 | 1 | joint_best.pt | latent_cos | 0.4972 | 0.4714 | 0.4408 | 0.6696 | 0.6580 | 0.6395 | 0.4108 | 0.0165 | 0.1005 |
| single | 0.05 | 1 | joint_best.pt | pixel | 0.4261 | 0.5731 | 0.5544 | 0.6424 | 0.7320 | 0.7295 | 0.5848 | 0.0766 | 0.3450 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.05 | 1 | joint_best.pt | pixel | 0.5201 | 0.5201 | 0.5614 | 0.8249 | 0.8080 | 0.8373 | 0.3502 | 0.0116 | 0.0715 |
| carpet | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.3985 | 0.5999 | 0.5510 | 0.7529 | 0.8404 | 0.8228 | 0.5806 | 0.0179 | 0.1309 |
| carpet | single | 0.05 | 1 | joint_best.pt | disc | 0.5273 | 0.4354 | 0.4490 | 0.8112 | 0.7459 | 0.7522 | 0.6717 | 0.0221 | 0.0870 |
| grid | single | 0.05 | 1 | joint_best.pt | pixel | 0.3960 | 0.4277 | 0.3968 | 0.6898 | 0.7285 | 0.6908 | 0.5408 | 0.0075 | 0.2172 |
| grid | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.7368 | 0.4135 | 0.3935 | 0.8640 | 0.7062 | 0.6909 | 0.6530 | 0.0103 | 0.2547 |
| grid | single | 0.05 | 1 | joint_best.pt | disc | 0.4762 | 0.4921 | 0.4946 | 0.7153 | 0.7243 | 0.7206 | 0.4376 | 0.0056 | 0.0867 |
| hazelnut | single | 0.05 | 1 | joint_best.pt | pixel | 0.4879 | 0.9032 | 0.8182 | 0.6528 | 0.9531 | 0.9038 | 0.8447 | 0.3185 | 0.8188 |
| hazelnut | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.5075 | 0.4896 | 0.4036 | 0.6173 | 0.6377 | 0.5898 | 0.2542 | 0.0133 | 0.0045 |
| hazelnut | single | 0.05 | 1 | joint_best.pt | disc | 0.7911 | 0.4139 | 0.3864 | 0.8674 | 0.6029 | 0.5903 | 0.5917 | 0.0237 | 0.0144 |
| screw | single | 0.05 | 1 | joint_best.pt | pixel | 0.3023 | 0.6391 | 0.6005 | 0.6183 | 0.8397 | 0.8189 | 0.8575 | 0.0118 | 0.5428 |
| screw | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.4001 | 0.4845 | 0.4409 | 0.7076 | 0.7663 | 0.7347 | 0.1459 | 0.0014 | 0.0000 |
| screw | single | 0.05 | 1 | joint_best.pt | disc | 0.5106 | 0.5313 | 0.5384 | 0.7729 | 0.7774 | 0.7846 | 0.3320 | 0.0017 | 0.0594 |
| transistor | single | 0.05 | 1 | joint_best.pt | pixel | 0.4242 | 0.3754 | 0.3950 | 0.4261 | 0.3310 | 0.3965 | 0.3307 | 0.0335 | 0.0746 |
| transistor | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.4433 | 0.3696 | 0.4150 | 0.4062 | 0.3394 | 0.3590 | 0.4202 | 0.0397 | 0.1124 |
| transistor | single | 0.05 | 1 | joint_best.pt | disc | 0.5158 | 0.5933 | 0.6054 | 0.4690 | 0.4866 | 0.5430 | 0.4625 | 0.0427 | 0.1793 |
