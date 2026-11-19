# RAE-DiT IAD results (J5_latent_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.5025 | 0.8496 | 0.8480 | 0.8028 | 0.9488 | 0.9465 | 0.9234 | 0.4467 | 0.8053 |
| single | 1.0 | 1 | best | latent_cos | 0.8841 | 0.9598 | 0.9603 | 0.9527 | 0.9869 | 0.9868 | 0.9829 | 0.3367 | 0.9192 |
| single | 1.0 | 1 | best | pixel | 0.4585 | 0.6492 | 0.6521 | 0.7419 | 0.8574 | 0.8556 | 0.7699 | 0.0604 | 0.4370 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6677 | 0.6545 | 0.7159 | 0.8862 | 0.8652 | 0.8981 | 0.6723 | 0.0971 | 0.3093 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9822 | 0.3877 | 0.9283 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2813 | 0.9165 | 0.9053 | 0.7290 | 0.9759 | 0.9729 | 0.9848 | 0.7273 | 0.9397 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2492 | 0.6440 | 0.5882 | 0.5976 | 0.8496 | 0.8131 | 0.8675 | 0.0236 | 0.5647 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7682 | 0.9197 | 0.9207 | 0.9053 | 0.9738 | 0.9736 | 0.9836 | 0.2857 | 0.9101 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.7237 | 0.7827 | 0.7907 | 0.8766 | 0.9217 | 0.9202 | 0.8620 | 0.1661 | 0.6708 |
