# SSE-Bench Overview

| Dataset | Events | SSE | SSE Rate | Slice (min) | Headline Window | Task1 XGB AUROC | Task1 XGB AUPRC | Task1 Graph AUROC | Task2 MAE | Task3 RMSE | Task4 Recall@k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snap_infopath_keywords | 60000 | 371 | 0.006 | 60 | 360 | 0.903 | 0.157 | 0.979 | 481.567 | 0.418 | 0.168 |
| uci_news_social_feedback | 260515 | 1507 | 0.006 | 20 | 360 | 0.767 | 0.190 | - | 346.687 | 1.244 | 0.129 |

### snap_infopath_keywords Window Sweep

| Window (min) | Task1 XGB AUROC | Task1 XGB AUPRC | Task3 RMSE | Task4 Recall@k |
| --- | --- | --- | --- | --- |
| 60 | 0.602 | 0.059 | 0.695 | 0.045 |
| 360 | 0.903 | 0.157 | 0.418 | 0.168 |
| 1440 | 0.978 | 0.405 | 0.167 | 0.338 |

### uci_news_social_feedback Window Sweep

| Window (min) | Task1 XGB AUROC | Task1 XGB AUPRC | Task3 RMSE | Task4 Recall@k |
| --- | --- | --- | --- | --- |
| 20 | 0.557 | 0.029 | 1.562 | 0.024 |
| 60 | 0.597 | 0.084 | 1.520 | 0.046 |
| 360 | 0.767 | 0.190 | 1.244 | 0.129 |
| 1440 | 0.937 | 0.300 | 0.548 | 0.337 |

