# Mean wait time results

Mean wait time across 30 seeds per cell, with bootstrap 95% confidence intervals.
All values in seconds.

| Arrival rate (per min) | StrictFIFO | ShortestQueue | Random | UnstrictFIFO | RoundRobin |
|---|---|---|---|---|---|
| 1.5 | 12.4 [10.5, 14.7] | 12.4 [10.4, 14.6] | 10.3 [8.5, 12.6] | 7.9 [6.4, 9.7] | 387.6 [360.0, 413.3] |
| 1.8 | 19.2 [16.6, 22.1] | 19.2 [16.7, 22.1] | 16.4 [14.0, 19.2] | 15.5 [12.9, 18.6] | 523.9 [489.9, 555.7] |
| 2.0 | 26.1 [22.9, 29.9] | 26.1 [22.9, 29.8] | 22.8 [19.5, 26.5] | 21.0 [17.9, 24.4] | 647.8 [604.6, 691.6] |
| 2.2 | 38.1 [34.1, 43.1] | 38.1 [34.0, 43.0] | 35.7 [31.5, 41.1] | 32.7 [28.1, 37.8] | 802.2 [750.4, 854.4] |
| 2.4 | 49.7 [44.6, 54.4] | 49.7 [44.4, 54.6] | 49.4 [43.9, 55.1] | 46.3 [40.7, 52.4] | 974.7 [919.0, 1031.9] |

**Reading the table.** Each cell is `mean [95% CI low, high]` over 30 seeds.
Non-overlapping CIs indicate the difference is unlikely to be due to chance.