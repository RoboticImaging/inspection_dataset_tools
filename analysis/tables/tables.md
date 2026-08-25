# Result tables

## T1_main_results

**Headline scores on the three assets every method completed**

_mean of the per-asset medians (glass_house2, old_house, unfinished_build). The last two columns come from all seven assets and show how far each method scales. COLMAP and VGGT-SLAM are scored only on the frames they registered (see T6)._

| method        |   AUC@30 ↑ |   RTA@5 ↑ |   RRA@5 ↑ |   rel. MAE ↓ |   Acc@1.05 ↑ |   F@0.25m ↑ |   Chamfer ↓ | assets done   |   OOM on |
|:--------------|-----------:|----------:|----------:|-------------:|-------------:|------------:|------------:|:--------------|---------:|
| colmap/no_gt  |      0.992 |      99.3 |      99.3 |        0.148 |        0.766 |       0.584 |       0.325 | 6/7           |        0 |
| mpsfm         |      0.815 |      79.5 |      92.7 |        0.233 |        0.256 |       0.317 |       0.962 | 3/7           |        0 |
| vggt          |      0.702 |      42.7 |      94.4 |        0.077 |        0.556 |       0.405 |       0.355 | 4/7           |        3 |
| vggt-long     |      0.75  |      52.5 |      94.1 |        0.103 |        0.393 |       0.284 |       0.51  | 6/7           |        0 |
| vggt-slam     |      0.78  |      51.2 |      93.4 |        0.147 |        0.189 |       0.282 |       0.744 | 6/7           |        0 |
| vggt-omega    |      0.913 |      85   |      99.3 |        0.032 |        0.857 |       0.595 |       0.297 | 3/7           |        4 |
| da3           |      0.979 |      98.8 |      99.3 |        0.027 |        0.898 |       0.658 |       0.282 | 3/7           |        4 |
| da3_streaming |      0.979 |      99   |      99.3 |        0.031 |        0.895 |       0.629 |       0.293 | 6/7           |        0 |
| pi3           |      0.897 |      80.4 |      99.3 |        0.035 |        0.821 |       0.55  |       0.299 | 4/7           |        2 |
| pi3x          |      0.692 |      85.9 |      73.3 |        0.146 |        0.251 |       0.384 |       0.403 | 6/7           |        1 |
| pi-long       |      0.923 |      87.2 |      99.6 |        0.061 |        0.472 |       0.563 |       0.307 | 6/7           |        0 |


## T2_scaling

**How far each method scales, in frames**

_longest sequence scored and shortest sequence that ran out of GPU memory. Every one-shot method has a ceiling; no chunked or streaming method hit one._

| method        | mode                |   longest_ok |   shortest_OOM |   assets_ok |   assets_OOM |   assets_not_run |
|:--------------|:--------------------|-------------:|---------------:|------------:|-------------:|-----------------:|
| colmap/no_gt  | chunked / streaming |          803 |     nan        |           6 |            0 |                1 |
| vggt-long     | chunked / streaming |          803 |     nan        |           6 |            0 |                1 |
| vggt-slam     | chunked / streaming |          803 |     nan        |           6 |            0 |                1 |
| da3_streaming | chunked / streaming |          803 |     nan        |           6 |            0 |                1 |
| pi3x          | chunked / streaming |          803 |       1.04e+03 |           6 |            1 |                0 |
| pi-long       | chunked / streaming |          803 |     nan        |           6 |            0 |                1 |
| mpsfm         | chunked / streaming |           99 |     nan        |           3 |            0 |                4 |
| vggt          | one-shot            |          522 |     566        |           4 |            3 |                0 |
| pi3           | one-shot            |          522 |     803        |           4 |            2 |                1 |
| vggt-omega    | one-shot            |           99 |     522        |           3 |            4 |                0 |
| da3           | one-shot            |           99 |     522        |           3 |            4 |                0 |


## T3_asset_difficulty

**Asset difficulty, ordered by sequence length**

_'median' is over the methods that completed that asset - so the long assets are already scored by a survivor set of the more capable methods, and are still far worse._

| asset            |   frames |   methods scored |   methods OOM | best AUC@30           |   median AUC@30 | best rel. MAE          |   median rel. MAE | best F@0.25m           |   median F@0.25m |
|:-----------------|---------:|-----------------:|--------------:|:----------------------|----------------:|:-----------------------|------------------:|:-----------------------|-----------------:|
| glass_house2     |       97 |               11 |             0 | colmap/no_gt (0.979)  |          0.894  | da3 (0.0433)           |            0.107  | pi3 (0.495)            |           0.382  |
| old_house        |       99 |               11 |             0 | colmap/no_gt (0.999)  |          0.936  | vggt-omega (0.0215)    |            0.051  | da3 (0.658)            |           0.593  |
| unfinished_build |       99 |               11 |             0 | mpsfm (1)             |          0.865  | da3 (0.0158)           |            0.0578 | da3 (0.848)            |           0.678  |
| crane            |      522 |                8 |             2 | da3_streaming (0.246) |          0.0576 | da3_streaming (0.716)  |            0.855  | da3_streaming (0.0403) |           0.0166 |
| bridge           |      566 |                6 |             3 | da3_streaming (0.881) |          0.474  | da3_streaming (0.0799) |            0.478  | da3_streaming (0.413)  |           0.0719 |
| ship             |      803 |                6 |             4 | colmap/no_gt (0.982)  |          0.755  | da3_streaming (0.0705) |            0.209  | colmap/no_gt (0.386)   |           0.11   |
| office2          |     1042 |                0 |             5 | vggt-slam (0.0427)    |          0.0384 | vggt-long (0.411)      |            0.5    | vggt-long (0.194)      |           0.116  |


## T4_axis_sensitivity

**How much each axis of scene variation costs, relative to its reference level**

_positive = worse than reference. Trajectory and illumination move the numbers; soiling and reflectivity move almost nothing except for one method each._

|                                            |   median worst-level cost % |   mean cost at worst level % | worst level   | most-affected method   |
|:-------------------------------------------|----------------------------:|-----------------------------:|:--------------|:-----------------------|
| ('trajectory', 'pose · AUC@30')            |                         3.3 |                          7.2 | trendspek     | vggt-slam (+37%)       |
| ('trajectory', 'depth · Relative MAE')     |                        82.9 |                         68.2 | trendspek     | vggt-slam (+183%)      |
| ('trajectory', 'cloud · F-score @0.25m')   |                         5.9 |                          4.4 | trendspek     | pi3x (+49%)            |
| ('illumination', 'pose · AUC@30')          |                         5.3 |                         10.1 | sunrise       | vggt (+33%)            |
| ('illumination', 'depth · Relative MAE')   |                        46.4 |                         47.4 | sunrise       | vggt-long (+107%)      |
| ('illumination', 'cloud · F-score @0.25m') |                         5.4 |                          1.5 | sunrise       | vggt (+37%)            |
| ('soiling', 'pose · AUC@30')               |                         0.1 |                          4.5 | high          | mpsfm (+58%)           |
| ('soiling', 'depth · Relative MAE')        |                         1.9 |                          7.6 | high          | mpsfm (+97%)           |
| ('soiling', 'cloud · F-score @0.25m')      |                         0   |                          0.6 | medium        | mpsfm (+64%)           |
| ('reflectivity', 'pose · AUC@30')          |                         1.5 |                         12.5 | 1.0           | mpsfm (+74%)           |
| ('reflectivity', 'depth · Relative MAE')   |                         2.9 |                         28.2 | 1.0           | mpsfm (+184%)          |
| ('reflectivity', 'cloud · F-score @0.25m') |                         5.8 |                         11.5 | 1.0           | mpsfm (+78%)           |


## T5_robustness

**Worst level on each appearance axis, and what it costs in pose AUC@30**

_core assets for illumination; glass_house2 for soiling and reflectivity, the only asset that varies them._

| method        | illumination worst   | soiling worst   | reflectivity worst   |
|:--------------|:---------------------|:----------------|:---------------------|
| colmap/no_gt  | overcast (−5%)       | verylow (−0%)   | 0.25 (−0%)           |
| mpsfm         | sunrise (−29%)       | high (−58%)     | 1.0 (−74%)           |
| vggt          | sunrise (−33%)       | verylow (−0%)   | 1.0 (−19%)           |
| vggt-long     | sunrise (−21%)       | low (−0%)       | 1.0 (−14%)           |
| vggt-slam     | sunrise (−17%)       | verylow (−0%)   | 1.0 (−24%)           |
| vggt-omega    | sunrise (−2%)        | medium (−0%)    | 0.5 (−1%)            |
| da3           | sunrise (−2%)        | verylow (−0%)   | 1.0 (−1%)            |
| da3_streaming | sunrise (−1%)        | high (−0%)      | 0.5 (−2%)            |
| pi3           | sunrise (−2%)        | medium (−0%)    | 1.0 (−0%)            |
| pi3x          | night (−8%)          | medium (−1%)    | 1.0 (−4%)            |
| pi-long       | night (−2%)          | low (−0%)       | 0.75 (−0%)           |


## T6_frame_registration

**Share of each sequence's frames a method is actually scored on**

_1.0 = scored on every frame any method saw. COLMAP and VGGT-SLAM drop frames they cannot register and are then scored only on the rest, so their scores are not measured on the same footing as the rest._

| method        |   median reg. rate |   mean |   min |   % of runs below 90% |
|:--------------|-------------------:|-------:|------:|----------------------:|
| colmap/no_gt  |              0.96  |  0.831 | 0.312 |                  45.5 |
| mpsfm         |              1     |  0.999 | 0.99  |                   0   |
| vggt          |              1     |  1     | 1     |                   0   |
| vggt-long     |              1     |  1     | 1     |                   0   |
| vggt-slam     |              0.959 |  0.912 | 0.552 |                  32.3 |
| vggt-omega    |              1     |  1     | 1     |                   0   |
| da3           |              1     |  0.996 | 0.38  |                   0.6 |
| da3_streaming |              1     |  1     | 1     |                   0   |
| pi3           |              1     |  1     | 1     |                   0   |
| pi3x          |              1     |  1     | 1     |                   0   |
| pi-long       |              1     |  1     | 1     |                   0   |


## T7_completion

**How much of the benchmark grid is filled in**

_the grid is 7 assets x 11 methods x that asset's variants = 4,048 rows per category._

| category   |   scored |   OOM |   not_run |    total |   pct_scored |
|:-----------|---------:|------:|----------:|---------:|-------------:|
| pose       | 2.31e+03 |   864 |       872 | 4.05e+03 |         57.1 |
| depth      | 2.3e+03  |   880 |       869 | 4.05e+03 |         56.8 |
| cloud      | 2.31e+03 |   880 |       856 | 4.05e+03 |         57.1 |
