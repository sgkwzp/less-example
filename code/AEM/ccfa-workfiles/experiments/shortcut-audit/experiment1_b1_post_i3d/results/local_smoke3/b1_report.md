# G1-B1 Post-I3D temporal resampling

The intervention rescales cached I3D trajectories and annotated segment coordinates while keeping the released AEM checkpoint and GT action labels fixed.

- rates: `[1.0]`
- rows: 24
- segments with rate=1 baseline: 24
- sham mean/max absolute delta: 0.0 / 0.0

## Video-level slopes

Slope is Δscore against |log(rate)|, fitted separately for rates below and above 1.0. CI is a 2000-draw video bootstrap.

| stratum | n videos | mean beta | 95% CI |
|---|---:|---:|---|
| g0_fast | 0 | None | [None, None] |
| g0_slow | 0 | None | [None, None] |
| g1_fast | 0 | None | [None, None] |
| g1_slow | 0 | None | [None, None] |

A B1 pilot is evidence about the downstream temporal path only. It does not establish sensitivity of the physical RGB/I3D representation; that requires B2.
