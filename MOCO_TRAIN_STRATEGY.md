Stage 0: normalize the domains
Convert all images to a consistent pipeline: grayscale→3ch, same crop size (e.g. 224/256), similar normalization.
Optional but helpful: keep dataset ID metadata so you can do controlled sampling later.

Stage 1: “generic SEM prior”
Pretrain MoCo on NFFA + nanoparticle dataset (exclude Carinthia for now)
Goal: learn general SEM textures/edges/contrast/noise invariances
Length: ~100–200 epochs (or a fixed number of steps)

Stage 2: domain adapt to wafer SEM
Continue MoCo pretraining with Carinthia-heavy sampling
Use a mixture like:
70–90% Carinthia
10–30% NFFA + nanoparticle
Length: 50–150 epochs
Goal: shift embeddings toward wafer backgrounds/defect morphology while retaining some generalization

Stage 3 (optional): in-domain “polish”
Final MoCo continuation on Carinthia only for 20–50 epochs
Helps if your downstream task is strictly “wafer SEM defects”
