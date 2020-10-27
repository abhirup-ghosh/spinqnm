#pSEOBNRv4HM files
# C01 files
## GR
rsync -ztave gsissh hypatia1.aei.mpg.de:/home/abhirup.ghosh/Documents/Work/spinqnm/runs/XLALSimInspiralChooseTDWaveformFromCache_runs/ligo_events/S190521g/C01/GR/cbcBayes/posterior_samples.dat pSEOBNRv4HM_GR_c01.dat

## 220
rsync -ztave gsissh hypatia1.aei.mpg.de:/home/abhirup.ghosh/Documents/Work/spinqnm/runs/XLALSimInspiralChooseTDWaveformFromCache_runs/ligo_events/S190521g/C01/220/cbcBayes/posterior_samples.dat pSEOBNRv4HM_frac_220_c01.dat

# C01 nonsens files
## GR
rsync -ztave gsissh hypatia1.aei.mpg.de:/home/abhirup.ghosh/Documents/Work/spinqnm/runs/XLALSimInspiralChooseTDWaveformFromCache_runs/ligo_events/S190521g/C01_nonsens/GR/cbcBayes/posterior_samples.dat pSEOBNRv4HM_GR_c01_nonsens.dat

## 220
rsync -ztave gsissh hypatia1.aei.mpg.de:/home/abhirup.ghosh/Documents/Work/spinqnm/runs/XLALSimInspiralChooseTDWaveformFromCache_runs/ligo_events/S190521g/C01_nonsens/220/cbcBayes/posterior_samples.dat pSEOBNRv4HM_frac_220_c01_nonsens.dat

# pEOBNRv2HM files
# C01 files
## GR
rsync -ztave ssh abhirup.ghosh@ldas-grid.ligo.caltech.edu:/home/richard.brito/public_html/LVC/projects/pEOBNRv2HM/O3/S190521g/C01/GR/1126259462.39-0/H1L1/posterior_samples.dat pEOBNRv2HM_frac_richard_GR_c01.dat

# 220 large priors
rsync -ztave ssh abhirup.ghosh@ldas-grid.ligo.caltech.edu:/home/richard.brito/public_html/LVC/projects/pEOBNRv2HM/O3/S190521g/C01/220_largeposterior/1126259462.39-0/H1L1/posterior_samples.dat pEOBNRv2HM_frac_richard_220_c01.dat

# C01 files
## GR
rsync -ztave ssh abhirup.ghosh@ldas-grid.ligo.caltech.edu:/home/richard.brito/public_html/LVC/projects/pEOBNRv2HM/O3/S190521g/C01/GR_sub60Hz/1126259462.39-0/H1L1/posterior_samples.dat pEOBNRv2HM_frac_richard_GR_c01_nonsens.dat

# 220 large priors
rsync -ztave ssh abhirup.ghosh@ldas-grid.ligo.caltech.edu:/home/richard.brito/public_html/LVC/projects/pEOBNRv2HM/O3/S190521g/C01/220_sub60Hz/1126259462.39-0/H1L1/posterior_samples.dat pEOBNRv2HM_frac_richard_220_c01_nonsens.dat
