# EMDiarization2
An EM algorithm for joint source separation and diarisation of multichannel convolutive speech mixtures

```
D. Kounades-Bastian, L. Girin, X. Alameda-Pineda, R. Horaud and S. Gannot, "Exploiting the intermittency of speech for joint separation and diarization," 2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA,
pp. 41-45, doi: 10.1109/WASPAA.2017.8169991.
```



### MATLAB

This repo implements the above paper in MATLAB.

```python
# Running example.m generates a stereo mix with 2 sources 
# (by loading trueSrc1.wav,..) and then calls dnd.m 
# to diarize and separate this mix.
#
# OUTPUTS
#
# The separated source-images are written in .wav files 
# (estimatedSrc1.wav,..). 
# The diarization is written # in diarization.rttm (NIST '06) 
#
# Each line in the .rttm file is an interval of activity, e.g. 
#
#   SPEAKER ID 1 2.08 4.44 <NA> <NA> estimatedSrc1 <NA>
#
# indicates that SPEAKER with LABEL=estimatedSource2 
# starts talking at 2.08s for a duration of 4.44s, 
# hence going silent at 6.52s.
#
# In MATLAB command run:
>>> example
```

## PAPER
  - [pdf](https://inria.hal.science/hal-01568813/document) / [Poster](https://team.inria.fr/perception/files/2018/05/poster.pdf)






