# Project-Neuron
A repository that holds source code relating to Project Neuron.

### File Description
* `demo-pipeline.py`: Works with sample data associated with CaImAn project to generate a full pipeline system that produces a visual output regarding the detected spikes and regions.
* `denoise-spikes.py`: Reads in 1-D data from a file to generate figures that map its waveforms and indicated spiking region. The example uses `data.mat` for its input data.
* `full-pipeline.py`: Produces a full pipeline alongside a visual output that does a complete tagging of an input movie. The example uses `movie.tif` as the movie and `masks-output.mat` as the regions of interest, though a MaskRCNN can be used instead to locate these by commenting out the code in Part 4.
* `readBinMov.m`: Reads a binary file that contains movie data to produce a multi-dimensional TIF movie that can be used for analysis.
* `readMask.m`: Reads a binary file that contains mask data to produce a multi-dimensional matrix that can be used to map the regions of interest.
* `experimental-parameters.txt`: Provides insight into the parameters that govern the movie.
