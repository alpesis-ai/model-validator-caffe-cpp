##############################################################################
Model Validator Caffe (C++)
##############################################################################

Extracting weights and layer outputs from caffe model and forward process.

==============================================================================
Getting Started
==============================================================================

::

    $ cd validator
    $ vim extractor.cpp   # update the layer names in WEIGHT_NAMES

    $ make all
    $ ./_build/extractor <prototxt> <caffemodel> <outpath>
