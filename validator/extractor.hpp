#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/blob.hpp>

using namespace caffe;
using namespace std;

/* -------------------------------------------------------------------------------------- */

typedef struct
{
  unsigned int n;
  unsigned int channels;
  unsigned int height;
  unsigned int width;
} tensor_shape;


typedef struct
{
  tensor_shape shape;
  unsigned long capacity;

  float * data;
  float * data_gpu;
} tensor_float;


typedef enum 
{
  conv,
  pool,
  relu,
  fc,
  softmax,
  rpn,
  roi_pool,
  reshape
} layer_type;


typedef struct
{
  int index;
  layer_type type;
  tensor_float * weights;
  tensor_float * biases;
} model_layer;

typedef struct
{
  unsigned int layer_size;    // counter of total number of conv and fc layers
  model_layer * layers;
} model;

/* -------------------------------------------------------------------------------------- */

void tensor_out (const char * filepath, tensor_float * t)
{
  FILE * f = fopen(filepath, "wb");
  if (f == NULL)
  {
    printf("IOError: File at %s not found\n", filepath);
  }

  fwrite(t, sizeof(tensor_float), 1, f);
  fwrite(t->data, t->capacity*sizeof(float), 1, f);

  fclose(f);
}

/* -------------------------------------------------------------------------------------- */

template <typename Dtype>
void layers_extractor (Net<Dtype>* net)
{
  // layers
  for (int i = 0; i < net->layers().size(); ++i)
  {
    const string layerName = net->layer_names()[i];
    bool layerHasLayer = net->has_layer(layerName);
    const char * layerType = net->layers()[i]->type();
    string layerParamName = net->layers()[i]->layer_param().name();
    LOG(INFO) << layerName << " "
              << layerHasLayer << " "
              << layerType << " "
              << layerParamName << " " << endl;

  }
}


template <typename Dtype>
void weights_extractor (Net<Dtype>* net, string outpath, string weightnames[])
{
  // layer params
  for (int i = 0; i < net->params().size(); ++i)
  {
    string paramsName = net->param_display_names()[i];
    map<string, int> paramsNameIndex = net->param_names_index();
    const boost::shared_ptr<Blob<float> > thisParams = net->params()[i];
    const float * paramsOut = thisParams->cpu_data();
    LOG(INFO) << i << " (n, c, h, w) " << paramsName << ", "
                                       << thisParams->num() << " " 
                                       << thisParams->channels() << " "
                                       << thisParams->height() << " "
                                       << thisParams->width() << " " << endl;
    tensor_float * weights = (tensor_float*)malloc(sizeof(tensor_float));
    weights->shape.n = thisParams->num();
    weights->shape.channels = thisParams->channels();
    weights->shape.height = thisParams->height();
    weights->shape.width = thisParams->width();
    weights->capacity = thisParams->num() * thisParams->channels() * 
                        thisParams->height() * thisParams->width();
    weights->data = (float*)malloc(weights->capacity*sizeof(float));
    for (int index = 0; index < weights->capacity; ++index)
    {
      weights->data[index] = paramsOut[index];
    }

    string fpath;
    if (paramsName == "0")
      fpath = outpath + "weights_" + weightnames[i] +  "_w.data";
    else if (paramsName == "1")
      fpath = outpath + "weights_" + weightnames[i] + "_b.data";
    else
    {
      fpath = outpath; 
    }
    LOG(INFO) << fpath << endl;
    tensor_out (fpath.c_str(), weights);
    
    free (weights->data);
    free (weights);
  }
}


template <typename Dtype>
void outs_extractor (Net<Dtype>* net, string outpath)
{
  // layer out
  for (int i = 0; i < net->blobs().size(); ++i)
  {
    string layerName = net->blob_names()[i];
    const boost::shared_ptr<Blob<float> > thisLayer = net->blobs()[i];
    const float * layerOut = thisLayer->cpu_data();
    LOG(INFO) << i << " (n, c, h, w) " << layerName << ": " 
                                       << thisLayer->num() << " "
                                       << thisLayer->channels() << " " 
                                       << thisLayer->height() << " "
                                       << thisLayer->width() << endl;

    tensor_float * outs = (tensor_float*)malloc(sizeof(tensor_float));
    outs->shape.n = thisLayer->num();
    outs->shape.channels = thisLayer->channels();
    outs->shape.height = thisLayer->height();
    outs->shape.width = thisLayer->width();
    outs->capacity = thisLayer->num() * thisLayer->channels() * thisLayer->height() * thisLayer->width();
    outs->data = (float*)malloc(outs->capacity*sizeof(float));
    for (int i = 0; i < outs->capacity; ++i)
    {
      outs->data[i] = layerOut[i];
    }

    string fpath = outpath + "outs_" + layerName + ".data";
    LOG(INFO) << fpath << endl;
    tensor_out (fpath.c_str(), outs);

    free (outs->data);
    free (outs);
  }
}


