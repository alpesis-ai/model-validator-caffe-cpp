#include "extractor.hpp"


string WEIGHT_NAMES[8] = {"conv1", "conv1",
                          "conv2", "conv2",
                          "ip1", "ip1",
                          "ip2", "ip2"};

template <typename Dtype>
void net_extractor (char * pt, char * model, char * outpath)
{
  Caffe::set_mode (Caffe::CPU);
  Net<Dtype> net (pt,caffe::TEST);
  net.CopyTrainedLayersFrom(model);


  Dtype loss = 0.0;
  vector<Blob<Dtype>*> results = net.ForwardPrefilled(&loss);

  // net
  string name = net.name();
  LOG(INFO) << name << endl;

  layers_extractor(&net);
  weights_extractor(&net, (string)outpath, WEIGHT_NAMES);
  outs_extractor(&net, outpath);

}


/*
 * usage: ./_build/extractor <prototxt> <caffemodel>
 */
int main(int argc, char ** argv)
{
  cout << "num of args: " << argc << endl;
  net_extractor<float>((char*)argv[1], (char*)argv[2], (char*)argv[3]);

  return 0;
}

