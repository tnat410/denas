import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
##LBANN stuff
import lbann
import lbann.contrib.args
import lbann.contrib.launcher
import lbann.models
import lbann.models.resnet
import lbann.contrib.args
import lbann.contrib.models.wide_resnet
import lbann.contrib.launcher
import data.cifar10



try:
    import configparser
except ImportError:
    import ConfigParser as configparser



file_path = os.path.dirname(os.path.realpath(__file__))

def common_parser(parser):

    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(file_path, 'cifar10_default_model.txt'),
                        help="specify model configuration file")
    parser.add_argument("--nodes", type=int, default=16) 
    parser.add_argument("--ppn", type=int, default=4) 
    parser.add_argument("--ppt", type=int, default=2) 
    parser.add_argument('--block-type', action='store', default=None, type=str,
    	choices=('basic', 'bottleneck'),
    	help='ResNet block type')
    parser.add_argument('--blocks', action='store', default=None, type=str,
    	help='ResNet block counts (comma-separated list)')
    parser.add_argument('--block-channels', action='store', default=None, type=str,
    	help='Internal channels in each ResNet block (comma-separated list)')
    parser.add_argument('--warmup', action='store_true', help='use a linear warmup')
    parser.add_argument('--mini-batch-size', action='store', default=256, type=int,
    	help='mini-batch size (default: 256)', metavar='NUM')
    parser.add_argument('--num-epochs', action='store', default=90, type=int,
    	help='number of epochs (default: 90)', metavar='NUM')
    parser.add_argument('--num-classes', action='store', default=10, type=int,
    	help='number of Cifar10 classes (default: 1000)', metavar='NUM')
    parser.add_argument('--random-seed', action='store', default=0, type=int,
    	help='random seed for LBANN RNGs', metavar='NUM')


    return parser

def get_model_parser():

	parser = argparse.ArgumentParser(prog='cifar10_baseline', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='CIFAR10 LBANN ')

	return common_parser(parser).parse_args()

def read_config_file(file):
    #print("Reading default config (param) file : ", file)
    config=configparser.ConfigParser()
    config.read(file)
    section=config.sections()
    fileParams={}

    fileParams['model_name']=eval(config.get(section[0],'model_name'))
    fileParams['save']=eval(config.get(section[0], 'save'))
    fileParams['run_id']=eval(config.get(section[0], 'run_id')) 
    fileParams['width']=eval(config.get(section[0],'width'))
    fileParams['resnet']=eval(config.get(section[0],'resnet'))
    fileParams['bn_statistics_group_size']=eval(config.get(section[0],'bn_statistics_group_size'))
    fileParams['rotation']=eval(config.get(section[0], 'rotation'))
    fileParams['shear']=eval(config.get(section[0], 'shear'))
    fileParams['translation']=eval(config.get(section[0], 'translation'))
    
    return fileParams

def initialize_parameters(args):
    # Get command-line parameters
    #args = get_model_parser()
    #args = parser.parse_args()
    # Get parameters from configuration file
    gParameters = read_config_file(args.config_file)
    return gParameters

   

def run(gParameters,run_args,exp_dir=None):
  
  # Due to a data reader limitation, the actual model realization must be
  # hardcoded to 10 labels for Cifar10.
  cifar10_labels = 10

  # Choose ResNet variant
  resnet_variant_dict = {18: lbann.models.ResNet18,
                       34: lbann.models.ResNet34,
                       50: lbann.models.ResNet50,
                       101: lbann.models.ResNet101,
                       152: lbann.models.ResNet152}
  wide_resnet_variant_dict = {50: lbann.contrib.models.wide_resnet.WideResNet50_2}
  block_variant_dict = {
    'basic': lbann.models.resnet.BasicBlock,
    'bottleneck': lbann.models.resnet.BottleneckBlock
  }


  if gParameters['width'] == 1:
    # Vanilla ResNet.
    resnet = resnet_variant_dict[gParameters['resnet']](
        cifar10_labels,
        bn_statistics_group_size=gParameters['bn_statistics_group_size'])
  elif gParameters['width'] == 2 and gParameters['resnet'] == 50:
    # Use pre-defined WRN-50-2.
    resnet = wide_resnet_variant_dict[gParameters['resnet']](
        cifar10_labels,
        bn_statistics_group_size=gParameters['bn_statistics_group_size'])
  else:
    # Some other Wide ResNet.
    resnet = resnet_variant_dict[gParameters['resnet']](
        cifar10_labels,
        bn_statistics_group_size=gParameters['bn_statistics_group_size'],
        width=gParameters['width'])


  # Construct layer graph
  input_ = lbann.Input(target_mode='classification')
  images = lbann.Identity(input_)
  labels = lbann.Identity(input_)

  # Data argumentation
  # Rotation
  mag_rot = str(gParameters['rotation'])
  rot_weights = lbann.Weights(initializer=lbann.ValueInitializer(values=mag_rot), name='rot_weights', optimizer=lbann.NoOptimizer())
  rot = lbann.WeightsLayer(dims='1', weights=rot_weights, name='rot', device='CPU')

  # Shear
  mag_shear = str(gParameters['shear']) + ' ' + str(gParameters['shear'])
  shear_weights = lbann.Weights(initializer=lbann.ValueInitializer(values=mag_shear), name='shear_weights', optimizer=lbann.NoOptimizer())
  shear = lbann.WeightsLayer(dims='2', weights=shear_weights, name='shear', device='CPU')

  # Translation
  mag_trans = str(gParameters['translation']) + ' ' + str(gParameters['translation'])
  trans_weights = lbann.Weights(initializer=lbann.ValueInitializer(values=mag_trans), name='trans_weights', optimizer=lbann.NoOptimizer())
  trans = lbann.WeightsLayer(dims='2', weights=trans_weights, name='trans', device='CPU')

  images = lbann.Composite_Image_Translation(images, rot, shear, trans, device='CPU')


  preds = resnet(images)
  probs = lbann.Softmax(preds)
  cross_entropy = lbann.CrossEntropy(probs, labels)
  top1 = lbann.CategoricalAccuracy(probs, labels)
  top5 = lbann.TopKCategoricalAccuracy(probs, labels, k=5)
  layers = list(lbann.traverse_layer_graph(input_))

  # Setup tensor core operations (just to demonstrate enum usage)
  tensor_ops_mode = lbann.ConvTensorOpsMode.NO_TENSOR_OPS
  for l in layers:
    if type(l) == lbann.Convolution:
        l.conv_tensor_op_mode=tensor_ops_mode

  # Setup objective function
  l2_reg_weights = set()
  for l in layers:
    if type(l) == lbann.Convolution or type(l) == lbann.FullyConnected:
        l2_reg_weights.update(l.weights)
  l2_reg = lbann.L2WeightRegularization(weights=l2_reg_weights, scale=1e-4)
  obj = lbann.ObjectiveFunction([cross_entropy, l2_reg])

  # Setup model
  metrics = [lbann.Metric(top1, name='accuracy', unit='%'),
             lbann.Metric(top5, name='top-5', unit='%')]
  callbacks = [lbann.CallbackPrint(),
               lbann.CallbackTimer(),
               lbann.CallbackDropFixedLearningRate(
                 drop_epoch=[30, 60, 80], amt=0.1)]

  # Perturb callback
  callbacks.append(lbann.CallbackPerturbWeights(output_name='rot_weights',batch_interval=400,lower=-45,upper=45,scale=90,perturb_probability=0.5))
  
  callbacks.append(lbann.CallbackPerturbWeights(output_name='shear_weights',batch_interval=400,lower=-0.15,upper=0.15,scale=0.2,perturb_probability=0.5))
  
  callbacks.append(lbann.CallbackPerturbWeights(output_name='trans_weights',batch_interval=400,lower=-5,upper=5,scale=10,perturb_probability=0.5))
 

  if run_args.warmup:
    callbacks.append(
        lbann.CallbackLinearGrowthLearningRate(
            target=0.1 * args.mini_batch_size / 256, num_epochs=5))
  
  model = lbann.Model(run_args.num_epochs,
                      layers=layers,
                      objective_function=obj,
                      metrics=metrics,
                      callbacks=callbacks)

  # Setup optimizer
  opt = lbann.SGD(learn_rate=0.001, momentum=0.9)

  # Setup data reader 
  data_reader = data.cifar10.make_data_reader(num_classes=10)

  job_name = "t"+ str(gParameters['run_id']-1)

  # Setup the training algorithm
  RPE = lbann.RandomPairwiseExchange
  SGD = lbann.BatchedIterativeOptimizer
  metalearning = RPE(
                   metric_strategies={'accuracy': RPE.MetricStrategy.HIGHER_IS_BETTER})
  ltfb = lbann.LTFB("ltfb",
                   metalearning=metalearning,
                   local_algo=SGD("local sgd",
                                   num_iterations=400),
                   metalearning_steps=40)

  # Setup trainer
  trainer = lbann.Trainer(mini_batch_size=128,
                          training_algo=ltfb)

  # Run experiment
 
  status = lbann.contrib.launcher.run(
        trainer,
        model,
        data_reader,
        opt,
        work_dir=exp_dir,
        nodes=run_args.nodes,
        procs_per_node=run_args.ppn,
        partition ='pdebug',
        proto_file_name="experiment.prototext.trainer"+str(gParameters['run_id']-1),
        job_name=job_name,
        setup_only = True,
        time_limit = 60, 
        lbann_args=f"--generate_multi_proto --procs_per_trainer={run_args.ppt}"
     )

def main():

    args = get_model_parser()
    gParameters = initialize_parameters(args)
    run(gParameters,args) 

if __name__ == '__main__':
    main()
