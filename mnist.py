import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
##LBANN stuff
import lbann
import data.mnist
import lbann.contrib.args
import lbann.contrib.launcher

try:
    import configparser
except ImportError:
    import ConfigParser as configparser



file_path = os.path.dirname(os.path.realpath(__file__))

def common_parser(parser):

    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(file_path, 'mnist_default_model.txt'),
                        help="specify model configuration file")
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--ppn", type=int, default=4)
    parser.add_argument("--ppt", type=int, default=2)

    return parser

def get_model_parser():

	parser = argparse.ArgumentParser(prog='mnist_baseline', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='MNIST LBANN ')

	return common_parser(parser).parse_args()

def read_config_file(file):
    #print("Reading default config (param) file : ", file)
    config=configparser.ConfigParser()
    config.read(file)
    section=config.sections()
    fileParams={}

    fileParams['model_name']=eval(config.get(section[0],'model_name'))
    fileParams['conv']=eval(config.get(section[0],'conv'))
    fileParams['dense']=eval(config.get(section[0],'dense'))
    fileParams['activation']=eval(config.get(section[0],'activation'))
    fileParams['pool_mode']=eval(config.get(section[0],'pool_mode'))
    #fileParams['optimizer']=eval(config.get(section[0],'optimizer'))
    fileParams['epochs']=eval(config.get(section[0],'epochs'))
    fileParams['batch_size']=eval(config.get(section[0],'batch_size'))
    fileParams['classes']=eval(config.get(section[0],'classes'))  
    fileParams['save']=eval(config.get(section[0], 'save'))
    fileParams['lr']=eval(config.get(section[0], 'lr'))
    fileParams['run_id']=eval(config.get(section[0], 'run_id')) 
    fileParams['weights']=eval(config.get(section[0], 'weights'))


    return fileParams

def initialize_parameters(args):
    # Get command-line parameters
    #args = get_model_parser()
    #args = parser.parse_args()
    # Get parameters from configuration file
    gParameters = read_config_file(args.config_file)
    return gParameters

def get_activation(name, x):
      if name  == 'relu':
          return lbann.Relu(x)
      elif name == 'tanh' :
           return lbann.Tanh(x)
      elif name == 'elu' :
           return lbann.Elu(x)
      elif name == 'selu' :
           return lbann.Selu(x)
      elif name == 'leaky_relu' :
           return lbann.LeakyRelu(x)
      elif name == 'softplus' :
           return lbann.Softplus(x)
    

def run(gParameters,run_args,exp_dir=None):
    
    #convs: out_c, conv_dim, conv_stride
    conv_outc= []
    conv_dim = []
    conv_stride = []
    conv_params = list(range(0, len(gParameters['conv']), 3))
    for l, i in enumerate(conv_params):
        conv_outc.append(gParameters['conv'][i])
        conv_dim.append(gParameters['conv'][i+1])
        conv_stride.append(gParameters['conv'][i+2])
    
    # Input data
    input_ = lbann.Input(target_mode='classification')
    images = lbann.Identity(input_)
    labels = lbann.Identity(input_)


    # Data arguments
    mag =  gParameters['weights']
    crop_mag = mag
    resize_mag = np.floor(15 + mag*28).astype(int)

    #pos_weights = lbann.Weights(initializer=lbann.ValueInitializer(values='0.5 0.5 0.5'),name='crop_weights',optimizer=lbann.NoOptimizer())
    #pos = lbann.WeightsLayer(dims='3', weights=pos_weights, name='pos')
    #images = lbann.Crop(images,pos, dims='3 20 20')
    
    rot_weights = lbann.Weights(initializer=lbann.ValueInitializer(values='90'),name='rot_weights',optimizer=lbann.NoOptimizer())
    rot = lbann.WeightsLayer(dims='1', weights=rot_weights, name='pos')
    images = lbann.Rotation(images,rot,device='CPU')

 
    images = lbann.BilinearResize(images, height=28, width=28, name='resize') 
		
    # LeNet
    x = lbann.Convolution(images,
                      num_dims = 2,
                      num_output_channels = conv_outc[0],
                      num_groups = 1,
                      conv_dims_i = conv_dim[0],
                      conv_strides_i = conv_stride[0],
                      conv_dilations_i = 1,
                      has_bias = True)
    x = get_activation(gParameters['activation'],x)
    x = lbann.Pooling(x,
                  num_dims = 2,
                  pool_dims_i = 2,
                  pool_strides_i = 2,
                  pool_mode = str(gParameters['pool_mode']))
    x = lbann.Convolution(x,
                      num_dims = 2,
                      num_output_channels = conv_outc[1],
                      num_groups = 1,
                      conv_dims_i = conv_dim[1],
                      conv_strides_i = conv_stride[1],
                      conv_dilations_i = 1,
                      has_bias = True)
    x = get_activation(gParameters['activation'],x)
    x = lbann.Pooling(x,
                  num_dims = 2,
                  pool_dims_i = 2,
                  pool_strides_i = 2,
                  pool_mode = str(gParameters['pool_mode']))
    x = lbann.FullyConnected(x, num_neurons = gParameters['dense'][0], has_bias = True)
    x = get_activation(gParameters['activation'],x)
    x = lbann.FullyConnected(x, num_neurons = gParameters['dense'][1], has_bias = True)
    x = get_activation(gParameters['activation'],x)
    x = lbann.FullyConnected(x, num_neurons = gParameters['classes'], has_bias = True)
    probs = lbann.Softmax(x)

    # Loss function and accuracy
    loss = lbann.CrossEntropy(probs, labels)
    acc = lbann.CategoricalAccuracy(probs, labels)
    
    lr = gParameters['lr']
    opt = lbann.SGD(learn_rate=lr, momentum=0.9)
    ##Uncomment to support optimizer exchange
    '''
    if gParameters['optimizer'] == 'adam':
        opt = lbann.Adam(learn_rate=lr, beta1=0.9, beta2=0.99, eps=1e-8)
    elif gParameters['optimizer'] == 'adagrad':
        opt = lbann.AdaGrad(learn_rate=lr, eps=1e-8)
    '''
    #sendrecv_weights,
    callbacks = [lbann.CallbackPrint(),lbann.CallbackTimer()]#,lbann.CallbackPrintModelDescription()]
    #callbacks.append(lbann.CallbackPerturbLearningRate(learning_rate_factor = 1.0,
    #                                                   perturb_during_training = True,
    #                                                   batch_interval = 400))
    
    #callbacks.append(lbann.CallbackPerturbWeights(output_name='rot_weights',batch_interval=400))
                                
	
    layers = list(lbann.traverse_layer_graph(input_))
    #layers.append(w_layer)

    metrics = [lbann.Metric(acc, name='accuracy', unit='%')]
    #metrics.append(lbann.Metric(pos_weights, name='weights'))

    model = lbann.Model(10,
                    layers=layers,
                    objective_function=loss,
                    metrics=metrics,
                    callbacks=callbacks)

    # Setup data reader
    data_reader = data.mnist.make_data_reader()

    # Setup trainer
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

    #trainer = lbann.Trainer(mini_batch_size=128,
    #                    training_algo=ltfb)
# Setup trainer
    trainer = lbann.Trainer(name=job_name, mini_batch_size=128)
    status = lbann.contrib.launcher.run(
        trainer,
        model,
        data_reader,
        opt,
        #work_dir=gParameters['save'],
        work_dir=exp_dir,
        nodes=run_args.nodes,
        procs_per_node=run_args.ppn,
        partition ='pdebug',
        #proto_file_name=job_name+"exp.prototext",
        proto_file_name="experiment.prototext.trainer"+str(gParameters['run_id']-1),
        job_name=job_name,
        setup_only = True,
        time_limit = 60, 
        #batch_job = True,
        lbann_args=f"--generate_multi_proto --procs_per_trainer={run_args.ppt}"
        #lbann_args=['--generate_multi_proto']
     )

def main():

    args = get_model_parser()
    gParameters = initialize_parameters(args)
    run(gParameters,args) #new

if __name__ == '__main__':
    main()
