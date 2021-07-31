import argparse
import lbann
import lbann.models
import lbann.models.resnet
import lbann.contrib.args
import lbann.contrib.models.wide_resnet
import lbann.contrib.launcher
import data.cifar10

# Command-line arguments
desc = ('Construct and run ResNet on Cifar10 data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_resnet', type=str,
    help='scheduler job name (default: lbann_resnet)')
parser.add_argument(
    '--resnet', action='store', default=50, type=int,
    choices=(18, 34, 50, 101, 152),
    help='ResNet variant (default: 50)')
parser.add_argument(
    '--width', action='store', default=2, type=float,
    help='Wide ResNet width factor (default: 2)')
parser.add_argument(
    '--block-type', action='store', default=None, type=str,
    choices=('basic', 'bottleneck'),
    help='ResNet block type')
parser.add_argument(
    '--blocks', action='store', default=None, type=str,
    help='ResNet block counts (comma-separated list)')
parser.add_argument(
    '--block-channels', action='store', default=None, type=str,
    help='Internal channels in each ResNet block (comma-separated list)')
parser.add_argument(
    '--bn-statistics-group-size', action='store', default=1, type=int,
    help=('Group size for aggregating batch normalization statistics '
          '(default: 1)'))
parser.add_argument(
    '--warmup', action='store_true', help='use a linear warmup')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=90, type=int,
    help='number of epochs (default: 90)', metavar='NUM')
parser.add_argument(
    '--num-classes', action='store', default=10, type=int,
    help='number of Cifar10 classes (default: 1000)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()

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

if (any([args.block_type, args.blocks, args.block_channels])
    and not all([args.block_type, args.blocks, args.block_channels])):
    raise RuntimeError('Must specify all of --block-type, --blocks, --block-channels')
if args.block_type and args.blocks and args.block_channels:
    # Build custom ResNet.
    resnet = lbann.models.ResNet(
        block_variant_dict[args.block_type],
        cifar10_labels,
        list(map(int, args.blocks.split(','))),
        list(map(int, args.block_channels.split(','))),
        zero_init_residual=True,
        bn_statistics_group_size=args.bn_statistics_group_size,
        name='custom_resnet',
        width=args.width)
elif args.width == 1:
    # Vanilla ResNet.
    resnet = resnet_variant_dict[args.resnet](
        cifar10_labels,
        bn_statistics_group_size=args.bn_statistics_group_size)
elif args.width == 2 and args.resnet == 50:
    # Use pre-defined WRN-50-2.
    resnet = wide_resnet_variant_dict[args.resnet](
        cifar10_labels,
        bn_statistics_group_size=args.bn_statistics_group_size)
else:
    # Some other Wide ResNet.
    resnet = resnet_variant_dict[args.resnet](
        cifar10_labels,
        bn_statistics_group_size=args.bn_statistics_group_size,
        width=args.width)

# Construct layer graph
input_ = lbann.Input(target_mode='classification')
images = lbann.Identity(input_)
labels = lbann.Identity(input_)


# Argumentation
mag_rot = '0'
rand_rot = lbann.Uniform(min=-1, max=1, neuron_dims='1', training_only=True)
rot_weights = lbann.Weights(initializer=lbann.ValueInitializer(values=mag_rot), name='rot_weights', optimizer=lbann.NoOptimizer())
rot = lbann.WeightsLayer(dims='1', weights=rot_weights, name='rot', device='CPU')
rot = lbann.Multiply(rot, rand_rot, device='CPU')

mag_shear = '0'
rand_shear = lbann.Uniform(min=-1, max=1, neuron_dims='1', training_only=True)
shear_weights = lbann.Weights(initializer=lbann.ValueInitializer(values=mag_shear),name='shear_weights',optimizer=lbann.NoOptimizer())
shear = lbann.WeightsLayer(dims='1', weights=shear_weights, name='shear')
shear = lbann.Multiply(shear, rand_shear, device='CPU')

images = lbann.Rotation(images, rot, shear, device='CPU')

images = lbann.BilinearResize(images, height=32, width=32,name='resize')


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
             lbann.CallbackTimer()]#,
             #lbann.CallbackDropFixedLearningRate(
             #    drop_epoch=[30, 60, 80], amt=0.1)]

callbacks.append(lbann.CallbackPerturbWeights(output_name='rot_weights',batch_interval=400,lower=-45,upper=45,scale=45))
callbacks.append(lbann.CallbackPerturbWeights(output_name='shear_weights',batch_interval=400,lower=-0.2,upper=0.2,scale=0.5))

if args.warmup:
    callbacks.append(
        lbann.CallbackLinearGrowthLearningRate(
            target=0.1 * args.mini_batch_size / 256, num_epochs=5))

model = lbann.Model(args.num_epochs,
                    layers=layers,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks)

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Setup data reader
data_reader = data.cifar10.make_data_reader(num_classes=args.num_classes)

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

lbann_args="--procs_per_trainer=2"
nodes_args=8
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt, nodes=nodes_args, lbann_args = lbann_args,job_name="resnet_LTBF_w40_s",**kwargs)
