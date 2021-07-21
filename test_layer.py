import numpy as np
import lbann
import lbann.modules
from lbann.util import str_list


# Test provided layer for user-provided image
if __name__ == "__main__":

    # Imports
    import argparse
    import matplotlib.image

    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image', default='Lenna.png',action='store', type=str,
        help='image file', metavar='FILE',
    )
    args = parser.parse_args()

    # Load image
    image = matplotlib.image.imread(args.image)
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    assert image.ndim == 3, f'failed to load 2D image from {args.image}'
    if image.shape[-1] == 1:
        image = np.tile(image, (1,1,3))
    elif image.shape[-1] == 4:
        image = image[:,:,:3]
    assert image.shape[-1] == 3, f'failed to load RGB image from {args.image}'
    image = np.transpose(image, (2,0,1))

    # Dummy input
    reader = lbann.reader_pb2.DataReader()
    def add_data_reader(role):
        _reader = reader.reader.add()
        _reader.name = 'synthetic'
        _reader.role = role
        _reader.num_samples = 1
        _reader.num_labels = 1
        _reader.synth_dimensions = '1'
        _reader.percent_of_data_to_use = 1.0
    add_data_reader('train')
    add_data_reader('test')


    # Image
    x = lbann.WeightsLayer(
        weights=lbann.Weights(
            lbann.ValueInitializer(values=str_list(image.flatten())),
        ),
        dims=str_list(image.shape),
    )

    input_ = lbann.Input()

    # Rotation
    rot_weights = lbann.Weights(initializer=lbann.ValueInitializer(values='25'),name='rot_weights',optimizer=lbann.NoOptimizer())
    rot = lbann.WeightsLayer(dims='1', weights=rot_weights, name='pos')
    images = lbann.Rotation(x,rot,device='CPU')

    rotated = lbann.Identity(images, name='rotated')

    # Construct model
    callbacks = [
        lbann.CallbackSaveImages(layers='rotated',image_format='jpg'),
    ]
    model = lbann.Model(
        epochs=0,
        layers=lbann.traverse_layer_graph([input_,x]),
        callbacks=callbacks,
    )

    # Run LBANN
    lbann.run(
        trainer=lbann.Trainer(mini_batch_size=1),
        model=model,
        data_reader=reader,
        optimizer=lbann.NoOptimizer(),
        job_name='lbann_rotate_test',
    )
