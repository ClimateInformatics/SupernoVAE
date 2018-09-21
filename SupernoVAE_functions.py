import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_normal_initializer
import numpy as np
import os

# Load settings
from configobj import ConfigObj

# set the name of the world and the folder the ouput should be stored to
info        = ConfigObj("Config_Svae.ini")
verbosity   = info["Verbosity"]

# Setting verbosity of tensorflow
if verbosity == 'No information':
    tf.logging.set_verbosity(tf.logging.ERROR)
if verbosity == 'Information':
    tf.logging.set_verbosity(tf.logging.INFO)
if verbosity == 'Debug':
    tf.logging.set_verbosity(tf.logging.DEBUG)

# General parameters of the network
theta           = float(info["Theta"])
max_lat         = int(info["Max_lat"])
max_lon         = int(info["Max_lon"])
time_size       = int(info["Time_size"])
encoding_size   = int(np.round(time_size * theta))

def make_model_fn(l_rate, random_seed):
    """
    This function creates and returns a model function using two arguments.

    Keyarguments
    l_rate      --  The learning rate.
    random_seed --  The random seed that should be used for all random number generators.
    """

    def ave_model_fn(features, labels, mode):
        """ 
        The variational autoencoder function
            
        The input should be the following
                features    --  a dict containing two keys 'x' and 'y'. The 'x' value should contain the time series
                                of the area around the point of interrest. The format should be 9*9*696. However, the
                                correct format is not enforced.
                                'y' should contain the position of the example. We might want to change it into a 'lat'
                                and a 'lon' parameter.
                labels      --  this is meaningless and should be set to a constant zero. Labels are not needed for
                                a variational autoencoder
                mode        --  This contains the mode the network is used in and should be one of the following values
                                tf.estimator.ModeKeys.TRAIN     - if the network is trained at the moment
                                tf.estimator.ModeKeys.PREDICT   - if the network is used to get information on the examples
                                tf.estimator.ModeKeys.EVAL      - if the network is used to get general information about
                                                                    the performance of the network like a mean error on
                                                                    all data samples
                                if the value equals non of the above, the network will use the EVAL-Key
        """

        # the input layer reshapes the input into the disired form
        input_layer = tf.reshape(features['x'], [-1, time_size, 1, 1, 1]) # HAS TO BE CHANGED
        input_layer = tf.transpose(input_layer, [0, 2, 3, 1, 4])

        input_slice_center  = input_layer

        # All of the tensors used in the encoding part will start with the word 'encoding/'
        with tf.variable_scope('encoding'):
            
            # WE HAVE TO INCLUDE THE AFECTED LAYERS DEPENDING ONT THEIR SIZE. LET'S START WITH 696
            
            if time_size == 696:

                # We use a batch norm at the beginnin. This sets the mean of the inputdata in every step to 0 and the
                # biggest variance to one. The idea is to approximate a normal distribution. We should be careful here
                # if the batchsize is to low
                batch_norm_1 = tf.layers.batch_normalization(inputs=input_slice_center,
                                                             training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                             name='batchnorm1')

                # The first convolution layer. We use strided convolution as it leads to better reproducebility
                # compared to max pooling. << acording to 'Unsupervised Representation learning with deep convolutional
                # genartive adversarial networks' by Radford et al.
                # We also followed there advise to put batch normalization in between every layer and use ReLU
                # activations
                conv_1 = tf.layers.conv3d(inputs=batch_norm_1,
                                          filters=128,
                                          kernel_size=(1, 1, 4),
                                          strides=(1, 1, 2),
                                          padding='valid',
                                          activation=tf.nn.relu,
                                          kernel_initializer=glorot_normal_initializer(),
                                          name='conv1')
                pass # End of the if 696
            
            if time_size == 1392:
                
                batch_norm_0    = tf.layers.batch_normalization(inputs      = input_slice_center,
                                                            training    = (mode == tf.estimator.ModeKeys.TRAIN),
                                                            name        = 'batchnorm0')
            
                #The first convolution layer. We use strided convolution as it leads to better reproducebility 
                #compared to max pooling. << acording to 'Unsupervised Representation learning with deep convolutional
                #genartive adversarial networks' by Radford et al.
                #We also followed there advise to put batch normalization in between every layer and use ReLU 
                #activations
                conv_0          = tf.layers.conv3d( inputs                  = batch_norm_0,
                                                    filters                 = 128,
                                                    kernel_size             = (1,1,4),
                                                    strides                 = (1,1,2),
                                                    padding                 = 'valid',
                                                    activation              = tf.nn.relu,
                                                    kernel_initializer      = glorot_normal_initializer(),
                                                    name                    = 'conv0')

                batch_norm_1    = tf.layers.batch_normalization(inputs      = conv_0,
                                                                training    = (mode == tf.estimator.ModeKeys.TRAIN),
                                                                name        = 'batchnorm1')

                conv_1          = tf.layers.conv3d( inputs                  = batch_norm_1,
                                                    filters                 = 128,
                                                    kernel_size             = (1,1,4),
                                                    strides                 = (1,1,2),
                                                    padding                 = 'valid',
                                                    activation              = tf.nn.relu,
                                                    kernel_initializer      = glorot_normal_initializer(),
                                                    name                    = 'conv1')
                
                pass # End of the if 1392 

            # batch normalization makes the activation almost standart normaly distributed in every layer
            batch_norm_2 = tf.layers.batch_normalization(inputs=conv_1,
                                                         training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                         name='batchnorm2')

            # a second convolutional layer
            conv_2 = tf.layers.conv3d(inputs=batch_norm_2,
                                      filters=64 * 3,
                                      kernel_size=(1, 1, 4),
                                      strides=(1, 1, 2),
                                      padding='valid',
                                      activation=tf.nn.relu,
                                      kernel_initializer=glorot_normal_initializer(),
                                      name='conv2')
            # again a batch normalization layer
            batch_norm_3 = tf.layers.batch_normalization(inputs=conv_2,
                                                         training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                         name='batchnorm3')
            # third convolutional layer
            conv_3 = tf.layers.conv3d(inputs=batch_norm_3,
                                      filters=64 * 4,
                                      kernel_size=(1, 1, 4),
                                      strides=(1, 1, 2),
                                      padding='valid',
                                      activation=tf.nn.relu,
                                      kernel_initializer=glorot_normal_initializer(),
                                      name='conv3')

            # batch normalization in between every layer
            batch_norm_4 = tf.layers.batch_normalization(inputs=conv_3,
                                                         training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                         name='batchnorm4')

            # fourth convolutional layer
            conv_4 = tf.layers.conv3d(inputs=batch_norm_4,
                                      filters=64 * 4,
                                      kernel_size=(1, 1, 4),
                                      strides=(1, 1, 2),
                                      padding='valid',
                                      activation=tf.nn.relu,
                                      kernel_initializer=glorot_normal_initializer(),
                                      name='conv4')

            # batch normalization in between every layer
            batch_norm_5 = tf.layers.batch_normalization(inputs=conv_4,
                                                         training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                         name='batchnorm5')

            # 5. convolutional layer
            conv_5 = tf.layers.conv3d(inputs=batch_norm_5,
                                      filters=64 * 4,
                                      kernel_size=(1, 1, 4),
                                      strides=(1, 1, 2),
                                      padding='valid',
                                      activation=tf.nn.relu,
                                      kernel_initializer=glorot_normal_initializer(),
                                      name='conv5')

            # batch normalization in between every layer
            batch_norm_6 = tf.layers.batch_normalization(inputs=conv_5,
                                                         training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                         name='batchnorm6')

            # 6. convolutional layer
            conv_6 = tf.layers.conv3d(inputs=batch_norm_6,
                                      filters=64 * 6,
                                      kernel_size=(1, 1, 4),
                                      strides=(1, 1, 2),
                                      padding='valid',
                                      activation=tf.nn.relu,
                                      kernel_initializer=glorot_normal_initializer(),
                                      name='conv6')

            # batch normalization in between every layer
            batch_norm_7 = tf.layers.batch_normalization(inputs=conv_6,
                                                         training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                         name='batchnorm7')

            # the tensor is flattened with reshape instead of flatten to allow for older versions of tf. The drawback
            # is that we need to put the size of the vector in manually
            flatten = tf.contrib.layers.flatten(batch_norm_7)

            # against the advise of Radford et al.,we use a fully connected layer here. We calculate the means of the
            # latent distributions. Since the mean of such a distribution could as well be negative we use no ReLU activation
            # but instead a linear activation
            means = tf.layers.dense(inputs=flatten,
                                    units=encoding_size,
                                    activation=None,
                                    kernel_initializer=glorot_normal_initializer(),
                                    name='fullyconnected_means')

            # for the standart diviations we again use a fully connected layer. Since these should not be negative, we use
            # ReLU activations here.
            deviations = tf.layers.dense(inputs=flatten,
                                         units=encoding_size,
                                         activation=tf.nn.relu,
                                         kernel_initializer=glorot_normal_initializer(),
                                         name='fullyconnected_dev')

        # This completes the encoding part since the latent distributions are assumed to be normal and therfore are completely
        # defined by the mean and diviation
        # Here we begin the middle part. All tensors from the middle part, where we draw the encoding from the latent distributions
        # start with 'vae/'
        with tf.variable_scope('vae'):

            # Draw numbers from a standart normal distribution. The numbers have the same shape as means and as seed we use the
            # seed set above. If no seed is set, the time will be used but the experiment will not be reproducible.
            random_numbers = tf.random_normal(shape=tf.shape(means),
                                              mean=0.0,
                                              stddev=1.0,
                                              seed=random_seed,
                                              name='random_number_generator')

            # Since drawing from a normal distribution with standart diviation σ is the same as drawing from a standart normal
            # distribution and multypling by σ, we multiply the drawn values by the deviations
            scaled_random = tf.multiply(x=random_numbers,
                                        y=deviations,
                                        name='adjust_variance')

            # Since drawing from a normal distribution with mean μ is the same as drawing from a normal distribution with mean
            # 0 and adding μ we add the means
            encoding = tf.add(x=scaled_random,
                              y=means,
                              name='adjust_means')

            # now the encodings contain values wich are normally distributed with means 'means' and deviations 'deviations' but we
            # can still use backpropagation since we do not need to propagate back through the random number generator

        # Here begins the decoding part of the network. All tensors here will begin with 'decoding/'
        with tf.variable_scope('decoding'):

            # We start withe a fully connected layer. This gives the network a chance to rearange the features and makes the architecture
            # of the decoder somehow independent of the compression factor ϑ, since the dimensionaliety will be the same after this step
            fc_decoding = tf.layers.dense(inputs=encoding,
                                          units=20 * 64 * 4,
                                          activation=tf.nn.relu,
                                          kernel_initializer = glorot_normal_initializer(),
                                          name='fullyconnected')

            # We reverse the flattening of the tensors and regain a shape suitable for a inverse convolution
            input_layer_decoding = tf.reshape(fc_decoding, [-1, 1, 1, 20, 64 * 4])

            # Since the centering is a big part of the encoding we do not use a batch normalization at this position in the network

            # for reconstruction we use transposed convolutional layers. These produce an inverse of a convolutional layer.
            deconv_1 = tf.layers.conv3d_transpose(
                inputs=input_layer_decoding,
                filters=64 * 6,
                kernel_size=(1, 1, 4),
                strides=(1, 1, 2),
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=glorot_normal_initializer(),
                use_bias=True,
                name='deconv1')

            # From here on again batch normalization in between every layer
            deconv_1_bn = tf.layers.batch_normalization(
                inputs=deconv_1,
                training=(mode == tf.estimator.ModeKeys.TRAIN),
                name='batchnorm2')

            # Since we incoded using three convolutions, we decode using 3 transposed convolutions
            deconv_2 = tf.layers.conv3d_transpose(
                inputs=deconv_1_bn,
                filters=64 * 4,
                kernel_size=(1, 1, 4),
                strides=(1, 1, 2),
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=glorot_normal_initializer(),
                use_bias=True,
                name='deconv2')

            # Batch normalization in between every layers
            deconv_2_bn = tf.layers.batch_normalization(
                inputs=deconv_2,
                training=(mode == tf.estimator.ModeKeys.TRAIN),
                name='batchnorm3')

            # the third deconvolution should leed to the exact same size as the input
            deconv_3 = tf.layers.conv3d_transpose(
                inputs=deconv_2_bn,
                filters=64 * 3,
                kernel_size=(1, 1, 4),
                strides=(1, 1, 2),
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=glorot_normal_initializer(),
                use_bias=True,
                name='deconv3')

            # Batch normalization in between every layers
            deconv_3_bn = tf.layers.batch_normalization(
                inputs=deconv_3,
                training=(mode == tf.estimator.ModeKeys.TRAIN),
                name='batchnorm4')

            # the 4.  deconvolution should leed to the exact same size as the input
            deconv_4 = tf.layers.conv3d_transpose(
                inputs=deconv_3_bn,
                filters=128,
                kernel_size=(1, 1, 4),
                strides=(1, 1, 2),
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=glorot_normal_initializer(),
                use_bias=True,
                name='deconv4')

            # Batch normalization in between every layers
            deconv_4_bn = tf.layers.batch_normalization(
                inputs=deconv_4,
                training=(mode == tf.estimator.ModeKeys.TRAIN),
                name='batchnorm5')

            # the 5.  deconvolution should leed to the exact same size as the input
            deconv_5 = tf.layers.conv3d_transpose(
                inputs=deconv_4_bn,
                filters=1,
                kernel_size=(1, 1, 4),
                strides=(1, 1, 2),
                padding='valid',
                activation=None,
                kernel_initializer=glorot_normal_initializer(),
                use_bias=True,
                name='deconv5')
            
            # ADD EXTRA CONVOLUTIONS FOR 1392 SIZE
            if time_size == 1392:
                
                #Batch normalization in between every layers
                deconv_5_bn             = tf.layers.batch_normalization(
                                                inputs=deconv_5,
                                                training=(mode==tf.estimator.ModeKeys.TRAIN),
                                                name='batchnorm6')

                #the 6. deconvolution should leed to the exact same size as the input
                deconv_6                = tf.layers.conv3d_transpose(
                                                inputs=deconv_5_bn,
                                                filters=1,
                                                kernel_size=(1,1,4),
                                                strides=(1,1,2),
                                                padding='valid',
                                                activation=None,
                                                kernel_initializer=glorot_normal_initializer(),
                                                use_bias = True,
                                                name='deconv6')
                
                deconv_slice = tf.slice(deconv_6, [0,0,0,0,0], [-1,1,1,1392,1])
                
                pass # END OF THE 1392 IF Start the other if
            
            if time_size == 696: 

                deconv_slice = tf.slice(deconv_5, [0, 0, 0, 0, 0], [-1, 1, 1, 696, 1])

        # The network can report the following quantities:
        #   The reconstructed time series, mostly for comparison to the original
        #   The position given as the 'y' component. This should maybe be substituted for 'lat' and 'lon'
        #   The latent distributions
        predictions = {
            'timeseries': deconv_slice,
            'position': features['y'],
            'encoding_mean': means,
            'encoding_dev': deviations,
            'input': input_slice_center
        }

        # If the mode was to predict, the above quanteties are reportet back
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # If the mode was not set to predict, the loss gets calculatet.
        # The epsilon is a parameter to avoid numerical problems if the diviation of one variable goes to zero.
        epsilon = 0.0000001

        # The reconstruction error is the mean squared error of the original time series of the middle point and the
        # reconstructed time series of the middle point.

        reconstrucktion_error = tf.losses.mean_squared_error(labels=input_slice_center, predictions=deconv_slice)
        
        # The latent loss is the KL-Divergence between the latent distributions and a multivariat normal distribution
        latent_loss = tf.reduce_mean(
            0.5 * tf.reduce_sum(tf.square(means) + tf.square(deviations) - tf.log(tf.square(deviations) + epsilon) - 1, 1))

        # as loss we use the sum of both losses above weighted by a factor of λ. if λ = 1 both losses are weighted equally, if λ = 0 only the reconstruction loss
        # is taken into account and if λ is big the latent loss is much more important than the reconstruction loss
        lamb = 0.0000001
        loss = reconstrucktion_error + lamb * latent_loss

        # if the mode is set to train, the network now uses a minimizer to minimize the loss
        if mode == tf.estimator.ModeKeys.TRAIN:
            # As an optimizer we use Adam.
            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)

            # Since the training error does not depend on the sliding avarage used in the batch normalizations,
            # they are not automatically updatet. Therefore, they have to be updatet by hand
            batch_norm_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # update the weights and the batch norm parameters
            with tf.control_dependencies(batch_norm_update):
                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # if the mode is neighter predict nor train, we assume it to be eval

        # The evaluation metics are calculated. Here we calculate the mean squarred error and the
        # mean absolute error on the whole region, not just on the center
        eval_metric_ops = {
            'squared_error': tf.metrics.mean_squared_error(
                labels=input_slice_center,
                predictions=deconv_slice),
            'absolute_error': tf.metrics.mean_absolute_error(
                labels=input_slice_center,
                predictions=deconv_slice)
        }

        # the metrics are reported back
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return ave_model_fn


def save_encodings(generator):
    """ 
    A Function to save the latent distributions used by the variational auto encoder

    Inputs:
            generator -- a generator producing dictionarys containing at leas 2 key value pairs
                            'encoding_mean' --  means of the latent distributions,
                            'encoding_dev'  --  deviations of the latent distributions
    """
    # import os
    # from configobj import ConfigObj
    # set the name of the world and the folder the ouput should be stored to
    info = ConfigObj("Config_Svae.ini")
    folder = './' + info["Network_folder"]
    verbosity = info["Verbosity"]
    name = info["Name_datafile"]
    
    # set the directorys for the means and deviations
    directory_mean = folder + '/output/mean/'
    directory_dev = folder + '/output/dev/'
    directory_time = folder + '/output/out_time/'
    directory_input = folder + '/output/in_time/'

    # if the folders do not exist create them
    if not os.path.exists(directory_mean):
        os.makedirs(directory_mean)

    if not os.path.exists(directory_dev):
        os.makedirs(directory_dev)

    if not os.path.exists(directory_time):
        os.makedirs(directory_time)
    if not os.path.exists(directory_input):
        os.makedirs(directory_input)

    # create a list out of the generator
    predictions = list(generator)

    # make sure the list has the expected number of elements

    # create a counter to go through the list
    i = 0

    # go through the lat and lon values in the order they are in the dataset
    for lat in range(0, max_lat):
        for lon in range(0, max_lon):
            # create the name for the next files
            name_mean = directory_mean + name + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin'
            name_dev = directory_dev + name + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin'
            name_time = directory_time + name + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin'
            name_input = directory_input + name + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin'

            # store the data
            predictions[i]['encoding_mean'].tofile(name_mean)
            predictions[i]['encoding_dev'].tofile(name_dev)
            predictions[i]['timeseries'].tofile(name_time)
            predictions[i]['input'].tofile(name_input)

            # TO FIX
            print("Iteration saved: " +str(i), end='\r')
            
            if verbosity == 'Debug':
                print("sample saved " + name_mean)
            
            # increse the counter
            i += 1
 
    print('Total predictions computed: ' + str(len(predictions)))


def parser(serialized_example):
    """ parses a single tf.example into inmaeg and lable tensor """

    # Decode the string read from a file into an tensor of float32
    example = tf.decode_raw(serialized_example, tf.float32)

    # create a meaningless location and label. Here we need to produce the
    # correct label but it can not be recovert since it is not in the file
    y = tf.constant([[3, 3]])
    labels = tf.constant([0])

    # return the correct input format for the vae model
    return example, y, labels


def train_input_fn():
    """function to read in files and create an infinite stream of inputs for the vae model"""

    batch_size = 8
    
    # set the name and the folder of the files the dataset should be created out of
    info = ConfigObj("Config_Svae.ini")
    name = info["Name_datafile"]
    folder = './' + info["Network_folder"] + '/input_data'
    verbosity = info["Verbosity"]

    # create an empty list to hold the names of the files the dataset should be created out of
    names = []

    # find the files coressponding to the naming sceme folder/name_positon.bin
    for lat in range(0, max_lat):
        for lon in range(0, max_lon):
            directory = folder + '/' + name + '_' + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin'

            # if the file with this naming sceme exists
            if os.path.exists(directory):
                
                # print a sucsess message
                if verbosity == 'Debug':
                    print("sample found! " + directory)
                
                # add the file to the list of files to bild the dataset out of
                names.append(folder + '/' + name +'_' + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin')

    # set the size of data in bytes that should be converted into one example
    record_bytes = time_size * 1 * 1 * 4  # 4 for float32
    
    # create a dataset by loading blocks of size 'record_size' from the files in the list 'names'
    dataset = tf.data.FixedLengthRecordDataset(names, record_bytes)

    # parse every example by useing the parser function. This takes the string read from the file and create a
    # x, y and a label
    dataset = dataset.map(parser)

    # shuffle the first n examples of the dataset. here we set n to be the length of the list containing the
    # input files. This should lead to all the examples beeing shuffled. The dataset gets than repeated infinitly often
    # and split into batches of length 'batch_size'
    dataset = dataset.shuffle(len(names)).repeat().batch(batch_size)

    # create an iterator that iterates through every example onece
    iterator = dataset.make_one_shot_iterator()

    # this creates three tensors wich, if called upon, return the next example from the dataset
    x, y, label = iterator.get_next()

    # adjust the form of the input and return the example in the disired form
    return {'x': x, 'y': y}, label


def test_input_fn():
    """methode to iterate through the files exactly once"""
    
    # set the name and the folder of the files the dataset should be created out of
    info = ConfigObj("Config_Svae.ini")
    name = info["Name_datafile"]
    folder = './' + info["Network_folder"] + '/input_data'
    verbosity = info["Verbosity"]

    # prepare an empty list to hold the file names
    names = []

    # go through the names of the files that should exist
    for lat in range(0, max_lat):
        for lon in range(0, max_lon):
            directory = folder + '/' + name + '_' + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin'

            # if the file exists report sucsess to the stdout and add the name to the list of files
            if os.path.exists(directory):
                if verbosity == 'Debug':
                    print("sample found! " + directory)
                names.append(folder + '/' + name + '_' + str(lat).zfill(3) + '_'+ str(lon).zfill(3) + '.bin')
            else:
                print('SAMPLE NOT FOUND::::! ' + str(directory))

    # define the lenght every example shoud have in byts
    record_bytes = time_size * 1 * 1 * 4  # 4 for float32

    # read the length as discribed above from the files in the list and create a dataset out of them
    dataset = tf.data.FixedLengthRecordDataset(names, record_bytes)

    # call parser for every example to convert the string into an example a position and a label
    dataset = dataset.map(parser)

    # create an iterator presenting every triplet of example position and label once
    iterator = dataset.make_one_shot_iterator()

    # get the next triple from the dataset
    x, y, label = iterator.get_next()

    # return the triple in the disired form
    return {'x': x, 'y': y}, label
