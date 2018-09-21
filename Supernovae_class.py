class supernovae:
    import tensorflow as tf
    import numpy as np
    from sklearn.decomposition import PCA
    import itertools as it
    from configobj import ConfigObj
    import os

    def __init__(self, random_seed=1, N_steps=100000):

        # Check if config file exists
        if self.os.path.exists("Config_Svae.ini"):
            print("Configuration file found")
            self.__config__ = self.ConfigObj("Lorenz_model.ini")
            self.network_folder = self.__config__["Network_folder"]
            self.description = self.__config__["Description"]
            self.name_model = self.__config__["Name_model"]
            self.name_datafile = self.__config__["Name_datafile"]
            self.verbosity = self.__config__["Verbosity"]
            self.theta = self.__config__["Theta"]
            self.time_size = self.__config__["Time_size"]

            # Create the structure of the folders
            self._create_folders()
        else:
            self.network_folder = None
            self.description = None
            self.name_model = None
            self.verbosity = None
            self.theta = None
            self.time_size = None
            self.name_datafile = None
            print("No configuration file found, use create_conf_file() to create one")

        # Setting verbosity
        if self.verbosity == 'No information' or self.verbosity == None:
            self.tf.logging.set_verbosity(self.tf.logging.ERROR)
        if self.verbosity == 'Information':
            self.tf.logging.set_verbosity(self.tf.logging.INFO)
        if self.verbosity == 'Debug':
            self.tf.logging.set_verbosity(self.tf.logging.DEBUG)

    def create_conf_file(self, Name_model, Description, Network_folder, Name_datafile, Verbosity, Theta, Time_size):

        # Create the object and assign it values
        config_file = self.ConfigObj()
        config_file.filename = "Config_Svae.ini"
        config_file["Network_folder"] = Network_folder
        config_file["Name_model"] = Name_model
        config_file["Description"] = Description
        config_file["Name_datafile"] = Name_datafile
        config_file["Verbosity"] = Verbosity
        config_file["Theta"] = Theta
        config_file["Time_size"] = Time_size
        # Save the file
        config_file.write()

        # Included in the object
        self.network_folder = config_file["Network_folder"]
        self.description = config_file["Description"]
        self.name_model = config_file["Name_model"]
        self.verbosity = config_file["Verbosity"]
        self.theta = config_file["Theta"]
        self.time_size = config_file["Time_size"]
        self.name_datafile = config_file["Name_datafile"]

        # Setting verbosity
        if self.verbosity == 'No information':
            self.tf.logging.set_verbosity(self.tf.logging.ERROR)
        if self.verbosity == 'Information':
            self.tf.logging.set_verbosity(self.tf.logging.INFO)
        if self.verbosity == 'Debug':
            self.tf.logging.set_verbosity(self.tf.logging.DEBUG)
            
        self._create_folders()
        
                
    def _create_folders(self):

        # Check if the folder structure is created, otherwise we created
        # set the directorys for the means and deviations
        directory_main = self.network_folder
        directory_model = directory_main + '/model/'
        directoy_input = directory_main + '/input_data/'
        directory_output = directory_main + '/output/'
        directory_mean = directory_output + '/mean/'
        directory_dev = directory_output + '/dev/'
        directory_time = directory_output + '/reconstruction/'
        directory_input = directory_output + '/input_data/'
        
        # We check if the filesystem exists
        if self.verbosity == 'information' or self.verbosity == 'Debug':
            print('Cheking if file system exists')

        # if the folders do not exist create them
        if not self.os.path.exists(directory_main):
            self.os.makedirs(directory_main)
            if self.verbosity == 'information' or self.verbosity == 'Debug':
                print(directory_main + 'Directory has been created')

        if not self.os.path.exists(directory_model):
            self.os.makedirs(directory_model)
            if self.verbosity == 'information' or self.verbosity == 'Debug':
                print(directory_model + 'Directory has been created')

        if not self.os.path.exists(directoy_input):
            self.os.makedirs(directoy_input)
            if self.verbosity == 'information' or self.verbosity == 'Debug':
                print(directoy_input + 'Directory has been created')

        if not self.os.path.exists(directory_output):
            self.os.makedirs(directory_output)
            if self.verbosity == 'information' or self.verbosity == 'Debug':
                print(directory_output + 'Directory has been created')

        if not self.os.path.exists(directory_mean):
            self.os.makedirs(directory_mean)
            if self.verbosity == 'information' or self.verbosity == 'Debug':
                print(directory_mean + 'Directory has been created')

        if not self.os.path.exists(directory_dev):
            self.os.makedirs(directory_dev)
            if self.verbosity == 'information' or self.verbosity == 'Debug':
                print(directory_dev + 'Directory has been created')

        if not self.os.path.exists(directory_time):
            self.os.makedirs(directory_time)
            if self.verbosity == 'information' or self.verbosity == 'Debug':
                print(directory_time + 'Directory has been created')

        if not self.os.path.exists(directory_input):
            self.os.makedirs(directory_input)
            if self.verbosity == 'information' or self.verbosity == 'Debug':
                print(directory_input + 'Directory has been created')

    def train_svae(self, N_steps=10000, random_seed=1):  # ENTRENAMENT DE VERITAT
        import SupernoVAE_functions as svae

        MODEL_DIR = self.network_folder + '/model/'
        l_rate = 0.0006282
        N_steps_rise = N_steps

        # define an estimator using the model named above and storing checkpoints to the
        # directory difined by MODEL_DIR
        for l in range(1):
            l_rate = l_rate * 1
            supernovae = self.tf.estimator.Estimator(model_fn=svae.make_model_fn(l_rate, random_seed), model_dir=MODEL_DIR)

            # train the classifier with the training function for the specified number of steps. This will be
            # skipped automatically if the model was trained at leas N_steps in an earlier run of the programm
            supernovae.train(
                input_fn=svae.train_input_fn,
                max_steps=N_steps)
            N_steps += N_steps_rise

    def predict(self, random_seed=1, save_output = True):
        
        import SupernoVAE_functions as svae
        MODEL_DIR = self.network_folder + '/model/'
        l_rate = 0.0006282

        # evaluate the model on all examples and output the mean square error and the mean absolute error
        # to the std out. Here not only the time series for the center point is taken into account but
        # all 81 locations are compared
        supernovae = self.tf.estimator.Estimator(model_fn=svae.make_model_fn(l_rate, random_seed), model_dir=MODEL_DIR)

        eval_out = supernovae.evaluate(input_fn=svae.test_input_fn)
        print(eval_out)

        # Set up logging hooks to log the progress during predictions
        logging_hook = self.tf.train.StepCounterHook(every_n_steps=1)

        # report the time series, the position and the encoding for every example
        output = supernovae.predict(input_fn=svae.test_input_fn, hooks=[logging_hook])
        
        # Save the output on the object
        self.output = output
        
        # Save save the encodings of the output
        if save_output:
            svae.save_encodings(output)
            
    def load_predictions(self, max_lat = 180, max_lon = 360):
        
        folder = self.network_folder + "/output/mean/"
        size = self.time_size 
        dtype=self.np.float32,
        name = self.name_datafile

        names = [] # Store the names and find names in the directory
        p = 0
        
        # Find the file in the folder
        for i, j in self.it.product(range(1, max_lat +1), range(1, max_lon +1)):
            if self.verbose == 'Information' or self.verbose == 'Debug':
                if self.np.random.uniform() > 0.9995:
                    print(str(p+1) + "/" + str(max_lat*max_lon) + ":" + 
                          str(round(p*100/(max_lat*max_lon), 2)) + "%", end = "\r")
            p += 1
            file_name = folder + '/' + name + str(i).zfill(3) + '_' + str(j).zfill(3) + '.bin'


            # check if the file exists, store its name into the name list
            if self.os.path.exists(file_name):
                names.append(file_name)

        # Stores the embeddings without masking
        self.embeddings = []

        #read the data from the files
        for n in names:
            self.embeddings.append(self.np.fromfile(n, dtype="float32"))
        
        #check that the first element has the correct length
        assert(ret_values[0].shape[0] == size)

        #return the list of arrays
        self.embeddings = self.np.array(self.embeddings)
        
        if self.verbose == 'Information' or self.verbose == 'Debug':
            print('Predictions loaded ...')
            
    def save_data_tf(self, dataset, max_lon = 180, max_lat = 360):
        dataset = dataset.astype('float64')

        folder = self.network_folder + '/input_data/'
        Name = self.name_datafile + '_'
        name = None

        # Save it
        for lat in range(1,max_lon):
            for lon in range(1,max_lat):
                i = lat - 1
                j = lon - 1
                sample = dataset[i, j,:]
                name = str(folder)+'/' +str(Name) + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin'
                sample.tofile(name)
                if self.verbosity == 'Debug':
                    print(name)
                    pass
                pass
            pass
        print('The dataset has been saved in TF format')
        
    def PCA_embeddings(self):
        self.principal_components = self.PCA()
        self.pca.fit(self.np.asarray(self.principal_components).transpose())