from configobj import ConfigObj

class supernovae:
    """
    The central class of the methode. This class is mainly an enviroment that holds the hyperparameters of the task you
    want to solve using SupernoVAE.
    """
    import tensorflow   as tf
    import numpy        as np
    import itertools    as it
    import os
    
    from sklearn.decomposition import PCA

    def __init__(self, random_seed=1, N_steps=100000):
        """
        The initialization function of the class. This function will set the hyperparameters of the enviroment of the 
        task you want to solve using SupernoVAE. Also the structure of the Folders will be created. If the config 
        file can not be found, the values will be said to 'None' and a message will be put to the std-out.


        Keyargument
        random_seed -- The random seed to use for all random number generators.
        N_steps     -- The number of steps that the neural network is trained for.
        """
        # Check if config file exists. If the config file exist, all enviroment variables will be loaded from it. The 
        # config file has to be named 'Config_Svae.ini'
        if self.os.path.exists("Config_Svae.ini"):
            print("Configuration file found")
            
            self.__config__     = ConfigObj("Config_Svae.ini")
            self.network_folder = self.__config__["Network_folder"]
            self.description    = self.__config__["Description"]
            self.name_model     = self.__config__["Name_model"]
            self.name_datafile  = self.__config__["Name_datafile"]
            self.verbosity      = self.__config__["Verbosity"]
            self.theta          = self.__config__["Theta"]
            self.time_size      = self.__config__["Time_size"]
            self.max_lat        = int(self.__config__["Max_lat"])
            self.max_lon        = int(self.__config__["Max_lon"])

            # Call the function to create the structure of folders.
            self._create_folders()

        else:
            #Set all values to 'None'
            self.network_folder = None
            self.description    = None
            self.name_model     = None
            self.verbosity      = None
            self.theta          = None
            self.time_size      = None
            self.name_datafile  = None

            #Put a message on the std-out.
            print("No configuration file found, use create_conf_file() to create one")

        # Setting verbosity of tensorflow to the verbosity of the enviroment. If the verbosity of the enviroment is 
        # 'None', the verbosity will be set to only output error
        if self.verbosity == 'No information' or self.verbosity == None:
            self.tf.logging.set_verbosity(self.tf.logging.ERROR)

        if self.verbosity == 'Information':
            self.tf.logging.set_verbosity(self.tf.logging.INFO)

        if self.verbosity == 'Debug':
            self.tf.logging.set_verbosity(self.tf.logging.DEBUG)

    def create_conf_file(   self, 
                            Name_model      = 'World_toy_Example', 
                            Description     = None, 
                            Network_folder  = 'World_test_main_folder', 
                            Name_datafile   = 'World_toy', 
                            Verbosity       = None,
                            Theta           = 0.1, 
                            Time_size       = 696, 
                            Max_lat         = 60, 
                            Max_lon         = 60):
        """
        The function that creates the config file. The config file will be stored and the configuration will be used 
        for this instance of supernovae

        Keyarguments
        Name_model      --  The name of the model. If not set, the name will be 'World_toy_Example'
        Description     --  A short description of the task. No default value will be set.
        Network_folder  --  The name of the direction in which all input and output data will be stored. This can be 
                            changed to save different runs of the same model in different locations. The default value
                            is 'World_test_main_folder'.
        Name_datafile   --  This will be the begining of the name of all outputs of the method. If not set it will be 
                            'World_toy'.
        Verbosity       --  Will set the amount of information Tensorflow will output during the training and 
                            prediction. There are three possible values 'No information', for only errors and important
                            information. 'Information' will give more information and 'Debug' will also include debug
                            information. The default value is 'None' which is the same as 'No information'.
        Theta           --  Theta is the ratio of the dimensionality of the bottelnack of the autoencoder and the 
                            dimensionality of the input. The default is 0.1 meaning that the input is ten times bigger
                            than the bottelnack.
        Time_size       --  The number of timesteps in each timeseries. The default value is 696.
        Max_lat         --  The number of rows in the data grid. Default value is 60.
        Max_lon         --  The number of columns in the data grid. Default value is 60.
        """
        # Create the object and assign it values
        config_file = ConfigObj()
        config_file.filename            = "Config_Svae.ini"
        config_file["Network_folder"]   = Network_folder
        config_file["Name_model"]       = Name_model
        config_file["Description"]      = Description
        config_file["Name_datafile"]    = Name_datafile
        config_file["Verbosity"]        = Verbosity
        config_file["Theta"]            = Theta
        config_file["Time_size"]        = Time_size
        config_file["Max_lat"]          = Max_lat
        config_file["Max_lon"]          = Max_lon
        # Save the file
        config_file.write()

        # Use the new configuration for this instance of SupernoVAE
        self.network_folder             = config_file["Network_folder"]
        self.description                = config_file["Description"]
        self.name_model                 = config_file["Name_model"]
        self.verbosity                  = config_file["Verbosity"]
        self.theta                      = config_file["Theta"]
        self.time_size                  = config_file["Time_size"]
        self.name_datafile              = config_file["Name_datafile"]
        self.max_lat                    = config_file["Max_lat"]
        self.max_lon                    = config_file["Max_lon"]

        # Setting verbosity
        if self.verbosity == 'No information' or self.verbosity == None:
            self.tf.logging.set_verbosity(self.tf.logging.ERROR)
        if self.verbosity == 'Information':
            self.tf.logging.set_verbosity(self.tf.logging.INFO)
        if self.verbosity == 'Debug':
            self.tf.logging.set_verbosity(self.tf.logging.DEBUG)
            
        self._create_folders()
        
                
    def _create_folders(self):
        """ The methods expect a certain folder structure. This method will create this folder structure """
        
        # set the name for all directorys needed in the later process.
        directory_main      = self.network_folder
        directory_model     = directory_main    + '/model/'
        directoy_input      = directory_main    + '/input_data/'
        directory_output    = directory_main    + '/output/'
        directory_mean      = directory_output  + '/mean/'
        directory_dev       = directory_output  + '/dev/'
        directory_time      = directory_output  + '/reconstruction/'
        directory_input     = directory_output  + '/input_data/'
        
        # Check if the folder structure is created, otherwise we created
        # set the directorys for the means and deviations
        if self.verbosity == 'information' or self.verbosity == 'Debug':
            print('Cheking if file system exists')

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
        """
        This function will call the learning process as difined in the SupernoVAE_functions.py file.

        Keyarguments:
        N_steps     -- The number of steps that the algorithm will train for. Default is 10000.
        random_seed -- The random seed used for all random generators.
        """        
        import SupernoVAE_functions as svae

        #Define the parameters needed for the training process
        MODEL_DIR = self.network_folder + '/model/'
        l_rate = 0.0006282
        N_steps_rise = N_steps

        # define an estimator using the model named above and storing checkpoints to the
        # directory difined by MODEL_DIR
        for l in range(1):
            # This loop is created to make learning rate sceduals more easy. The learning rate can be changed due to
            # a scedual in this loop. Notice that the number of steps will be carried out for every scedualed learning
            # rate.
            l_rate = l_rate * 1
            supernovae = self.tf.estimator.Estimator(   model_fn    =svae.make_model_fn(l_rate, random_seed), 
                                                        model_dir   =MODEL_DIR)

            # train the classifier with the training function for the specified number of steps. This will be
            # skipped automatically if the model was trained at least N_steps in an earlier run of the programm
            supernovae.train(
                input_fn    = svae.train_input_fn,
                max_steps   = N_steps)
            N_steps += N_steps_rise

    def predict(self, random_seed=1, save_output = True):
        """
        This function is used to predict the embedding space for all input examples.
        
        Keyarguments
        random_seed --  The seed used for all random generators. The default is 1.
        save_output --  This boolean is true if the output should be saved to the disk. If it is false, the output will
                        only be held in the instance of Supernovae
        """
        import SupernoVAE_functions as svae
        
        #set parameters for the network
        MODEL_DIR = self.network_folder + '/model/'
        l_rate = 0.0006282

        # evaluate the model on all examples and output the mean square error and the mean absolute error
        # to the std out.
        supernovae = self.tf.estimator.Estimator(model_fn=svae.make_model_fn(l_rate, random_seed), model_dir=MODEL_DIR)
        eval_out = supernovae.evaluate(input_fn=svae.test_input_fn)
        print(eval_out)

        # Set up logging hooks to log the progress during predictions
        logging_hook = self.tf.train.StepCounterHook(every_n_steps=1)

        # report the time series, the position and the encoding for every example
        output = supernovae.predict(input_fn=svae.test_input_fn, hooks=[logging_hook])
        
        # Save the output on the object
        self.output = output
        
        # Save the encodings of the output
        if save_output:
            svae.save_encodings(output)
            
    def load_predictions(self):
        """ This method will load the predictions to later carry out the PCA over them"""

        #set the folder location of the examples and set the values needed to calculate the size.
        folder = self.network_folder + "/output/mean/"
        size = self.time_size 
        dtype=self.np.float32,
        name = self.name_datafile

        names = [] # Store the names and find names in the directory
        p = 0
        
        # Find all files in the folder
        for lat, lon in self.it.product(range(0, self.max_lat), range(0, self.max_lon)):
            if self.verbosity == 'Information' or self.verbosity == 'Debug':
                if self.np.random.uniform() > 0.9995:
                    print(str(p+1) + "/" + str(self.max_lat*self.max_lon) + ":" + 
                          str(round(p*100/(self.max_lat*self.max_lon), 2)) + "%", end = "\r")
            p += 1
            file_name = folder + '/' + name + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin'


            # check if the file exists, store its name into the name list
            if self.os.path.exists(file_name):
                names.append(file_name)

        # Stores the embeddings without masking
        self.embeddings = []

        #read the data from the files
        for n in names:
            self.embeddings.append(self.np.fromfile(n, dtype="float32"))
        
        #check that the first element has the correct length
        print(len(self.embeddings))
        #assert(len(self.embeddings) == self.max_lat*self.max_lon)

        #return the list of arrays
        self.embeddings = self.np.array(self.embeddings)
        
        if self.verbosity == 'Information' or self.verbosity == 'Debug':
            print('Predictions loaded ...')
            
    def save_data_tf(self, dataset):
        """Save the data into a binary file"""
        dataset = dataset.astype('float32')

        folder = self.network_folder + '/input_data/'
        Name = self.name_datafile + '_'
        name = None

        # Save it
        for lat in range(0,self.max_lat):
            for lon in range(0,self.max_lon):
                sample = dataset[lat, lon,:]
                name = str(folder)+'/' +str(Name) + str(lat).zfill(3) + '_' + str(lon).zfill(3) + '.bin'
                sample.tofile(name)
                if self.verbosity == 'Debug':
                    print(name)
                    pass
                pass
            pass
        print('The dataset has been saved in binary format')
        
    def PCA_embeddings(self):
        """Use sklearn to carry out the PCA"""
        self.principal_components = self.PCA()
        self.principal_components.fit(self.np.asarray(self.embeddings).transpose())
