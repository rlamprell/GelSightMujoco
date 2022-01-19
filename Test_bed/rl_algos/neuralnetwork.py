"""
    Class for creating neural networks (nn) for the Mujoco environment.
    -- It is utilised twice for DDQN and four times for HDDQN.
    -- Obsevation types, [vision, tactile, kinetic, raw] should match those output from the environment.
    -- The number of dense layers (and the neurons per layer) pre and post concatenation are changable 

    You may need to adjust the amount of vram (memory_limit=''), dependent on your gpu and the resolution of 
    the inputs provided from the environment observation.
"""

# Package imports
import  tensorflow                                          as     tf
from    tensorflow.keras.models                             import Model, Sequential
from    tensorflow.keras.layers                             import InputLayer, concatenate, Flatten, Dense, InputLayer, BatchNormalization, ReLU
from    tensorflow.keras.optimizers                         import Adam
from    tensorflow.keras.losses                             import Huber
from    tensorflow.keras.initializers                       import HeUniform, HeNormal, GlorotNormal, GlorotUniform
from    tensorflow.keras.applications.resnet_v2             import ResNet50V2
from    tensorflow.keras.applications.mobilenet_v2          import MobileNetV2 
from    tensorflow.keras.applications.inception_v3          import InceptionV3
from    tensorflow.keras.applications.inception_resnet_v2   import InceptionResNetV2
from    classification_models.tfkeras                       import Classifiers


# DISABLED (enabled by default) - Runs faster
tf.compat.v1.disable_eager_execution()

# try to enable jit
# -- should run faster with jit enabled
# -- tested on an RTX 3090, unsure if this works on AMD
try:
    tf.config.optimizer.set_jit(True)
except:
    pass

# attempt to use a gpu for processing and limit the amount of memory it can use
# -- may encounter warning messages if not enough vram is alloted for the nn
try:
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=10000)])
except:
    pass



# Construct neural networks 
class NeuralNetwork:
    def __init__(   self,
                    lr,
                    n_actions, 
                    vision_dims     = None, 
                    tact_dims       = None,
                    num_inp         = 0,
                    nn_type         = [False, False, False, True],
                    net_name        = None,
                    concate_dense   = None,
                    numeric_dense   = None,
                    transfer_mod    = "resnet18"
                    ):

        self.lr                     = lr
        self.n_actions              = n_actions
        self.tact_dims              = tact_dims
        self.vision_dims            = vision_dims
        self.num_inp                = num_inp
        self.nn_type                = nn_type
        self.net_name               = net_name
        self.concate_dense_layers   = concate_dense
        self.numeric_dense_layers   = numeric_dense
        self.counter                = 0
        self.transfer_model         = self.__get_transfer_model(transfer_mod)

        # need at least one dense layer
        assert len(self.concate_dense_layers) > 1

    
    # return the model we wisht to use
    def __get_transfer_model(self, transfer_mod):
        if transfer_mod=='resnet50':
            return ResNet50V2
        elif transfer_mod=='resnet18':
            ResNet18, preprocess_input = Classifiers.get('resnet18')
            return ResNet18
        elif transfer_mod=='mobilenet':
            return MobileNetV2 
        elif transfer_mod=='inception':
            return InceptionV3
        elif transfer_mod=='inceptionresnet':
            return InceptionResNetV2
        else:
            raise ValueError('Error, invalid transfer model selected.  Please choose either "resnet50", "resnet18", "mobilenet", "inception", or "inceptionresnet".')


    # build a neural network based on the constructor inputs
    def build(self):
        # set all observations to None initially
        envCam  = None
        tact0   = None
        tact1   = None
        arms    = None

        # if using vision
        if self.nn_type[0]:
            envCam = self.add_resnet(self.vision_dims)

        # if using tactile
        if self.nn_type[1]:
            tact0  = self.add_resnet(self.tact_dims)
            tact1  = self.add_resnet(self.tact_dims)

        # if using kinematics or raw
        if self.nn_type[2] or self.nn_type[3] or self.num_inp!=0:
            arms   = self.add_numeric(self.num_inp)

        # combine the included models
        all_outputs = self.combine_output(envCam, tact0, tact1, arms)

        # -- if more than one use the combined models
        # -- else use the single model
        combined = concatenate(all_outputs)

        # first Dense layer post concate
        z = Dense(self.concate_dense_layers[0], kernel_initializer=HeNormal, name="combined_layer")(combined)
        z = BatchNormalization()(z)
        z = ReLU()(z)

        # append any additional dense layers
        for layer in range(1, len(self.concate_dense_layers)):
            z = Dense(self.concate_dense_layers[layer], kernel_initializer=HeNormal, name=f"combined{layer}")(z)
            z = BatchNormalization()(z)
            z = ReLU()(z)

        z = Dense(self.n_actions, activation="softmax", dtype='float32', name="outputs")(z)

        # combined all the model inputs
        all_inputs = self.combine_input(envCam, tact0, tact1, arms)

        # build the complete model
        model = Model(inputs=all_inputs, outputs=z, name=self.net_name)

        # Compile the model using the layers above, the Adam optimiser and a loss function
        model.compile(optimizer=Adam(learning_rate=self.lr, clipnorm=500), loss=Huber())

        # display a summary of the model produced
        model.summary()
        print()
        print()

        return model

    
    # add a pre-trained resnet model to the structure
    def add_resnet(self, dims):
        model = Sequential()
        model.add(InputLayer(input_shape=(dims[0], dims[1], 3)))
        model.add(self.transfer_model(
            input_shape=(dims[0], dims[1], 3),
            include_top=False,
            weights='imagenet'
        ))
        model.add(Flatten())

        # rename the model layers in case we use more than one
        # -- layer names will conflict otherwise
        for layer in model.layers:
            layer._name = layer.name + str(self.counter)

        # update the layer name modifer, ready for the next input
        self.counter += 1

        return model


    # add a numeric input to the structure
    def add_numeric(self, dims):
        model = Sequential()
        model.add(InputLayer(input_shape=(dims)))

        # add any additional numeric layers before concatenation with 
        # any ResNet Inputs
        for layer in range(len(self.numeric_dense_layers)):
            model.add(Dense(self.numeric_dense_layers[layer]))
            model.add(BatchNormalization())
            model.add(ReLU())

        return model


    # combine all the used inputs for concat
    def combine_input(self, envCam, tact0, tact1, arms):
        combined = []
        # if using vision
        if envCam!=None:
            combined = [envCam.input]
        # if using tactile
        if tact0!=None:
            if len(combined)==0:
                combined = [tact0.input]
            else:
                combined.append(tact0.input)
            combined.append(tact1.input)
        # if using kinematics or raw
        if arms!=None:
            if len(combined)==0:
                combined = [arms.input]
            else:
                combined.append(arms.input)

        return combined


    # combined all the used outputs for concat
    def combine_output(self, envCam, tact0, tact1, arms):
        combined = []
        # if using vision
        if envCam!=None:
            combined = [envCam.output]
        # if using tactile
        if tact0!=None:
            if len(combined)==0:
                combined = [tact0.output]
            else:
                combined.append(tact0.output)
            combined.append(tact1.output)
        # if using kinematics or raw
        if arms!=None:
            if len(combined)==0:
                combined = [arms.output]
            else:
                combined.append(arms.output)

        return combined