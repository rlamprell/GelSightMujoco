"""
    This is a commandline input handler to access some of the headline features of the software.
    It will allow you to choose between:
        -- standard ddqn and hddqn agents
        -- different environments for the robotarm: pickandplace, push, reach or slide
        -- testing or training (Note: a nn model can be provided for either test or train but MUST be provided for the former)
    
    The main limitation of this format is that the underlying neural structure and some key parameters of the ddqn model(s) are
    not directly accessible to the user.  Some of these additionally settings can be manually updated in the methods below, whereas
    others require modifications directly to their associated class/method.
"""


# sys used for commandline inputs
import sys

# both standard double deep q-learning (ddqn) and hierarchical-ddqn (hddqn) agents
from agents                 import bot_ddqn
from agents                 import bot_hddqn

# all available environments - Note: the base class is excluded as it is not intended to be called on its own
from envs                   import RobotArm__pick_and_place
from envs                   import RobotArm__push
from envs                   import RobotArm__reach
from envs                   import RobotArm__slide

# import the ddqn algorithm - which contains the neural network building
from rl_algos.ddqn_model    import ddqn
from rl_algos.neuralnetwork import NeuralNetwork
from rl_algos.replaybuffer  import ReplayBuffer



# handler changing some of the features of the environment and neural network settings
class handler:
    def __init__(self,
                 # env related params
                 # - note: image_rescale should be numeric to enable or, None to disable
                 env                = "reach",
                 render             = True,
                 random_spawns      = True,
                 gelsight_dims      = [224, 224],
                 envcam_dims        = [720, 720],
                 image_rescale      = 32,
                 image_grayscale    = False,
                 image_normalise    = False,
                 framestack         = False,
                 k_frames           = 1,
                 
                 # agent related params
                 agent              = "ddqn", 
                 mode               = "train", 
                 number_of_episodes = 1000000,
                 model_weights0     = None,   
                 model_weights1     = None,

                 # DDQN_model params
                 replay_buffer_size = 10000,
                 batch_size         = 128,

                 # Neural Network params
                 alpha_lr           = 0.0001,
                 beta_lr            = None,
                 nn_dense           = [1024, 512, 256],
                 nn_num_dense       = [256, 256, 256],
                 ext_nn_dense       = None,
                 ext_nn_num_dense   = None,
                 transfer_model     = 'resnet18',

                 # misc params 
                 # -- when using hddqn, optical observations must be paired with a numeric one
                 #    for the intrinsic agent only (examples: type_=raw_vision; type_=tactile_kinematics)
                 # -- This is due to the inclusion a goal within the intrinsic agent's network and replaybuffer
                 type_              = "raw",
                 extrinsic_type2_   = "vision",
                 seed               = 42
                 ):

        # agent: ddqn or hddqn
        # env:   Reach, pickandplace, push or slide
        # mode:  train or test
        # epis:  number of episodes performed by the agent
        self.input_agent    = agent
        self.input_env      = env
        self.input_mode     = mode
        self.n_episodes     = number_of_episodes
        self.render         = render

        # if running ddqn remove the extrinsic observation type
        if self.input_agent=="ddqn":
            extrinsic_type2_=None


        # weights for training or testing the network 
        # -- some networks are setup to load pretrained weights already
        self.input_model    = model_weights0
        self.input_model0   = model_weights1


        # camera information
        # -- dimensions of the gelsight
        # -- should the images be rescaled, grayscaled or normalised 
        #       (applies to all cameras)
        self.gelsight_dims  = gelsight_dims
        self.envCam_dims    = envcam_dims
        self.image_rescale  = image_rescale
        self.image_gray     = image_grayscale
        self.image_norm     = image_normalise
        self.framestack     = framestack
        self.k_frames       = k_frames


        # should the objects be randomised
        # -- applies to both the target object and the target position
        self.random_spawns  = random_spawns
        self.seed           = seed
        

        # learning rates for the neural networks
        # -- beta sets the extrinsic agent's learning rate for hddqn
        #       (if no beta is selected it will be the same as alpha)
        self.nn_lr          = alpha_lr
        self.nn_lr1         = beta_lr
        if beta_lr is None:
            self.nn_lr1     = self.nn_lr


        # what type of neural network are we usin
        # -- raw, vision, kinematics, tact or some combination of the four
        self.nn_type        = type_
        self.nn_type2       = extrinsic_type2_
        if extrinsic_type2_ is None:
            self.nn_type2   = self.nn_type


        # number of dense layers and their neuron counts in the network
        self.nn_dense       = nn_dense
        self.nn_num_dense   = nn_num_dense    


        # transfer model to be used
        # -- 'resnet', 'mobilenet' or 'inceptionresnet'
        self.transfer_model = transfer_model


        # if there are no layers specified for an extrinsic model, use the intrinsic's
        self.ext_nn_dense    = ext_nn_dense
        self.ext_nn_num_dense= ext_nn_num_dense
        if ext_nn_dense==None:
            self.ext_nn_dense     = self.nn_dense
        if ext_nn_num_dense==None:
            self.ext_nn_num_dense = self.nn_num_dense

        self.batch_size     = batch_size
        self.replaybuffsize = replay_buffer_size


        # if there is more than one input type, then combine them so the environment
        # can return both sets of dimensions
        nn_type_bool        = self.__get_agent_returns(self.nn_type)
        nn_type_bool0       = self.__get_agent_returns(self.nn_type2)
        env_nn              = nn_type_bool.copy()

        if nn_type_bool!=nn_type_bool0:
            for i in range(len(nn_type_bool)):
                if nn_type_bool[i] or nn_type_bool0[i]:
                    env_nn[i] = True


        # check the parameters provided are valid
        active_agent        = self.__get_agent()
        active_env          = self.__get_env(env_nn)
        self.__check_mode()


        # get the actions and dimensions of the environment
        self.n_actions      = len(active_env.action_space.sample())*2
        self.env_dims       = active_env.get_observation_space()


        # get all the associated dimensions for the networks and replaybuffer
        # -- int: intrinsic agent dims (also used on ddqn)
        # -- ext: extrinsic agent dims
        self.int_vision_dims, self.int_tact_dims, self.int_arm_dims = self.__get_env_dims(self.nn_type)
        self.ext_vision_dims, self.ext_tact_dims, self.ext_arm_dims = self.__get_env_dims(self.nn_type2)

        # create the agent within the environment and get the dims
        if self.input_agent=="ddqn":
            myAgent             = active_agent.Bot(environment=active_env)
        else:
            myAgent             = active_agent.Bot(environment=active_env,
                                                   int_returns=nn_type_bool, ext_returns=nn_type_bool0)

        # pass the dims to create the correct reinforcement and neural network structures
        self.__create_an_rf_model_and_nn(myAgent, self.nn_lr, self.nn_lr1)

        # run the model (test or train)
        self.__run(myAgent)


    # convert input agent to a usable class
    # -- ddqn:  double deep q-learning
    # -- hddqn: hierarchicial double deep q-learning
    def __get_agent(self):
        if   self.input_agent=="ddqn":
            return bot_ddqn

        elif self.input_agent=="hddqn":
            return bot_hddqn

        else:
            raise ValueError("Error,", self.input_agent, "is not a recogneised agent!  Use 'ddqn' or 'hddqn'.")


    # convert the input environment to a usable env
    def __get_env(self, env_type):

        real, tact, kine, raw = env_type

        # generic keywords
        kwargs  = { 'render':           self.render,
                    'return_real':      real, 
                    'return_tact':      tact, 
                    'return_kine':      kine, 
                    'return_raw':       raw,
                    'gel_width':        self.gelsight_dims[0], 
                    'gel_height':       self.gelsight_dims[1],
                    'envcam_width':     self.envCam_dims[0],
                    'envcam_height':    self.envCam_dims[1],
                    'image_grayscale':  self.image_gray,
                    'image_normalise':  self.image_norm,
                    'rescale':          self.image_rescale,
                    'frame_stack':      self.framestack,
                    'k_frames':         self.k_frames,
                    'random_target':    self.random_spawns,
                    'seed':             self.seed
        }
        # object specific keywords
        object_k = {
                    'random_object':    self.random_spawns,
        }

        if  self.input_env=="pickandplace":
            return RobotArm__pick_and_place.RobotArmPickAndPlace(   **kwargs, **object_k)

        elif self.input_env=="push":
            return RobotArm__push.RobotArmPush(                     **kwargs, **object_k)

        elif self.input_env=="slide":
            return RobotArm__slide.RobotArmSlide(                   **kwargs, **object_k)

        elif self.input_env=="reach":
            return RobotArm__reach.RobotArm__Reach(                 **kwargs)

        else:
            raise ValueError("Error,", self.input_env, "is not a recogneised environment!"
                              "Use 'pickandplace', 'push', 'reach', or 'slide'.")


    # from the nn_type provided, establish which prebuilt neural network is required
    # -- [vision, tactile, kinematics, raw]
    def __get_agent_returns(self, type):
        nn_format = [False, False, False, False]
        if type.find("vision")!=-1:
            nn_format[0]=True

        if type.find("tact")!=-1:
            nn_format[1]=True

        if type.find("kinematics")!=-1:
            nn_format[2]=True
        
        if type.find("raw")!=-1:
            nn_format[3]=True

        return nn_format


    # check the mode input is valid
    def __check_mode(self):
        if self.input_mode=="train":
            return 

        elif self.input_mode=="test":
            if self.input_model==None:
                raise ValueError("Error, a model must be selected when testing an agent, HINT: model='3000-ddqn_model.h5'")
            return 

        else:
            raise ValueError("Error,", self.input_mode, "is not a recogneised mode!  Use 'test' or 'train'.")


    # dimensions for the environemnt
    def __get_env_dims(self, type):

        vision_dims= None
        tact_dims  = None
        arm_dims   = [0]

        if type.find("vision")!=-1:
            vision_dims     = self.env_dims['real'].shape

        if type.find("tact")!=-1:
            tact_dims       = self.env_dims['tact'].shape

        if type.find("kinematics")!=-1 or type.find("raw")!=-1:
            arm_dims        = self.env_dims['arm'].shape

        return vision_dims, tact_dims, arm_dims


    # make a neural network and reinforcement learning algo
    # -- lr0: is the default learning rate and will apply to the inner agent for hddqn
    # -- lr1: only used for hddqn and is applied to the extrinsic agent
    def __create_an_rf_model_and_nn(self, agent, lr0, lr1):

        kwargs = {
            'n_actions':    self.n_actions,
            'vision_dims':  self.int_vision_dims,
            'tact_dims':    self.int_tact_dims,
            'num_inp':      self.int_arm_dims,
            'nn_type':      self.nn_type
        }
        network = {
            'concat_dense': self.nn_dense,
            'numeri_dense': self.nn_num_dense,
            'transfer_mod': self.transfer_model
        }
        # create a nn and rb for the ddqn agent
        if self.input_agent=="ddqn":
            # construct the agent model utilities
            # -- evalutation neural net
            # -- target neural net
            # -- replaybuffer
            eval_nn = self.__build_nn(**kwargs, **network, lr=lr0, net_name="eval_network")
            targ_nn = self.__build_nn(**kwargs, **network, lr=lr0, net_name="targ_network")
            rb      = self.__build_rb(**kwargs)

            # create the deep model and pass the nn and rb to it
            self.deep_model  = ddqn(gamma=0.99, n_actions=self.n_actions, epsilon=1, epsilon_decay=0.99996,
                                    epsilon_minim=0.1, batch_size=self.batch_size,
                                    eval_neuralnetwork=eval_nn, targ_neuralnetwork=targ_nn,
                                    replaybuffer=rb)
                                    

        # create nn's and rb's for the hddqn agent                               
        elif self.input_agent=="hddqn":
            # goals for this environment 
            # -- used as the network output for the hddqn networks
            n_goals = len(agent.get_env_goals())

            # construct inner agent model utilities
            # -- evalutation neural net
            # -- target neural net
            # -- replaybuffer
            eval_nn     = self.__build_nn(**kwargs, **network, lr=lr0, net_name="intrinsic_eval_network", goal=1)
            targ_nn     = self.__build_nn(**kwargs, **network, lr=lr0, net_name="intrinsic_targ_network", goal=1)
            rb          = self.__build_rb(**kwargs,                                                       goal=1)

            # contruct extrinsic agent model utilities
            # -- evalutation neural net
            # -- target neural net
            # -- replaybuffer
            h_kwargs    = {
                'n_actions':    n_goals,
                'vision_dims':  self.ext_vision_dims,
                'tact_dims':    self.ext_tact_dims,
                'num_inp':      self.ext_arm_dims,
                'nn_type':      self.nn_type2,
            }
            h_network   = {
                'concat_dense': self.ext_nn_dense,
                'numeri_dense': self.ext_nn_num_dense,
                'transfer_mod': self.transfer_model
            }
            
            eval_h_nn   = self.__build_nn(**h_kwargs, **h_network, lr=lr1, net_name="extrinsic_eval_network")
            targ_h_nn   = self.__build_nn(**h_kwargs, **h_network, lr=lr1, net_name="extrinsic_targ_network")
            h_rb        = self.__build_rb(**h_kwargs)


            # create the intrinsic ddqn model and pass the nn and rb to it
            self.deep_model  = ddqn(gamma=0.99, n_actions=self.n_actions, epsilon=1, epsilon_decay=0.99996,
                                    epsilon_minim=0.1, batch_size=self.batch_size,
                                    eval_neuralnetwork=eval_nn, targ_neuralnetwork=targ_nn, 
                                    replaybuffer=rb)

            # create the extrinsic ddqn model and pass the nn and rb to it
            self.h_model     = ddqn(gamma=0.99, n_actions=n_goals, epsilon=1, epsilon_decay=0.99996,
                                    epsilon_minim=0.1, batch_size=self.batch_size,
                                    eval_neuralnetwork=eval_h_nn, targ_neuralnetwork=targ_h_nn,
                                    replaybuffer=h_rb)

        else:
            raise ValueError("Unknown error when creating learning models from handler.")


    # create the neural_networks we need for ddqn and hddqn
    def __build_nn( self,
                    lr,
                    n_actions, 
                    vision_dims = None,
                    tact_dims   = None,
                    num_inp     = None,
                    nn_type     = None,
                    concat_dense= None,
                    numeri_dense= None,
                    net_name    = None,
                    transfer_mod= None,
                    goal        = 0
                ):

        # 'goal' should be 0 or 1, only use 1 for the deep_model of a hddqn bot
        # -- adds goal to the numerical inputs of the network
        nn = NeuralNetwork(
            lr              = lr,
            n_actions       = n_actions, 
            vision_dims     = vision_dims,
            tact_dims       = tact_dims,
            num_inp         = num_inp[0]+goal,
            nn_type         = self.__get_agent_returns(nn_type),
            net_name        = net_name,
            concate_dense   = concat_dense,
            numeric_dense   = numeri_dense,
            transfer_mod    = transfer_mod
        )

        network = nn.build()

        return network


    # create the replay buffer to store the state transitions
    def __build_rb( self, 
                    n_actions, 
                    vision_dims = None,
                    tact_dims   = None,
                    num_inp     = None,
                    goal        = 0,
                    nn_type     = None):
        
        rb = ReplayBuffer(  n_actions   = n_actions, 
                            vision_dims = vision_dims,
                            tact_dims   = tact_dims,
                            num_inp     = num_inp[0]+goal,
                            nn_type     = self.__get_agent_returns(nn_type),
                            max_size    = self.replaybuffsize,
                            goal        = goal
                            )
        return rb

        
    # run the model
    def __run(self, agent):
        # if ddqn then load one deep model
        # -- this contains two gelsight cameras and arm related properties (actuation forces, joint positions and joint velocities)
        # run double deep q-learning
        if self.input_agent=="ddqn":
            kwargs   = {
                'deep_model': self.deep_model,
                'model_weights': self.input_model,
                'n_episodes': self.n_episodes
            }

            # run training
            if self.input_mode=="train":
                agent.train(**kwargs)
            # run testing
            elif self.input_mode=="test":
                agent.test( **kwargs)
            # error
            else:
                raise ValueError("Error, input_mode is invalid, please use either 'train' or 'test'.")

        # if hddqn then load two deep models 
        # -- one for the intrinsic agent and another for the extrinsic agent
        elif self.input_agent=="hddqn":

            kwargs = {
                'intrinsic_model':  self.deep_model,
                'extrinsic_model':  self.h_model,
                'model0_weights':   self.input_model,
                'model1_weights':   self.input_model0,
                'n_episodes':       self.n_episodes
            }

            if self.input_mode=="train":
                agent.train(**kwargs)

            elif self.input_mode=="test":
                agent.test( **kwargs)
            
            raise ValueError("Error, input_mode is invalid, please use either 'train' or 'test'.")

        else:
            raise ValueError("An issue occured when attempting to load the agent with the supplied parameters.")




# run the handler when using the commandline prompt
if __name__=='__main__':
    # take the command line inputs and pass them to the handler class
    handler(*sys.argv[1:])