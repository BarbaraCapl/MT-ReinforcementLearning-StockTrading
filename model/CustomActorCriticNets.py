import numpy as np
from typing import Tuple, Optional, Union, Dict, NoReturn, Lists 
import torch
from torch import nn
from torch.distributions import Normal, Dirichlet

########################################################################
# CUSTOM NETWORK ARCHITECTURES FOR:
#           FEATURE EXTRACTOR MODULE
#           ACTOR NETWORK MODULE
#           CRITIC NETWORK MODULE
#           AGENT NETWORK (Combination of all of the above)
# 
# HELPER FUNCTIONS FOR WEIGHT INITIALIZATION OF AGENT NETWORK
########################################################################

class FeatureExtractorNet(nn.Module):
    """ FEATURE EXTRACTOR (FE)

    This class implements ths Shared Feature Extractor Network Module.
        Input:  Current State / "State Observations" of the agent
        Output: Intermediate Features to be passed to the Actor and Critic Network Modules.

    MLP = Multilayer Perceptron
    LSTM = Long short-term memory
    """
    def __init__(self,
                 observations_size: int = None,
                 mid_features_size: int = 64,
                 hidden_size: int = 64,
                 net_arch: str = "mlp_separate",
                 lstm_observations_size: int = None,
                 lstm_hidden_size: int = 32,
                 lstm_num_layers: int = 2, # 
                 ):
        """        
        Args:
            observations_size (int, optional):  Size of the observation space, input dimension. (Usually n_assets * n_asset_features + 1 (cash) + 1 (VIX). Defaults to None.
            mid_features_size (int, optional):  Number of neurons in the mlp layer. Defaults to 64.
            hidden_size (int, optional):        Number of neurons in the hidden layer. Defaults to 64.
            net_arch (str, optional):           Architecture of choice. Defaults to "mlp_separate".
                "mlp_separate":     MLP with no shared layers between actor and critic.
                "mlp_shared":       LP with layers shared between actor and critic. Output layer separate.
                "mlplstm_separate": Combination of MLP and LSTM horozontically, no shared layers between actor and critic.
                "mlplstm_shared":   Combination of MLP and LSTM horozontically, shared layers between actor and critic. Output layer separate.
            lstm_observations_size (int, optional): Input dimenstion for LSTM. Usually n_assets (returns) + 1 (VIX)), as it is only applied to returns. Defaults to None.
            lstm_hidden_size (int, optional):       Number of neurons in the LSTM hidden layer. Defaults to 32.
            lstm_num_layers (int, optioal):         Number of LSTM layers. Parameter can be directly passed to the nn.LSTM. Defaults to 2
        
        Note on LSTM module (in PyTorch): 
            refer to: 
                -   https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
                -   https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
                -   https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
        
        """
        super(FeatureExtractorNet, self).__init__()
        
        self.observations_size = observations_size
        self.mid_features_size = mid_features_size
        self.hidden_size = hidden_size
        self.net_arch = net_arch
        self.lstm_observations_size = lstm_observations_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        if net_arch == "mlp_separate" or net_arch == "mlp_shared":
            print("(FE) net_arch = only mlp (shared or separate)")
            
            self.feature_extractor = nn.Sequential(
                nn.Linear(in_features=observations_size, out_features=hidden_size, bias=True),
                nn.Tanh(),
                #nn.Softmax(),
                #nn.ReLU(), # relu and softmax here didn't work well; made both layers have 0 gradient more often
                nn.Linear(in_features=hidden_size, out_features=mid_features_size, bias=True),
                nn.Tanh(),
            )

        elif net_arch == "mlplstm_separate" or net_arch == "mlplstm_shared": # using mlp as before and adding an lstm for additional lags extraction.
            print("(FE) net_arch = mlp + lstm")
            self.feature_extractor_mlp = nn.Sequential(
                nn.Linear(in_features=observations_size, out_features=hidden_size, bias=True),
                nn.Tanh(),
                nn.Linear(in_features=hidden_size, out_features=mid_features_size, bias=True),
                nn.Tanh(),
            )
           
            self.feature_extractor_lstm = nn.LSTM(
                input_size=lstm_observations_size, 
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers, 
                batch_first=True, # means that input is provided as (batch_size, sequence_length, n_features(=input_size)
                bias=True, 
                dropout=0.2
            )
            
            # we use a summary layer to decrease the MLP-LSTM-combined larger feature space to 64.
            self.summary_layer = nn.Sequential(
                nn.Linear(in_features=mid_features_size+lstm_hidden_size, out_features=mid_features_size),
                nn.Tanh()
            )
            
        else:
            raise NotImplementedError(f"(FE) net_arch {self.net_arch} specified is not valid.")

    def create_initial_lstm_state(self, sequence_length: int=1):
        """Method that initializes the LSTM hidden state.
        
        Refer to: https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384
        Shape of hidden state (h) and cell state (c) is each: (num_layers, number_timesteps, hidden_size)
        E.g. if we have 2 layers (default), a sequence of 5 timesteps and hidden_size=64 (default) => shape = (2, 5, 64).
        If we do a forward pass or predict on one sample (day) only (sequence length = 1), we have shape = (2, 1, 64).
    
        Args:
            sequence_length (int, optional): Length of time sequence (e.g. 1 if 1 day). Defaults to 1.
    
        Returns:
            Tuple[torch.zeros]: Contains pairs of hidden and cell state at initialization of the LSTM.
        """
        return  (
            torch.zeros(self.lstm_num_layers, sequence_length, self.lstm_hidden_size),
            torch.zeros(self.lstm_num_layers, sequence_length, self.lstm_hidden_size)
        )

    def forward(self, 
                observations: torch.Tensor=None, 
                lstm_observations: torch.Tensor=None, 
                lstm_states: bool=None):
        """Method that implements the forward pass.

        Args:
            observations (torch.Tensor, optional):      MLP-embedded features from the Feature Extractor. Defaults to None.
            lstm_observations (torch.Tensor, optional): LSTM-embedded features from the Feature Extractor. Defaults to None.
            lstm_states (Tuple[torch.zeros], optional): LSTM states (hidden (h), cell (c)). Defaults to None.

        Returns:
            mid_features (torch.Tensor):   Embedded features to be used as input for actor and critic modules
            lstm_states (torch.Tensor):    Tuple vector with (h, c) being h=hidden state, c=cell state.
        """
        # if length of shape vector is 1 (e.g. shape = 2), then we have a tensor like this: [features], => batch of 1
        # and we want to make it compatible with concatenation later with the lstm output, so we reshape it to [[features]] (shape = batch_size, n_features)
        if len(observations.shape) == 1 and self.net_arch in ["mlplstm_separate", "mlplstm_shared"]:
            observations = observations.reshape(1, self.observations_size)
            # reshape the lstm input dimension_ must be 3D (batch_number=1, time steps per batch = 1, features_number)
            lstm_observations = lstm_observations.reshape(1, 1, self.lstm_observations_size)

        elif len(observations.shape) > 1 and self.net_arch in ["mlplstm_separate", "mlplstm_shared"]:
            # reshape the lstm input dimension_ must be 3D (batch_number=1, time steps per batch, features_number)
            lstm_observations = lstm_observations.reshape(1, len(observations), self.lstm_observations_size)

        if self.net_arch == "mlp_separate" or self.net_arch == "mlp_shared":
            mid_features = self.feature_extractor(observations)

        elif self.net_arch == "mlplstm_separate" or self.net_arch == "mlplstm_shared":
            mid_features_mlp = self.feature_extractor_mlp(input=observations)
            mid_features_lstm, lstm_states = self.feature_extractor_lstm(input=lstm_observations, hx=lstm_states)
            mid_features_mlp_reshaped = mid_features_mlp.reshape(len(observations),  self.mid_features_size)
            mid_features_lstm_reshaped = mid_features_lstm.reshape(len(observations),  self.lstm_hidden_size)
            mid_features = torch.cat((mid_features_mlp_reshaped, mid_features_lstm_reshaped), dim=1) # concatenate features on dim 1
            mid_features = self.summary_layer(mid_features)

        else:
            raise NotImplementedError(f"(FE) net_arch {self.net_arch} specified is not valid.")

        return mid_features, lstm_states


class ActorNet(nn.Module):
    """ ACTOR NETWORK MODULE (AN)
    
    This class implements the Actor module of the Actor-Critic network architecture.
    The Actor network redicts the action means.
    """
    def __init__(
        self,
        actions_num: int,
        mid_features_size: int = 64,
        hidden_size: int = 64,
        net_arch: str = "mlp_separate",
        env_step_version: str = "default",
        ):
        """
        Args:
            actions_num (int):                  output dimension of the actor network, defines the umber of actions for the agent.
            mid_features_size (int, optional):  Size of the encoded observation space / features by feature extractor. Defaults to 64.
            hidden_size (int, optional):        Number of neurons in hidden layer if any. Defaults to 64.
            net_arch (str, optional):           Net architecture. Defaults to "mlp_separate".
            env_step_version (str, optional):   ?. Defaults to "default".
        """
        super(ActorNet, self).__init__()

        if env_step_version == "default":
            self.action_mean_net = nn.Sequential(
                nn.Linear(in_features=mid_features_size, out_features=actions_num, bias=True),
                #nn.Tanh(), # Using squashing function didn't perfprm well 
                )
            
            # Adding log stdev as a parameter, initializes as 0 by default (log(std) = 0 => std = 1, since log(1)=0)
            #   Note: sometimes stdev may "explode", refer to: https://www.reddit.com/r/reinforcementlearning/comments/fdzbs9/ppo_entropy_and_gaussian_standard_deviation/
            #   Workaround used based on the following source: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L84
            log_stdev = -0.5 * np.ones(actions_num, dtype=np.float32)
            self.log_stdev = torch.nn.Parameter(data=torch.as_tensor(log_stdev), requires_grad=True)
        
        elif env_step_version == "newNoShort" or env_step_version == "newNoShort2":
            self.action_mean_net = nn.Sequential(
            nn.Linear(in_features=mid_features_size, out_features=actions_num, bias=True),
        )
        else: 
            raise NotImplementedError(f"(AN) env_step_version {env_step_version} not implemented.")

    def forward(self, mid_features):
        """Method that implements the forward pass.
        
        Args:
            mid_features (torch.Tensor): Intermediate features received from the feature extractor
        Returns:
            actions_mean (torch.Tensor): Vector of predicted actions mean for each action (used to construct the actions distribution)
        """
        return self.action_mean_net(mid_features)


class CriticNet(nn.Module):
    """ CRITIC NETWORK (CN)
    
    This class implements the Critic module of the Actor-Critic network architecture.
    The Critic estimates V, the estimated (Present) Value of all future actions given the current action. 
    """
    def __init__(self,
                 mid_features_size: int=64,
                 hidden_size: int=64,
                 net_arch: str="mlp_separate"):
        """
        Args:
            mid_features_size (int, optional):  Size of the encoded observation space / features by feature extractor. Defaults to 64.
            hidden_size (int, optional):        Number of neurons in hidden layer if any. Defaults to 64.
            net_arch (str, optional):           Net architecture. Defaults to "mlp_separate".
        """
        super(CriticNet, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(in_features=mid_features_size, out_features=1, bias=True),
        )

    def forward(self, mid_features):
        """ Method that takes the intermediate features (given by feature extractor) and
        returns the value (V) estimate for the current state.
        
        Args:
            mid_features (torch.tensor): Intermediate features received from the feature extractor, vector of length mid_features_size.

        Returns:
            (float) : Estimate of value V for current state. 
        """
        return self.value_net(mid_features)



# Helper functions for weights intialization (since it is quite cumbersome if we have a nn.Sequential() model).
# Weight initialization is important to overcome the problem of exploding / vanishing gradients in very deep neural networks. 
# Since the network here is rather shallow, the way we initialize the weights doesn't have a huge impact.
# refer to: 
#   -   https://pytorch.org/docs/stable/nn.init.html
#   -   https://discuss.pytorch.org/t/initialising-weights-in-nn-sequential/76553
#   -   about weights initialization, see also: https://towardsdatascience.com/all-ways-to-initialize-your-neural-network-16a585574b52
#   -   https://ml-compiled.readthedocs.io/en/latest/initialization.html
#   -   https://jamesmccaffrey.wordpress.com/2020/11/20/the-gain-parameter-for-the-pytorch-xavier_uniform_-and-xavier_normal_-initialization-functions/

# TODO: weights functions seem to repeat themselves apart from a few hyperparam (gain), maybe pack in class or in one function only.

def init_weights_feature_extractor_net(module: Optional[nn.Linear], gain: float=np.sqrt(2)):
    """Function to initialize weights for feature extractor network

    Args:
        module (Optional[nn.Linear]): NN layer for which the weights are to be initialized.
        gain (float, optional): Parameter for xavier normal weights initialization. Defaults to np.sqrt(2).
    """
    if isinstance(module, nn.Linear):
        #nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)  # fill bias with 0 if there is any

def init_weights_actor_net(module: Optional[nn.Linear], gain: float=0.01):
    """Function to initialize weights for actor network

    Args:
        module (Optional[nn.Linear]): NN layer for which the weights are to be initialized.
        gain (float, optional): Parameter for xavier normal weights initialization. Defaults to 0.01.
    """
    if isinstance(module, nn.Linear):
        #nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0) # fill bias with 0 if there is any
            
def init_weights_critic_net(module: Optional[nn.Linear], gain: float=1.):
    """Function to initialize weights for critic network

    Args:
        module (Optional[nn.Linear]): NN layer for which the weights are to be initialized.
        gain (float, optional): Parameter for xavier normal weights initialization. Defaults to 0.01.
    """
    if isinstance(module, nn.Linear):
        #nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0) # fill bias with 0 if there is any

class AgentActorCritic(nn.Module):
    """ ACTOR-CRITIC AGENT

    This class implements the Actor-Critic Reinforcement Learning Agent, comining the previously defined architectures.
    """
    
    def __init__(self,
                 observations_size: int,
                 actions_num: int,
                 feature_extractor_class=FeatureExtractorNet,
                 actor_class=ActorNet,
                 critic_class=CriticNet,
                 init_weights_feature_extractor_net = init_weights_feature_extractor_net,
                 init_weights_actor_net = init_weights_actor_net,
                 init_weights_critic_net = init_weights_critic_net,
                 mid_features_size: int = 64,
                 hidden_size_actor: int = 64,
                 hidden_size_critic: int = 64,
                 hidden_size_features_extractor: int = 64,
                 # for lstm, only actually used if verison = mlplstm
                 lstm_observations_size: int = None,
                 lstm_hidden_size_feature_extractor: int = 64,
                 lstm_num_layers: int = 2, #default is 2 because that worked well
                 # optimizer
                 optimizer: torch.optim = torch.optim.Adam,
                 learning_rate: float = 0.00025,
                 # net_archs
                 net_arch: str = "mlp_separate",
                 # step version in env
                 env_step_version: str = "default",
                 ):
        super(AgentActorCritic, self).__init__()
        
        self.net_arch = net_arch
        self.env_step_version = env_step_version
        if self.env_step_version == "newNoShort2":
            actions_num = actions_num + 1 # one for "cash" position in the portfolio

        if net_arch == "mlp_shared" or net_arch == "mlplstm_shared":
            # only one feature extractor for both actor and critic (shared)
            self.feature_extractor = feature_extractor_class(observations_size=observations_size,#FeatureExtractorNet
                                                         mid_features_size=mid_features_size,
                                                         hidden_size=hidden_size_features_extractor,
                                                         net_arch=net_arch,
                                                         # if we have net_arch == mlplstm, else these will be ignored
                                                         lstm_hidden_size=lstm_hidden_size_feature_extractor,
                                                         lstm_observations_size=lstm_observations_size,
                                                         lstm_num_layers=lstm_num_layers)
            # initialize weights for feature extractor
            self.init_weights_feature_extractor_net = init_weights_feature_extractor_net
            self.feature_extractor.apply(self.init_weights_feature_extractor_net)
        
        elif net_arch == "mlp_separate" or net_arch == "mlplstm_separate":
            # separate feature extraction layers between actor and critic
            self.feature_extractor_actor = FeatureExtractorNet(observations_size=observations_size,
                                                               mid_features_size=mid_features_size,
                                                               hidden_size=hidden_size_features_extractor,
                                                               net_arch=net_arch,
                                                               # if we have verison == mlplstm, else these will be ignored
                                                               lstm_hidden_size=lstm_hidden_size_feature_extractor,
                                                               lstm_observations_size=lstm_observations_size,
                                                               lstm_num_layers=lstm_num_layers
                                                               )
            self.feature_extractor_critic = FeatureExtractorNet(observations_size=observations_size,
                                                               mid_features_size=mid_features_size,
                                                               hidden_size=hidden_size_features_extractor,
                                                               net_arch=net_arch,
                                                               # if we have verison == mlplstm, else these willmbe ignored
                                                               lstm_hidden_size=lstm_hidden_size_feature_extractor,
                                                               lstm_observations_size=lstm_observations_size,
                                                               lstm_num_layers=lstm_num_layers
                                                               )
            # initialize weights for both feature extractor nets separately
            self.init_weights_feature_extractor_net = init_weights_feature_extractor_net
            self.feature_extractor_actor.apply(self.init_weights_feature_extractor_net)
            self.feature_extractor_critic.apply(self.init_weights_feature_extractor_net)
        
        else:
            raise NotImplementedError(f"(AgentAC) net_arch specified ({self.net_arch}) not valid.")

        # initialize actor net
        self.actor = actor_class(actions_num=actions_num,
                              mid_features_size=mid_features_size,
                              hidden_size=hidden_size_actor,
                              net_arch=net_arch,
                              env_step_version=env_step_version)

        # initialize acritic net
        self.critic = critic_class(mid_features_size=mid_features_size,
                                hidden_size=hidden_size_critic,
                                net_arch=net_arch)

        #initialize weights for actor and critic
        self.init_weights_actor_net = init_weights_actor_net
        self.init_weights_critic_net = init_weights_critic_net
        self.actor.apply(self.init_weights_actor_net)
        self.critic.apply(self.init_weights_critic_net)

        # Setup optimizer with learning rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        if net_arch == "mlp_shared" or net_arch == "mlplstm_shared":
            # for Adam, apply optimizer to all the parameters in all the three networks
            self.optimizer = self.optimizer(list(self.feature_extractor.parameters()) +
                                            list(self.actor.parameters()) +
                                            list(self.critic.parameters()),
                                            lr=learning_rate, # 0.00025 by default
                                            eps=1e-08, # by default
                                            betas=(0.9, 0.999), # by default
                                            weight_decay=0 # L2 penalty, float, 0 by default
                                            )
        
        elif net_arch == "mlp_separate" or net_arch == "mlplstm_separate": # separate layers between actor and critic
            self.optimizer = self.optimizer(list(self.feature_extractor_actor.parameters()) +
                                            list(self.feature_extractor_critic.parameters()) +
                                            list(self.actor.parameters()) +
                                            list(self.critic.parameters()),
                                            lr=learning_rate, # 0.00025 by default
                                            eps=1e-08, # by default
                                            betas=(0.9, 0.999), # by default
                                            weight_decay=0 # L2 penalty, float, 0 by default
                                            )
            
        else:
            raise NotImplementedError(f"(AgentAC) net_arch specified ({self.net_arch}) not valid.")

    def forward(self,
                observations: torch.Tensor,
                evaluation_mode: bool=False,
                actions: torch.Tensor=None,
                actions_deterministic: bool=False,
                lstm_observations: torch.Tensor=None,
                lstm_states: Tuple=None,
                lstm_states_actor: Tuple=None,
                lstm_states_critic: Tuple=None,
        ):

        if self.net_arch == "mlp_shared" or self.net_arch == "mlplstm_shared": # shared feature extactor between actor and critic
            ### FEATURES EXTRACTION
            mid_features, lstm_states = self.feature_extractor(observations=observations,
                                                              lstm_observations=lstm_observations,
                                                              lstm_states=lstm_states)
            ### CRITIC
            # get value estimate from critic (value network) using these mid_features
            value_estimate = self.critic(mid_features)
            ### ACTOR
            # get estimated action means from actor (policy network) using these mid features
            action_means = self.actor(mid_features)

            # set variables not needed
            lstm_states_actor = None
            lstm_states_critic = None
            if self.net_arch == "mlp_shared":
                lstm_states=None

        elif self.net_arch == "mlp_separate" or self.net_arch == "mlplstm_separate": # separate layers between actor and critic
            ### FEATURES EXTRACTION
            mid_features_actor, lstm_states_actor = self.feature_extractor_actor(observations=observations,
                                                                                lstm_observations=lstm_observations,
                                                                                lstm_states=lstm_states_actor)
            mid_features_critic, lstm_states_critic = self.feature_extractor_critic(observations=observations,
                                                                                   lstm_observations=lstm_observations,
                                                                                   lstm_states=lstm_states_critic)
            ### CRITIC
            # get value estimate from critic (value network) using these mid_features
            value_estimate = self.critic(mid_features_critic)
            ### ACTOR
            # get estimated action means from actor (policy network) using these mid features
            action_means = self.actor(mid_features_actor)

            # set variables not needed
            lstm_states = None
            if self.net_arch == "mlp_separate":
                lstm_states_actor=None
                lstm_states_critic=None
        else:
            raise NotImplementedError(f"(AgentAC) net_arch specified ({self.net_arch}) not valid.")

        if self.env_step_version == "default":
            # get estimated log stdev parameter (like a bias term of the last layer) appended to the actor network
            # Note: it is a nn.Parameter, which is like a tensor added to the module of Pytorch and it is a bit badly
            # documented but apparently this is like a bias term that gets changed as well when the network trains.
            log_stdev = self.actor.log_stdev
            # convert log standard deviation to stdev
            stdev = log_stdev.exp()
            # Note: this is one value. But we need to get a vector of this same value (one for each action)
            # so we can then use it to create a distribution around each action mean
            stdev_vector = torch.ones_like(action_means) * stdev
            # now that we have the means for each action and the standard deviation (one estimate only, same for all actions),
            # we can create a distribution around each action mean; we define a normal distribution for our continuous actions,
            # but we could also use something else
            actions_distribution = Normal(action_means, stdev_vector)
            # see also: https://pytorch.org/docs/stable/distributions.html
        elif self.env_step_version == "newNoShort" or self.env_step_version == "newNoShort2":
            concentrations = action_means.exp() # we interpret the output of the actor (named action means)
                                                # as log alpha vector (=Dirichlet concentrations), and we transform from [-inf, inf] interval
                                                # to a [0, inf] interval
            actions_distribution = Dirichlet(concentration=concentrations)
            # now the action means are the mean of the dirichlet distribution
            action_means = actions_distribution.mean
            # there is no stdev, so we set it to None
            stdev = None
        else:
            raise NotImplementedError(f"(AgentAC) net_arch specified ({self.net_arch}) not valid.")

        #### EVALUATE
        if evaluation_mode:
            # in evaluation mode, we want to get the new log probabilities and new actions distribution based
            # on the actions we feed the agent. Before the first backward pass, the actions distribution and probability are still the same.
            # after the first backward pass, they change.
            actions_joint_log_proba, actions_distr_entropy = self.evaluate_actions(actions, actions_distribution)
            return value_estimate, actions_joint_log_proba, actions_distr_entropy, action_means, actions_distribution, stdev

        #### FORWARD PASS / PREDICT
        else: # by default, forward pass
            if actions_deterministic == True: # only used sometimes for prediction, never in a normal forward pass
                # if we sample deterministically, our actions are the action means
                action_samples = action_means

            elif actions_deterministic == False:
                # if we sample non-deterministically, our actions are sampled from a distribution, which
                # gives it some randomness within a seed and gives it some exploration
                action_samples = actions_distribution.rsample()
                # Note: rsample() supports gradient calculation through the sampler, in contrast to sample()
                # https://forum.pyro.ai/t/sample-vs-rsample/2344

            # get action log probabilities with current action samples.
            # This part of the function is used when we forward pass during trajectories collection into the buffer
            actions_log_probs = actions_distribution.log_prob(action_samples)
            # now we want to calculate the joint distribution of all action mean distributions over all stocks
            # we make the assumption that the actions are independent from each other (could be violated)
            # and we have log probas, hence we can sum the log probabilities up
            if len(actions_log_probs.shape) > 1:
                actions_joint_log_proba = actions_log_probs.sum(dim=1)
            else:
                actions_joint_log_proba = actions_log_probs.sum()

            return value_estimate, action_samples, \
                   actions_joint_log_proba, action_means, actions_distribution, stdev, \
                   lstm_states, lstm_states_actor, lstm_states_critic

    def evaluate_actions(self, actions, actions_distribution):
        # If we are in "evaluation mode", we don't sample a new action but instead
        # use the old action as input to the current distribution.
        # We want to find out: how likely are the actions we have taken during the trajectories sampling
        # (into the Buffer) now, after we have updated our policy with a backward pass?
        # Note: in the first round, we have not yet updated our policy, hence the probabilities we will get will be the same
        # get new action log probabilities for the old actions using peviously defined Normal distribution (actions_distribution)
        actions_log_probs = actions_distribution.log_prob(actions)
        actions_distr_entropy = actions_distribution.entropy()

        if self.env_step_version == "default":
            # now we want to calculate the joint distribution of all action mean distributions over all stocks
            # we make the assumption that the actions are independent from each other (could be violated)
            # and we have log probas, hence we can sum the log probabilities up

            # IMPORTANT: if our batch is of length 1, we sum across the first dimension,
            # because we don't want to sum action log probabilities over all days, but just the action log probabilities overall actions of ONE day
            # (at first I got a lot of errors because of not considering his and the actor had no gradient)
            # same for entropy below
            if len(actions_log_probs.shape) > 1:
                actions_joint_log_proba = actions_log_probs.sum(dim=1)
            else:
                actions_joint_log_proba = actions_log_probs.sum()
            # calculate joint entropy the same way:
            if len(actions_distr_entropy.shape) > 1:
                actions_distr_entropy = actions_distr_entropy.sum(dim=1)
            else:
                actions_distr_entropy = actions_distr_entropy.sum()
        
        # here we use Dirichlet distribution, which already gives a joint log prob and entropy
        elif self.env_step_version == "newNoShort" or self.env_step_version == "newNoShort2":
            actions_joint_log_proba = actions_log_probs
            actions_distr_entropy = actions_distr_entropy
        else:
            raise NotImplementedError(f"(AgentAC) net_arch specified ({self.net_arch}) not valid.")
        # Note: our log_stdev is initialized as a vector of -0.5's.
        # but if we take the exp(-0.5), it returns a vector of 0.6065 (the standard deviations).
        # If we would initialize it like the base net_arch, as a vector of zeroes, then the std would be exp(0) = 1.
        # entropy of a Normal distribution with std 1 =~1.4189
        # https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived/1804829
        # so this will be the first entropy value we will get before any backpropagation
        # (this is good to know for debugging / to check if code works properly)
        return actions_joint_log_proba, actions_distr_entropy




