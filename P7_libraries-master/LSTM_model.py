# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:01:28 2018

@author: Tobias
"""


from keras.models import Model
from keras.layers import Input, LSTM, concatenate
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers import Dense
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from utilities import createFolder, return_to_root
from forecast_dataframe import fc_df
import pandas as pd
import numpy as np
from datetime import time as d_time
from datetime import datetime

from recurrentshop import RecurrentModel, LSTMCell

class SPP_LSTM():
    """
    Class designed creating various LSTM designs to forecast SPP.
    """
    def __init__(self,output_dim,input_dim = None,loss = 'mean_squared_error',
                 optm = 'RMS',
                 LSTM_args = {'activation': None, 
                              'recurrent_activation': 'hard_sigmoid'},
                 learning_rate = 1e-2, 
                 LSTM_output_dim = None,mode = 'vanilla',Tan = False,
                 DO = False,DTs = False, Tan_args = None, DO_args = None,
                 DTs_args = None,sLSTM_args = None):
        """
        Initialises LSTM model and compiles it.
        
        Currently only supports 1 layer LSTM 
        
        Parameters
        ----------
        
        input_dim : int or None
            Length of the vector used for input. If P_X_fnn is enabled this 
            arugment is ignored.
        output_dim: int
            Length of the vector used for output
        loss : str
            String for Keras loss function, defaults to mean_squared_error
        activation : str
            Keras keyword for activation function. Defaults to 'tanh'
        learning_rate : float
            Initial learning rate in gradient decent. Defaults to 10^(-2)
        LSTM_output_dim: int
            Used to set costum LSTM_output dim if dense out is True
        mode: str
            'vanilla': Plain LSTM model with no added depth
            'DTs': Dense transition shortcut mode
            'sLSTM': Stacked LSTM model 
        Tan/DO/DTs/sLSTM_args : dict or None
            See ?_process_input/_DO/DTs/sLSTM for more info on args
        **lstmargs : keyword arguments
            Additional parameters for LSTM layer
            
        """        
        self.M = input_dim
        self.output_dim = output_dim 
        #If dense output is True, LSTM output_dim can be different than
        #output_dim
        if LSTM_output_dim is None:
            LSTM_output_dim = self.output_dim
        self.LSTM_output_dim = LSTM_output_dim
        self.callbacks = None
        
        #Firstly setup input, if prompted fnn will make P_X from GHI
        if Tan:
            if Tan_args is None:
                raise(ValueError("In P_X_fnn mode, arguments for dense layers should be specified in dict as {'GHI_dim': x, 'SPP_dim' : y,'kwargs':**kwargs}"))
            self.inputs,self.inputs_processed = \
            self._process_input(**Tan_args)
            #Updating M can be nescceary
            self.M = int(self.inputs_processed.shape[2])
        
        else: #No processing done to input
            self.inputs = Input(shape = (None,input_dim))
            self.inputs_processed = self.inputs
        
        #Setup main part of RNN using different combinations of LSTM setups
        if mode == 'vanilla': #Plain LSTM
            self._LSTM_vanilla(**LSTM_args)
        
        elif mode == 'DTs': #Deep Transition shortcut
            self._DTs(**DTs_args)
        
        elif mode == 'sLSTM': #Stacked LSTM
            self._sLSTM(**sLSTM_args)
            
        #If prompted output from LSTM will go thorugh ff network (Deep output)
        if DO:
            self.prediction = self._DO(**DO_args)
        
        else: #If no deep output, prediction will be from LSTM
            self.prediction = self.LSTM_output
        
        #Now define model based on inputs and outputs
        self.model = Model(inputs = self.inputs, outputs = self.prediction)
        
        #RMS propergation is recconmended for recurrent networks
        if optm == 'RMS':
            optimizer = RMSprop(lr=learning_rate)
        elif optm == 'SGD': # SGD learningrate should be lower than that of RMSprop
            optimizer = SGD(lr=learning_rate)
        else:
            print('Choose a valid optimization algorithm')
        self.model.compile(loss=loss, optimizer=optimizer)
        
    def _LSTM_vanilla(self,activation = None,
                      recurrent_activation = 'hard_sigmoid',
                      **lstmargs):
        """
        Builds vanilla (shallow) LSTM and saves in instance. 
        """
        self.LSTM_output = LSTM(units = self.LSTM_output_dim,
                            return_sequences = True,
                            input_shape = (None,self.M),
                            recurrent_activation = recurrent_activation,
                            activation = activation,
                            **lstmargs)(self.inputs_processed)

    def _sLSTM(self, activation = 'tanh', recurrent_activation = 'sigmoid',
               last_activation = None,depth = 4, direct_flow = True,
               **lstmargs):
        """
        Build a stacked LSTM of various depth. If direct_flow == True,
        input at each layer (except first) will be last state and input, 
        else will only be fed as first layer. 
        """
        #Setup first layer only have normal input
        out = LSTM(units=self.M,
                   return_sequences=True,
                   input_shape=(None, self.M),
                   recurrent_activation=recurrent_activation,
                   activation=activation,
                   **lstmargs)(self.inputs_processed)
        
        #Then setup intermediate layers
        #First and last layer is counted (therefore "n - 2")
        if direct_flow:
            stacked_dim = 2*self.M
        else:
            stacked_dim = self.M
        
        for _ in range(depth-2): 
            #Concatenate output from last LSTM-cell with input at time t
            if direct_flow:
                in_p = concatenate([self.inputs_processed,out])
                
            else:
                in_p = out
            out = LSTM(units=self.M,
                       input_shape = (None,stacked_dim),
                       return_sequences=True,
                       recurrent_activation=recurrent_activation,
                       activation=activation,
                       **lstmargs)(in_p)
        
        #Lastly setup the last layer which output dimention matching to prompted      
        if direct_flow:
            in_p = concatenate([self.inputs_processed,out])
        else:
            in_p = out
        self.LSTM_output= LSTM(units=self.LSTM_output_dim,
                               input_shape = (None,stacked_dim),
                               return_sequences=True,
                               recurrent_activation=last_activation,
                               activation=activation,
                               **lstmargs)(in_p)
                    

    def _process_input(self,GHI_size,SPP_size,N_layers = 5,activation = 'relu',
                    last_activation = None,
                    dense_output_dim = None,**kwargs):
        """
        Make radiation model from GHI using a feed forward neuaral network.
        Then returns GHI concatenated with SPP
        Needs GHI and SPP size in order to work
        """
        x_GHI = Input(shape = (None,GHI_size),name = "GHI_input")
        x_SPP = Input(shape = (None,SPP_size),name = "SPP_input")

        
        #Set output dim
        if dense_output_dim is None:
            dense_output_dim = GHI_size
        
        
        P_X_fnn = self._add_fnn(x_GHI,input_dim = GHI_size,
                                output_dim = dense_output_dim,
                                N_layers = N_layers,activation = activation,
                                last_activation = last_activation,**kwargs)
        
        x_processed = concatenate([x_SPP,P_X_fnn])
        
        return([x_SPP,x_GHI],x_processed)
    
    def _DO(self,N_layers = 5,activation = 'relu',**kwargs):
        """
        Adds deep layer to LSTM output 
        """
        dense_out = self._add_fnn(self.LSTM_output,self.LSTM_output_dim,
                                  self.output_dim,N_layers = N_layers, 
                                  activation = activation,**kwargs)
        return(dense_out)
    
    def _DTs(self,**DTs_args):
        """
        Builds DTs and saves in instance. 
        """
        LSTM_c = self._DT_LSTM(self.M,self.output_dim,**DTs_args)
        self.LSTM_output = LSTM_c(self.inputs_processed)
        
    
    def _add_fnn(self,f_input,input_dim,output_dim,N_layers = 1,
                   activation = 'relu',last_activation = None,**kwargs):
        """
        As part of Keras functional model - adds dense layers to input to build
        an fnn. All layers except the last have the same activation function. 
        """
        out = f_input
        for _ in range(N_layers-1):
            out = Dense(input_dim,activation = activation,**kwargs)(out)
        out = Dense(output_dim,activation = last_activation,**kwargs)(out)
        return(out)
    
    def _DT_LSTM(self,input_dim,output_dim,
                 recurrent_activation = 'hard_sigmoid',activation = None,
                 dense_input = False,
                 dense_states = True,
                 inp_depth =  3, 
                 inp_activation =  'hard_sigmoid',
                 input_dim_mode = 0,
                 inp_dense_states = ['h','c'],
                 st_depth =  3, 
                 st_activation = 'hard_sigmoid',
                 st_dense_states = ['h','c'],
                 **lstm_args):
        """
        Defines an LSTM cell with a dense transition under different settings.
        
        parameters
        ----------
        dense_state_mode (in args) : int
            0: Send both c and h through dense layers
            1: Send only h through dense layers
            2: Send only c thorugh dense layers
        inp_* : various
            arguments starting with inp_* are argument to dense input 
            transition
        st_* : various
            arguments starting with st_* are argument to dense state 
            transition
            
                
        """
        #Set input
        inputs = Input((input_dim,))
        h_tm1 = Input((output_dim,)) #time minus 1
        c_tm1 = Input((output_dim,)) #time minus 1
        
        #Pass input and previous states through dense layers
        if dense_input:
            x_d = self._set_DTs_dense_input(inputs,input_dim,
                                            self.output_dim,
                                            input_dim_mode,
                                            h_tm1,c_tm1,
                                            inp_depth,
                                            inp_dense_states,
                                            inp_activation)
        else:
            x_d = inputs
        
        #Pass states through dense layers based on argument
        if dense_states:
            if 'h' in st_dense_states:
                h_tm1_d = self._add_fnn(h_tm1,output_dim,output_dim,st_depth,
                   activation = st_activation,last_activation = st_activation)
            else:
                h_tm1_d = h_tm1
            if 'c' in st_dense_states:
                c_tm1_d = self._add_fnn(c_tm1,output_dim,output_dim,st_depth,
                   activation = st_activation,last_activation = st_activation)
            else:
                c_tm1_d = c_tm1
        else:
            h_tm1_d = h_tm1
            c_tm1_d = c_tm1
            
        #Then pass through LSTM cell
        lstm_output, state1_t, state2_t = LSTMCell(output_dim,
                                                   activation = activation,
                                recurrent_activation = recurrent_activation)\
                                                   ([x_d,h_tm1_d,c_tm1_d])    
        #Initialise model
        LSTM_layer = RecurrentModel(input=inputs,
                                    output=lstm_output, 
                                    initial_states = [h_tm1,c_tm1], 
                                    final_states = [state1_t, state2_t],
                                    output_length = input_dim)
        return(LSTM_layer)
    
    def _set_DTs_dense_input(self,inputs,input_dim,output_dim,input_dim_mode,
                             h_tm1,c_tm1,depth,dense_states,activation):
        """
        Setup dense layer with previous states and input and concatenates them.
        """
        concat = [inputs]
        #Chose input passed thorugh
        if 'h' in dense_states:
            concat.append(h_tm1)
        if 'c' in dense_states:
            concat.append(c_tm1)
        xhc_concat = concatenate(concat)
        #There are some choises for input dimention to LSMT cell
        #Either have dim(x) + dim(c) + d(h)
        if input_dim_mode == 0: 
            LSTM_input_dim = input_dim + 2*output_dim
        #Or have dim(c) = dim(h)
        elif input_dim_mode == 1: 
            LSTM_input_dim = output_dim
        else:
            raise(ValueError("Chose input_dim mode to:\n0: keep concatenated dimention for LSTM_input\n1:reduce LSTM_input dimention to output dimention"))
        x_d = self._add_fnn(xhc_concat,
                            input_dim + 2*output_dim,
                            LSTM_input_dim,
                            N_layers = depth,
                            activation = activation,
                            last_activation = activation)
        return(x_d)

    def set_callbacks(self,checkpoint = False,early_stop = False,
                      reduce_lr = True, tns_board = False,
                      checkpoint_args = {'verbose': 0,
                                        'save_weights_only':True,
                                        'mode': 'min'}, #Load model with lowest loss after fitting
                      early_stop_args = {'min_delta': 0, #minimum value countes as improvement
                                         'patience': 10, #How long to wait if no improbement is seen
                                         'verbose':0},
                      reduce_lr_args = {'factor' : 0.1,#How much to reduce to lr
                                        'min_lr': 1e-4, #mimimum learning rate
                                        'patience': 3, #How long to wait if no improvement is seen
                                        'verbose': 0}):
        """
        Setup various callback functions that are called during training.
        Currently ModelCheckpoint, EarlyStopping and ReduceLROnPlateau
        are available. At default ModelCheckpoint and ReduceLROnPlateau
        is called during training. 
        
        Se the keras documentation for more info on these callback methods. 
        
        Note: 'min_delta' = float(data.SPP_out_avg/10) is an alternetive to zero
        """
        
        #Create folder for model check points if it does not exist
        root = return_to_root()
        file_path =  root + "scripts/keras_temp_files/"
        checkpoint_path = file_path + "mod_check_points/"
        
        #Create unique path for TensorBoard logs
        now = datetime.now()
        tns_board_path = file_path + "logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        createFolder(checkpoint_path)
        
        #Collect callbacks in list
        callbacks = []
        if checkpoint:
            raise(NotImplementedError("Modelcheckpoint currently not available due to bug"))
            callbacks.append(ModelCheckpoint(filepath=checkpoint_path,
                                             **checkpoint_args))
        if early_stop: 
            callbacks.append(EarlyStopping(**early_stop_args))
        if reduce_lr:
            callbacks.append(ReduceLROnPlateau(**reduce_lr_args))
        
        if tns_board:
            callbacks.append(TensorBoard(log_dir=tns_board_path))
        self.callbacks = callbacks
    
    def fit(self,batch_generator,epochs,steps_pr_epoc,verbose = 0,
            **kwargs):
        """
        Fits the model given batch generator and number of epocs
        
        Parameters
        ----------
        batch_generator : generator
            Data generator from the setup class 
        epochs : int
            Number of epochs to fit the data on
        steps_per_epoch : int
            Number of batches trained on in each epoc, should corrospond to 
            the number of steps_per_epoch in batch generator
        **kwargs : keyword arguments
            Additional keyword arguments for keras fit_generator function
        """
        #Set default callbacks if none are given.
        if self.callbacks == None:
            self.set_callbacks()
            
        #Fit using keras fi_generator function. 
        self.history = self.model.fit_generator(generator = batch_generator,
                                                steps_per_epoch = steps_pr_epoc,
                                                epochs = epochs,
                                                verbose = verbose,
                                                callbacks = self.callbacks,
                                                **kwargs)
    
    def __call__(self,tns_in,return_df = True,horizon = d_time(1),
                 muni_out = None,index = None):
        """
        Calls the model on some input data from the setup class. Returns target
        as dataframe
        
        Parameters
        ----------
        
        tns_in : numpy tensor
            Input data setup via the RNN_data_setup lib as tensor
        return_df : bool
            If True, return time series dataframe. Else return tensor. Note
            needs index parameter for this to be handled
        muni_out : None / list
            If list is given, columns names are added to outputted dataframe
        
        """
        res = self.model.predict(tns_in)
        if return_df and (not index is None): 
            if muni_out is None:
                print("Warning: Muni_names not given")
                muni_out = range(res.shape[1])
            #Setup dataframe to with with fc_df type
            #Idx is shifted by horizon
            idx = index - pd.Timedelta(hours = horizon.hour,
                                              minutes = horizon.minute)
            #Columns are multi index with muni name and horizon
            h_repeat = np.repeat(horizon,len(muni_out))
            col_multi = list(zip(*(muni_out,h_repeat)))
            pd_lab = pd.MultiIndex.from_tuples(col_multi,
                                               names=['Muni','Horizon'])
            N = res.shape[0]
            df = pd.DataFrame(res.reshape(N,self.output_dim),index = idx,
                              columns = pd_lab)
            df.index.name = "Time"
            return(fc_df(df))
        else:
            raise(ValueError("Index arugment should be given as pandas timeseriesindex"))
            
        return(tns_in)