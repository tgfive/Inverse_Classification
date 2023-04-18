from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle as pkl
import csv
import tensorflow as tf

from absl import flags #Consistent with TF 2.0 API

FLAGS = flags.FLAGS


def make_model(data_dict,indirect_model):

    train_dat = data_dict['train']
    opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    #For training indirect models
    if indirect_model:
        in_dim = len(data_dict['xU_ind'])+len(data_dict['xD_ind'])        
        out_dim = len(data_dict['xI_ind'])
        model = tf.keras.models.Sequential()
        if FLAGS.hidden_units > 0:
            model.add(tf.keras.layers.Dense(FLAGS.hidden_units,input_dim=in_dim,activation='relu'))
            model.add(tf.keras.layers.Dense(out_dim,input_dim=FLAGS.hidden_units,activation='relu'))
        else:
            model.add(tf.keras.layers.Dense(out_dim,input_dim=in_dim,activation='relu'))
        
        model.compile(
            loss="mse",
            optimizer=opt,
            metrics=["mse","mae"]
        )

        return model

    #For training regular models
    out_dim = len(data_dict['xI_ind'])+len(data_dict['xD_ind'])
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(units=out_dim)
    ])

    model.compile(
        loss='mse',
        optimizer=opt,
        metrics=['mse','mae']
    )

    return model

def load_indices(data_path,util_file):
    """
        data_path: Path to data files.

        util_file: Name of the file containing the index designations, cost
                   parameters, direction of change parameters and observation boolean. Should be
                   of the form:

                        index, designation, cost increase, cost decrease, direction, observed
                      
                        e.g.:

                        0,id,,,obs
                        1,dir,0,2,-1,
                        2,dir,3,0,1,obs
                        3,dir,4,3,0,
                        4,unch,,,obs
                        5,ind,,,,
                         ...
                        p,target,,,,
    """
	
	obs_indices = []
    unch_indices = []
    ind_indices = []
    dir_indices = []
    cost_inc = []
    cost_dec = []
    direct_chg = []
    id_ind = -1
    target_ind = -1
    with open(data_path+util_file,'rU') as rF:
        fReader = csv.reader(rF,delimiter=',')
        for i, row in enumerate(fReader):
			if row[5] == 'obs':
				obs_indices.append(int(row[0]))
				
            if row[1] == 'id':
                id_ind = int(row[0])
            elif row[1] == 'target':
                target_ind = int(row[0])
            elif row[1] == 'ind':
                ind_indices.append(int(row[0]))
            elif row[1] == 'unch':
                unch_indices.append(int(row[0]))
            elif row[1] == 'dir':
                dir_indices.append(int(row[0]))
                cost_inc.append(int(row[2]))
                cost_dec.append(int(row[3]))
                direct_chg.append(int(row[4]))
            else:
                raise Exception("Problem loading index file. Unrecognized designation '{}' found on row\
                          {}".format(row[0],str(i+1)))
				
    return obs_indices, unch_indices,ind_indices,dir_indices,cost_inc,cost_dec,direct_chg,id_ind,target_ind

def load_data(data_path,data_file,file_type="csv",obs_indices=[],unchange_indices=[],indirect_indices=[],
                direct_indices=[],id_ind=0,target_ind=-1,seed=1234,val_prop=0.10,test_prop=0.10,
                opt_params={},save_file=""):

    """
        data_path: Path to the data file. The output data will be written to this 
		   location.

        data_file: File containing the data to be loaded.

        file_type: The type of file, either 'csv' or 'pkl'.

                   (1 ) 'csv' assumes the following:

                         a. Has a header and is the first line in the file.
                         b. The first column identifies the instance.
                         c. The last column is the target variable.
                         d. ALL VARIABLES ARE NUMERIC (including identifiers
                            and target).

                   (2) 'pkl' file type is assumed to have been generated
                        according to this code.
        
        obs_indices: The indices of the observable features.
        
        unchange_indices: The indices of the unchangeable features.

        indirect_indices: The indices of the indirectly changeable features.
 
        direct_indices: The indices of the directly changeable features

        seed: Seed to randomly partition data.

        val_prop: Proportion of data to be used for the validation set.

        test_prop: Proportion of data to be used for the test set.
    """

    if file_type == "pkl":
        with open(data_path+data_file,'rb') as rF:
            load_data = pkl.load(rF)
            return load_data
    
    elif file_type == "csv":
        sep=","
    else:
        raise Exception("Unsupoorted file type {}. Support file types are 'csv' and 'pkl'.".format(file_type))

    dset_df = pd.read_csv(data_path+data_file,sep=sep)
    nan_df = pd.read_csv(data_path+data_file[:-4]+'_nan.csv',sep=sep)

    header = dset_df.columns


    id_col_name = header[id_ind]
    target_col_name = header[target_ind]
    obs_col_names = header[obs_indices]
    indirect_col_names = header[indirect_indices]
    direct_col_names = header[direct_indices]
    unchange_col_names = header[unchange_indices]

    dset_ids = dset_df[id_col_name].values
    dset_targets = dset_df[target_col_name].values
    X_data = dset_df.drop([id_col_name, target_col_name],axis=1)
    nan_data = nan_df.drop([id_col_name, target_col_name],axis=1)

	obs_indices = [X_data.columns.get_loc(c) for c in obs_col_names]
    unchange_indices = [X_data.columns.get_loc(c) for c in unchange_col_names]
    indirect_indices = [X_data.columns.get_loc(c) for c in indirect_col_names]
    direct_indices = [X_data.columns.get_loc(c) for c in direct_col_names]

    X_data = X_data.values
    nan_data = nan_data.values


    #Define train, val, test indices according to test_prop, val_prop
    nfull = dset_ids.shape[0]
    train_indices = [i for i in range(int(nfull * (1 - test_prop - val_prop)))]
    val_indices = [i + int(nfull * (1 - test_prop - val_prop)) for i in range(int(nfull * val_prop))]
    test_indices = [i + int(nfull * (1 - test_prop)) for i in range(int(nfull * test_prop))]
      

    #Partition data into train,val,test according to the above defined indices
    
    #Train
    train_X = X_data[train_indices]
    train_nan = nan_data[train_indices]
    train_target = dset_targets[train_indices]
    train_ids = dset_ids[train_indices]

    if test_prop != 1:
        #Obtain normalization values
        min_X = np.amin(train_X,axis=0)
        max_X = np.amax(train_X,axis=0) 

        #Normalize training data
        norm_train_X = np.divide(train_X - min_X,max_X - min_X)
    else:
        norm_train_X = np.array([])
   
    train_dict = {"X":norm_train_X, "nan":train_nan, "target":train_target, "ids":train_ids}

    #Val
    val_X= X_data[val_indices]
    val_nan = nan_data[val_indices]
    val_target = dset_targets[val_indices]
    val_ids = dset_ids[val_indices]
    
    if test_prop != 1:
        #Normailze validation data
        norm_val_X = np.divide(val_X - min_X,max_X - min_X)
    else:
        norm_val_X = np.array([])

    val_dict = {"X":norm_val_X, "nan":val_nan, "target":val_target, "ids":val_ids}

    #Test
    test_X = X_data[test_indices]
    test_nan = nan_data[test_indices]
    test_target = dset_targets[test_indices]
    test_ids = dset_ids[test_indices]
    
    if test_prop == 1:
        min_X = np.amin(test_X,axis=0)
        max_X = np.amax(test_X,axis=0)

    #Normalize test data
    norm_test_X = np.divide(test_X - min_X,max_X - min_X)

    test_dict = {"X":norm_test_X, "nan":test_nan, "target":test_target, "ids":test_ids}


    return_dict = {'train':train_dict,
                   'val':val_dict,
                   'test':test_dict,
                   'train_indices':train_indices,
                   'val_indices':val_indices,
                   'test_indices':test_indices,
                   'min_X':min_X,
                   'max_X':max_X,
                   'opt_params':opt_params,
                   'xO_ind':obs_indices,
                   'xU_ind':unchange_indices,
                   'xI_ind':indirect_indices,
                   'xD_ind':direct_indices
                   }

    #If a save file is defined, write the defined data out.
    if save_file != "":
        with open(data_path+save_file,'wb') as sF:
            pkl.dump(return_dict,sF)

    return return_dict

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, data_dict):

        self.train_dat = data_dict['train']['X']
        self.val_dat = data_dict['val']['X']
        self.test_dat = data_dict['test']['X']

        self.label_column_indices = data_dict['xI_ind'] + data_dict['xD_ind']
        self.column_indices = data_dict['xU_ind'] + data_dict['xI_ind'] + data_dict['xD_ind']

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift 

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column indices: {self.label_column_indices}'
        ])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]
        labels = tf.stack(
            [labels[:, :, self.column_indices[i]] for i in self.label_column_indices],
            axis=-1
        )

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def make_dataset(self, data):
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=data.shape[0]
        )

        ds = ds.map(self.split_window)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_dat)

    @property
    def val(self):
        return self.make_dataset(self.val_dat)
    
    @property
    def test(self):
        return self.make_dataset(self.test_dat)
