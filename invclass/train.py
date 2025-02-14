from __future__ import division
from __future__ import print_function

__author__ = "Trenton Gerew and Michael T. Lash, PhD"
__copyright__ = "Copyright 2019, Michael T. Lash"
__credits__ = [None]
__license__ = "MIT"
__version__ = "1.2.0"
__maintainer__ = "Trenton Gerew"
__email__ = "tgerew@anl.gov"
__status__ = "Prototype" #"Development", "Production"

import tensorflow as tf
import numpy as np
import os
from absl import flags,app #Consistent with TF 2.0 API

from invclass.utils import load_data, load_indices, make_model, WindowGenerator


# Settings
#flags = absl.flags
FLAGS = flags.FLAGS

#core params..
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for optimizer. Optional (default: 0.01)')
flags.DEFINE_string('data_path', '', 'Path to the data. Required.')
flags.DEFINE_string('data_file', '', 'Name of the file containing the data. Required.')
flags.DEFINE_string('file_type', 'csv', 'Type of data file. Either "csv" or "pkl". Optional (default: "csv")')
flags.DEFINE_string('util_file', '', 'Name of the file containing index designations. Required.')
flags.DEFINE_string('save_file', '', 'Name of the file to save the processed data to. Optional.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train the model. Optional (default: 200)')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability). Optional (default: 0)')
flags.DEFINE_integer('hidden_units', 20, 'Number of hidden nodes in hidden layer. If 0, then logistic regression\
                      model is used. Optional (default: 10).')
flags.DEFINE_boolean('indirect_model', False, 'Whether or not we are training a model to predict the\
                     indirect features or not. Default: False')
flags.DEFINE_float('val_prop',0.10,'Proportion of dataset to use for validation. Default: 0.10')
flags.DEFINE_float('test_prop',0.10,'Proportion of dataset to use for testing. Default: 0.10')
flags.DEFINE_float('weight_decay',0.,'Weight decay on l2 regularization of model weights.')
flags.DEFINE_integer('input_width', 6, 'Number of time steps of the input window. Default: 6')
flags.DEFINE_integer('label_width', 6, 'Number of time steps of the label window. Default: 6')
flags.DEFINE_integer('shift', 6, 'Time offset between input window and label window. Default: 1')



#os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)
#GPU_MEM_FRACTION = 0.8

def log_dir():
    
    if FLAGS.indirect_model:
        mod_type = 'indirect'
    else:
        mod_type = 'reg'

    log_dir = FLAGS.data_path + "/sup-" + FLAGS.data_file.split(".")[-2]
    log_dir += "/{model_type:s}_{model_size:d}_{lr:0.4f}/".format(
            model_type=mod_type,
            model_size=FLAGS.hidden_units,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def train(data_dict):

    #Make a window
    window = WindowGenerator(
        input_width = FLAGS.input_width,
        label_width = FLAGS.label_width,
        shift = FLAGS.shift,
        data_dict = data_dict
    )
    
    print(window)

    train_dat = data_dict['train']
    val_dat = data_dict['val']
    model = make_model(data_dict,FLAGS.indirect_model)
    csv_logger = tf.keras.callbacks.CSVLogger(log_dir()+'training.log')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
    
    if FLAGS.indirect_model:
        tr_X = train_dat['X']
        val_X = val_dat['X']

        X_train = np.hstack([tr_X[:,data_dict['xU_ind']],tr_X[:,data_dict['xD_ind']]])
        X_val = np.hstack([val_X[:,data_dict['xU_ind']],val_X[:,data_dict['xD_ind']]])
      
        Y_train = tr_X[:,data_dict['xI_ind']]
        Y_val =  val_X[:,data_dict['xI_ind']]

        history = model.fit(X_train, Y_train, epochs=FLAGS.epochs, batch_size=64,
                       validation_data=(X_val,Y_val),
                       callbacks = [csv_logger, early_stopping])
        
        model.save(log_dir()+"model.h5")   

        return
    
    history = model.fit(window.train, epochs=FLAGS.epochs,
                       validation_data=window.val,
                       callbacks = [csv_logger, early_stopping])
    

    model.save(log_dir()+"model.h5")

    return

def main(argv):
    print("Loading data...")
    obs_indices,unch_indices,indir_indices,dir_indices,cost_inc,cost_dec,direct_chg,id_ind,target_ind = load_indices(FLAGS.data_path,FLAGS.util_file)
    opt_params = {'cost_inc':cost_inc,'cost_dec':cost_dec,'direct_chg':direct_chg}

    data_dict = load_data(FLAGS.data_path,FLAGS.data_file,FLAGS.file_type,obs_indices,
                          unch_indices,indir_indices,dir_indices,id_ind=id_ind,
                          target_ind=target_ind,val_prop=FLAGS.val_prop,
                          test_prop=FLAGS.test_prop,opt_params=opt_params,
                          save_file=FLAGS.save_file)
	
    print(f'Observable indices: {obs_indices}')
    print(f'Unchangeable indices: {unch_indices}')
    print(f'Indirectly changeable indices: {indir_indices}')
    print(f'Directly changeable indices: {dir_indices}')
	
    print("Done loading. Training model...")
    train(data_dict=data_dict)


if __name__ == '__main__':
    app.run(main)
