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

import numpy as np
import pickle as pkl
import tensorflow as tf
import copy
from absl import flags #Consistent with TF 2.0 API

FLAGS = flags.FLAGS

def obj_fun(model, inputs, labels):
    prediction = model(inputs)
    observed = tf.cast(labels, tf.float32)

    loss = tf.norm(prediction[:,-1,:] - observed[:,-1,:], ord='euclidean')
    
    return loss

def inv_gradient(model, inputs, labels):
    if type(inputs) == np.ndarray:
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    with tf.GradientTape() as t:
        t.watch(inputs)

        loss = obj_fun(model, inputs, labels)

        grads = t.gradient(loss,inputs).numpy()

        return grads
    
def inv_gradient_ind(model, x, num_loss=0):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    #print('x_tensor:', x_tensor)

    grad = []
    for i in range(num_loss):
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            #print('output:', model(x_tensor))

            loss = model(x_tensor)[:,i]
            #print('loss:', loss)
            grad.append(t.gradient(loss,x_tensor).numpy()[0])
    #print('grad:', np.array(grad))
    return np.array(grad)

def set_parameters(data_dict):
    
    opt_params = data_dict['opt_params']

    param_dict = {}
    
    index_dict = {'xU_ind': data_dict['xU_ind'], 'xI_ind': data_dict['xI_ind'], 
                 'xD_ind':data_dict['xD_ind'], 
                 'xD_ind_ind': [i for i in range(len(data_dict['xU_ind']),
                               len(data_dict['xU_ind'])+len(data_dict['xD_ind']))]}
    param_dict['inds'] = index_dict

    if FLAGS.budget_start - FLAGS.budget_end == 0:
        budgets = [FLAGS.budget_start]
    else:
        budgets = [i for i in np.arange(FLAGS.budget_start,FLAGS.budget_end+0.0001,
                  FLAGS.budget_interval)]

    param_dict['budgets'] = budgets

    param_dict['c+'] = opt_params['cost_inc']
    param_dict['c-'] = opt_params['cost_dec']
    param_dict['d'] = opt_params['direct_chg'] 
    
    return param_dict

def save_result(result_dict):

    direct = FLAGS.data_path
    prefix_data = FLAGS.data_file.split(".")[0]
    save_name = direct+prefix_data+"-"+FLAGS.save_file
    with open(save_name, 'wb') as sF:
        pkl.dump(result_dict,sF)
    return

def set_bounds(inputs,grads,param_dict):
    fd = []
    fc = []
    fl = []
    fu = []

    for x in inputs:
        index_dict = param_dict['inds']
        d = param_dict['d']
        txD = x[index_dict['xD_ind']]
        tl = np.minimum(np.zeros((len(index_dict['xD_ind']),)),x[index_dict['xD_ind']])
        tu = np.maximum(np.ones((len(index_dict['xD_ind']),)),x[index_dict['xD_ind']])
        c = np.zeros((len(param_dict['c-']),))
        pos_d = np.where(d >np.zeros((len(index_dict['xD_ind']),)))
        neg_d = np.where(d < np.zeros((len(index_dict['xD_ind']),)))
        ambig_d = np.where(d == np.zeros((len(index_dict['xD_ind']),)))
        c[pos_d] = np.array(param_dict['c+'])[pos_d]
        c[neg_d] = np.array(param_dict['c-'])[neg_d]
        if len(ambig_d[0]) >0:
            ambig_d = ambig_d[0]
            pos_amb_d = ambig_d[np.where(grads[ambig_d] > np.zeros((len(ambig_d),)))]
            neg_amb_d = ambig_d[np.where(grads[ambig_d] < np.zeros((len(ambig_d),)))]
            c[pos_amb_d] = np.array(param_dict['c+'])[pos_amb_d]
            c[neg_amb_d] = np.array(param_dict['c-'])[neg_amb_d]
            d = np.array(d)
            d[pos_amb_d] = np.ones((len(pos_amb_d),))
            d[neg_amb_d] = -1*np.ones((len(neg_amb_d),))
        u = copy.deepcopy(tu)
        l = copy.deepcopy(tl)
        pos_d = np.where(d >np.zeros((len(index_dict['xD_ind']),)))
        neg_d = np.where(d < np.zeros((len(index_dict['xD_ind']),)))
        u[neg_d] = txD[neg_d] - tl[neg_d] #Can only decrease
        u[pos_d] = tu[pos_d] - txD[pos_d]

        fd.append(d)
        fc.append(c)
        fl.append(l)
        fu.append(u)

    fd = np.array(fd)
    fc = np.array(fc)
    fl = np.array(fl)
    fu = np.array(fu)

    return fd, fc, fl, fu
