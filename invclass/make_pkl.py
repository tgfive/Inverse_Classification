import sys
from absl import flags,app
from invclass.utils import load_data, load_indices

flags.DEFINE_string('data_path', '', 'Path to the data. Required.')
flags.DEFINE_string('data_file', '', 'Name of the file containing the data. Required.')
flags.DEFINE_string('file_type', 'csv', 'Type of data file. Either "csv" or "pkl". Optional (default: "csv")')
flags.DEFINE_string('util_file', '', 'Name of the file containing index designations. Required.')
flags.DEFINE_string('save_file', '', 'Name of the file to save the processed data to. Optional.')
flags.DEFINE_float('val_prop',0,'Proportion of dataset to use for validation. Default: 0')
flags.DEFINE_float('test_prop',1,'Proportion of dataset to use for testing. Default: 1')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

print('Creating test data...')

obs_indices,unch_indices,indir_indices,dir_indices,cost_inc,cost_dec,direct_chg,id_ind,target_ind = load_indices(FLAGS.data_path,FLAGS.util_file)
opt_params = {'cost_inc':cost_inc,'cost_dec':cost_dec,'direct_chg':direct_chg}

data_dict = load_data(FLAGS.data_path,FLAGS.data_file,FLAGS.file_type,obs_indices,
    unch_indices,indir_indices,dir_indices,id_ind=id_ind,
    target_ind=target_ind,val_prop=FLAGS.val_prop,
    test_prop=FLAGS.test_prop,opt_params=opt_params,
    save_file=FLAGS.save_file)

print('Done creating test data.')
