import sys
import csv
import seaborn as sns
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from absl import flags, app

flags.DEFINE_string('data_path', '', 'Path to the data file. Required.')
flags.DEFINE_string('data_file', '', 'Name of the data file. Required.')
flags.DEFINE_string('util_file', '', 'Name of the csv data file with column headers. Required.')
flags.DEFINE_string('result_file', '', 'Name of the result file. Required.')
flags.DEFINE_string('image_path', '', 'Path to the image file. Required.')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

print('Loading data...')

with open(FLAGS.data_path + FLAGS.result_file,'rb') as file:
    result_dict = pkl.load(file)
with open(FLAGS.data_path + FLAGS.data_file,'rb') as file:
    data_dict = pkl.load(file)
with open(FLAGS.data_path + FLAGS.util_file, newline='') as file:
    reader = csv.reader(file)
    header = next(reader)
del header[0]
del header[-1]

xI_ind = data_dict['xI_ind']
xD_ind = data_dict['xD_ind']

xI_obs = data_dict['test']['X'][:,xI_ind]
xD_obs = data_dict['test']['X'][:,xD_ind]

xI_nan = data_dict['test']['nan'][:,xI_ind]
xD_nan = data_dict['test']['nan'][:,xD_ind]
time_steps = np.arange(0,xI_nan.shape[0])

budgets = result_dict['budgets']
improv_mat = result_dict['improv_mat']
time_mat = result_dict['time_mat']

print('PLotting indirectly changeable features...')

for count, x_ind in enumerate(xI_ind):
	feat_name = header[x_ind]
	fig, ax = plt.subplots(layout='constrained', figsize=(8,5))
	ax.set_title(f'Time series for feature {feat_name} (indirectly changeable)')
	
	xI_feat_nan = xI_nan[:,count]
	plt.xticks(ticks=time_steps[xI_feat_nan==True], minor=True)
	ax.tick_params(axis='x', which='minor', colors='red')
	
	ax.plot(xI_obs[:,count], 'k.-', label='Observed')
	
	for b, bud in enumerate(budgets[1:]):
		ax.plot(result_dict['xI'][b,:,count], label=bud)
	
	ax.legend(title='Budgets', loc='center right', bbox_to_anchor=(1.15,0.5))
	ax.set_xlabel('Time')
	ax.set_ylabel(feat_name)
	fig.savefig(FLAGS.image_path + FLAGS.util_file[:-4] + f'_values_feat_{x_ind}')
	
	plt.close('all')


print('PLotting directly changeable features...')

for count, x_ind in enumerate(xD_ind):
	feat_name = header[x_ind]
	fig, ax = plt.subplots(layout='constrained', figsize=(8,5))
	ax.set_title(f'Time series for feature {feat_name} (directly changeable)')
	
	xD_feat_nan = xD_nan[:,count]
	plt.xticks(ticks=time_steps[xD_feat_nan==True], minor=True)
	ax.tick_params(axis='x', which='minor', colors='red')
	
	ax.plot(xD_obs[:,count], 'k.-', label='Observed')
	
	for b, bud in enumerate(budgets[1:]):
		ax.plot(result_dict['xD'][b,:,count], label=bud)
	
	ax.legend(title='Budgets', loc='center right', bbox_to_anchor=(1.15,0.5))
	ax.set_xlabel('Time')
	ax.set_ylabel(feat_name)
	fig.savefig(FLAGS.image_path + FLAGS.util_file[:-4] + f'_values_feat_{x_ind}')
	
	plt.close('all')
