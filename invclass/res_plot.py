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

print('Begin plotting...')

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


print('Plotting indirectly changeable features...')

for count, x_ind in enumerate(xI_ind):
    feat_name = header[x_ind]
    fig, ax = plt.subplots(layout='constrained', figsize=(8,5))
    ax.set_title(f'Perturbations in feature {feat_name} (indirectly changeable)')
    
    xI_feat_nan = xI_nan[:,count]
    plt.xticks(ticks=time_steps[xI_feat_nan==True], minor=True)
    ax.tick_params(axis='x', which='minor', colors='red')

    for b, bud in enumerate(budgets[1:]):
        xI_est = result_dict['xI'][b,:,count]
        xI_diff = xI_est - xI_obs[:,count]
        ax.plot(xI_diff, label=bud)

        plt.figure('histogram')
        hist = sns.histplot(data=xI_diff)
        hist.set(xlabel='Perturbation',
                 title=f'Perturbations in feature {feat_name} (indirectly changeable) with budget {bud}')
        plt.savefig(FLAGS.image_path + FLAGS.util_file[:-4] + f'_hist_feat-{x_ind}_bud-{b}')
        plt.close('histogram')
        
    ax.legend(title='Budgets', loc='center right', bbox_to_anchor=(1.15,0.5))
    ax.set_xlabel('Time Series')
    ax.set_ylabel('Perturbation')
    fig.savefig(FLAGS.image_path + FLAGS.util_file[:-4] + f'_pert_feat_{x_ind}')

    plt.close('all')

print('Plotting directly changeable features...')

for count, x_ind in enumerate(xD_ind):
    feat_name = header[x_ind]
    fig, ax = plt.subplots(layout='constrained', figsize=(8,5))
    ax.set_title(f'Perturbations in feature {feat_name} (directly changeable)')
    
    xD_feat_nan = xD_nan[:,count]
    plt.xticks(ticks=time_steps[xD_feat_nan==True], minor=True)
    ax.tick_params(axis='x', which='minor', colors='red')

    for b, bud in enumerate(budgets[1:]):
        xD_est = result_dict['xD'][b,:,count]
        xD_diff = xD_est - xD_obs[:,count]
        ax.plot(xD_diff, label=bud)

        plt.figure('histogram')
        hist = sns.histplot(data=xD_diff, bins=40)
        hist.set(xlabel='Perturbation',
                 title=f'Perturbations in feature {feat_name} (directly changeable) with budget {bud}')
        plt.savefig(FLAGS.image_path + FLAGS.util_file[:-4] + f'_hist_feat-{x_ind}_bud-{b}')
        plt.close('histogram')

    ax.legend(title='Budgets', loc='center right', bbox_to_anchor=(1.15,0.5))
    ax.set_xlabel('Time Series')
    ax.set_ylabel('Perturbation')
    fig.savefig(FLAGS.image_path + FLAGS.util_file[:-4] + f'_pert_feat_{x_ind}')

    plt.close('all')

print('PLotting compute time...')

fig, ax = plt.subplots(layout='constrained', figsize=(8,5))
ax.set_title('Compute Time')
ax.plot(budgets, time_mat)
ax.set_yscale('log')
ax.set_xlabel('Budget')
ax.set_ylabel('CPU Time (seconds)')
fig.savefig(FLAGS.image_path + FLAGS.util_file[:-4] + 'compute_time')


print('Plotting average loss...')

fig, ax = plt.subplots(layout='constrained', figsize=(8,5))
ax.set_title('Average Loss')
ax.plot(budgets, np.mean(improv_mat, axis=0))
ax.set_xlabel('Budget')
ax.set_ylabel('Loss')
fig.savefig(FLAGS.image_path + FLAGS.util_file[:-4] + '_avg_loss')

print('Plotting complete.')
