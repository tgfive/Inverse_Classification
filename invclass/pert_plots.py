import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from absl import flags, app

FLAGS = flags.FLAGS

DATA_PATH = '../brazil_data/'
IMAGE_PATH = '../documentation/'

with open(DATA_PATH+'brazil_result.pkl','rb') as file:
    result_dict = pkl.load(file)
with open(DATA_PATH+'processed_brazil.pkl','rb') as file:
    data_dict = pkl.load(file)

xI_ind = data_dict['xI_ind']
xD_ind = data_dict['xD_ind']

xI_obs = data_dict['test']['X'][:,xI_ind]
xD_obs = data_dict['test']['X'][:,xD_ind]

budgets = result_dict['budgets']
improv_mat = result_dict['improv_mat']
time_mat = result_dict['time_mat']

for count, x_ind in enumerate(xI_ind):
    fig, ax = plt.subplots(layout='constrained', figsize=(8,5))
    ax.set_title(f'Perturbations in feature {x_ind}')

    for b, bud in enumerate(budgets[1:]):
        xI_est = result_dict['xI'][b,:,count]
        xI_diff = xI_est - xI_obs[:,count]
        ax.plot(xI_diff, label=bud)

    ax.legend(title='Budgets', loc='center right', bbox_to_anchor=(1.15,0.5))
    ax.set_xlabel('Time Series')
    ax.set_ylabel('Perturbation')
    fig.savefig(IMAGE_PATH + f'pert_feat_{x_ind}')

for count, x_ind in enumerate(xD_ind):
    fig, ax = plt.subplots(layout='constrained', figsize=(8,5))
    ax.set_title(f'Perturbations in feature {x_ind}')

    for b, bud in enumerate(budgets[1:]):
        xD_est = result_dict['xD'][b,:,count]
        xD_diff = xD_est - xD_obs[:,count]
        ax.plot(xD_diff, label=bud)

    ax.legend(title='Budgets', loc='center right', bbox_to_anchor=(1.15,0.5))
    ax.set_xlabel('Time Series')
    ax.set_ylabel('Perturbation')
    fig.savefig(IMAGE_PATH + f'pert_feat_{x_ind}')
