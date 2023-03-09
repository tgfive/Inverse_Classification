import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from absl import flags, app

FLAGS = flags.FLAGS

with open(r'brazil_data/brazil_result.pkl','rb') as file:
    result_dict = pkl.load(file)
with open(r'brazil_data/processed_brazil.pkl','rb') as file:
    data_dict = pkl.load(file)

xI_ind = data_dict['xI_ind']
xD_ind = data_dict['xD_ind']

xI_obs = data_dict['test']['X'][:,xI_ind]
xD_obs = data_dict['test']['X'][:,xD_ind]

budgets = result_dict['budgets']
improv_mat = result_dict['improv_mat']
time_mat = result_dict['time_mat']

for i, b in enumerate(budgets):
    xI_est = result_dict['xI'][i]
    xI_diff = np.mean(np.absolute(xI_obs - xI_est), axis=1)
    plt.plot(xI_diff, label=b)

plt.legend(title='Budgets')
plt.title('xI: MAE for Prediction vs Observed')
plt.xlabel('Time Series')
plt.ylabel('MAE')
_ = plt.show()

for i, b in enumerate(budgets):
    xI_est = result_dict['xI'][i]
    xI_diff = np.mean(np.square(xI_obs - xI_est), axis=1)
    plt.plot(xI_diff, label=b)

plt.legend(title='Budgets')
plt.title('xI: MSE for Prediction vs Observed')
plt.xlabel('Time Series')
plt.ylabel('MSE')
_ = plt.show()

for i, b in enumerate(budgets):
    xD_est = result_dict['xD'][i]
    xD_diff = np.mean(np.absolute(xD_obs - xD_est), axis=1)
    plt.plot(xD_diff, label=b)

plt.legend(title='Budgets')
plt.title('xD: MAE for Prediction vs Observed')
plt.xlabel('Time Series')
plt.ylabel('MAE')
_ = plt.show()

for i, b in enumerate(budgets):
    xD_est = result_dict['xD'][i]
    xD_diff = np.mean(np.square(xD_obs - xD_est), axis=1)
    plt.plot(xD_diff, label=b)

plt.legend(title='Budgets')
plt.title('xD: MSE for Prediction vs Observed')
plt.xlabel('Time Series')
plt.ylabel('MSE')
_ = plt.show()

plt.plot(budgets, time_mat)
plt.yscale('log')

plt.title('Compute Time')
plt.xlabel('Budget')
plt.ylabel('CPU Time (seconds)')

_ = plt.show()

plt.plot(budgets, np.mean(improv_mat, axis=0))

plt.title('Average Loss')
plt.xlabel('Budget')
plt.ylabel('Loss')

_ = plt.show()