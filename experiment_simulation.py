'''
Code based on the MATLAB implementation from: https://github.com/sjgershm/RL-tutorial
'''
import numpy as np
from numpy import matlib
from construct_stimulus import construct_stimulus
from TD import TD
from construct_CSC import construct_CSC
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os


results_path = './Results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)


class stimulus_params:
	def __init__(self, trial_length, onset, dur):
		self.trial_length = trial_length
		self.onset = onset # stimulus onset
		self.dur = dur # stimulus duration


def simulate_experiment(nTrials, trial_length, type_plot):
	# construct stimuli
	s = construct_stimulus(stimulus_params(trial_length, 5, 2))  # A -> + ; predictive stimulus

	x = construct_CSC(s) # -> complete serial compound (CSC) representation of the predictive stimulus
	X = matlib.repmat(x, nTrials + 1, 1) 
	# nTrials for training; +1 Trial for testing one of the following situations:
	# reward omission (or), early reward (er), delayed reward (dr)

	r = construct_stimulus(stimulus_params(trial_length, 6, 1))  # -> reward stimulus (only for training)

	if type_plot ==  'or':
	    # omission reward experiment
	    r = np.concatenate((np.matlib.repmat(r,nTrials,1), np.zeros((trial_length, 1))), axis=0)
	    # The last trial is with reward omission -- this is why we concatenate a 0 vector of rewards   
	elif type_plot == 'er':
	    # early reward experiment
	    reward_early = construct_stimulus(stimulus_params(trial_length, 2, 1));
	    r = np.concatenate((np.matlib.repmat(r,nTrials,1), reward_early), axis=0)
	else: 
	    # delayed reward experiment
	    reward_delayed = construct_stimulus(stimulus_params(trial_length, 8, 1));
	    r = np.concatenate((np.matlib.repmat(r,nTrials,1), reward_delayed), axis=0)

	# run TD model
	model = TD(X, r, None)

	# TD error
	dt = [model[i].dt for i in range(len(model))]
	dt = np.asarray(dt)
	dt = np.reshape(dt, (trial_length, nTrials + 1), order="F").T

	sns.set_style("darkgrid")
	ax = sns.lineplot(data=dt[0, :])
	ax.set_title('Before learning')
	plt.plot()
	file_name = 'before_learning'
	plt.savefig(os.path.join(results_path, file_name + '.png'))
	plt.show()

	ax = sns.lineplot(data=dt[-2, :])
	title = 'After learning (' + str(nTrials) + ' trials)'
	ax.set_title(title)
	plt.plot()
	file_name = 'after_learning_' + str(nTrials)
	plt.savefig(os.path.join(results_path, file_name + '.png'))	
	plt.show()


	if type_plot == 'or':
		# plot omission response
		ax = sns.lineplot(data=dt[-1, :])
		title = 'Reward omission (' + str(nTrials) + ' trials)'
		ax.set_title(title)
		plt.plot()
		file_name = 'reward_omission_' + str(nTrials)
		plt.savefig(os.path.join(results_path, file_name + '.png'))
		plt.show()
	elif type_plot == 'er':
	    # plot early reward
		ax = sns.lineplot(data=dt[-1, :])
		title = 'Early reward (' + str(nTrials) + ' trials)'
		ax.set_title(title)
		plt.plot()
		file_name = 'early_reward_' + str(nTrials)
		plt.savefig(os.path.join(results_path, file_name + '.png'))
		plt.show()
	else: 
	    # plot delayed reward
		ax = sns.lineplot(data=dt[-1, :])
		title = 'Delayed reward (' + str(nTrials) + ' trials)'
		ax.set_title(title)
		plt.plot()
		file_name = 'delayed_reward_' + str(nTrials)
		plt.savefig(os.path.join(results_path, file_name + '.png'))
		plt.show()

	# Value function	
	V = [model[i].V for i in range(len(model))]
	V = np.asarray(V)
	V = np.reshape(V, (trial_length, nTrials + 1), order="F").T


simulate_experiment(5, 10, 'or')
simulate_experiment(5, 10, 'er') 
simulate_experiment(5, 10, 'dr')
