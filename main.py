import pandas as pd
import matplotlib.pyplot as plt
import numpy
import warnings
import weakref
import itertools
import gc
warnings.simplefilter('ignore', numpy.RankWarning)

def regression_models(df, plot=False):
	# This function is used to train the different regression models used to predict the DAG parameters at each stage

	global SV_Exec
	global SV_Mem
	global EF_Exec
	global EF_Mem
	global CF_Exec
	global CF_Mem
	SV_Exec = numpy.poly1d(numpy.polyfit(df['Input Size'], df['SV_Exec'], 3))
	SV_Mem = numpy.poly1d(numpy.polyfit(df['Input Size'], df['SV_Mem'], 3))
	EF_Exec = numpy.poly1d(numpy.polyfit(df['Input Size'], df['EF_Exec'], 3))
	EF_Mem = numpy.poly1d(numpy.polyfit(df['Input Size'], df['EF_Mem'], 3))
	CF_Exec = numpy.poly1d(numpy.polyfit(df['Input Size'], df['CF_Exec'], 3))
	CF_Mem = numpy.poly1d(numpy.polyfit(df['Input Size'], df['CF_Mem'], 3))
	
	if plot == True:
		plot_regression(df,SV_Exec,"SV_Exec")
		plot_regression(df,SV_Mem,"SV_Mem")
		plot_regression(df,EF_Exec,"EF_Exec")
		plot_regression(df,EF_Mem,"EF_Mem")
		plot_regression(df,CF_Exec,"CF_Exec")
		plot_regression(df,CF_Mem,"CF_Mem")

def plot_regression(df, model, name):
	# This function is used to plot and visualize the regression models generated

	myline = numpy.linspace(10, 54, 100)
	plt.scatter(df['Input Size'], df[name])
	plt.plot(myline, model(myline))
	plt.show()

class VM:
	# A class which defines a virtual machine

	id_iter = itertools.count()
	def __init__(self, type, memory, compute):
		# Constructor to initialize a VM

		self.type = type
		self.id = next(self.id_iter)
		self.memory = memory
		self.compute = compute
		self.lambdas = []
		self.display = [] 

	def assign_lambda(self, lambda_props):
		# Function to assign a lambda function to the VM

		for i in lambda_props:
			id = i
			memory = lambda_props[i]

		while memory > self.memory:
			print(self.memory)
			print(memory)
			self.memory = self.memory + next(iter(self.lambdas.pop(0).values()))
		self.memory = self.memory - memory
		self.lambdas.append(lambda_props)
		self.display.append({id: abs(memory)})


	def list_lambda(self):
		# Display the lambda functions currently assigned to the VM

		print(f'VM ID : ' + str(self.id))
		print(self.display)

df = pd.read_csv('dataset.csv')

# Initialize fixed parameters
block_size = 1 # 1 MB
frame_size = 0.1 # 0.1 MB
VM_startup_time = 4 # 4 seconds
VM_bandwidth = 2	# 2 MBPS

num_jobs = input("Enter the number of jobs ")
for i in range(int(num_jobs)):
	if i < 10:
		# Trains the regression models used to determine the DAG parameters
		regression_models(df.head(i+1), plot = False)
	else:
		video_input_size = int(input("Enter the size of the video in MBs "))
		fanout_degree = int(video_input_size/block_size)

		# Calculate the DAG parameters for the Split Video stage
		sv_exec = SV_Exec.__call__(video_input_size)
		sv_mem = SV_Mem.__call__(video_input_size)

		AWS_large_1 = VM('AWS_large', 8000, 2)
		AWS_large_1.assign_lambda({'1_1': sv_mem})

		# Calculate the DAG parameters for the Extract Frame stage
		ef_exec = EF_Exec.__call__(video_input_size)
		ef_mem = EF_Mem.__call__(video_input_size)

		# Calculate the time take for different data passing methods for the Extract Frame stage
		T_VM_EF = VM_startup_time + ef_exec*(fanout_degree)
		T_Direct_Scatter_EF = block_size/(VM_bandwidth/fanout_degree) + ef_exec + VM_startup_time
		# Remote Storage - no equation

		# Choosing the appropriate data passing method and assigning the lambda functions accordingly
		if T_VM_EF < T_Direct_Scatter_EF:
			for j in range(fanout_degree):
				lambda_id = '2_'+ str(j+1)
				AWS_large_1.assign_lambda({lambda_id: ef_mem})
			print("Data Passing Method from Split Video to Extract Frame: VM-Storage")
		else:
			VMs_EF = [VM('AWS_large', 8000, 2) for j in range(fanout_degree)]
			for VM_EF, count in zip(VMs_EF, range(fanout_degree)):
				lambda_id = '2_' + str(count+1)
				VM_EF.assign_lambda({lambda_id: ef_mem})
			print("Data Passing Method from Split Video to Extract Frame: Direct-Passing")

		# Calculate the DAG parameters for the Classify Frame stage
		cf_exec = CF_Exec.__call__(video_input_size)
		cf_mem = CF_Mem.__call__(video_input_size)

		# Calculate the time take for different data passing methods for the Classify Frame stage
		T_VM_CF = VM_startup_time + cf_exec*(fanout_degree)
		T_Direct_Scatter_CF = frame_size/(VM_bandwidth/fanout_degree) + cf_exec + VM_startup_time
		# Remote Storage - no equation

		# Choosing the appropriate data passing method and assigning the lambda functions accordingly
		if T_VM_CF < T_Direct_Scatter_CF:
			for VM_EF, count in zip(VMs_EF, range(fanout_degree)):
				lambda_id = '3_' + str(count+1)
				VM_EF.assign_lambda({lambda_id: cf_mem})
			print("Data Passing Method from Extract Frame to Classify Frame: VM-Storage")
		else:
			VMs_CF = [VM('AWS_large', 8000, 2) for j in range(fanout_degree)]
			for VM_CF, count in zip(VMs_CF, range(fanout_degree)):
				lambda_id = '3_' + str(count+1)
				VM_CF.assign_lambda({lambda_id: cf_mem})
			print("Data Passing Method from Extract Frame to Classify Frame: Direct-Passing")

		print("The lambda functions assigned to each VM are: ")	
		for obj in gc.get_objects():
			if isinstance(obj, VM):
				obj.list_lambda()
				del obj