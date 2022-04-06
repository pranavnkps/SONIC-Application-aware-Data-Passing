# Sonic: Application-Aware Data Passing

This work is based on the paper "Sonic: Application-aware Data Passing for Chained Serverless Applications" and is done as part of the course assignment for the Cloud Computing (CS466) course at National Institute of Technology, Karnataka.
https://www.usenix.org/conference/atc21/presentation/mahgoub

## Team Members
1. Arnav Nair (181CO209)
2. Dhanwin Rao (181CO217)
3. Pranav Kumar (181CO239)

## Implementation
The implementation shows the working of SONIC with respect to the Video Analytics applications. The Video Analytics application has three stages:
1. Split Video
2. Extract Frame
3. Classify Frame

The function of SONIC is to place the lambda functions associated with each of these stages in the appropriate VMs and choose the data passing methods which minimizes the execution time. The data passing method chosen can either be VM-Storage, Direct-Passing or Remote-Storage.

For the specific case of the Video Analytics application, SONIC takes as input the size the of the video, the number of stages as well as the details of each stage.
SONIC uses regression models at each stage of the application to predict the different DAG parameters such as execution time, memory footprint, intermediate size and fanout degree. The regression models are trained using the information gathered during the first 10 runs on different input sizes. The generation and training of these models are done in "Regression Models.ipynb".

"main.py" is the driver program. It takes video input size from the user. The regression models trained are used to predict the DAG parameters for the each stage. These value are then used to predict the appropriate data passing method along with the right placement of the different lambda function.
