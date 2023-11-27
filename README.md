# Federated_Learning
Code for FL in Pytorch 

## Structure 
    .
    ├── main.py                   # Runs a FL training on the input arguments (run `python main.py --help`)
    ├── models.py                 # MLP model that will be trained using FedAvg. This MLP model is same as you used in Lab1.
    ├── server.py                 # Contains `ServerTrainer` class that simulates FL and performs FedAvg
    ├── client.py                 # Contains `ClientTrainer` class that that train a model for specified epochs on a specific data partition.
    ├── lr_scheduler.py           # Changes learning rate per round. Use the provided lr_scheduler
    ├── partition_data.py         # partitions  the data based on the specified data either in iid or non-iid fashion.
    ├── params.py                 # input params to the main.py
    └── execute_experiments.sh    # You will be running this script as is to get experimental results. 


## Pseudo Code for FL simulation

```
Mg <- global_model
For round_idx in {1...num_rounds}
    sampledClients <- Randomly sample "total_clients_per_round" clients from  "total_num_clients"
    local_updates = []
    for ci in sampledClients:
         Mci <- train Mg on client ci's data parition for specified local_epochs
         local_updates.append(Mci)
    Update Mg based on aggregate(local_updates) using FedAvg algorithm.
    Evaluate Mg on global test dataset
```
## Running Experiments
To run the experiments, use the command 
```
mkdir output
./execute_experiments.sh ./output
```
