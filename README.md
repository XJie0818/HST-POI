# HST-POI
HST-POI: Joint Hypergraph and Phased Spatio-Temporal Learning for Next POI Recommendation

### Requirements
To run this model, ensure you have Python 3.9 installed.
```shell
pip install -r requirement.txt
```

## Running
To run experiments on the Gowalla dataset, you must first generate the required hypergraph structure:

Friendship Hypergraph: Run friend2id.py followed by friendgraph.py.


After generating both graphs, you can train the model with default hyperparameters using:
```
python ./model/run.py --dataset GW --dim 8
python ./model/run.py --dataset MP --dim 8
python ./model/run.py --dataset TC --dim 16
```
