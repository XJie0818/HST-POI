<img width="1152" height="934" alt="主图" src="https://github.com/user-attachments/assets/6bcc5cde-545c-460b-96a9-a7594a6e848b" /># HST-POI
HST-POI: Joint Hypergraph and Phased Spatio-Temporal Learning for Next POI Recommendation

<img width="1152" height="934" alt="主图" src="https://github.com/user-attachments/assets/e4c3f754-2ee8-4469-a771-79fb00c2aa31" />

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
