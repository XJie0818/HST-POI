**HST-POI: Joint Hypergraph and Phased Spatio-Temporal Learning for Next POI Recommendation**

In this paper, we propose a Joint Hypergraph Learning and Phased Spatial-Temporal Learning model (HST-POI) for the next POI recommendation. HST-POI integrates hypergraph-based global preference learning, ST-Former-driven phase-aware spatio-temporal modeling and visit time prediction to fuse multi-source context , substantially enhancing the robustness and adaptability of Next POI Recommendation.


&#8226;  We design a **hypergraph learning module** that constructs two distinct hypergraphs to comprehensively model user preferences from personal historical, social, and transitional perspectives. 

&#8226;  We propose the **ST-Former Encoder**, which uses window attention to capture phased mobility patterns and explicit spatio-temporal interaction layers to model the interdependence between spatial and temporal features.

&#8226;  We develop a **visit time prediction module** that leverages multi-head attention to forecast the next check-in time, providing valuable contextual information for the subsequent POI recommendation.


The overall framework of our proposed HST-POI model is illustrated in Figure 1.

<div align="center">
  <figure>
    <img width="1152" height="934" alt="主图" src="https://github.com/user-attachments/assets/e4c3f754-2ee8-4469-a771-79fb00c2aa31" />
    <figcaption style="margin-top: 10px; color: #666;">
      <strong style="color: #333;">Figure 1.The overall framework of the proposed HST-POI</strong><br>
    </figcaption>
  </figure>
</div>

### Requirements
To run this model, ensure you have Python 3.9 installed.
```shell
pip install -r requirement.txt
```

## Running
To run experiments on the Gowalla dataset, you must first generate the required hypergraph structure:

**Friendship Hypergraph**: Run friend2id.py followed by friendgraph.py.


After generating both graphs, you can train the model with default hyperparameters using:
```
python ./model/run.py --dataset GW --dim 8
python ./model/run.py --dataset MP --dim 8
python ./model/run.py --dataset TC --dim 16
```
