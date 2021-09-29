# EGASII: Efficient Graph Architecture Search with Initial Residual and Identity Mapping

[[Paper]]()

## Overview
Recently, Graph Convolutional Network (GCN) architectures are getting deeper, and the model size and inference time of GCN are increasing. This paper proposes EGASII, which uses Neural Architecture Search (NAS) to automatically derive an efﬁcient GCN architecture. EGASII combines PDARTS and SGAS to narrow the accuracy gap between the search phase and the evaluation phase. Initial residual and identity mapping is added to the candidate operations. EGASII tries to ﬁnd the combination and connection of these candidate operations to derive an architecture with improved model size and inference time efﬁciency. As a result, the derived architecture of EGASII has less model size or inference time. That is, given the same constraints on model size or inference time, the derived architecture of EGASII has improved accuracy. For the PPI dataset, model size and inference time efﬁciency are improved. For the ModelNet dataset, model size efﬁciency is improved. Both datasets are node classification tasks under inductive setting.

<p align="center">
  <img src='./pic/overview.png' width=900>
</p>

## Requirements
* Pytorch 1.8.0
* CUDA 10.2
* torch-cluster 1.5.9
* torch-geometric 1.6.3
* torch-scatter 2.0.6
* torch-sparse 0.6.9
* torch-spline-conv 1.2.1
* numpy 1.19.2

## Citation
Please cite our paper if you find anything helpful,
```

```

## Acknowledgement
This code is modified from [SGAS](https://github.com/lightaime/sgas), and borrowed from [PDARTS](https://github.com/chenxin061/pdarts) and [GCNII](https://github.com/chennnM/GCNII).

## Contact
bryan07270@gmail.com
