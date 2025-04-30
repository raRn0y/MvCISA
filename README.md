# MvCISA

This is the code for paper Contrastive Independent Subspace Analysis Network for Multi-view Spatial Information Extraction.

Tengyu Zhang, Deyu Zeng, Wei Liu, Zongze Wu, Chris Ding, Xiaopin Zhong,
Contrastive independent subspace analysis network for multi-view spatial information extraction,
Neural Networks,
Volume 185,
2025,
107105,
ISSN 0893-6080,
https://doi.org/10.1016/j.neunet.2024.107105.
(https://www.sciencedirect.com/science/article/pii/S0893608024010347)
Abstract: Multi-view classification integrates features from different views to optimize classification performance. Most of the existing works typically utilize semantic information to achieve view fusion but neglect the spatial information of data itself, which accommodates data representation with correlation information and is proven to be an essential aspect. Thus robust independent subspace analysis network, optimized by sparse and soft orthogonal optimization, is first proposed to extract the latent spatial information of multi-view data with subspace bases. Building on this, a novel contrastive independent subspace analysis framework for multi-view classification is developed to further optimize from spatial perspective. Specifically, contrastive subspace optimization separates the subspaces, thereby enhancing their representational capacity. Whilst contrastive fusion optimization aims at building cross-view subspace correlations and forms a non overlapping data representation. In k-fold validation experiments, MvCISA achieved state-of-the-art accuracies of 76.95%, 98.50%, 93.33% and 88.24% on four benchmark multi-view datasets, significantly outperforming the second-best method by 8.57%, 0.25%, 1.66% and 5.96% in accuracy. And visualization experiments demonstrate the effectiveness of the subspace and feature space optimization, also indicating their promising potential for other downstream tasks. Our code is available at https://github.com/raRn0y/MvCISA.
Keywords: Multi-view classification; Subspace learning; Contrastive learning; Data representation


If this code helps you, please cite:
'''
@article{ZHANG2025107105,
title = {Contrastive independent subspace analysis network for multi-view spatial information extraction},
journal = {Neural Networks},
volume = {185},
pages = {107105},
year = {2025},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.107105},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024010347},
author = {Tengyu Zhang and Deyu Zeng and Wei Liu and Zongze Wu and Chris Ding and Xiaopin Zhong},
keywords = {Multi-view classification, Subspace learning, Contrastive learning, Data representation},
abstract = {Multi-view classification integrates features from different views to optimize classification performance. Most of the existing works typically utilize semantic information to achieve view fusion but neglect the spatial information of data itself, which accommodates data representation with correlation information and is proven to be an essential aspect. Thus robust independent subspace analysis network, optimized by sparse and soft orthogonal optimization, is first proposed to extract the latent spatial information of multi-view data with subspace bases. Building on this, a novel contrastive independent subspace analysis framework for multi-view classification is developed to further optimize from spatial perspective. Specifically, contrastive subspace optimization separates the subspaces, thereby enhancing their representational capacity. Whilst contrastive fusion optimization aims at building cross-view subspace correlations and forms a non overlapping data representation. In k-fold validation experiments, MvCISA achieved state-of-the-art accuracies of 76.95%, 98.50%, 93.33% and 88.24% on four benchmark multi-view datasets, significantly outperforming the second-best method by 8.57%, 0.25%, 1.66% and 5.96% in accuracy. And visualization experiments demonstrate the effectiveness of the subspace and feature space optimization, also indicating their promising potential for other downstream tasks. Our code is available at https://github.com/raRn0y/MvCISA.}
}
'''
