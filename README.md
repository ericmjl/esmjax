# esmjax

This repository provides a reimplementation of the 650M parameter ESM-1b protein language model originally introduced in [Rives et. al. (2021)](https://www.pnas.org/content/118/15/e2016239118). The original implementation was written in PyTorch, which you can find [here](https://github.com/facebookresearch/esm) and based on the fairseq framework. This implementation is written in Haiku/JAX, with the model weights ported over from the original PyTorch implementation. Since this effectively is pure JAX, all the standard features (jit, pmap, multi-TPU training with ease) come for free.

## Feature Generation
Right now, this implementation is inference only, aimed at feature generation, and there's two types of features you can generate from this that may be of interest:

* **Per residue/whole sequence embeddings:** You can directly generate embeddings for the final transformer layer (that's layer 33, producing 1280 dimensional embeddings), which you can then use per-residue, or average them to get a whole protein embedding. For example, here's a subset of protein domains in the all-alpha helice folding class in [SCOPe](https://scop.berkeley.edu/) embedded using the model, then projected down to 3D space using [TriMap](https://github.com/eamid/trimap)

![Features](../images/embeddings.png?raw=true)

(This plot is interactive, and you can find it [here](https://htmlpreview.github.io/?https://github.com/irhum/esm-jax/blob/main/notebooks/embeddings_vis.html))

* **Per head attention weights:** You can also generate the per-head attention weights for each of the 20 heads of the 33 self-attention layers in the model for downstream tasks. One provided with this implementation is the weights for the contact prediction head, which uses the 660 values per residue pair to compute 1 value between 0 and 1, indicating whether the residues are in contact in 3D space. (Note: d1n3ya_ is the same protein used in Fig. 5 of the original ESM paper)

![Contacts](../images/contactpred.jpg?raw=true)

## What's here

### Weights
You can download the weights need to run the model [here](https://drive.google.com/file/d/1bM_Lt8Tbkf3njWC4nnLcCsAmS7a5Vc0_/view?usp=sharing).

### Notebooks
There's three notebooks in this repository (in the `/notebooks` folder) to help you get started:

* `inference.ipynb`: This is the first one you should go through. Covers how to load in the weights and obtain a fully working version of the model, and shows how to convert a protein sequence into embeddings and the attention weights.

* `embeddings.ipynb`: This dives deeper into embeddings, by going through a full pipeline from loading in a FASTA file, tokenizing it, computing the embeddings and then visualizing them by projecting them down with TriMap.

* `weight_porting.ipynb`: A more "optional" notebook, that shows the scripts used to convert the PyTorch weights into a Haiku-ready format.

### Requirements
* `jax` - JAX's installation differs on whether you need hardware support or not, so follow the instructions [here](https://github.com/google/jax#installation)

* `haiku` - You can install `haiku` from [here](https://github.com/deepmind/dm-haiku)
* `h5py` - Needed for model saving/loading, installation instructions [here](https://docs.h5py.org/en/stable/build.html).
### Remarks
This repository exists independently of that of the original authors; I personally just found the model very fascinating, and wanted to dig deeper. Figured it may be of interest to others too! (There's still a bit more to do, primarily with packaging, but it's already capable of generating protein embeddings on acceleration hardware)

Access to TPUs was generously provided through the [TPU Research Cloud](https://sites.research.google/trc/about/). 
