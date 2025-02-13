# How to install
RNAgg was tested in a Linux environment. It works with Python 3.10.2 (and CUDA 11.8 for GPU). If you use other Python versions, you may need to install the appropriate versions of the packages listed in requirements.txt.

```
git clone https://github.com/gterai/RNAgg
cd RNAgg
pip install -r requirements.txt
cd scripts
```
You are ready to use RNAgg. If you want to use a GPU, you may need to install the appropriate version of the torch package and/or CUDA 11.8.


# Training
```
python RNAgg_train.py ../examples/sample_input.txt
```
This will output RNAgg_model.pth which contains the parameters of a trained VAE.

# Generation
```
python RNAgg_generate.py 10 RNAgg_model.pth output.txt
```
This will output 10 generated RNA sequences and their secondary structure in the output.txt file.

# Training considering RNA avtivity
```
python RNAgg_train.py ../examples/sample_input.txt --act_fname ../examples/sample_act.txt
```
This will output RNAgg_model.pth which contains the parameters of a trained VAE considering
RNA activity as well as sequence and structure.

# Drawing latent space
```
python gg_get_embedding.py ../examples/sample_input.txt RNAgg_model.pth emb.pickle
```
This will convert input data into 8-dimensional latent vectors and save it as the emb.pickle file.

```
python python gg_draw_latent.py emb.pickle
```
This will convert the latent vectors to 2 dimensionals vectors with the UMAP algorithm and output them as the latent.png file.

```
python python gg_draw_latent.py emb.pickle --color_fname ../example/sample_act.txt
```
If you want to color the latent vectors, specify the color using the --color_fname option.

# File format

