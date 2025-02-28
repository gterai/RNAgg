# How to install
RNAgg has been tested in a **Linux environment** running on an **Intel64 (x86_64)** architecture. It is compatible with **Python 3.10**.

## Installation Instructions
To install RNAgg, please follow these steps:
```
git clone https://github.com/gterai/RNAgg # Clone the repository
cd RNAgg                                  # Navigate to the RNAgg directory
pip install -r requirements.txt           # Install the required dependencies
cd scripts                                # Navigate to the scripts directory
```
Optionally, you can use GPU acceleration. However, the requirements.txt file was created for CUDA 11.8. The above instructions will work with GPU support if you use CUDA 11.8.

## Using Different Environments
If you plan to use **a different operating system, Python version, or CUDA version**, you may need to install appropriate package versions that are compatible with your environment. Below is a list of key dependencies required for RNAgg:
```
numpy
torch
matplotlib
joblib
umap-learn
```
We recommend ensuring that the versions of these libraries are compatible with your system configuration to avoid potential issues.

# Training
```
python RNAgg_train.py ../example/sample_input.txt
```
This will create the model_RNAgg.pth file which contains the parameters of a trained VAE.

# Generation
```
python RNAgg_generate.py 10 model_RNAgg.pth output.txt
```
This will output 10 generated RNA sequences and their secondary structures in the output.txt file.

# Training considering RNA avtivity
```
python RNAgg_train.py ../example/sample_input.txt --act_fname ../example/sample_act.txt
```
This will output model_RNAgg.pth which contains the parameters of a trained VAE considering
RNA activity as well as sequence and structure.

# Drawing latent space
```
python gg_get_embedding.py ../example/sample_input.txt model_RNAgg.pth emb.pickle
```
This will convert input data into 8-dimensional latent vectors and save it as the emb.pickle file.

```
python gg_draw_latent.py emb.pickle
```
This will convert the latent vectors to 2 dimensionals vectors with the UMAP algorithm and output them as the latent.png file.

```
python gg_draw_latent.py emb.pickle --color_fname ../example/sample_act.txt
```
If you want to color the latent vectors, specify the color using the --color_fname option.

# File format
The input file (sample_input.txt) has three columns separeted by a white space, where the first, second, and third column are identifier, RNA sequence and secondary structure, respectively (for example, see the sample_input.txt file).
The format of output file is the same as the input file. The activity file has two columns separeted by a white space, where the first and second column are identifier and activity value, respectively.

# Reference
Terai G and Asai K. Deep generative model of RNAs based on variational autoencoder with context free grammar. submitted. 
