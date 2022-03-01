# T5 Pre-training Using FLAX Framework
<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="logo/jax_logo.png" width="300" height="200">
  </a>
  <h3 align="center">T5 Pre-training Using FLAX</h3>
  <p align="center">
  </p>
</p>

<!-- Table of contents -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#how-to-set-up-your-environment">How to set up your environment</a>
    </li>
    <li>
      <a href="#how-to-pre-train-a-t5-model-using-transformers-and-flax">How to pre-train a T5 model using Transformers and Flax</a>
    </li>
  </ol>
</details>

## How to set up your environment
### First, install the "jax", "jaxlib" properly by running the following commands in your own conda environment:
```python
pip install --upgrade jax jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
Note. you can check all available versions [here](https://storage.googleapis.com/jax-releases/jax_releases.html)

### Second, install "flax" by running the following commands in your own conda environment:
```python
pip install --user flax
```
  
### Third, install "tensorflow" by running the following commands in your own conda environment:
```python
pip install --user tensorflow
```

### Forth, install "transformers" by running the following commands in your own conda environment:
```python
pip install --user transformers
```

## How to pre-train a T5 model using Transformers and Flax

### Step 1, cd to pre-training dir:
```python
cd Flax_T5_Pre-training/transformers/examples/flax/language-modeling
```

### Step 2, make dir to save your pre-trained model
```python 
mkdir pretrained_model
```

### Step 3, run the following commands to start pre-training:
```python
python run_t5_mlm_flax.py --output_dir="./pretrained_model" \
                          --train_file="./data/train.txt" \
                          --validation_file="./data/val.txt" \
                          --model_type="t5" \
                          --model_name_or_path="Salesforce/CodeT5" \
                          --config_name="Salesforce/CodeT5" \
                          --tokenizer_name="Salesforce/CodeT5" \
                          --from_pt \
                          --max_seq_length="512" \
                          --per_device_train_batch_size="8" \
                          --per_device_eval_batch_size="8" \
                          --adafactor \
                          --learning_rate="0.005" \
                          --weight_decay="0.001" \
                          --warmup_steps="2000" \
                          --overwrite_output_dir \
                          --logging_steps="500" \                            
                          --save_steps="10000" \
                          --eval_steps="2500"
```

### Important Note for Pre-training setting
#### The pre-training setting above is the default setting provided by authors of Transformers library. 
#### Please modify to fit your needs. 
**--model_name_or_path** / **--config_name** / **--tokenizer_name**

These parameters are related to the model checkpoint used to initialize your T5 model to be pre-trained, this can either be a local path or the model provided on the API provided by Huggingface Team.
#### If your checkpoint model is a flax model, please change "--from_pt" to "--from_flax"
#### By default, the script accepts a checkpoint model in PyTorch format.
