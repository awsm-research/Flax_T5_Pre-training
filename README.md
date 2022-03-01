# T5 Pre-training Using FLAX Framework
<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="logo/flax_logo.png" width="200" height="200">
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
