# Ternary quantization
Training models with ternary quantized weights. PyTorch implementation of https://arxiv.org/abs/1612.01064

### Work in progress
- [x] Train MNIST model in original format (float32)
- [x] Train MNIST model with quantized weights
- [x] Add training logs
- [ ] Analyze quantized weights
- [ ] Quantize weights keeping w_p and w_n fixed

### Repo Guide
- A simple model (`model_full`) defined in `model.py` was trained on MNIST data using full precision weights. The trained weight is stored as `weights/original.ckpt`.
  - Code for training can be found under `main_original.py`.
- A copy of the above model (loaded with trained weights) was created (`model_to_quantify`) and was trained using quantization. The trained weight is stored as `weights/quantized.ckpt`.
  - Code for training can be found under `main_ternary.py`. The logs can be found inside the file `logs/quantized_wp_wn_trainable.txt`.

### Notes:
- Full precision model gives an accuracy of 98.8%
- Quantized model gives an accuracy of 97.8%.
  - However the training was not stable and the model would stop learning a couple of times.
  - I got good results by using a very small learning rate (`0.00001`) for updating scaling parameters and full precision weights. I also slightly changed the way gradients are calculated. Using mean instead of sum in lines 15 an 16, `quantification.py` gave better results:
  ```
  w_p_grad = (a * grad_data).mean() # not (a * grad_data).sum()
  w_n_grad = (b * grad_data).mean() # not (b * grad_data).sum()
  ```
