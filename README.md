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
- I also tried updating the weights __by an equal amount__ in the direction of their gradients. In other words, I took the sign of every parameter's gradient and updated the parameter by a small value (`0.001`) like so:
  `param.grad.data = torch.sign(param.grad.data) * 0.001`
  - I got decent results but didn't dig deeper into it. The weights for this model are `weights/autoquantize.ckpt`.

### Notes:
- Full precision model gives an accuracy of 98.8%
- Quantized model gives an accuracy of as high as 98.52%
  - I slightly changed the way gradients are calculated. Using mean instead of sum in lines 15 an 16, `quantification.py` gave better results:
  ```python
  w_p_grad = (a * grad_data).mean() # not (a * grad_data).sum()
  w_n_grad = (b * grad_data).mean() # not (b * grad_data).sum()
  ```
