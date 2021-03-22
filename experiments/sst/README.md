# Scripts for the STS (Spatial Time Series) Benchmark

- Use `last_frame_prediction.py` to evaluate the naive baseline which uses the last frame to predict all the future frames.
It's a good example of how to test your own models using the offline setting of the benchmark.

- Use `sst_main.py` to train the RNN models for precipitation nowcasting and use `sst_rnn_test.py` to test these models. The training scripts support multiple GPUs.
If you find you cannot train the model using a single GPU, try to decrease the batch_size.

    1. Commands for training the RNN models:
    ```
    # Train ConvGRU model with B-MSE + B-MAE
    python3 sst_main.py --cfg configurations/convgru_55_55_33_1_64_1_192_1_192_b4.yml --save_dir convgru_55_55_33_1_64_1_192_1_192_b4 --ctx gpu0,gpu1
    # Train TrajGRU model with B-MSE + B-MAE
    python3 sst_main.py --cfg configurations/trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml --save_dir trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4 --ctx gpu0,gpu1
    # Train ConvGRU model without B-MSE + B-MAE
    python3 sst_main.py --cfg configurations/convgru_55_55_33_1_64_1_192_1_192_nobal_b4.yml --save_dir convgru_55_55_33_1_64_1_192_1_192_nobal_b4 --ctx gpu0,gpu1
    ```

- Use `deconvolution.py` to run the 2D/3D models.

    1. Commands for trainnig the CNN models:
    ```
    # Train Conv2D model
    python deconvolution.py --cfg configurations/conv2d_3d/conv2d.yml --save_dir Conv2D --ctx gpu0
    # Train Conv3D model
    python deconvolution.py --cfg configurations/conv2d_3d/conv2d.yml --save_dir Conv3D --ctx gpu0
    ```

# Test with Pretrained Models

```
# Test ConvGRU model in the offline setting
python sst_rnn_test.py \
 --cfg ConvGRU/cfg0.yml \
 --load_dir ConvGRU \
 --load_iter 49999 \
 --finetune 0 \
 --ctx gpu0 \
 --save_dir ConvGRU \
 --mode fixed \
 --dataset test

# Test ConvGRU model in the online setting
python sst_rnn_test.py \
 --cfg ConvGRU/cfg0.yml \
 --load_dir ConvGRU \
 --load_iter 49999 \
 --finetune 1 \
 --lr 1E-4 \
 --ctx gpu0 \
 --save_dir ConvGRU \
 --mode online \
 --dataset test

# Test TrajGRU model in the offline setting
python sst_rnn_test.py \
 --cfg TrajGRU/cfg0.yml \
 --load_dir TrajGRU \
 --load_iter 79999 \
 --finetune 0 \
 --ctx gpu0 \
 --save_dir TrajGRU \
 --mode fixed \
 --dataset test

# Test TrajGRU model in the online setting
python sst_rnn_test.py \
 --cfg TrajGRU/cfg0.yml \
 --load_dir TrajGRU \
 --load_iter 79999 \
 --finetune 1 \
 --lr 1E-4 \
 --ctx gpu0 \
 --save_dir TrajGRU \
 --mode online \
 --dataset test

# Test ConvGRU model without balanced loss
python sst_rnn_test.py \
 --cfg ConvGRU-nobal/cfg0.yml \
 --load_dir ConvGRU-nobal \
 --load_iter 59999 \
 --finetune 0 \
 --ctx gpu0 \
 --save_dir ConvGRU-nobal \
 --mode fixed \
 --dataset test

# Test ConvGRU model without balanced loss
python sst_rnn_test.py \
 --cfg ConvGRU-nobal/cfg0.yml \
 --load_dir ConvGRU-nobal \
 --load_iter 59999 \
 --finetune 1 \
 --lr 1E-4 \
 --ctx gpu0 \
 --save_dir ConvGRU-nobal \
 --mode online \
 --dataset test
```
