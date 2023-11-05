1. `store()` 是 ema 的 `collected_params` 存储 model 权重

2. `copy_to()` 是 将 ema 的 buffer 赋值给 model 的权重。buffer 在 init 中初始化，在 forward 中更新

3. `restore()` 是 将 ema 的 `collected_params` 赋值给 model 的权重。

三者依次顺序的意思是，先存 model 的权重是训练时的权重，copy ema 的buffer得到平滑的权重，这时比如存储ckpt，然后恢复训练的权重来继续训练。



`only_model=False` 表示没有 ema