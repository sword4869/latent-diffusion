## requirements and modify 

**用原本的 `environment.yaml` 不行**。

### from pytorch\_lightning.utilities.distributed import rank\_zero\_only

- [ldm/models/diffusion/ddpm.py#l19](./ldm/models/diffusion/ddpm.py#L19), [github](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/diffusion/ddpm.py#L19)
- [main.py#l17](./main.py#L17), [github](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/main.py#L17)
```
File "/home/lab/project/project/latent-diffusion/ldm/models/diffusion/ddpm.py", line 19, in <module>
from pytorch_lightning.utilities.distributed import rank_zero_only
ModuleNotFoundError: No module named 'pytorch_lightning.utilities.distributed'
```


```
- from pytorch_lightning.utilities.distributed import rank_zero_only
+ from pytorch_lightning.utilities.rank_zero import rank_zero_only
```

## ImportError: cannot import name 'VectorQuantizer2 from 'taming.modules.vqvae.quantize'

- [ldm/models/autoencoder.py#l6](./ldm/models/autoencoder.py#L6), [github](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/autoencoder.py#L6)
```
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
ImportError: cannot import name 'VectorQuantizer2' from 'taming.modules.vqvae.quantize'
```

原因：新的[taming](https://so.csdn.net/so/search?q=taming&spm=1001.2101.3001.7020)版本不存在VectorQuantizer2


```
- from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
+ from taming.modules.vqvae.quantize import VectorQuantizer
```

## 2\. parser = Trainer.add\_argparse\_args(parser)

问题详细描述：

```
parser = Trainer.add_argparse_args(parser)
AttributeError: type object 'Trainer' has no attribute 'add_argparse_args'
```

原因：因为**pytorch-lightening 2.x**已经不兼容1.x。

解决方案：将 **pytorch-lightning**降级, 可以尝试

```
pip install pytorch-lightning==1.9.4
```