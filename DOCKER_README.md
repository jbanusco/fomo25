# FOMO25 - SSL MultiModalUNetVAE

Pre-built Docker image for the FOMO25 Challenge with pretrained weights included.

## Quick Start

```bash
docker pull jbanusco/sslmmunetave:1.0.1
```

## What's Included

- ✅ MultiModalUNetVAE architecture
- ✅ Pretrained weights (FOMO-60K, 100 epochs)
- ✅ All dependencies pre-installed
- ✅ Ready for finetuning on downstream tasks

## Pretrained Weights Location

Inside the container:
```
/usr/src/weights/fomo25_mmunetvae_pretrained.ckpt
```

Or via environment variable:
```bash
$FOMO25_WEIGHTS_PATH
```

## Tasks Supported

1. **Task 1**: Infarct Detection (Classification)
2. **Task 2**: Meningioma Segmentation
3. **Task 3**: Brain Age Regression

## Links

- GitHub: https://github.com/jbanusco/fomo25
- Release: https://github.com/jbanusco/fomo25/releases/tag/v1.0.1
