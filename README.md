## Performance research of image-gs

This is an unofficial variant of image-gs, where I add new features to measure performance of image-gs. For official description refer to original: https://github.com/NYU-ICL/image-gs.

This is WORK IN PROGRESS and may not fully work and isn't documented (yet).

### How to run

Guide below is meant for `runpod.io`, because this is what I use.

- Create a new pod. Newest GPUs like RTX 5090 may not work (they are too new). It should work for `RTX 4090` or `RTX A6000 ADA` (cheap, works fine)
- Open the (jupyter) notebook
- Open terminal
- Import project using `git clone https://github.com/MichaelEight/image-gs`
- Exit terminal
- In the file explorer enter `image-gs` directory
- Open `quick-start.ipynb`
- Start executing all cells
- While it is downloading and installing, import your images. Open `input/` directory and drag-and-drop your images
- Then look for configuration step (probably step 3):

```py
from quick_start import set_config

config = set_config(
    input_filenames="cat.png",  # or ["cat.png", "dog.png"] for multiple
    gaussians=[1000],
    steps=[2000],
    use_progressive=False,
    init_gaussian_file=None,  # or "output/session_1/cat-5000-3500/model.pt"
    allow_partial=False
)
```

Modify the input_filenames to your image or images name e.g. `dog.png` or `["cat.png", "dog.png"]`. Images may be of any type, but PNG is recommended (because image-gs has potential to "compress" image while keeping its quality).

Modify number gaussians and steps. It's an array, because you can do e.g. `gaussians=[1000, 2000, 3000]` and `steps=[1000, 2000]`. This will cause the script to rerun training for all combinations of image-gaussians-steps.

Each of the trained models will be placed into `output/` directory. You'll find there rendered image, trained model, logs and training metrics. Do what you want with that.

## Disclaimer

Claude AI was used for some quick features and refactors. This was done to quickly explore options and test ideas. There should be no breaking changes to the core code of image-gs, aside for logging some additional data. That being said, at this point on time (as long as this disclaimer is in place) the code may contain _some_ errors and differences to original code and shouldn't be used as a base for any official research. Still, you can use it to test it out yourself and have fun with it.
