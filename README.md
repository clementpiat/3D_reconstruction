# 3D Stereo Reconstruction

## Data

Data can be loaded on the [TUM computer vision group](https://vision.in.tum.de/data/datasets/3dreconstruction).

Please unzip and move the loaded folders into the data folder of this project.

## Project structure

There are three main Python files, and three notebook that can be used to execute these functions and vizualise the results.

```
.
├── X_list_bird.csv      - csv results (can be vizualised on vizualisation.ipynb)
├── X_list_pig.csv       - ^
├── X_list_pig_big.csv   - ^
├── X_candidates.json    - json candidates (can be vizualised on many_images.ipynb)
├── data
│   └── bird_data        - dataset name
│       ├── calib        - calibration data
│       │   ├── 0000.txt
│       │   ├── ...
│       │   └── 0020.txt
│       ├── images       - images
│       │   ├── 0000.ppm
│       │   ├── ...
│       │   └── 0020.ppm
│       └── silhouettes  - images silhouettes 
│           ├── 0000.pgm
│           ├── ...
│           └── 0020.pgm
├── epipolar.ipynb       - Notebook: epipolar lines and keypoints
├── many_images.ipynb    - Notebook: reconstruction using multiple images
├── reconstruct.py       - Python file on reconstruction
├── stereo.py            - Python file on stereo
├── utils.py             - Utils functions about stereo
└── vizualisation.ipynb  - Notebook: results vizualisation
```

## Loading a dataset

The `utils.read` function loads an entire dataset. It's used on the notebook (e.g on `many_images.ipynb`), and you can change the dataset using the parameter obj, e.g. `utils.read(number, obj="pig")`.