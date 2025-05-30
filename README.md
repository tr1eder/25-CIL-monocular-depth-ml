# ETH CIL Monocular Depth estimation

Prerequisistes

- assumes `test/test/testfiles` and `train/train/trainfiles` as subfolders
- virtual environment requirements (installed pip packages) in `venv-requirements.txt`

## Canny edge detector preprocessing (if wanted, the code does not rely on this)

Run in `canny_edge_detector.py`

```python
preprocess_images_from_file('train_list.txt', 'train/train')
preprocess_images_from_file('test_list.txt', 'test/test')
```

## Training files

- file `Mon_Depth_efficientnetb3_canny_cannyreg.py`
  - efficientnet base and using canny as 4th channel input and combined MSE and cannyreg loss
- file `Mon_Depth_midas_cannyreg.py`
  - midas base and appended 2-layer conv. with canny, BerHu and cannyreg loss
- remaining files are different versions of the final form, notably also `Mon_Depth_efficientnetb3_run_predictions_again.py` can load an efficientnetb3 model from `.pth`, ex. to run validation/tests again on an already trained version

## Running the models

- change hyperparameters inside the files and directly run them, only dependency is `canny_edge_detector.py`
- `Mon_Depth_efficientnetb3_canny_cannyreg.py` can be run from command line, can pass hyperparameters directly as command line arguments, example below

```powershell
python Mon_Depth_efficientnetb3_canny_cannyreg.py [-n 10000] [-e 10] [-lr 0.00002] [-cannyregweight 0.1] [-fff 3]
```

- `job.slurm` can be slightly adjusted to run any of the models in slurm (supports command line arguments forwarding)
- most notable hyperparameters are `TRAIN_FIRST_N`, `NUM_EPOCHS`, `LEARNING_RATE`, `CANNY_REG_WEIGHT`
- additional hyperparameter is `FIXED_FOR_FIRST_N` (enb3) or `FIXED_DEC_FOR_FIRST_N` and `FIXED_ENC_FOR_FIRST_N` (midas)
- set `NUM_WORKERS=0` if there are problems with parallelism (eg in jupyter notebooks), otherwise, 4 is a good value

## Generating compressed prediction.csv

```powershell
    python create_prediction_csv.py [-o <foldername_of_output_dir(relative_location)>]
```
