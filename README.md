# ETH CIL Monocular Depth estimation

Prerequisistes

- assumes test/test/testfiles and train/train/trainfiles as subfolders

## Canny edge detector preprocessing

Run in canny_edge_detector.py

```python
preprocess_images_from_file('train_list.txt', 'train/train')
preprocess_images_from_file('test_list.txt', 'test/test')
```

## MON. DEPTH TRAINING

Run Tim_Mon_Depth.py, adjust TRAIN_FIRST_N, NUM_EPOCHS, BATCH_SIZE, USE_CANNY, NUM_WORKERS
