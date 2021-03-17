import tensorflow_datasets as tfds


# Create google cloud storage to save the tensorflow datasets.
STORAGE_BUCKET = 'gs://CLOUD_STORAGE_BUCKET'
data_dir = f'{STORAGE_BUCKET}/data'


# Make sure that you put ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar 
# into the cache_dir.
cache_dir = 'IMAGENET_TAR_FILE_DIR/'


ds = tfds.load("imagenet2012:5.0.0", split="train", data_dir=data_dir,
               download_and_prepare_kwargs={'download_kwargs':
                                            tfds.download.DownloadConfig(manual_dir=cache_dir)})
tfds.as_numpy(ds)