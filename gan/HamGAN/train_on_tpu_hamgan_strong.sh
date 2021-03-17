export STORAGE_BUCKET=gs://CLOUD_STORAGE_BUCKET
export TPU_NAME=YOUR-TPU-NAME
export PROJECT_ID=YOUR-PROJECT-ID
export TPU_ZONE=europe-west4-a

python3 train_experiment_main.py \
      --use_tpu=true \
      --eval_on_tpu=true \
      --use_tpu_estimator=true \
      --mode=train \
      --max_number_of_steps=1000000 \
      --train_batch_size=1024 \
      --eval_batch_size=1024 \
      --predict_batch_size=1024 \
      --num_eval_steps=49 \
      --train_steps_per_eval=1000 \
      --max_ckpts=100 \
      --tpu=$TPU_NAME \
      --gcp_project=$PROJECT_ID \
      --tpu_zone=$TPU_ZONE \
      --model_dir=$STORAGE_BUCKET/logdir/hamgan_strong \
      --data_dir=$STORAGE_BUCKET/data \
      --alsologtostderr \
      --G_module hamburger \
      --D_module hamburger \
      --G_ham_type NMF     \
      --D_ham_type NMF     \
      --G_version v1 \
      --D_version v1 \
      --G_s 1        \
      --D_s 1        \
      --G_d 256      \
      --D_d 128      \
      --G_r 32       \
      --D_r 16       \
      --G_K 6        \
      --D_K 6        \
      --G_steps 1    \
      --D_steps 1    \
      --generator_lr 0.0001 \
      --discriminator_lr 0.0004
