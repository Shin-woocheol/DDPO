# config.sample.batch_size (= 8) * config.sample.num_batches_per_epoch (= 4) * config.num_gpus (=8) * config.num_epochs(=60) = 15360 reward queries
# config.train.batch_size (= 4) * config.train.gradient_accumulation_steps (= 2) * config.num_gpus (=8) = 64 batch_size

# DDPO-KL
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/train_ddpo_kl.py --config config/dgx.py:aesthetic_kl \
#     --config.seed=0 \
#     --config.kl_weight=0.04 \

# # DDPO
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/train.py --config config/dgx.py:aesthetic \
#     --config.sample.batch_size=16 \
#     --config.sample.num_batches_per_epoch=4 \
#     --config.train.batch_size=4 \
#     --config.train.gradient_accumulation_steps=4 \
#     --config.save_img_freq=40 \

# sample.batch_size * sample.num_batches_per_epoch * num_gpus = 256
# train.batch_size * train.gradient_accumulation_steps * num_gpus = 64
# then num_epochs = 500 -> 2000 steps update
# save_img_freq = 40 (40번 update 할 때마다 이미지 저장, sqdf와 동일)
# 그럼 train.batch_size = 1로 해야하나....?
# 이 세팅에서 train.batch_size = 2로 하면 31G 메모리 사용
# 이 세팅에서 train.batch_size = 4로 하면 48G 넘침

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/train.py --config config/dgx.py:aesthetic \
    --config.sample.batch_size=16 \
    --config.sample.num_batches_per_epoch=4 \
    --config.train.batch_size=2 \
    --config.train.gradient_accumulation_steps=32 \
    --config.save_img_freq=50 \