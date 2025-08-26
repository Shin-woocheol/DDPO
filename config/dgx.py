import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    config = base.get_config()

    config.pretrained.model = "runwayml/stable-diffusion-v1-5"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 10
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config


def incompressibility():
    config = compressibility()
    config.reward_fn = "jpeg_incompressibility"
    return config


def aesthetic():
    config = compressibility()
    config.num_epochs = 500
    config.reward_fn = "aesthetic_score"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    #* CUDA_VISIBLE_DEVICES GPU 갯수 *  batch_size * gradient_accumulation_steps = 64가 되도록 설정.
    #* accumulation이 줄어들수록 빨라짐.
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 16

    #* seed 0, 1, 2로 실험
    config.seed = 0

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config

def aesthetic_debug():
    config = compressibility()
    config.num_epochs = 200
    config.reward_fn = "aesthetic_score"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 1
    config.sample.batch_size = 1
    config.sample.num_batches_per_epoch = 1

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config

def prompt_image_alignment():
    config = compressibility()

    config.num_epochs = 200
    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 6 * 6 = 288.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 6

    # again, this one is harder to optimize, so I used (8 * 6) / (4 * 6) = 2 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 6

    # prompting
    config.prompt_fn = "nouns_activities"
    config.prompt_fn_kwargs = {
        "nouns_file": "simple_animals.txt",
        "activities_file": "activities.txt",
    }

    # rewards
    config.reward_fn = "llava_bertscore"

    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config

def hps():
    config = compressibility()

    config.num_epochs = 500
    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 6 * 6 = 288.
    #* batch size는 같게 해주고, num batches를 gradient accumualtion의 배수로 늘려주면 됨. 
    config.sample.batch_size = 4
    config.sample.num_batches_per_epoch = 32

    # again, this one is harder to optimize, so I used (8 * 6) / (4 * 6) = 2 gradient updates per epoch.
    # 만약 GPU 2개 사용한다면, 
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 32

    config.lora_rank = 32

    # prompting
    config.prompt_fn = "hps_v2_all"
    config.eval_prompt_fn = "hps_v2_all_eval"
    # rewards
    config.reward_fn = "hps"

    config.seed = 0
    config.use_eval_prompts = True

    config.save_img_freq = 40

    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config

'''
아래 제약이 main 에 있음.
    samples_per_epoch = (
        config.sample.batch_size #* 한 GPU가 생성할 이미지 갯수
        * accelerator.num_processes #* GPU 수
        * config.sample.num_batches_per_epoch #* 한 epoch에 몇번 sampling반복할지
    )
    total_train_batch_size = (
        config.train.batch_size #* 한 GPU가 한번에 학습할 이미지 갯수
        * accelerator.num_processes #* GPU 수
        * config.train.gradient_accumulation_steps #* gradient를 몇번 누적할지지
    )
    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0
'''

def get_config(name):
    return globals()[name]()
