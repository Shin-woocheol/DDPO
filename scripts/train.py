from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker #* pip install -e . 해주니까 이렇게 상대경로 상관없이 from해서 쓰네.
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import torch.distributed as dist
import torchvision.utils as vutils
import random
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    now = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if not config.run_name:
        config.run_name = now
    else:
        config.run_name += "_" + now

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on #* 모델 몇 step 학습시킬지.
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps #* objective function의 expectation을 몇개 sample로 근사할지.
        * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": f"{config.run_name}_DDPO_bs{config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps}"}},
        )
    logger.info(f"\n{config}")

    if accelerator.num_processes > 1:
        now = now if accelerator.is_main_process else ""
        obj = [now]
        dist.broadcast_object_list(obj, src=0)
        now = obj[0]
    now = now

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    # freeze parameters of models to save more memory
    #! stable diffusion이 아닌 다른 model사용하면 아래 no grad해주는 부분 수정해야할듯.
    pipeline.vae.requires_grad_(False) #* stable diffusion은 latent diffusion이어서서
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora) #* LoRA사용시엔, model param tuning이 아닌 adapter layer 학습이니까.
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler #* noise scheduler 설정
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    #* precision 줄여서 메모리 절약.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    #* stable diffusion의 Unet 내부에 attention layer에 LoRA 적용. conditoinal이라서 attention 존재.
    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys(): 
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"): #* bottleneck
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"): #* feature up sample(decoder)
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"): #* feature down sample(encoder)
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            #* diffuser의 attention-layer용 LoRA 삽입모듈. attention weights들에 대해서 W = W + alpha AB같은 구조 삽입.
            lora_attn_procs[name] = LoRAAttnProcessor( 
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        pipeline.unet.set_attn_processor(lora_attn_procs) 
        #* corssattention class내에 processor를 LoRAattention processor로 바꾼 것임.
        #* optimizer에 param넘기려면 attnprocslayer라는 객채로 감싸서 prameters()뽑아줘야함
        #* 이 객체는 forward를 구현하지 않았어서 wrapper로 감싸줌. forward는 UNet을 상요하고 Parameters는 LoRA만 가져가게 하는것.

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers): #* 기존 class상속해서 추가하고 싶은 함수 추가한 것. 
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32: #* float32연산 속도 증가가
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    if config.use_eval_prompts:
        eval_prompt_fn = getattr(ddpo_pytorch.prompts, config.eval_prompt_fn)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)() #* reward function 실행시, 함수 return으로 설정되어잇어서 실행한 것.

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder( #* 뭐 stable diffusion 안의 text encoder사용법이겠지
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1) #* (batch_size, neg_prompt_embed.shape[1], neg_prompt_embed.shape[2])
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # initialize stat tracker
    if config.per_prompt_stat_tracking: #* 각 prompt별로 reward 통계 추적. prompt별로 난이도가 다르니까.
        stat_tracker = PerPromptStatTracker( #* ddpo_pytorch안에 있음.
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    #* float32 대신 float 16 또는 bfloat16을 자동으로 적용해 속도, 메모리 효율 높여주는 pytorch 기능.
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`. #* training에 필요한 모든 객체 prepare에 넣기. 그럼 분산 환경에 최적화됨.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2) #* 최대 2개의 thread를 사용할 수 있는 스레드 풀 생성
    #* 나중에 future = executor.submit()을 통해서 async로 실행할 수 있고, future.result()로 결과 받아옴.

    # Train!
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

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch), #* 한 epoch에 진행할 sampling 반복횟수수
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[
                    prompt_fn(**config.prompt_fn_kwargs) #* keyword argument가 필요한 prompt의 함수들이 있음.
                    for _ in range(config.sample.batch_size)
                ]#* 함수 return되서 나온, 각 list의 index에 들어가있는 tuple들, list unpack해서 zip의 argument로 넣어주면, tuple의 index에 맞게 zip object를 만듦듦
            )

            # encode prompts
            prompt_ids = pipeline.tokenizer( #* 사전에 정의된 id로 단어들을 mapping
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0] #* (batch_size, 77, 768) CLIP tokenizer

            # sample
            with autocast(): #* mixed precision
                images, _, latents, log_probs = pipeline_with_logprob( #* 정해진 step denoise하고, x0 이미지, 각 step에서의 latent x_t, log prob을 저장함.
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                ) #* latents : (1, 4, 64, 64)가 num_steps개

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat( #* noise scheduler상의 시간 단계계
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously #* 이걸 다른 thread에 할당함.
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata) #* 이미지에 대한 reward 계산
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[ #* xT to x1
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[ #* xT-1 to x0
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result() #* 이걸로 thread넘겼던거 결과 기다림.
            # accelerator.print(reward_metadata)
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        #* samples라는 dict로 batch 묶는거. 첫번째 sample의 key들 가져와서 돌리고, sample들 안에서 그 key에 대한 값 가져와서 붙이고 새로운 dict에는 k를 키로해서 넣음.
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.25} | {reward:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(
                            zip(prompts, rewards)
                        )  # only log rewards from process 0
                    ],
                },
                step=global_step,
            )

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode( #* prompt str로 복원
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index] #* advantage를 각 process의 숫자에 맞게 reshape한 후, accelerator를 통해서는 각 process index에 알맞게 가져가니 분배가 됨.
            .to(accelerator.device)
        )

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample.batch_size * config.sample.num_batches_per_epoch
        )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs): #* 같은 sample로 몇 번 학습시킬지.
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device) #* random 순열
            samples = {k: v[perm] for k, v in samples.items()} #* 순열에 맞춰서 dict 다시 정렬

            # shuffle along time dimension independently for each sample
            perms = torch.stack( #* time step도 random으로 바꿈.
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]: #* samples도 timestep따라서 바꿈
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            #* 여기까지 sample randomize, time step randomize하고 batch 묶어줌줌 
            # train
            pipeline.unet.train()
            info = defaultdict(list) #* 새로운 key들어오면 자동으로 value에 empty list할당.
            for i, sample in tqdm( #* 한 x0 image를 만드는 seq에 대한게 sample.
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg: #* classifier-free guidance
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps), #* 보통 50
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ): #* 각 x_t에 대해서 loss 계산산
                    with accelerator.accumulate(unet): #* gradient accumulation 이걸로 묶여있으면 자동으로 acccumulate check하고 step해줌.
                        with autocast():
                            if config.train.cfg:
                                noise_pred = unet( #* noise는 각 latent의 elem마다 존재
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) #* 첫 번쨰 차원 균등하게 나눔.
                                noise_pred = (
                                    noise_pred_uncond
                                    + config.sample.guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model #* like PPO loss
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp( #* 원래 PPO에는 없는 것이지만 안정화를 위해서 advantage도 clip하는듯.
                            sample["advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        print(f"log_prob: {log_prob.shape}, sample['log_probs'][:, j]: {sample['log_probs'][:, j].shape}")
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j]) #* current / previous
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) #* -advantage니까, maximum

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                unet.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step() #* 결국 gradient accumulation step만큼의 gradient가 쌓임. 그리고 그건 update하려는 step만큼으로 설정됨. 그래서 objective function과 같음.
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

                        #* 이곳은 update가 될 때만 들어오게 됨. 고로 이곳에서 evaluation fig저장 로직을 작성.
                        with torch.no_grad():
                            if global_step % config.save_img_freq == 0:
                                eval_save_dir = f"eval/{now}/epoch{global_step}"
                                os.makedirs(eval_save_dir, exist_ok=True)
                                if config.use_eval_prompts:
                                    eval_prompts_all = eval_prompt_fn(return_all=True)[0]
                                else:
                                    eval_prompts_all = prompt_fn(return_all=True)[0]

                                num_repeats = 32 // accelerator.num_processes
                                eval_rewards = []

                                # Fix the random seed for reproducibility
                                torch.manual_seed(config.seed + accelerator.process_index)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed_all(config.seed + accelerator.process_index)

                                for i in range(num_repeats):
                                    # seed = config.seed + accelerator.process_index + i * accelerator.num_processes
                                    # print(f"[Rank: {accelerator.process_index}, Seed: {seed}]")
                                    # torch.manual_seed(seed)
                                    # if torch.cuda.is_available():
                                    #     torch.cuda.manual_seed_all(seed)
                                    # eval_prompts_all = eval_prompts_all[:3]
                                    for i in tqdm(range(0, len(eval_prompts_all), config.train.batch_size)):
                                        batch_prompts = eval_prompts_all[i:i + config.train.batch_size]
                                        batch_prompts_metadata = {}
                                        batch_prompt_ids = pipeline.tokenizer(
                                            batch_prompts,
                                            return_tensors="pt",
                                            padding="max_length",
                                            truncation=True,
                                        ).input_ids.to(accelerator.device)
                                        batch_prompt_embeds = pipeline.text_encoder(batch_prompt_ids)[0]
                                        # sample_neg_prompt_embeds = neg_prompt_embed.repeat(
                                        #     config.train.batch_size, 1, 1
                                        # )
                                        eval_neg_prompt_embeds = train_neg_prompt_embeds[:len(batch_prompts)]
                                        with autocast():
                                            images, _, latents, log_probs = pipeline_with_logprob(
                                                pipeline,
                                                prompt_embeds=batch_prompt_embeds,
                                                negative_prompt_embeds=eval_neg_prompt_embeds,
                                                num_inference_steps=config.sample.num_steps,
                                                guidance_scale=config.sample.guidance_scale,
                                                eta=config.sample.eta,
                                                output_type="pt",
                                            )
                                        rewards = executor.submit(reward_fn, images, batch_prompts, batch_prompts_metadata)
                                        time.sleep(0)
                                        rewards, _ = rewards.result()
                                        print(f"Rewards: {rewards}")
                                        
                                        for j, (prompt, image) in enumerate(zip(batch_prompts, images)):
                                            eval_rewards.append(rewards[j].item())
                                            filename = f"{prompt}_{seed}_{rewards[j].item():.4f}.png"
                                            save_path = os.path.join(eval_save_dir, filename)
                                            vutils.save_image(image, save_path, normalize=True, value_range=(0, 1))
                                accelerator.wait_for_everyone()
                                seed = random.randint(0, 100)
                                torch.manual_seed(seed)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed_all(seed) 

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
