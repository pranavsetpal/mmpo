import torch
from tensordict.nn import TensorDictModule

from math import prod

from src.param import n_agents, frames_per_batch, total_frames, minibatch_size, lr
from src.env import create_env
from src.models.utils import DoneTransform

from torchrl.data import OneHotDiscreteTensorSpec
from torchrl.modules import MultiAgentMLP, QValueActor, ProbabilisticActor, TanhNormal, AdditiveGaussianWrapper, ValueOperator

from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.objectives import DDPGLoss

# from src.param import n_agents, frames_per_batch, total_frames, minibatch_size, gamma, lr, epochs, max_grad_norm


torch.manual_seed(0)
device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"

env = create_env(seed=0)


# ## Policy
actor_net = MultiAgentMLP(
    n_agent_inputs=prod(tuple(env.observation_spec["agents", "observation"].shape[2:])),
    n_agent_outputs=env.observation_spec["action_mask"].shape[-1],
    n_agents=n_agents,
    centralised=False,
    share_params=False,
    device=device,
    depth=2,
    num_cells=256,
)
policy_module = TensorDictModule(
    actor_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "param")]
)
policy = QValueActor(
    module=policy_module,
    spec=OneHotDiscreteTensorSpec(env.observation_spec["action_mask"].shape[-1]),
    action_value_key=("agents", "param"),
    action_mask_key=("action_mask")
    )
# policy = ProbabilisticActor(
#     module=policy_module,
#     spec=env.action_spec,
#     in_keys=[("agents", "param")],
#     out_keys=[env.action_key],
#     distribution_class=TanhNormal,
#     distribution_kwargs={
#         "min": env.action_spec.space.low,
#         "max": env.action_spec.space.high
#     },
# )
policy_explore = AdditiveGaussianWrapper(
    policy,
    annealing_num_steps=total_frames // 2, # steps to explore; epsilson shift from 1.0 to 0.1
    action_key=env.action_key
)


## Critic
value_net = MultiAgentMLP(
    n_agent_inputs=prod(tuple(env.observation_spec["agents", "observation"].shape[2:])) + env.action_spec.shape[-1],
    n_agent_outputs=1,
    n_agents=n_agents,
    centralised=True,
    share_params=True,
    device=device,
    depth=2,
    num_cells=256
)
value_module = ValueOperator(
    module=value_net,
    in_keys=[("agents", "param")],
    out_keys=[("agents", "state_action_value")]
)


## Collectors
collector = SyncDataCollector(
    create_env_fn=env,
    policy=policy_explore,
    device=device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys) # idk what this does
)

# replay_buffer = TensorDictReplayBuffer(
#     storage=LazyTensorStorage(frames_per_batch, device=device),
#     sampler=SamplerWithoutReplacement(),
#     batch_size=minibatch_size
# )

## Loss
loss_module = DDPGLoss(
    actor_network=policy,
    value_network=value_module
)
loss_module.set_keys(
    state_action_value=("agents", "state_action_value"),
    reward=env.reward_key,
    done=("done"),
    terminated=("terminated")
)

optim = torch.optim.Adam(loss_module.parameters(), lr)


# ## Training
# episode_reward_mean_list = []
# # for tensordict_data in collector:
# #     current_frames = tensordict_data.numel()
# #     total_frames += current_frames
# #
# #     data_view = tensordict_data.reshape(-1)
# #     replay_buffer.extend(data_view)
# #
# #     for _ in range(epochs):
# #         for _ in range(frames_per_batch // minibatch_size):
# #             subdata = replay_buffer.sample()
# #             loss_vals = loss_module(subdata)
# #
# #             loss_value = loss_vals["loss_actor"] + loss_vals["loss_value"]
# #
# #             loss_value.backward()
# #
# #             total_norm = torch.nn.utils.clip_grad_norm_(
# #                 loss_module.parameters(),
# #                 max_grad_norm
# #             )
# #             optim.step()
# #             optim.zero_grad()
# #
# #         # policy_explore.step(frames=current_frames) #TODO
# #         collector.update_policy_weights_()
