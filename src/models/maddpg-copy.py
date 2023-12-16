import torch

from math import prod

from src.env import create_env
from src.param import n_vehicles, frames_per_batch, total_frmaes, minibatch_size, gamma, lr, epochs, max_grad_norm

from torchrl.modules import MultiAgentMLP


torch.manual_seed(0)
device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"

env = create_env()


## Policy
actor_net = MultiAgentMLP(
    n_agent_inputs=prod(tuple(env.observation_spec["agents", "observation"].shape[1:])),
    n_agent_outputs=1,
    n_agents=n_vehicles,
    centralised=False,
    share_params=False,
    device=device,
    depth=2,
    num_cells=256
)
policy_module = TensorDictModule(
    actor_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "param")]
)

print(actor_net)


## Critic
critic_net = MultiAgentMLP(
    n_agent_inputs=prod(tuple(env.observation_spec["agents", "observation"].shape[1:])) + env.action_spec.shape[-1],
    n_agent_ouputs=1,
    n_agents=env.n_agents,
    centralised=True,
    share_params=True,
    device=device,
    depth=2,
    num_cells=256
)
critic_module = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "param")],
    out_keys=[("agents", "state_action_value")]
)


## Collectors
collector = SyncDataCollector(
    env,
    policy,#_explore
    device=device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size
)

## Loss
loss_module = DDPGLoss(
    actor_network=actor, #TODO
    value_network=value_module #THIS IS CRITIC????
)

loss_module.set_keys(
    state_action_value=("agents", "state_action_value"),
    reward=env.reward_key,
    # done=("done"),
    # terminated=("terminated")
)

optim = torch.optim.Adam(loss_module.parameters(), lr)


## Training
episode_reward_mean_list = []
for tensordict_data in collector:
    current_frames = tensordict_data.numel()
    total_frames += current_frames

    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    for _ in range(epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = loss_vals["loss_actor"] + loss_vals["loss_value"]

            loss_value.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(),
                max_grad_norm
            )
            optim.step()
            optim.zero_grad()

        # policy_explore.step(frames=current_frames) #TODO
        collector.update_policy_weights_()
