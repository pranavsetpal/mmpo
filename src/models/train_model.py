import torch

# Env
from src.env import create_env

# Networks
from torchrl.modules import MultiAgentMLP
from tensordict.nn import TensorDictModule

torch.manual_seed(0)

## Env
env = create_env(batch_size=10)


## Networks
policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        # n_agent_inputs=
        # n_agent_outputs=
        n_agents=env.num_vehicles,
        centralised=False,
        share_params=True,
        device=device,
        depth=2,
        num_cells=256,
        # activation_class=
    ),
    # If you need to separate the action values, uncoummet v
    # NormalParamExtractor()
)

policy_module = TensorDictModule()
