from math import prod
from src.param import n_customers, n_vehicles

# https://github.com/instadeepai/jumanji/blob/main/jumanji/environments/routing/multi_cvrp/env.py#L69
import jumanji
from jumanji.environments.routing.multi_cvrp.generator import UniformRandomGenerator
from src.reward import DistanceReward

from torchrl.envs.libs.jumanji import JumanjiWrapper
from torchrl.envs.transforms import TransformedEnv, Compose, CatTensors, FlattenObservation, UnsqueezeTransform, ExcludeTransform
from torchrl.envs.utils import check_env_specs


def create_env(batch_size=1, seed=None):
    global n_customers, n_vehicles

    generator = UniformRandomGenerator(
        num_customers=n_customers,
        num_vehicles=n_vehicles
    )
    map_max = generator._map_max

    env = jumanji.make(
        "MultiCVRP-v0",
        generator=generator,
        reward_fn=DistanceReward(n_customers, n_vehicles, map_max)
    )

    env = TransformedEnv(
        JumanjiWrapper(env, batch_size=[batch_size]),
        Compose(
            CatTensors(
                in_keys=[("nodes", "coordinates"), ("vehicles", "coordinates")],
                out_key=("agent", "observation"),
                dim=1,
            ),
            FlattenObservation(
                in_keys=[("agent", "observation")],
                first_dim=-2,
                last_dim=-1
            ),
            UnsqueezeTransform(in_keys=[("agent", "observation")], unsqueeze_dim=-2),
            CatTensors(
                in_keys=[("agent", "observation")],
                out_key=("agents", "observation"),
                del_keys=False
            ),
            *[ CatTensors(
                in_keys=[("agent", "observation"), ("agents", "observation")],
                out_key=("agents", "observation"),
                dim=1,
                del_keys=False
            ) for _ in range(n_vehicles-1) ],
            ExcludeTransform( ("agent", "observation") )
        )
    )

    if seed is not None:
        env.set_seed(seed)

    check_env_specs(env)
    return env


if __name__ == "__main__":
    env = create_env(batch_size=10)
