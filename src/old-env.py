import jumanji
from jumanji.environments.routing.multi_cvrp.generator import UniformRandomGenerator

from torchrl.envs.libs.jumanji import JumanjiWrapper
from torchrl.envs.transforms import TransformedEnv, ExcludeTransform
from torchrl.data.tensor_specs import CompositeSpec

def create_env(batch_size=1):
    # https://github.com/instadeepai/jumanji/blob/main/jumanji/environments/routing/multi_cvrp/env.py#L69
    env = jumanji.make(
        "MultiCVRP-v0",
        generator=UniformRandomGenerator( # Change env specs
            num_customers=20,
            num_vehicles=3
        )
        # reward_fn= # Change reward function
        # viewer= # Change render details
    )
    env = JumanjiWrapper(env, batch_size=[batch_size])
    env = TransformedEnv(
        env,
        ExcludeTransform(("full_observation_spec", "state", "nodes", "demands"))
    )

    return env

if __name__ == "__main__":
    env = create_env(batch_size=10)

    print(env.observation_spec.keys(True, True))
    observation_keys = [
        ("nodes", "coordinates"),
        ("vehicles", "coordinates"),
        ("action_mask"),
        ("state", "nodes", "coordinates"),
        ("state", "vehicles", "positions"),
        ("state", "vehicles", "distances"),
        ("state", "step_count"),
        ("state", "action_mask"),
        ("state", "key"),
    ]

    # observation_items = [
    #     (key, env.observation_spec[key])
    #     for key in observation_keys
    # ]

    observation_spec = CompositeSpec({
        key: env.observation_spec[key]
        for key in observation_keys
    })

    env.observation_spec = observation_spec

    print(observation_spec)
    print(env.observation_spec)
