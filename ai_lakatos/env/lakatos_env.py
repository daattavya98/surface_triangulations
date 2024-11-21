from gymnasium.spaces.space import Space
from pettingzoo import ParallelEnv
from beartype import beartype


class LakatosEnv(ParallelEnv):
    metadata = {
        "name": "lakatos_env_v0",
    }

    def __init__(self) -> None:
        pass

    @beartype
    def reset(self, seed=None, options=None) -> None:
        pass

    @beartype
    def step(self, actions) -> None:
        pass

    @beartype
    def render(self) -> None:
        pass

    @beartype
    def observation_space(self, agent) -> Space:
        return self().observation_space[agent]

    @beartype
    def action_space(self, agent) -> Space:
        return self().action_space[agent]
