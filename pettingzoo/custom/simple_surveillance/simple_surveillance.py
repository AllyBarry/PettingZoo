import numpy as np
import pygame
from gymnasium.utils import EzPickle
from pettingzoo.custom._custom_utils.core import Agent, Landmark, TargetRegion, World
from pettingzoo.custom._custom_utils.scenario import BaseScenario
from pettingzoo.custom._custom_utils.simple_env import SimpleEnvironment, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

# Colours
MID_ORANGE = (255, 153, 0)
MID_GREEN = (0, 204, 68)
MID_BLUE = (51, 153, 255)
MID_GREY = (153, 153, 153)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
AGENT_COLOURS = [MID_ORANGE, MID_GREEN, MID_BLUE]


class raw_env(SimpleEnvironment, EzPickle):
    def __init__(
        self,
        max_cycles=100,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world()
        SimpleEnvironment.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_surveillance_v0"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(3)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.color = AGENT_COLOURS[i]
            agent.size = 35
            agent.accel = 3.0
            agent.max_speed = 1.0
        # add target region
        world.target_region = TargetRegion()
        world.target_region.name = "target region"
        world.target_region.size = np.array([200, 150])
        world.target_region.color = MID_GREY

        return world

    def reset_world(self, world, np_random):
        # set random initial states
        world.target_region.state.p_pos = np_random.uniform(0, +1, world.dim_p)

        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(0, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.target_pos = np_random.uniform(0, +1, world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # TODO
        entity_pos = []
        for entity in world.agents:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate(entity_pos)

    def global_reward(self, env):
        env.draw()

        agents_mask = pygame.mask.from_surface(env.agents_overlay)
        target_region_mask = pygame.mask.from_surface(env.target_region_overlay)
        pre_observation = pygame.surfarray.pixels3d(target_region_mask.to_surface())
        target_region_mask.erase(agents_mask, (0, 0))
        observation = pygame.surfarray.pixels3d(target_region_mask.to_surface())

        pixels_post = np.sum(observation[:, :, 1])
        pixels_pre = np.sum(pre_observation[:, :, 1])

        reward_value = 1 - pixels_post / pixels_pre

        return reward_value
