import functools
import os
import random
from copy import copy

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.custom._custom_utils.core import Agent
from pettingzoo.utils import agent_selector, wrappers

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnvironment(AECEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "grid_surveillance_environment_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
    ):
        """The init method takes in environment arguments.

        Add text
        """
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 500
        self.screen = pygame.Surface([self.width, self.height], pygame.SRCALPHA)
        self.max_size = 1
        # self.game_font = pygame.freetype.Font(
        #     os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        # )

        # Set up the drawing window
        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = Box(low=0, high=1, shape=(space_dim,))
            else:
                self.action_spaces[agent.name] = Discrete(space_dim)
            self.observation_spaces[agent.name] = Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0

        self.current_actions = [None] * self.num_agents

        # For reward function
        self.target_region_overlay = None
        self.agents_overlay = None

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - observation
        - infos
        - ...

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = copy(self.possible_agents)
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = float(self.scenario.global_reward(self))

        for agent in self.world.agents:
            #     agent_reward = float(self.scenario.reward(agent, self.world))
            #     if self.local_ratio is not None:
            #         reward = (
            #             global_reward * (1 - self.local_ratio)
            #             + agent_reward * self.local_ratio
            #         )
            #     else:
            #         reward = agent_reward
            self.rewards[agent.name] = global_reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                agent.action.u[0] += action[0][2] - action[0][1]
                agent.action.u[1] += action[0][4] - action[0][3]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        """Takes in an actions for all agents.

        Needs to update:
        - ...
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render(self):
        """Renders the environment."""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255, 1))

        # draw target region
        self.target_region_overlay = pygame.Surface(
            (self.width, self.height), pygame.SRCALPHA
        )  # This surface will hold the target region
        self.target_region_overlay.fill(
            (0, 0, 0, 0)
        )  # Fill with a fully transparent background
        x, y = self.world.target_region.state.p_pos
        x = x * (self.width - self.world.target_region.size[0])
        y = self.height - (
            y * (self.height - self.world.target_region.size[1])
            + self.world.target_region.size[1]
        )
        pygame.draw.rect(
            self.target_region_overlay,
            self.world.target_region.color,
            pygame.Rect(
                x, y, self.world.target_region.size[0], self.world.target_region.size[1]
            ),
        )

        # update geometry and text positions
        self.agents_overlay = pygame.Surface(
            (self.width, self.height), pygame.SRCALPHA
        )  # This surface will hold the agents
        self.agents_overlay.fill(
            (0, 0, 0, 0)
        )  # Fill with a fully transparent background
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            x, y = entity.state.p_pos
            x = x * (self.width - entity.size) + entity.size // 2
            y = -1 * (y * (self.height - entity.size) + entity.size // 2) + self.height
            pygame.draw.circle(self.agents_overlay, entity.color, (x, y), entity.size)
            pygame.draw.circle(
                self.agents_overlay, (0, 0, 0), (x, y), 10
            )  # center point
            # assert (
            #     0 < x < self.width and 0 < y < self.height
            # ), f"Coordinates {(x, y)} are out of bounds."
            # TODO - add if show target
            # Draw estimate target
            x, y = entity.state.target_pos
            x = x * (self.width - entity.size) + entity.size // 2
            y = -1 * (y * (self.height - entity.size) + entity.size // 2) + self.height
            pygame.draw.circle(self.screen, entity.color, (x, y), entity.size // 2)

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

        # Blit the transparent surface (overlay) onto the screen
        self.screen.blit(self.target_region_overlay, (0, 0))
        self.screen.blit(self.agents_overlay, (0, 0))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.renderOn = False
