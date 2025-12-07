import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:17:19.618100
# Source Brief: brief_02385.md
# Brief Index: 2385
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for managing a network of interconnected pipes.

    The goal is to fill four target tanks to a specific level within a time limit,
    while managing flow rates to avoid emergent blockages and leaks. The game is
    designed with a clean, technical aesthetic and responsive controls for a
    satisfying gameplay experience.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) for pipe selection.
    - action[1]: Space button (0=released, 1=held) to increase flow.
    - action[2]: Shift button (0=released, 1=held) to decrease flow.

    Observation Space: Box(shape=(400, 640, 3), dtype=uint8)
    - A 640x400 RGB image of the current game state.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage a network of pipes to fill tanks to their target levels. "
        "Adjust flow rates carefully to prevent blockages and leaks before time runs out."
    )
    user_guide = (
        "Controls: Use ←/↓ and →/↑ arrow keys to select a pipe. "
        "Press space to increase flow and shift to decrease flow."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For simulation scaling
        self.MAX_STEPS = 1800 # 60 seconds at 30 FPS

        # Game constants
        self.NUM_PIPES = 7
        self.NUM_TANKS = 4
        self.TANK_CAPACITY = 60.0
        self.TARGET_FILL = 50.0
        self.MAX_FLOW = 10
        self.FLOW_SCALAR = 0.1 # Scales flow rate to level change per step
        self.LEAK_CHANCE = 0.002 # Per step chance for a leak on a high-flow pipe
        self.LEAK_RATE = 1.0
        self.BLOCK_FLOW_THRESHOLD = 8
        self.BLOCK_TIME_THRESHOLD = 3 * self.FPS # 3 seconds of high flow
        self.LEAK_FLOW_THRESHOLD = 6

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PIPE = (100, 110, 130)
        self.COLOR_PIPE_BLOCKED = (60, 60, 70)
        self.COLOR_PIPE_SELECTED = (255, 255, 0)
        self.COLOR_WATER = (50, 150, 255)
        self.COLOR_OVERFLOW = (255, 50, 50)
        self.COLOR_TARGET_LINE = (50, 255, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_VALUE = (255, 255, 255)
        self.COLOR_TEXT_WARN = (255, 180, 0)
        self.COLOR_TEXT_FAIL = (255, 80, 80)

        # --- Gym Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("Consolas", 14)
        self.font_md = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_lg = pygame.font.SysFont("Arial", 48, bold=True)

        # --- Game Layout ---
        self._define_layout()

        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_over_reason = ""
        self.pipes = []
        self.tanks = []
        self.leaks = []
        self.particles = []
        self.selected_pipe_idx = 0
        self.prev_action = np.array([0, 0, 0])
        self.actual_flows = [0.0] * self.NUM_PIPES

        # --- Finalization ---
        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # Not needed for submission

    def _define_layout(self):
        self.JUNCTION_POS = {
            'A': (150, 200),
            'B': (320, 100),
            'C': (320, 300)
        }
        source_pos = (50, 200)

        self.TANK_POS = [
            {'xy': (250, 180), 'size': (40, 80)},
            {'xy': (450, 80), 'size': (40, 80)},
            {'xy': (450, 180), 'size': (40, 80)},
            {'xy': (450, 280), 'size': (40, 80)}
        ]
        
        # Pipe connections: (start_pos, end_pos)
        self.PIPE_POS = [
            (source_pos, self.JUNCTION_POS['A']),                                  # P0
            (self.JUNCTION_POS['A'], (self.TANK_POS[0]['xy'][0], 200)),             # P1
            (self.JUNCTION_POS['A'], self.JUNCTION_POS['B']),                       # P2
            (self.JUNCTION_POS['B'], (self.TANK_POS[1]['xy'][0], 100)),             # P3
            (self.JUNCTION_POS['B'], self.JUNCTION_POS['C']),                       # P4
            (self.JUNCTION_POS['C'], (self.TANK_POS[2]['xy'][0], 200)),             # P5
            (self.JUNCTION_POS['C'], (self.TANK_POS[3]['xy'][0], 300)),             # P6
        ]
        
        self.PIPE_TO_TANK_MAP = {1: 0, 3: 1, 5: 2, 6: 3}

        self.NETWORK_TOPOLOGY = {
            'J_A': {'in': [0], 'out': [1, 2]},
            'J_B': {'in': [2], 'out': [3, 4]},
            'J_C': {'in': [4], 'out': [5, 6]},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_over_reason = ""
        self.selected_pipe_idx = 0
        self.prev_action = np.array([0, 0, 0])
        self.actual_flows = [0.0] * self.NUM_PIPES
        
        self.pipes = [{'flow_rate': 0, 'is_blocked': False, 'high_flow_timer': 0, 'just_blocked': False} for _ in range(self.NUM_PIPES)]
        self.tanks = [{'level': 0.0} for _ in range(self.NUM_TANKS)]
        self.leaks = []
        self.particles = []

        # No initial blockages or leaks for fair start
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        
        self._handle_actions(action)
        self._update_physics()
        self._update_events()
        self._update_particles()

        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if truncated and not terminated:
                self.game_over_reason = "Time limit reached!"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_press, shift_press = action[0], action[1], action[2]
        prev_movement, prev_space, prev_shift = self.prev_action

        # Cycle pipe selection on key press
        if (movement in [1, 4]) and not (prev_movement in [1, 4]):
            self.selected_pipe_idx = (self.selected_pipe_idx + 1) % self.NUM_PIPES
            # sfx: pipe_select
        if (movement in [2, 3]) and not (prev_movement in [2, 3]):
            self.selected_pipe_idx = (self.selected_pipe_idx - 1 + self.NUM_PIPES) % self.NUM_PIPES
            # sfx: pipe_select

        # Adjust flow rate on key press
        pipe = self.pipes[self.selected_pipe_idx]
        if not pipe['is_blocked']:
            if space_press and not prev_space:
                pipe['flow_rate'] = min(self.MAX_FLOW, pipe['flow_rate'] + 1)
                # sfx: flow_increase
            if shift_press and not prev_shift:
                pipe['flow_rate'] = max(0, pipe['flow_rate'] - 1)
                # sfx: flow_decrease
        
        self.prev_action = action

    def _update_physics(self):
        flows = [p['flow_rate'] if not p['is_blocked'] else 0 for p in self.pipes]
        self.actual_flows = [0.0] * self.NUM_PIPES
        junction_inflows = {'J_A': 0.0, 'J_B': 0.0, 'J_C': 0.0}

        # Source to J_A
        self.actual_flows[0] = flows[0]
        junction_inflows['J_A'] += self.actual_flows[0]

        # Process junctions
        for j_name, topo in self.NETWORK_TOPOLOGY.items():
            inflow = junction_inflows[j_name]
            requested_outflow = sum(flows[p_idx] for p_idx in topo['out'])
            
            scale = 1.0
            if requested_outflow > 0 and requested_outflow > inflow:
                scale = inflow / requested_outflow
            
            for p_idx in topo['out']:
                self.actual_flows[p_idx] = flows[p_idx] * scale
                if p_idx in self.PIPE_TO_TANK_MAP:
                    tank_idx = self.PIPE_TO_TANK_MAP[p_idx]
                    self.tanks[tank_idx]['level'] += self.actual_flows[p_idx] * self.FLOW_SCALAR
                else: # Flowing to another junction
                    next_j_list = [k for k,v in self.NETWORK_TOPOLOGY.items() if p_idx in v['in']]
                    if next_j_list:
                        junction_inflows[next_j_list[0]] += self.actual_flows[p_idx]

        # Apply leaks
        for leak in self.leaks:
            p_idx = leak['pipe_idx']
            leaked_amount = leak['rate'] * self.FLOW_SCALAR
            
            # Find which tank is affected downstream and drain it
            affected_tanks = []
            q = [p_idx]
            visited = {p_idx}
            while q:
                curr_p = q.pop(0)
                if curr_p in self.PIPE_TO_TANK_MAP:
                    affected_tanks.append(self.PIPE_TO_TANK_MAP[curr_p])
                else:
                    for j_name, topo in self.NETWORK_TOPOLOGY.items():
                        if curr_p in topo['in']:
                            for next_p in topo['out']:
                                if next_p not in visited:
                                    q.append(next_p)
                                    visited.add(next_p)
            
            if affected_tanks:
                drain_per_tank = leaked_amount / len(affected_tanks)
                for tank_idx in affected_tanks:
                    self.tanks[tank_idx]['level'] = max(0, self.tanks[tank_idx]['level'] - drain_per_tank)

    def _update_events(self):
        # Reset one-time flags
        for pipe in self.pipes:
            pipe['just_blocked'] = False

        # Check for new blockages and leaks
        for i, pipe in enumerate(self.pipes):
            if pipe['is_blocked']:
                continue

            # Blockages
            if pipe['flow_rate'] > self.BLOCK_FLOW_THRESHOLD:
                pipe['high_flow_timer'] += 1
                if pipe['high_flow_timer'] > self.BLOCK_TIME_THRESHOLD:
                    pipe['is_blocked'] = True
                    pipe['just_blocked'] = True
                    # sfx: pipe_block
            else:
                pipe['high_flow_timer'] = 0

            # Leaks
            if pipe['flow_rate'] > self.LEAK_FLOW_THRESHOLD:
                if self.np_random.random() < self.LEAK_CHANCE:
                    if not any(leak['pipe_idx'] == i for leak in self.leaks):
                        start, end = self.PIPE_POS[i]
                        leak_pos = (
                            start[0] + (end[0] - start[0]) * 0.5,
                            start[1] + (end[1] - start[1]) * 0.5
                        )
                        self.leaks.append({'pipe_idx': i, 'pos': leak_pos, 'rate': self.LEAK_RATE})
                        # sfx: leak_start

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

        for leak in self.leaks:
            if self.np_random.random() < 0.5: # Spawn particles intermittently
                angle = self.np_random.uniform(math.pi * 0.9, math.pi * 1.1)
                speed = self.np_random.uniform(0.5, 1.5)
                self.particles.append({
                    'pos': list(leak['pos']),
                    'vel': [speed * math.cos(angle), speed * math.sin(angle)],
                    'life': self.np_random.integers(20, 40)
                })

    def _calculate_reward(self):
        # Win condition
        if all(abs(t['level'] - self.TARGET_FILL) < 1.0 for t in self.tanks):
            return 100.0

        # Loss conditions
        if any(t['level'] > self.TANK_CAPACITY for t in self.tanks):
            return -50.0
        if self.steps >= self.MAX_STEPS:
            return -25.0

        # Step-based rewards
        reward = 0.0
        
        # Proximity reward
        for tank in self.tanks:
            if abs(tank['level'] - self.TARGET_FILL) <= 5.0:
                reward += 0.1
        
        # Penalty for newly blocked pipes
        for pipe in self.pipes:
            if pipe['just_blocked']:
                reward -= 5.0
        
        return reward

    def _check_termination(self):
        if all(abs(t['level'] - self.TARGET_FILL) < 1.0 for t in self.tanks):
            self.game_over_reason = "All tanks filled!"
            return True
        if any(t['level'] > self.TANK_CAPACITY for t in self.tanks):
            overflowing_tank = [i for i, t in enumerate(self.tanks) if t['level'] > self.TANK_CAPACITY][0]
            self.game_over_reason = f"Tank {overflowing_tank + 1} overflowed!"
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "selected_pipe": self.selected_pipe_idx,
            "tank_levels": [t['level'] for t in self.tanks],
        }

    def _render_game(self):
        # Draw pipes
        for i, (start, end) in enumerate(self.PIPE_POS):
            pipe = self.pipes[i]
            color = self.COLOR_PIPE_BLOCKED if pipe['is_blocked'] else self.COLOR_PIPE
            
            # Selected pipe glow
            if i == self.selected_pipe_idx:
                pygame.draw.line(self.screen, self.COLOR_PIPE_SELECTED, start, end, 10)

            pygame.draw.line(self.screen, color, start, end, 6)

            # Draw water flow in pipe
            if not pipe['is_blocked'] and self.actual_flows[i] > 0.1:
                flow_width = max(1, int(4 * (self.actual_flows[i] / self.MAX_FLOW)))
                pygame.draw.line(self.screen, self.COLOR_WATER, start, end, flow_width)

        # Draw junctions
        for pos in self.JUNCTION_POS.values():
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 8, self.COLOR_PIPE)
        
        # Draw tanks
        for i, tank_info in enumerate(self.TANK_POS):
            x, y = tank_info['xy']
            w, h = tank_info['size']
            level = self.tanks[i]['level']
            
            pygame.draw.rect(self.screen, self.COLOR_PIPE, (x, y, w, h), 2)
            
            fill_ratio = min(1.0, level / self.TANK_CAPACITY)
            fill_h = int(h * fill_ratio)
            color = self.COLOR_OVERFLOW if level > self.TANK_CAPACITY else self.COLOR_WATER
            pygame.draw.rect(self.screen, color, (x + 1, y + h - fill_h + 1, w - 2, fill_h - 1))

            target_y = y + h - int(h * (self.TARGET_FILL / self.TANK_CAPACITY))
            pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (x, target_y), (x + w, target_y), 1)

        # Draw leaks and particles
        for leak in self.leaks:
            pos = (int(leak['pos'][0]), int(leak['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_OVERFLOW)
        
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / 40))
            if alpha > 0:
                pygame.gfxdraw.pixel(self.screen, pos[0], pos[1], (*self.COLOR_WATER, alpha))

    def _render_ui(self):
        time_text = self.font_md.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - 150, 10))

        score_text = self.font_md.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        for i, tank_info in enumerate(self.TANK_POS):
            x, y = tank_info['xy']
            w, _ = tank_info['size']
            level = self.tanks[i]['level']
            text = f"{level:.1f}"
            color = self.COLOR_TEXT_VALUE
            if level > self.TANK_CAPACITY: color = self.COLOR_TEXT_FAIL
            elif abs(level - self.TARGET_FILL) < 1.0: color = self.COLOR_TARGET_LINE
            
            level_text = self.font_sm.render(text, True, color)
            self.screen.blit(level_text, (x + w/2 - level_text.get_width()/2, y - 20))
            
        for i, pipe in enumerate(self.pipes):
            start, end = self.PIPE_POS[i]
            mid_pos = ((start[0] + end[0]) / 2 - 5, (start[1] + end[1]) / 2 - 20)
            text = f"{pipe['flow_rate']}"
            color = self.COLOR_PIPE_SELECTED if i == self.selected_pipe_idx else self.COLOR_TEXT
            if pipe['is_blocked']: text, color = "X", self.COLOR_TEXT_FAIL
            elif pipe['flow_rate'] > self.BLOCK_FLOW_THRESHOLD: color = self.COLOR_TEXT_WARN

            flow_text = self.font_sm.render(text, True, color)
            self.screen.blit(flow_text, mid_pos)
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_lg.render(self.game_over_reason, True, self.COLOR_TEXT_VALUE)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for human play and visualization.
    # It will not be executed by the test suite.
    # The `SDL_VIDEODRIVER=dummy` is for headless execution in the test suite.
    # Running this script directly may still open a window.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pipe Network Manager")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keyboard to MultiDiscrete action space
        # Arrow keys are mapped to discrete movement actions
        # Space and Shift are binary actions
        
        # Note: The provided code uses key-down events for selection,
        # but the step function expects continuous state.
        # We will use get_pressed for continuous actions, and handle single presses in the env.
        
        # This mapping is based on the logic in _handle_actions
        # where 1 and 4 are 'next' and 2 and 3 are 'prev'.
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Reason: {env.game_over_reason}")
            print(f"Total reward: {total_reward:.2f}")
            total_reward = 0
            pygame.time.wait(2000)
            env.reset()

        clock.tick(env.FPS)
        
    env.close()