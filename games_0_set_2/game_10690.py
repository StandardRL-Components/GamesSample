import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:53:37.713112
# Source Brief: brief_00690.md
# Brief Index: 690
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Control the flow of water through a complex pipe system to fill designated tanks. "
        "Open and close valves strategically to manage pressure and prevent costly leaks."
    )
    user_guide = (
        "Controls: Use ← and → to select a valve. Use ↑ and ↓ to open or close the pipes connected to the selected valve."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_TIME = 120  # seconds
        self.FPS = 30
        self.MAX_STEPS = self.MAX_TIME * self.FPS
        self.TANK_CAPACITY = 50.0
        self.WIN_CONDITION_TANKS = 3
        self.SIMULATION_SUBSTEPS = 5
        self.FLOW_RATE = 0.1
        self.LEAKAGE_BASE_RATE = 0.0005 

        # Colors
        self.COLOR_BG = (26, 42, 58)
        self.COLOR_PIPE = (96, 112, 128)
        self.COLOR_WATER = (64, 160, 255)
        self.COLOR_VALVE = (255, 204, 0)
        self.COLOR_VALVE_SELECTED = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TANK_FILL = (64, 255, 128)
        self.COLOR_OBSTACLE = (255, 64, 64)
        self.COLOR_PARTICLE = (180, 220, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Initialize state variables to None
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.leakage_rate = None
        self.tanks_filled_count = None
        self.selected_valve_idx = None
        self.nodes = None
        self.pipes = None
        self.valves = None
        self.tanks = None
        self.obstacles = None
        self.particles = None
        
        self.reset()
        # self.validate_implementation(self) # This is a non-standard check, can be commented out

    def _define_network(self):
        # Define the static layout of the pipe network
        self.nodes = {
            'SOURCE': {'pos': (50, 200), 'level': 100.0, 'type': 'source'},
            'J1': {'pos': (150, 200), 'level': 0.0, 'type': 'junction'},
            'J2': {'pos': (320, 200), 'level': 0.0, 'type': 'junction'},
            'J3': {'pos': (490, 200), 'level': 0.0, 'type': 'junction'},
            'T1_IN': {'pos': (150, 100), 'level': 0.0, 'type': 'junction'},
            'T2_IN': {'pos': (320, 100), 'level': 0.0, 'type': 'junction'},
            'T3_IN': {'pos': (490, 100), 'level': 0.0, 'type': 'junction'},
            'D1': {'pos': (150, 300), 'level': 0.0, 'type': 'drain'},
            'O1_IN': {'pos': (320, 300), 'level': 0.0, 'type': 'junction'},
            'O1_OUT': {'pos': (420, 300), 'level': 0.0, 'type': 'drain'},
            'T1_OVERFLOW': {'pos': (50, 100), 'level': 0.0, 'type': 'junction'},
            'T1_OVERFLOW_END': {'pos': (50, 300), 'level': 0.0, 'type': 'junction'},
        }

        self.pipes = {
            'P_MAIN1': {'from': 'SOURCE', 'to': 'J1', 'width': 8},
            'P_MAIN2': {'from': 'J1', 'to': 'J2', 'width': 8},
            'P_MAIN3': {'from': 'J2', 'to': 'J3', 'width': 8},
            'P_V1_UP': {'from': 'J1', 'to': 'T1_IN', 'width': 6},
            'P_V1_DOWN': {'from': 'J1', 'to': 'D1', 'width': 6},
            'P_V2_UP': {'from': 'J2', 'to': 'T2_IN', 'width': 6},
            'P_V2_DOWN': {'from': 'J2', 'to': 'O1_IN', 'width': 6},
            'P_V3_UP': {'from': 'J3', 'to': 'T3_IN', 'width': 6},
            'P_OBSTACLE': {'from': 'O1_IN', 'to': 'O1_OUT', 'width': 6},
            'P_T1_OVERFLOW': {'from': 'T1_IN', 'to': 'T1_OVERFLOW', 'width': 4},
            'P_T1_OVERFLOW_DROP': {'from': 'T1_OVERFLOW', 'to': 'T1_OVERFLOW_END', 'width': 4},
            'P_T1_OVERFLOW_CONNECT': {'from': 'T1_OVERFLOW_END', 'to': 'O1_IN', 'width': 4},
        }

        self.valves = [
            {'node': 'J1', 'up_pipe': 'P_V1_UP', 'down_pipe': 'P_V1_DOWN', 'state': [False, False]},
            {'node': 'J2', 'up_pipe': 'P_V2_UP', 'down_pipe': 'P_V2_DOWN', 'state': [False, False]},
            {'node': 'J3', 'up_pipe': 'P_V3_UP', 'down_pipe': None, 'state': [False, False]},
        ]

        self.tanks = [
            {'node': 'T1_IN', 'pos': (125, 50), 'size': (50, 50), 'level': 0.0, 'filled': False},
            {'node': 'T2_IN', 'pos': (295, 50), 'size': (50, 50), 'level': 0.0, 'filled': False},
            {'node': 'T3_IN', 'pos': (465, 50), 'size': (50, 50), 'level': 0.0, 'filled': False},
        ]
        
        self.obstacles = [
            {'pipe': 'P_OBSTACLE', 'cleared': False}
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._define_network()
        for node in self.nodes.values():
            node['level'] = 0.0
        self.nodes['SOURCE']['level'] = 100.0

        for tank in self.tanks:
            tank['level'] = 0.0
            tank['filled'] = False
        
        for obstacle in self.obstacles:
            obstacle['cleared'] = False
            
        for valve in self.valves:
            valve['state'] = [False, False]

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        self.leakage_rate = 0.0
        self.tanks_filled_count = 0
        self.selected_valve_idx = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1.0 / self.FPS
        
        reward = self._handle_action(action)
        
        prev_tank_levels = [t['level'] for t in self.tanks]
        
        physics_rewards = self._update_physics()
        reward += physics_rewards
        
        # Continuous reward for filling tanks
        for i, tank in enumerate(self.tanks):
            delta = tank['level'] - prev_tank_levels[i]
            if delta > 0 and not tank['filled']:
                reward += delta * 0.1

        # Check for newly filled tanks
        for tank in self.tanks:
            if not tank['filled'] and tank['level'] >= self.TANK_CAPACITY:
                tank['filled'] = True
                self.tanks_filled_count += 1
                self.leakage_rate += self.LEAKAGE_BASE_RATE * 200 # Big jump on fill
                reward += 5.0
                # sfx: tank filled success chime

        terminated = self.tanks_filled_count >= self.WIN_CONDITION_TANKS or self.timer <= 0 or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.tanks_filled_count >= self.WIN_CONDITION_TANKS:
                reward += 100.0 # Win bonus
                # sfx: victory fanfare
            else:
                reward -= 100.0 # Loss penalty
                # sfx: failure sound
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]
        
        # Cycle through valves
        if movement == 3: # Left
            self.selected_valve_idx = (self.selected_valve_idx - 1) % len(self.valves)
            # sfx: ui click
        elif movement == 4: # Right
            self.selected_valve_idx = (self.selected_valve_idx + 1) % len(self.valves)
            # sfx: ui click

        # Toggle valve state
        selected_valve = self.valves[self.selected_valve_idx]
        if movement == 1: # Up
            if selected_valve['up_pipe'] is not None:
                selected_valve['state'][0] = not selected_valve['state'][0]
                # sfx: valve turn
        elif movement == 2: # Down
            if selected_valve['down_pipe'] is not None:
                selected_valve['state'][1] = not selected_valve['state'][1]
                # sfx: valve turn
        
        return 0.0

    def _is_pipe_open(self, pipe_id):
        for valve in self.valves:
            if pipe_id == valve['up_pipe']:
                return valve['state'][0]
            if pipe_id == valve['down_pipe']:
                return valve['state'][1]
        for obstacle in self.obstacles:
            if pipe_id == obstacle['pipe']:
                return obstacle['cleared']
        return True # Default to open if not controlled

    def _update_physics(self):
        reward = 0
        total_leakage = 0

        for _ in range(self.SIMULATION_SUBSTEPS):
            deltas = {node_id: 0.0 for node_id in self.nodes}
            
            # Water flow through pipes
            for pipe_id, pipe in self.pipes.items():
                if self._is_pipe_open(pipe_id):
                    node1 = self.nodes[pipe['from']]
                    node2 = self.nodes[pipe['to']]
                    
                    flow = (node1['level'] - node2['level']) * self.FLOW_RATE
                    deltas[pipe['from']] -= flow
                    deltas[pipe['to']] += flow

            # Apply deltas
            for node_id, delta in deltas.items():
                if self.nodes[node_id]['type'] != 'source':
                    self.nodes[node_id]['level'] += delta
            
            # Move water from input nodes to tanks
            for tank in self.tanks:
                in_node = self.nodes[tank['node']]
                if in_node['level'] > 0 and tank['level'] < self.TANK_CAPACITY:
                    transfer = min(in_node['level'], 1.0) # Transfer rate
                    tank['level'] += transfer
                    in_node['level'] -= transfer
            
            # Clamp levels
            for node_id, node in self.nodes.items():
                if node['type'] != 'source':
                    node['level'] = max(0, node['level'])

        # Handle tank overflows
        for tank in self.tanks:
            if tank['level'] > self.TANK_CAPACITY:
                overflow_amount = tank['level'] - self.TANK_CAPACITY
                tank['level'] = self.TANK_CAPACITY
                self.nodes[tank['node']]['level'] += overflow_amount # Spill back into input node
                for _ in range(5):
                    angle = random.uniform(0, math.pi * 2)
                    speed = random.uniform(1, 3)
                    self.particles.append({
                        'pos': [tank['pos'][0] + tank['size'][0]/2, tank['pos'][1]],
                        'vel': [math.cos(angle) * speed, -abs(math.sin(angle)) * speed],
                        'life': 20,
                        'radius': random.uniform(2, 4)
                    })
                # sfx: water splash
        
        # Check for obstacle clearing via overflow
        overflow_power = self.nodes['T1_OVERFLOW_END']['level']
        if overflow_power > 10 and not self.obstacles[0]['cleared']:
            self.obstacles[0]['cleared'] = True
            reward += 1.0
            # sfx: obstacle break
            
        # Handle leakage
        self.leakage_rate += self.LEAKAGE_BASE_RATE
        for node_id, node in self.nodes.items():
            if node['type'] in ['junction', 'drain']:
                leak = node['level'] * self.leakage_rate
                node['level'] -= leak
                total_leakage += leak
        
        reward -= total_leakage * 0.01
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render water first (background layer)
        for pipe_id, pipe in self.pipes.items():
            node1 = self.nodes[pipe['from']]
            node2 = self.nodes[pipe['to']]
            p1 = node1['pos']
            p2 = node2['pos']
            
            avg_level = (node1['level'] + node2['level']) / 2.0
            fill_ratio = min(1.0, avg_level / 50.0)

            if p1[0] == p2[0]: # Vertical pipe
                fill_h = abs(p1[1] - p2[1]) * fill_ratio
                rect = pygame.Rect(p1[0] - pipe['width'] // 2, min(p1[1], p2[1]), pipe['width'], abs(p1[1]-p2[1]))
                fill_rect = pygame.Rect(rect.left, rect.bottom - fill_h, rect.width, fill_h)
                pygame.draw.rect(self.screen, self.COLOR_WATER, fill_rect)
            elif p1[1] == p2[1]: # Horizontal pipe
                fill_h = pipe['width'] * fill_ratio
                rect = pygame.Rect(min(p1[0],p2[0]), p1[1] - pipe['width']//2, abs(p1[0]-p2[0]), pipe['width'])
                fill_rect = pygame.Rect(rect.left, rect.bottom - fill_h, rect.width, fill_h)
                pygame.draw.rect(self.screen, self.COLOR_WATER, fill_rect)

        # Render pipes
        for pipe_id, pipe in self.pipes.items():
            p1 = self.nodes[pipe['from']]['pos']
            p2 = self.nodes[pipe['to']]['pos']
            is_open = self._is_pipe_open(pipe_id)
            color = self.COLOR_PIPE if is_open else tuple(int(c*0.6) for c in self.COLOR_PIPE)
            pygame.draw.line(self.screen, color, p1, p2, pipe['width'])

        # Render obstacles
        for obstacle in self.obstacles:
            if not obstacle['cleared']:
                pipe = self.pipes[obstacle['pipe']]
                p1 = self.nodes[pipe['from']]['pos']
                p2 = self.nodes[pipe['to']]['pos']
                mid_pos = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                pygame.draw.circle(self.screen, self.COLOR_OBSTACLE, mid_pos, 10)
                pygame.draw.line(self.screen, self.COLOR_TEXT, (mid_pos[0]-5, mid_pos[1]-5), (mid_pos[0]+5, mid_pos[1]+5), 2)
                pygame.draw.line(self.screen, self.COLOR_TEXT, (mid_pos[0]-5, mid_pos[1]+5), (mid_pos[0]+5, mid_pos[1]-5), 2)

        # Render tanks
        for tank in self.tanks:
            pygame.draw.rect(self.screen, self.COLOR_PIPE, (*tank['pos'], *tank['size']), 3)
            fill_height = (tank['level'] / self.TANK_CAPACITY) * tank['size'][1]
            fill_height = max(0, min(tank['size'][1], fill_height))
            fill_rect = (tank['pos'][0], tank['pos'][1] + tank['size'][1] - fill_height, tank['size'][0], fill_height)
            
            color = self.COLOR_TANK_FILL if tank['filled'] else self.COLOR_WATER
            pygame.draw.rect(self.screen, color, fill_rect)

        # Render valves
        for i, valve in enumerate(self.valves):
            pos = self.nodes[valve['node']]['pos']
            is_selected = (i == self.selected_valve_idx)
            
            if is_selected:
                radius = 18
                for r in range(radius, 10, -2):
                    alpha = 80 * (1 - (r - 10) / (radius - 10))
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], r, (*self.COLOR_VALVE_SELECTED[:3], int(alpha)))

            pygame.draw.circle(self.screen, self.COLOR_VALVE, pos, 10)
            
            if valve['state'][0]: # Up
                pygame.draw.line(self.screen, self.COLOR_BG, pos, (pos[0], pos[1]-8), 3)
            if valve['state'][1]: # Down
                pygame.draw.line(self.screen, self.COLOR_BG, pos, (pos[0], pos[1]+8), 3)
        
        # Render particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = p['life'] / 20.0
                color = (*self.COLOR_PARTICLE, int(255 * alpha))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        timer_text = f"TIME: {max(0, int(self.timer))}"
        timer_surf = self.font_medium.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        tanks_text = f"TANKS FILLED: {self.tanks_filled_count} / {self.WIN_CONDITION_TANKS}"
        tanks_surf = self.font_medium.render(tanks_text, True, self.COLOR_TEXT)
        self.screen.blit(tanks_surf, (self.WIDTH // 2 - tanks_surf.get_width() // 2, 10))

        for tank in self.tanks:
            level_text = f"{min(self.TANK_CAPACITY, tank['level']):.1f}"
            level_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
            text_pos = (tank['pos'][0] + tank['size'][0]/2 - level_surf.get_width()/2, tank['pos'][1] - 20)
            self.screen.blit(level_surf, text_pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "tanks_filled": self.tanks_filled_count,
            "leakage_rate": self.leakage_rate,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    @staticmethod
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # You might need to unset the SDL_VIDEODRIVER dummy variable to see the window
    # e.g., run: SDL_VIDEODRIVER=x11 python your_script.py
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pipe Flow Manager")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")

        action = [movement, 0, 0] # space/shift not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}, Tanks: {info['tanks_filled']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()