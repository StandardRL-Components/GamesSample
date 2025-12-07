
# Generated: 2025-08-27T14:22:29.784023
# Source Brief: brief_00662.md
# Brief Index: 662

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓ to cycle tower locations, Shift to cycle tower types, Space to build."
    )

    game_description = (
        "Defend your base from descending waves of enemies by strategically placing defensive towers."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2500
        self.MAX_WAVES = 10

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PATH = (40, 50, 80)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_ENEMY = (230, 25, 75)
        self.COLOR_TOWER_SPOT = (255, 225, 25, 50)
        self.COLOR_TOWER_SPOT_SELECTED = (255, 225, 25)
        self.COLOR_UI_TEXT = (245, 245, 245)
        self.COLOR_HEALTH_BAR = (70, 240, 240)
        self.COLOR_HEALTH_BAR_BG = (128, 0, 0)
        self.COLOR_TOWER_1 = (210, 210, 210)
        self.COLOR_TOWER_2 = (120, 120, 120)
        self.COLOR_PROJ_1 = (0, 190, 255)
        self.COLOR_PROJ_2 = (255, 150, 0)

        # Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game assets (defined programmatically)
        self.path_waypoints = [
            (50, -20), (50, 100), (250, 100), (250, 250),
            (100, 250), (100, 350), (self.WIDTH // 2, 350)
        ]
        self.tower_locations = [
            (150, 50), (150, 150), (350, 175), (175, 300), (500, 100), (500, 300)
        ]
        self.tower_types = [
            {"name": "Gatling", "cost": 10, "range": 80, "fire_rate": 5, "damage": 1, "color": self.COLOR_TOWER_1, "proj_color": self.COLOR_PROJ_1, "proj_speed": 8},
            {"name": "Cannon", "cost": 25, "range": 120, "fire_rate": 30, "damage": 5, "color": self.COLOR_TOWER_2, "proj_color": self.COLOR_PROJ_2, "proj_speed": 6},
        ]

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.max_base_health = 50
        self.resources = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.wave_spawner = None
        self.selected_tower_location_idx = 0
        self.selected_tower_type_idx = 0
        self.last_action = np.array([0, 0, 0])
        self.step_reward = 0.0

        self.reset()
        # self.validate_implementation() # Uncomment to run validation check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.max_base_health
        self.resources = 40
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = deque(maxlen=200)
        self.selected_tower_location_idx = 0
        self.selected_tower_type_idx = 0
        self.last_action = np.array([0, 0, 0])
        self.wave_spawner = None
        self._start_next_wave()
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = -0.01  # Small penalty for time passing

        self._handle_input(action)
        
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        if self.wave_spawner and self.wave_spawner['timer'] <= 0:
            self._spawn_from_queue()
        elif self.wave_spawner:
            self.wave_spawner['timer'] -= 1

        if not self.enemies and not self.wave_spawner:
            if self.wave_number < self.MAX_WAVES:
                self._start_next_wave()
                self.step_reward += 5  # Wave completion bonus
                self.score += 50
            else: # Game won
                self.game_over = True

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and self.base_health > 0: # Win condition
            self.step_reward += 100
            self.score += 1000
        elif terminated and self.base_health <= 0: # Lose condition
            self.step_reward -= 100
            self.score -= 100

        reward = self.step_reward
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_press, shift_press = action[0], action[1], action[2]
        
        # Detect key presses (transition from 0 to 1)
        up_pressed = movement == 1 and self.last_action[0] != 1
        down_pressed = movement == 2 and self.last_action[0] != 2
        space_pressed = space_press == 1 and self.last_action[1] == 0
        shift_pressed = shift_press == 1 and self.last_action[2] == 0

        # Cycle tower locations
        if up_pressed:
            self.selected_tower_location_idx = (self.selected_tower_location_idx + 1) % len(self.tower_locations)
        if down_pressed:
            self.selected_tower_location_idx = (self.selected_tower_location_idx - 1 + len(self.tower_locations)) % len(self.tower_locations)
            
        # Cycle tower types
        if shift_pressed:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.tower_types)
            
        # Place tower
        if space_pressed:
            self._place_tower()

        self.last_action = action

    def _place_tower(self):
        loc_pos = self.tower_locations[self.selected_tower_location_idx]
        tower_spec = self.tower_types[self.selected_tower_type_idx]

        # Check if location is already occupied
        if any(t['pos'] == loc_pos for t in self.towers):
            return # sfx: error_buzz.wav
        
        # Check for sufficient resources
        if self.resources >= tower_spec['cost']:
            self.resources -= tower_spec['cost']
            new_tower = {
                'pos': loc_pos,
                'spec': tower_spec,
                'cooldown': 0,
                'target': None
            }
            self.towers.append(new_tower)
            # sfx: build_tower.wav

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return

        num_enemies = 2 + self.wave_number * 2
        enemy_health = 3 + self.wave_number
        enemy_speed = 1.0 + (self.wave_number // 2) * 0.1
        spawn_delay = 30 # Ticks between each enemy
        
        spawn_queue = []
        for i in range(num_enemies):
            spawn_queue.append({
                'health': enemy_health,
                'speed': enemy_speed + self.np_random.uniform(-0.1, 0.1)
            })
        
        self.wave_spawner = {
            'queue': spawn_queue,
            'timer': 0,
            'delay': spawn_delay
        }

    def _spawn_from_queue(self):
        if not self.wave_spawner or not self.wave_spawner['queue']:
            self.wave_spawner = None
            return

        enemy_spec = self.wave_spawner['queue'].pop(0)
        new_enemy = {
            'pos': np.array(self.path_waypoints[0], dtype=float),
            'max_health': enemy_spec['health'],
            'health': enemy_spec['health'],
            'speed': enemy_spec['speed'],
            'path_idx': 1,
            'bob_offset': self.np_random.uniform(0, 2 * math.pi)
        }
        self.enemies.append(new_enemy)
        self.wave_spawner['timer'] = self.wave_spawner['delay']
        
        if not self.wave_spawner['queue']:
            self.wave_spawner = None

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            if enemy['path_idx'] >= len(self.path_waypoints):
                self.base_health -= enemy['health']
                self.base_health = max(0, self.base_health)
                self.enemies.remove(enemy)
                self._create_particles(enemy['pos'], self.COLOR_BASE, 20)
                # sfx: base_damage.wav
                continue

            target_pos = np.array(self.path_waypoints[enemy['path_idx']], dtype=float)
            direction = target_pos - enemy['pos']
            distance = np.linalg.norm(direction)

            if distance < enemy['speed']:
                enemy['pos'] = target_pos
                enemy['path_idx'] += 1
            else:
                enemy['pos'] += (direction / distance) * enemy['speed']

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            # Find target
            target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = np.linalg.norm(np.array(tower['pos']) - enemy['pos'])
                if dist <= tower['spec']['range'] and dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = tower['spec']['fire_rate']
                self.projectiles.append({
                    'pos': np.array(tower['pos'], dtype=float),
                    'spec': tower['spec'],
                    'target': target,
                    'vel': (target['pos'] - np.array(tower['pos']))
                })
                # sfx: tower_shoot.wav
                # Muzzle flash particle
                self.particles.append({'pos': np.array(tower['pos']), 'vel': (self.np_random.random(2) - 0.5) * 2, 'life': 3, 'color': (255, 255, 150), 'radius': 4})


    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            # Move projectile
            direction = proj['target']['pos'] - proj['pos']
            distance = np.linalg.norm(direction)
            if distance < proj['spec']['proj_speed']:
                proj['pos'] = proj['target']['pos']
            else:
                proj['pos'] += (direction / distance) * proj['spec']['proj_speed']
            
            # Check for collision
            if np.linalg.norm(proj['pos'] - proj['target']['pos']) < 10:
                proj['target']['health'] -= proj['spec']['damage']
                self.step_reward += 0.1 # Reward for hitting
                self._create_particles(proj['pos'], proj['spec']['proj_color'], 5)
                # sfx: enemy_hit.wav

                if proj['target']['health'] <= 0:
                    if proj['target'] in self.enemies:
                        self.enemies.remove(proj['target'])
                        self.step_reward += 1.0 # Reward for kill
                        self.resources += 2
                        self._create_particles(proj['pos'], self.COLOR_ENEMY, 15)
                        # sfx: enemy_explode.wav
                
                if proj in self.projectiles:
                    self.projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            vel = (self.np_random.random(2) - 0.5) * self.np_random.uniform(1, 4)
            life = self.np_random.integers(10, 25)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color, 'radius': self.np_random.uniform(1, 3)})

    def _check_termination(self):
        return self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.game_over

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 20)

        # Draw base
        base_pos = self.path_waypoints[-1]
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_pos[0] - 20, base_pos[1] - 20, 40, 40))
        # Base health bar
        health_ratio = self.base_health / self.max_base_health
        bar_w = 40
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (base_pos[0] - 20, base_pos[1] - 30, bar_w, 5))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (base_pos[0] - 20, base_pos[1] - 30, bar_w * health_ratio, 5))

        # Draw tower locations
        for i, pos in enumerate(self.tower_locations):
            is_occupied = any(t['pos'] == pos for t in self.towers)
            if i == self.selected_tower_location_idx and not is_occupied:
                # Pulsating glow for selected spot
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                radius = int(18 + pulse * 4)
                alpha = int(150 + pulse * 105)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_TOWER_SPOT_SELECTED, alpha))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*self.COLOR_TOWER_SPOT_SELECTED, alpha))
            elif not is_occupied:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 20, self.COLOR_TOWER_SPOT)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 20, self.COLOR_TOWER_SPOT)

        # Draw towers
        for tower in self.towers:
            pos = tower['pos']
            color = tower['spec']['color']
            pygame.draw.circle(self.screen, (0,0,0), pos, 12)
            pygame.draw.circle(self.screen, color, pos, 10)
            pygame.draw.circle(self.screen, (255,255,255), (pos[0]-3, pos[1]-3), 2)


        # Draw enemies
        for enemy in self.enemies:
            bob = math.sin(self.steps * 0.1 + enemy['bob_offset']) * 2
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1] + bob))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0]-8, pos[1]-8, 16, 16))
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0]-8, pos[1]-14, 16, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0]-8, pos[1]-14, 16 * health_ratio, 3))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.circle(self.screen, proj['spec']['proj_color'], pos, 4)

        # Draw particles
        for p in list(self.particles):
            if p['life'] > 0:
                alpha = max(0, min(255, int(255 * (p['life'] / 25.0))))
                color = (*p['color'], alpha)
                surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(surf, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']))
            elif p in self.particles: # Check if it still exists before removing
                pass # deque handles removal automatically


    def _render_ui(self):
        # Top bar info
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        resources_text = self.font_small.render(f"RESOURCES: ${self.resources}", True, self.COLOR_UI_TEXT)
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))
        self.screen.blit(resources_text, (150, 10))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Selected tower info
        sel_tower = self.tower_types[self.selected_tower_type_idx]
        tower_info_text = self.font_small.render(f"BUILD: {sel_tower['name']} (Cost: ${sel_tower['cost']})", True, self.COLOR_UI_TEXT)
        self.screen.blit(tower_info_text, (10, 30))

        # Game Over / Win Text
        if self.game_over:
            if self.base_health <= 0:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            else:
                msg = "VICTORY!"
                color = self.COLOR_HEALTH_BAR
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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