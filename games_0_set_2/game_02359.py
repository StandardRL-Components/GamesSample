
# Generated: 2025-08-28T04:32:56.378276
# Source Brief: brief_02359.md
# Brief Index: 2359

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to place a tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of alien invaders by strategically placing defensive towers."
    )

    # Frames advance automatically at a conceptual 30fps.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000  # Increased for 10 waves
        self.GRID_SIZE = 40
        self.GRID_WIDTH = self.WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.GRID_SIZE
        self.MAX_WAVES = 10

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 40)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_ALIEN = (220, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (0, 200, 80)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        
        # --- Tower Definitions ---
        self.TOWER_SPECS = {
            0: { # Machine Gun
                "cost": 25, "range": 100, "damage": 5, "fire_rate": 6, # every 6 frames
                "color": (60, 120, 255), "proj_color": (255, 255, 100), "proj_size": 2, "proj_speed": 10
            },
            1: { # Cannon
                "cost": 75, "range": 150, "damage": 25, "fire_rate": 45, # every 45 frames
                "color": (100, 180, 255), "proj_color": (255, 200, 50), "proj_size": 5, "proj_speed": 7
            }
        }
        self.NUM_TOWER_TYPES = len(self.TOWER_SPECS)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.wave_cleared_timer = 0
        self.last_action_time = {
            "place": -100,
            "cycle": -100
        }
        self.np_random = None

        # --- Initialize and Validate ---
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.resources = 100
        self.wave_number = 1
        
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 3]
        self.selected_tower_type = 0
        self.wave_cleared_timer = 0
        
        self.last_action_time = {
            "place": -100,
            "cycle": -100
        }

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game State ---
        reward_from_updates = self._update_game_state()
        step_reward += reward_from_updates
        
        # --- Handle Wave Progression ---
        if not self.aliens and not self.game_over:
            if self.wave_cleared_timer == 0:
                self.wave_cleared_timer = self.FPS * 3 # 3 second delay
            
            self.wave_cleared_timer -= 1
            if self.wave_cleared_timer <= 0:
                if self.wave_number < self.MAX_WAVES:
                    self.wave_number += 1
                    self._spawn_wave()
                    self.wave_cleared_timer = 0
                else: # Victory condition
                    self.game_over = True
                    step_reward += 100
                    self.score += 100

        # --- Calculate Step Reward ---
        # Penalty for inaction
        if reward_from_updates == 0:
             step_reward -= 0.01

        self.score += step_reward

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # Loss condition
            self.game_over = True
            step_reward -= 100
            self.score -= 100

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _handle_input(self, action):
        """Processes player actions from the action space."""
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 2) # Prevent placing on base path

        # Cycle Tower Type
        if shift_held and (self.steps - self.last_action_time["cycle"] > self.FPS / 4):
            self.selected_tower_type = (self.selected_tower_type + 1) % self.NUM_TOWER_TYPES
            self.last_action_time["cycle"] = self.steps
            # sfx: UI_cycle

        # Place Tower
        if space_held and (self.steps - self.last_action_time["place"] > self.FPS / 2):
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.resources >= spec["cost"]:
                is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)
                if not is_occupied:
                    self.resources -= spec["cost"]
                    self.towers.append({
                        "grid_pos": list(self.cursor_pos),
                        "type": self.selected_tower_type,
                        "cooldown": 0,
                    })
                    self.last_action_time["place"] = self.steps
                    # sfx: place_tower

    def _update_game_state(self):
        """Master update function for all dynamic game elements."""
        reward = 0
        reward += self._update_aliens()
        reward += self._update_towers()
        reward += self._update_projectiles()
        self._update_particles()
        return reward

    def _update_aliens(self):
        """Moves aliens and checks if they reached the base."""
        reward = 0
        for alien in self.aliens[:]:
            alien['pos'][1] += alien['speed']
            if alien['pos'][1] >= self.HEIGHT - self.GRID_SIZE:
                self.aliens.remove(alien)
                self.base_health -= 10
                reward -= 5
                # sfx: base_hit
                self._create_explosion(alien['pos'], 40, (255,100,0))
        self.base_health = max(0, self.base_health)
        return reward

    def _update_towers(self):
        """Makes towers target and shoot at aliens."""
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            spec = self.TOWER_SPECS[tower['type']]
            tower_pos = (
                tower['grid_pos'][0] * self.GRID_SIZE + self.GRID_SIZE / 2,
                tower['grid_pos'][1] * self.GRID_SIZE + self.GRID_SIZE / 2,
            )

            target = None
            min_dist = spec['range'] ** 2
            
            for alien in self.aliens:
                dist_sq = (alien['pos'][0] - tower_pos[0])**2 + (alien['pos'][1] - tower_pos[1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = alien

            if target:
                tower['cooldown'] = spec['fire_rate']
                # sfx: shoot_laser or shoot_cannon
                self.projectiles.append({
                    "pos": list(tower_pos),
                    "target_pos": list(target['pos']),
                    "spec": spec
                })
        return 0

    def _update_projectiles(self):
        """Moves projectiles and checks for collisions with aliens."""
        reward = 0
        for proj in self.projectiles[:]:
            spec = proj['spec']
            dx = proj['target_pos'][0] - proj['pos'][0]
            dy = proj['target_pos'][1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < spec['proj_speed']:
                proj['pos'] = proj['target_pos']
            else:
                proj['pos'][0] += (dx / dist) * spec['proj_speed']
                proj['pos'][1] += (dy / dist) * spec['proj_speed']
            
            hit_alien = None
            for alien in self.aliens:
                if math.hypot(proj['pos'][0] - alien['pos'][0], proj['pos'][1] - alien['pos'][1]) < 10:
                    hit_alien = alien
                    break

            if hit_alien:
                self.projectiles.remove(proj)
                hit_alien['health'] -= spec['damage']
                reward += 0.1
                # sfx: alien_hit
                self._create_explosion(hit_alien['pos'], 15, spec['proj_color'])

                if hit_alien['health'] <= 0:
                    self.aliens.remove(hit_alien)
                    reward += 1
                    self.resources += 10
                    # sfx: alien_explode
                    self._create_explosion(hit_alien['pos'], 30, (255, 150, 50))
        return reward

    def _update_particles(self):
        """Animates and removes fading particles/explosions."""
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        """Creates a new wave of aliens."""
        num_aliens = 3 + (self.wave_number - 1)
        alien_speed = 1.0 + (self.wave_number - 1) * 0.05
        alien_health = 20 + (self.wave_number - 1) * 5
        
        for _ in range(num_aliens):
            self.aliens.append({
                'pos': [self.np_random.uniform(20, self.WIDTH - 20), self.np_random.uniform(-80, -20)],
                'health': alien_health,
                'max_health': alien_health,
                'speed': alien_speed,
            })

    def _check_termination(self):
        """Checks for game over conditions."""
        return self.base_health <= 0 or self.steps >= self.MAX_STEPS or (self.wave_number >= self.MAX_WAVES and not self.aliens)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "health": self.base_health, "resources": self.resources}

    def _render_game(self):
        """Renders all primary game elements."""
        self._render_grid()
        self._render_base()
        self._render_towers()
        self._render_aliens()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_base(self):
        base_rect = pygame.Rect(0, self.HEIGHT - self.GRID_SIZE, self.WIDTH, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, (100, 255, 150), base_rect, 3)

    def _render_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            pos = (
                int(tower['grid_pos'][0] * self.GRID_SIZE + self.GRID_SIZE / 2),
                int(tower['grid_pos'][1] * self.GRID_SIZE + self.GRID_SIZE / 2),
            )
            radius = self.GRID_SIZE // 3 + tower['type'] * 3
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, spec['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (255,255,255))
            # Cooldown indicator
            if tower['cooldown'] > 0:
                angle = (tower['cooldown'] / spec['fire_rate']) * 360
                arc_rect = pygame.Rect(pos[0]-radius, pos[1]-radius, radius*2, radius*2)
                pygame.draw.arc(self.screen, (255,255,255,100), arc_rect, math.radians(90), math.radians(90+angle), 2)


    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            p1 = (x, y)
            p2 = (x - 10, y - 15)
            p3 = (x + 10, y - 15)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_ALIEN)
    
    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            size = proj['spec']['proj_size']
            color = proj['spec']['proj_color']
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, (255,255,255,150))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            radius = int(p['radius'] * (1 - (p['life'] / p['max_life'])))
            if radius > 0:
                 pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _create_explosion(self, pos, radius, color):
        life = 15
        self.particles.append({'pos': list(pos), 'radius': radius, 'color': color, 'life': life, 'max_life': life})

    def _render_cursor(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        grid_x, grid_y = self.cursor_pos
        
        is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)
        can_afford = self.resources >= spec['cost']
        
        if is_occupied:
            cursor_color = (255, 0, 0, 100)
        elif not can_afford:
            cursor_color = (255, 255, 0, 100)
        else:
            cursor_color = (0, 255, 0, 100)

        # Draw placement box
        rect = pygame.Rect(grid_x * self.GRID_SIZE, grid_y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, cursor_color, surf.get_rect())
        self.screen.blit(surf, rect.topleft)

        # Draw range indicator
        center_pos = (
            int(grid_x * self.GRID_SIZE + self.GRID_SIZE / 2),
            int(grid_y * self.GRID_SIZE + self.GRID_SIZE / 2),
        )
        pygame.gfxdraw.aacircle(self.screen, center_pos[0], center_pos[1], spec['range'], (*cursor_color[:3], 150))

    def _render_ui(self):
        # Health Bar
        health_ratio = self.base_health / 100
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), bar_height))
        
        health_text = self.font_small.render(f"Base Health: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Resources
        res_text = self.font_small.render(f"Resources: {self.resources}", True, (255, 220, 100))
        self.screen.blit(res_text, (10, self.HEIGHT - 30))

        # Wave Number
        wave_text = self.font_large.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 15, 10))

        # Game Over / Win Text
        if self.game_over:
            if self.base_health <= 0:
                msg = "GAME OVER"
                color = self.COLOR_ALIEN
            else:
                msg = "VICTORY!"
                color = self.COLOR_HEALTH_BAR
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()