
# Generated: 2025-08-28T03:45:35.384640
# Source Brief: brief_02099.md
# Brief Index: 2099

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Arrow keys to move the cursor. Space to build/upgrade towers. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from zombie waves by placing and upgrading towers. Survive all 20 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Constants ---
        self.CURSOR_SPEED = 8
        self.MAX_TOWER_LEVEL = 3

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_BASE_DMG = (200, 50, 50)
        self.COLOR_ZOMBIE = (200, 40, 40)
        self.COLOR_TOWER_SPOTS = (60, 70, 80, 100)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.TOWER_COLORS = {
            1: (50, 150, 255),
            2: (100, 200, 255),
            3: (180, 240, 255)
        }
        self.PROJECTILE_COLOR = (255, 255, 100)
        
        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game Parameters
        self.BASE_MAX_HEALTH = 100
        self.INITIAL_GOLD = 150
        self.WAVE_INTERVAL = 600 # steps (20 seconds at 30fps)
        self.MAX_WAVES = 20
        self.MAX_STEPS = self.WAVE_INTERVAL * (self.MAX_WAVES + 1)
        
        self.TOWER_SPECS = {
            1: {'cost': 50, 'range': 80, 'damage': 10, 'cooldown': 30},
            2: {'cost': 75, 'range': 100, 'damage': 15, 'cooldown': 25},
            3: {'cost': 100, 'range': 120, 'damage': 25, 'cooldown': 20}
        }

        # Pathfinding & Layout
        self.BASE_POS = (self.WIDTH - 40, self.HEIGHT - 80)
        self.PATH_WAYPOINTS = [
            (-20, 80), (self.WIDTH - 150, 80), (self.WIDTH - 150, self.HEIGHT - 80), self.BASE_POS
        ]
        self.TOWER_SPOTS = [
            (100, 150), (200, 150), (300, 150), (400, 150),
            (100, 250), (200, 250), (300, 250), (400, 250),
            (self.WIDTH - 80, 150), (self.WIDTH - 80, 20)
        ]
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.gold = 0
        self.wave_number = 0
        self.wave_timer = 0
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 1
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                 self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.BASE_MAX_HEALTH
        self.gold = self.INITIAL_GOLD
        self.wave_number = 0
        self.wave_timer = self.WAVE_INTERVAL // 3 # First wave comes sooner
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.selected_tower_type = 1
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.001 # Small reward for surviving each step
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        if not self.game_over:
            self._handle_input(movement, space_held, shift_held)
            
            # Update game logic
            self.steps += 1
            self.wave_timer -= 1

            if self.wave_timer <= 0 and self.wave_number < self.MAX_WAVES:
                self._spawn_wave()
            
            zombie_kill_reward = self._update_game_state()
            reward += zombie_kill_reward
        
        self.score += reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.base_health <= 0:
                reward = -100.0
            else: # Won
                reward = 100.0
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)
        
        # Actions (on press)
        if space_held and not self.last_space_held:
            self._place_or_upgrade_tower()
        if shift_held and not self.last_shift_held:
            self._cycle_tower_type()
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        self._update_towers()
        self._update_projectiles()
        zombie_kill_reward = self._update_zombies()
        self._update_particles()
        return zombie_kill_reward

    def _spawn_wave(self):
        self.wave_number += 1
        self.wave_timer = self.WAVE_INTERVAL
        num_zombies = 5 + (self.wave_number - 1) * 2
        base_health = 20 + self.wave_number * 5
        
        for i in range(num_zombies):
            spawn_x = self.PATH_WAYPOINTS[0][0] - i * 20
            spawn_y = self.PATH_WAYPOINTS[0][1] + self.np_random.uniform(-10, 10)
            zombie_health = base_health * self.np_random.uniform(0.9, 1.1)
            self.zombies.append({
                'pos': np.array([spawn_x, spawn_y], dtype=float),
                'health': zombie_health,
                'max_health': zombie_health,
                'speed': self.np_random.uniform(0.8, 1.2),
                'waypoint_idx': 0
            })

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown_timer'] = max(0, tower['cooldown_timer'] - 1)
            if tower['cooldown_timer'] > 0:
                continue

            target = None
            min_dist = tower['range']
            for zombie in self.zombies:
                dist = np.linalg.norm(tower['pos'] - zombie['pos'])
                if dist < min_dist:
                    min_dist = dist
                    target = zombie
            
            if target:
                # Fire projectile
                self.projectiles.append({
                    'pos': np.array(tower['pos'], dtype=float),
                    'target': target,
                    'speed': 10,
                    'damage': tower['damage']
                })
                tower['cooldown_timer'] = tower['cooldown']
                # sfx: tower_shoot.wav

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.zombies:
                self.projectiles.remove(proj)
                continue
            
            direction = proj['target']['pos'] - proj['pos']
            dist = np.linalg.norm(direction)
            
            if dist < proj['speed']:
                proj['target']['health'] -= proj['damage']
                self._create_particles(proj['pos'], 5, self.PROJECTILE_COLOR)
                self.projectiles.remove(proj)
                # sfx: zombie_hit.wav
                continue

            direction = direction / dist
            proj['pos'] += direction * proj['speed']

    def _update_zombies(self):
        zombie_kill_reward = 0
        for z in self.zombies[:]:
            if z['health'] <= 0:
                self.zombies.remove(z)
                self.gold += 5 + self.wave_number
                zombie_kill_reward += 1.0
                # sfx: zombie_die.wav
                continue
            
            if z['waypoint_idx'] >= len(self.PATH_WAYPOINTS):
                self.base_health -= 10
                self._create_particles(z['pos'], 10, self.COLOR_BASE_DMG)
                self.zombies.remove(z)
                # sfx: base_damage.wav
                continue

            target_waypoint = np.array(self.PATH_WAYPOINTS[z['waypoint_idx']])
            direction = target_waypoint - z['pos']
            dist = np.linalg.norm(direction)

            if dist < z['speed']:
                z['waypoint_idx'] += 1
            else:
                z['pos'] += (direction / dist) * z['speed']
        
        return zombie_kill_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            p['radius'] -= 0.2
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _place_or_upgrade_tower(self):
        cursor_pos_tuple = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        
        # Find closest tower spot
        closest_spot = None
        min_dist = 30 # Cursor must be close to a spot
        for spot in self.TOWER_SPOTS:
            dist = np.linalg.norm(np.array(cursor_pos_tuple) - np.array(spot))
            if dist < min_dist:
                min_dist = dist
                closest_spot = spot
        
        if not closest_spot:
            return

        # Check if a tower is already on this spot
        existing_tower = None
        for t in self.towers:
            if t['spot_pos'] == closest_spot:
                existing_tower = t
                break
        
        if existing_tower:
            # Upgrade logic
            current_level = existing_tower['level']
            if self.selected_tower_type == current_level + 1 and current_level < self.MAX_TOWER_LEVEL:
                spec = self.TOWER_SPECS[self.selected_tower_type]
                if self.gold >= spec['cost']:
                    self.gold -= spec['cost']
                    existing_tower.update({
                        'level': self.selected_tower_type,
                        **spec
                    })
                    self._create_particles(existing_tower['pos'], 15, (255, 255, 255))
                    # sfx: upgrade_success.wav
        else:
            # Build new tower logic
            if self.selected_tower_type == 1:
                spec = self.TOWER_SPECS[1]
                if self.gold >= spec['cost']:
                    self.gold -= spec['cost']
                    self.towers.append({
                        'pos': np.array(closest_spot, dtype=float),
                        'spot_pos': closest_spot,
                        'level': 1,
                        'cooldown_timer': 0,
                        **spec
                    })
                    self._create_particles(closest_spot, 15, (255, 255, 255))
                    # sfx: build_success.wav

    def _cycle_tower_type(self):
        self.selected_tower_type += 1
        if self.selected_tower_type > self.MAX_TOWER_LEVEL:
            self.selected_tower_type = 1
        # sfx: ui_cycle.wav

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': velocity,
                'life': self.np_random.integers(10, 20),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })
    
    def _check_termination(self):
        win_condition = self.wave_number >= self.MAX_WAVES and not self.zombies
        lose_condition = self.base_health <= 0
        timeout = self.steps >= self.MAX_STEPS
        return win_condition or lose_condition or timeout
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_path()
        self._render_tower_spots()
        self._render_base()
        self._render_towers()
        self._render_zombies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_path(self):
        for i in range(len(self.PATH_WAYPOINTS) - 1):
            p1 = self.PATH_WAYPOINTS[i]
            p2 = self.PATH_WAYPOINTS[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, 80)
            pygame.gfxdraw.filled_circle(self.screen, int(p1[0]), int(p1[1]), 40, self.COLOR_PATH)
            pygame.gfxdraw.filled_circle(self.screen, int(p2[0]), int(p2[1]), 40, self.COLOR_PATH)

    def _render_tower_spots(self):
        for spot in self.TOWER_SPOTS:
            pygame.gfxdraw.aacircle(self.screen, spot[0], spot[1], 20, self.COLOR_TOWER_SPOTS)
    
    def _render_base(self):
        base_rect = pygame.Rect(self.BASE_POS[0] - 20, self.BASE_POS[1] - 20, 40, 40)
        health_perc = max(0, self.base_health / self.BASE_MAX_HEALTH)
        color = self.COLOR_BASE if health_perc > 0.3 else self.COLOR_BASE_DMG
        pygame.draw.rect(self.screen, color, base_rect)
        
        # Health bar
        bar_w = 60
        bar_h = 8
        bar_x = self.BASE_POS[0] - bar_w / 2
        bar_y = self.BASE_POS[1] + 30
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_w * health_perc, bar_h))

    def _render_towers(self):
        for tower in self.towers:
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            level = tower['level']
            # Draw range
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], tower['range'], (*self.TOWER_COLORS[level], 50))
            
            # Draw tower body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, self.TOWER_COLORS[1])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.TOWER_COLORS[level])
            if level >= 2:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.TOWER_COLORS[2])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.TOWER_COLORS[level])
            if level >= 3:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.TOWER_COLORS[3])

    def _render_zombies(self):
        for z in self.zombies:
            pos = (int(z['pos'][0]), int(z['pos'][1]))
            size = 12
            rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, rect)
            
            # Health bar
            health_perc = max(0, z['health'] / z['max_health'])
            bar_w = 20
            bar_h = 3
            bar_x = pos[0] - bar_w / 2
            bar_y = pos[1] - size/2 - 5
            pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0, 200, 0), (bar_x, bar_y, bar_w * health_perc, bar_h))

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.PROJECTILE_COLOR)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.PROJECTILE_COLOR)

    def _render_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

    def _render_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        
        # Pulsating glow
        glow_size = 15 + 5 * math.sin(self.steps * 0.2)
        pygame.gfxdraw.aacircle(self.screen, x, y, int(glow_size), (*self.COLOR_CURSOR, 50))
        
        # Crosshair
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x-10, y), (x+10, y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y-10), (x, y+10), 1)

        # Show selected tower type
        spec = self.TOWER_SPECS[self.selected_tower_type]
        text = f"L{self.selected_tower_type} ${spec['cost']}"
        text_surf = self.font_ui.render(text, True, self.COLOR_CURSOR)
        self.screen.blit(text_surf, (x + 15, y - 10))

    def _render_ui(self):
        ui_bar = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA)
        ui_bar.fill((0, 0, 0, 128))
        self.screen.blit(ui_bar, (0, 0))

        health_text = self.font_ui.render(f"♥ {int(self.base_health)}/{self.BASE_MAX_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 5))

        gold_text = self.font_ui.render(f"♦ {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (160, 5))

        wave_text = self.font_ui.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (280, 5))

        # Next wave timer bar
        if self.wave_number < self.MAX_WAVES:
            timer_w = 150
            timer_perc = self.wave_timer / self.WAVE_INTERVAL
            pygame.draw.rect(self.screen, (60,60,60), (self.WIDTH - timer_w - 10, 8, timer_w, 14))
            pygame.draw.rect(self.screen, (100,100,200), (self.WIDTH - timer_w - 10, 8, timer_w * timer_perc, 14))
            next_wave_text = self.font_ui.render("Next Wave", True, self.COLOR_TEXT)
            self.screen.blit(next_wave_text, (self.WIDTH - timer_w / 2 - 50, 5))
            
    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        text = "VICTORY" if self.base_health > 0 else "DEFEAT"
        color = (100, 255, 100) if self.base_health > 0 else (255, 100, 100)
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "gold": self.gold,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")