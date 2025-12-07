
# Generated: 2025-08-28T05:19:15.562297
# Source Brief: brief_02585.md
# Brief Index: 2585

        
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower type. Press Space to place a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Place towers to defend your base "
        "from waves of enemies. Survive 10 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_SPAWN = (230, 25, 75)
        self.COLOR_ENEMY = (250, 250, 250)
        
        # Tower definitions
        self.TOWER_SPECS = {
            0: {"name": "Rapid", "color": (220, 50, 50), "range": 80, "fire_rate": 0.2, "damage": 5, "projectile_speed": 8},
            1: {"name": "Slowing", "color": (70, 150, 255), "range": 100, "fire_rate": 1.0, "damage": 2, "projectile_speed": 6, "effect": "slow", "effect_duration": 60},
            2: {"name": "Splash", "color": (255, 225, 25), "range": 60, "fire_rate": 1.5, "damage": 8, "projectile_speed": 5, "effect": "splash", "splash_radius": 30},
        }
        
        # Game constants
        self.MAX_STEPS = 4000
        self.MAX_WAVES = 10
        self.CURSOR_SPEED = 8
        
        # Path definition
        self.path = [
            (0, 100), (100, 100), (100, 300), (400, 300), 
            (400, 50), (540, 50), (540, 200)
        ]
        self.base_pos = (540, 200)
        self.base_size = (40, 40)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.last_space_held = False
        self.last_shift_held = False
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""

        # Game state
        self.wave_number = 0
        self.wave_cooldown = 150 # Time before first wave
        self.enemies_in_wave = 0
        self.enemies_spawned_this_wave = 0
        self.base_enemy_speed = 1.0
        self.can_place_tower = True

        # Cursor and tower selection
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.selected_tower_type = 0

        # Action state
        self.last_space_held = True # Prevent placing tower on first frame
        self.last_shift_held = True

        # Clear game objects
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # 1. Handle player actions
        self._handle_input(movement, space_held, shift_held)
        
        # 2. Update game logic
        wave_clear_bonus = self._update_waves()
        reward += wave_clear_bonus
        
        if self.wave_cooldown <= 0:
            self._spawn_enemies()
            
        self._update_towers()
        reward += self._update_projectiles()
        
        enemy_defeat_reward, enemy_reach_base = self._update_enemies()
        reward += enemy_defeat_reward
        
        self._update_particles()
        
        self.score += reward

        # 3. Check for termination
        terminated, term_reward = self._check_termination(enemy_reach_base)
        reward += term_reward
        if term_reward != 0: self.score += term_reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Cycle tower type
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        
        # Place tower
        if space_held and not self.last_space_held and self.can_place_tower:
            if self._is_valid_placement(self.cursor_pos):
                spec = self.TOWER_SPECS[self.selected_tower_type]
                self.towers.append({
                    "pos": self.cursor_pos.copy(),
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    **spec
                })
                # sfx: place_tower.wav
                self.can_place_tower = False

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _is_valid_placement(self, pos):
        # Check distance to path segments
        for i in range(len(self.path) - 1):
            p1 = np.array(self.path[i])
            p2 = np.array(self.path[i+1])
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0:
                if np.linalg.norm(pos - p1) < 20: return False
            else:
                t = max(0, min(1, np.dot(pos - p1, p2 - p1) / l2))
                projection = p1 + t * (p2 - p1)
                if np.linalg.norm(pos - projection) < 20: return False
        
        for tower in self.towers:
            if np.linalg.norm(pos - tower['pos']) < 30: return False
        
        base_center = np.array([self.base_pos[0] + self.base_size[0]/2, self.base_pos[1] + self.base_size[1]/2])
        if np.linalg.norm(pos - base_center) < 40: return False
            
        return True

    def _update_waves(self):
        reward = 0
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            if self.wave_cooldown == 0:
                self.wave_number += 1
                if self.wave_number <= self.MAX_WAVES:
                    self.enemies_in_wave = 5 + self.wave_number * 2
                    self.enemies_spawned_this_wave = 0
                    self.spawn_timer = 0
        elif len(self.enemies) == 0 and self.enemies_spawned_this_wave >= self.enemies_in_wave:
            if self.wave_number <= self.MAX_WAVES and self.wave_number > 0:
                reward = 10 # Wave survival bonus
                self.wave_cooldown = 180 # Cooldown until next wave
                self.can_place_tower = True # Allow placing one tower per wave
                self.base_enemy_speed += 0.05
        return reward

    def _spawn_enemies(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and self.enemies_spawned_this_wave < self.enemies_in_wave:
            self.enemies.append({
                "pos": np.array(self.path[0], dtype=float),
                "health": 10 + self.wave_number * 5,
                "max_health": 10 + self.wave_number * 5,
                "speed": self.base_enemy_speed,
                "path_index": 1,
                "slow_timer": 0,
            })
            self.enemies_spawned_this_wave += 1
            self.spawn_timer = 30 # Ticks between spawns

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = None
                min_dist = tower['range']
                for enemy in self.enemies:
                    dist = np.linalg.norm(tower['pos'] - enemy['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    self.projectiles.append({
                        "pos": tower['pos'].copy(), "type": tower['type'],
                        "damage": tower['damage'], "speed": tower['projectile_speed'],
                        "target": target, "effect": tower.get('effect'),
                        "effect_duration": tower.get('effect_duration'),
                        "splash_radius": tower.get('splash_radius')
                    })
                    tower['cooldown'] = tower['fire_rate'] * 30 # fire_rate in seconds
                    # sfx: shoot.wav

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj['target']['pos']
            direction = target_pos - proj['pos']
            dist = np.linalg.norm(direction)
            
            if dist < proj['speed']:
                reward += self._handle_projectile_hit(proj)
                self.projectiles.remove(proj)
            else:
                proj['pos'] += (direction / dist) * proj['speed']
        return reward

    def _handle_projectile_hit(self, proj):
        reward = 0
        # sfx: hit.wav
        self._create_explosion(proj['pos'], self.TOWER_SPECS[proj['type']]['color'], 10)
        
        if proj['effect'] == 'splash':
            # sfx: explosion.wav
            self._create_explosion(proj['pos'], self.TOWER_SPECS[proj['type']]['color'], proj['splash_radius'], 15)
            for enemy in self.enemies:
                if np.linalg.norm(enemy['pos'] - proj['pos']) < proj['splash_radius']:
                    enemy['health'] -= proj['damage']
                    reward += 0.1
        else:
            target = proj['target']
            target['health'] -= proj['damage']
            reward += 0.1
            if proj['effect'] == 'slow':
                target['slow_timer'] = proj['effect_duration']
        return reward

    def _update_enemies(self):
        reward = 0
        base_reached = False
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.enemies.remove(enemy)
                reward += 1.0 # Defeated enemy bonus
                # sfx: enemy_die.wav
                self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 20)
                continue

            if enemy['path_index'] >= len(self.path):
                self.enemies.remove(enemy)
                base_reached = True
                # sfx: base_hit.wav
                self._create_explosion(np.array(self.base_pos), self.COLOR_BASE, 50, 30)
                continue

            target_pos = np.array(self.path[enemy['path_index']])
            direction = target_pos - enemy['pos']
            dist = np.linalg.norm(direction)
            
            speed = enemy['speed'] * (0.5 if enemy['slow_timer'] > 0 else 1.0)
            if enemy['slow_timer'] > 0: enemy['slow_timer'] -= 1
            
            if dist < speed:
                enemy['pos'] = target_pos
                enemy['path_index'] += 1
            else:
                enemy['pos'] += (direction / dist) * speed
                
        return reward, base_reached

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] *= 0.95
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, radius, num_particles=20):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel,
                'lifetime': self.np_random.integers(10, 20),
                'radius': radius / 2 * self.np_random.uniform(0.5, 1.2),
                'color': color
            })

    def _check_termination(self, enemy_reach_base):
        if enemy_reach_base:
            self.game_over = True
            self.game_over_message = "GAME OVER"
            return True, -100.0 # Loss
        if self.wave_number > self.MAX_WAVES:
            self.game_over = True
            self.game_over_message = "YOU WIN!"
            return True, 0.0 # Win (reward is from wave clears)
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_over_message = "TIME UP"
            return True, 0.0 # Time limit
        return False, 0.0

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)
        pygame.draw.circle(self.screen, self.COLOR_SPAWN, self.path[0], 15)
        
        base_rect = pygame.Rect(*self.base_pos, *self.base_size)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_BASE), base_rect, 3)

        for tower in self.towers:
            pos = tuple(map(int, tower['pos']))
            color = tower['color']
            p1, p2, p3 = (pos[0], pos[1] - 10), (pos[0] - 8, pos[1] + 6), (pos[0] + 8, pos[1] + 6)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], color)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], color)

        for enemy in self.enemies:
            pos = tuple(map(int, enemy['pos']))
            size = 10
            rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            health_ratio = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, (255, 0, 0), (rect.left, rect.top - 5, size, 3))
            pygame.draw.rect(self.screen, (0, 255, 0), (rect.left, rect.top - 5, size * health_ratio, 3))
            if enemy['slow_timer'] > 0:
                pygame.gfxdraw.box(self.screen, rect, (70, 150, 255, 100))

        for proj in self.projectiles:
            pos = tuple(map(int, proj['pos']))
            color = self.TOWER_SPECS[proj['type']]['color']
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, (*color, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, color)

        for p in self.particles:
            pos = tuple(map(int, p['pos']))
            alpha = int(255 * (p['lifetime'] / 20))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

    def _render_ui(self):
        if self.can_place_tower:
            cursor_pos = tuple(map(int, self.cursor_pos))
            spec = self.TOWER_SPECS[self.selected_tower_type]
            range_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            is_valid = self._is_valid_placement(self.cursor_pos)
            range_color = (255, 255, 255, 30) if is_valid else (255, 0, 0, 30)
            pygame.draw.circle(range_surf, range_color, cursor_pos, spec['range'])
            self.screen.blit(range_surf, (0,0))
            
            preview_surf = pygame.Surface((24, 24), pygame.SRCALPHA)
            p1, p2, p3 = (12, 2), (4, 18), (20, 18)
            preview_color = (*spec['color'], 150)
            pygame.gfxdraw.filled_polygon(preview_surf, [p1, p2, p3], preview_color)
            self.screen.blit(preview_surf, (cursor_pos[0] - 12, cursor_pos[1] - 12))

        bar_surf = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        bar_surf.fill((10, 10, 15, 200))
        
        wave_text = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        if self.wave_cooldown > 0 and self.wave_number < self.MAX_WAVES:
            wave_text = f"Next wave in: {self.wave_cooldown / 30:.1f}s"
        text_surf = self.font_m.render(wave_text, True, (220, 220, 220))
        bar_surf.blit(text_surf, (10, 8))
        
        score_text = f"Score: {int(self.score)}"
        text_surf = self.font_m.render(score_text, True, (220, 220, 220))
        bar_surf.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 8))
        
        if self.can_place_tower:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            tower_text = f"Selected: {spec['name']}"
            text_surf = self.font_m.render(tower_text, True, spec['color'])
            bar_surf.blit(text_surf, (self.WIDTH/2 - text_surf.get_width()/2, 8))
        else:
            tower_text = "Place one tower per wave"
            text_surf = self.font_s.render(tower_text, True, (150, 150, 150))
            bar_surf.blit(text_surf, (self.WIDTH/2 - text_surf.get_width()/2, 12))

        self.screen.blit(bar_surf, (0,0))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            text_surf = self.font_l.render(self.game_over_message, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0,0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "towers_placed": len(self.towers),
            "enemies_on_screen": len(self.enemies)
        }

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        if keys[pygame.K_r]:
            obs, info = env.reset()
            done = False
            
        action = [movement, space_held, shift_held]
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()