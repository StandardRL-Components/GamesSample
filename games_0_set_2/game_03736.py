
# Generated: 2025-08-28T00:15:29.030293
# Source Brief: brief_03736.md
# Brief Index: 3736

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to place or upgrade a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Minimalist tower defense. Survive 10 waves of enemies by placing and upgrading towers."
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Fonts and Colors
        self.font_s = pygame.font.SysFont("Consolas", 18)
        self.font_m = pygame.font.SysFont("Consolas", 24)
        self.font_l = pygame.font.SysFont("Consolas", 48)
        
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_PATH = (45, 50, 62)
        self.COLOR_BASE = (60, 120, 220)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TOWER = (255, 200, 0)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR_VALID = (0, 255, 0, 100)
        self.COLOR_CURSOR_UPGRADE = (255, 255, 0, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)
        
        # Game constants
        self.MAX_STEPS = 30 * 60 * 2 # 2 minutes at 30fps
        self.CURSOR_SPEED = 8
        self.TOWER_COSTS = {1: 75, 2: 150, 3: 300}
        self.TOWER_MAX_LEVEL = 3
        self.TOWER_SPECS = {
            1: {"range": 80, "cooldown": 45, "damage": 5}, # 1.5s
            2: {"range": 100, "cooldown": 30, "damage": 8}, # 1.0s
            3: {"range": 120, "cooldown": 15, "damage": 12} # 0.5s
        }
        self.ENEMY_PATH = self._define_path()
        self.BASE_POS = self.ENEMY_PATH[-1]
        self.BASE_RADIUS = 25
        self.MAX_WAVES = 10

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.wave_number = 0
        self.wave_state = "intermission" # spawning, active, intermission
        self.wave_timer = 0
        self.enemies_to_spawn = deque()
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.last_space_state = 0
        self.step_reward = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 100 # Starting score to build first tower
        self.game_over = False
        self.base_health = 100
        self.wave_number = 0
        self.wave_state = "intermission"
        self.wave_timer = 90 # 3 seconds
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.enemies_to_spawn.clear()

        self.cursor_pos = [self.screen_width // 2, self.screen_height // 2]
        self.last_space_state = 0
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.step_reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # Handle input
        self._handle_input(movement, space_held)
        
        # Update game logic
        self._update_wave_logic()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.screen_width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.screen_height)

        # Handle space press (on rising edge)
        if space_held and not self.last_space_state:
            self._place_or_upgrade_tower()
        self.last_space_state = space_held

    def _place_or_upgrade_tower(self):
        # Check for upgrade
        for tower in self.towers:
            dist = math.hypot(self.cursor_pos[0] - tower['pos'][0], self.cursor_pos[1] - tower['pos'][1])
            if dist < 10 and tower['level'] < self.TOWER_MAX_LEVEL:
                cost = self.TOWER_COSTS[tower['level'] + 1]
                if self.score >= cost:
                    self.score -= cost
                    tower['level'] += 1
                    spec = self.TOWER_SPECS[tower['level']]
                    tower.update(spec)
                    self._create_particles(tower['pos'], self.COLOR_TOWER, 30, 3, 5) # sfx: Upgrade
                    return

        # Check for placement
        cost = self.TOWER_COSTS[1]
        if self.score >= cost:
            # Prevent placing on path or too close to other towers
            if self._is_valid_placement(self.cursor_pos):
                self.score -= cost
                new_tower = {
                    'pos': list(self.cursor_pos),
                    'level': 1,
                    'fire_cooldown': 0,
                    **self.TOWER_SPECS[1]
                }
                self.towers.append(new_tower)
                self._create_particles(self.cursor_pos, self.COLOR_TOWER, 20, 2, 3) # sfx: PlaceTower

    def _is_valid_placement(self, pos):
        # Check path proximity
        for i in range(len(self.ENEMY_PATH) - 1):
            p1 = np.array(self.ENEMY_PATH[i])
            p2 = np.array(self.ENEMY_PATH[i+1])
            p3 = np.array(pos)
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1) if np.linalg.norm(p2-p1) != 0 else np.linalg.norm(p1-p3)
            if d < 30: return False
        
        # Check tower proximity
        for tower in self.towers:
            dist = math.hypot(pos[0] - tower['pos'][0], pos[1] - tower['pos'][1])
            if dist < 30: return False
            
        # Check base proximity
        dist_base = math.hypot(pos[0] - self.BASE_POS[0], pos[1] - self.BASE_POS[1])
        if dist_base < self.BASE_RADIUS + 15: return False
            
        return True

    def _update_wave_logic(self):
        if self.wave_state == "intermission":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()
        elif self.wave_state == "spawning":
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.enemies_to_spawn:
                self.enemies.append(self.enemies_to_spawn.popleft())
                self.wave_timer = 15 # 0.5s spawn interval
            elif not self.enemies_to_spawn:
                self.wave_state = "active"
        elif self.wave_state == "active":
            if not self.enemies and not self.enemies_to_spawn:
                self.step_reward += 100 # Wave clear reward
                if self.wave_number >= self.MAX_WAVES:
                    self.game_over = True # Win condition
                else:
                    self.wave_state = "intermission"
                    self.wave_timer = 90 # 3s break

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES: return

        self.wave_state = "spawning"
        self.wave_timer = 0
        
        num_enemies = 5 + (self.wave_number - 1) * 3
        enemy_health = 10 + (self.wave_number - 1) * 5
        enemy_speed = 1.0 + (self.wave_number - 1) * 0.1
        
        for _ in range(num_enemies):
            enemy = {
                'pos': list(self.ENEMY_PATH[0]),
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed,
                'path_index': 0,
            }
            self.enemies_to_spawn.append(enemy)

    def _update_towers(self):
        for tower in self.towers:
            tower['fire_cooldown'] = max(0, tower['fire_cooldown'] - 1)
            if tower['fire_cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    self._fire_projectile(tower, target)
                    tower['fire_cooldown'] = tower['cooldown']

    def _find_target(self, tower):
        closest_enemy = None
        min_dist = tower['range'] ** 2
        for enemy in self.enemies:
            dist_sq = (tower['pos'][0] - enemy['pos'][0])**2 + (tower['pos'][1] - enemy['pos'][1])**2
            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_enemy = enemy
        return closest_enemy

    def _fire_projectile(self, tower, target):
        # sfx: LaserShoot
        start_pos = list(tower['pos'])
        direction = math.atan2(target['pos'][1] - start_pos[1], target['pos'][0] - start_pos[0])
        projectile = {
            'pos': start_pos,
            'vel': [math.cos(direction) * 10, math.sin(direction) * 10],
            'damage': tower['damage'],
            'life': 30 # Lifespan in frames
        }
        self.projectiles.append(projectile)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['pos'][0] += proj['vel'][0]
            proj['pos'][1] += proj['vel'][1]
            proj['life'] -= 1

            hit = False
            for enemy in self.enemies[:]:
                if math.hypot(proj['pos'][0] - enemy['pos'][0], proj['pos'][1] - enemy['pos'][1]) < 8:
                    enemy['health'] -= proj['damage']
                    self.step_reward += 0.1 # Hit reward
                    self._create_particles(proj['pos'], self.COLOR_PROJECTILE, 5, 1, 2) # sfx: Hit
                    if enemy['health'] <= 0:
                        self._kill_enemy(enemy)
                    
                    self.projectiles.remove(proj)
                    hit = True
                    break
            
            if not hit and proj['life'] <= 0:
                self.projectiles.remove(proj)

    def _kill_enemy(self, enemy):
        # sfx: Explosion
        self._create_particles(enemy['pos'], self.COLOR_ENEMY, 40, 2, 8)
        self.enemies.remove(enemy)
        self.score += 10
        self.step_reward += 10 # Kill reward

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['path_index'] >= len(self.ENEMY_PATH) - 1:
                self.enemies.remove(enemy)
                self.base_health = max(0, self.base_health - 10)
                self.step_reward -= 5 # Health loss penalty
                self._create_particles(self.BASE_POS, self.COLOR_ENEMY, 50, 4, 10) # sfx: BaseDamage
                if self.base_health <= 0:
                    self.game_over = True
                    self.step_reward -= 200 # Game over penalty
                continue

            target_pos = self.ENEMY_PATH[enemy['path_index'] + 1]
            direction_vec = [target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1]]
            dist = math.hypot(*direction_vec)
            
            if dist < enemy['speed']:
                enemy['pos'] = list(target_pos)
                enemy['path_index'] += 1
            else:
                move_vec = [v / dist * enemy['speed'] for v in direction_vec]
                enemy['pos'][0] += move_vec[0]
                enemy['pos'][1] += move_vec[1]

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, speed_max, life_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, life_max * 10) / 10,
                'max_life': life_max,
                'color': color
            })

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.ENEMY_PATH, 20)
        
        # Render base
        pygame.gfxdraw.filled_circle(self.screen, int(self.BASE_POS[0]), int(self.BASE_POS[1]), self.BASE_RADIUS, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(self.BASE_POS[0]), int(self.BASE_POS[1]), self.BASE_RADIUS, self.COLOR_BASE)
        
        # Render towers and ranges
        for tower in self.towers:
            tx, ty = int(tower['pos'][0]), int(tower['pos'][1])
            color = self.COLOR_TOWER
            pygame.gfxdraw.filled_circle(self.screen, tx, ty, 8 + tower['level'], color)
            pygame.gfxdraw.aacircle(self.screen, tx, ty, 8 + tower['level'], color)
            
            # Range indicator
            range_color = (*color, 30)
            pygame.gfxdraw.filled_circle(self.screen, tx, ty, int(tower['range']), range_color)
            pygame.gfxdraw.aacircle(self.screen, tx, ty, int(tower['range']), (*color, 60))
        
        # Render enemies
        for enemy in self.enemies:
            ex, ey = int(enemy['pos'][0]), int(enemy['pos'][1])
            size = 7
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (ex - size, ey - size, size*2, size*2))
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, (0, 255, 0), (ex - size, ey - size - 5, size * 2 * health_pct, 3))

        # Render projectiles
        for proj in self.projectiles:
            px, py = int(proj['pos'][0]), int(proj['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, px, py, 3, self.COLOR_PROJECTILE)
        
        # Render particles
        for p in self.particles:
            alpha = max(0, (p['life'] / p['max_life']) * 255)
            color = (*p['color'], alpha)
            size = int(p['life'] / p['max_life'] * 3) + 1
            if size > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

        # Render cursor
        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        cursor_color = self.COLOR_CURSOR_INVALID
        can_upgrade = any(math.hypot(cx - t['pos'][0], cy - t['pos'][1]) < 10 and t['level'] < self.TOWER_MAX_LEVEL for t in self.towers)
        if can_upgrade:
            cursor_color = self.COLOR_CURSOR_UPGRADE
        elif self._is_valid_placement(self.cursor_pos):
            cursor_color = self.COLOR_CURSOR_VALID
        
        cursor_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(cursor_surf, 30, 30, 30, cursor_color)
        pygame.draw.line(cursor_surf, (255,255,255), (30, 20), (30, 40), 1)
        pygame.draw.line(cursor_surf, (255,255,255), (20, 30), (40, 30), 1)
        self.screen.blit(cursor_surf, (cx - 30, cy - 30))

        self._render_ui()

    def _render_ui(self):
        # Wave number
        wave_text = self.font_m.render(f"WAVE {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Score
        score_text = self.font_m.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.screen_width / 2, self.screen_height - 20))
        self.screen.blit(score_text, score_rect)

        # Base Health
        health_text = self.font_m.render(f"BASE HEALTH: {int(self.base_health)}%", True, self.COLOR_TEXT)
        health_rect = health_text.get_rect(topright=(self.screen_width - 10, 10))
        self.screen.blit(health_text, health_rect)

        # Game Over / Win message
        if self.game_over:
            if self.base_health <= 0:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            else:
                msg = "YOU WIN!"
                color = self.COLOR_TOWER
            
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_l.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.base_health,
        }
        
    def _define_path(self):
        return [
            (-20, 50),
            (100, 50),
            (100, 300),
            (250, 300),
            (250, 120),
            (450, 120),
            (450, 350),
            (self.screen_width - 50, 350),
            (self.screen_width - 50, 200),
        ]

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # For human play
    pygame.display.set_caption("Tower Defense Gym Environment")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    total_reward = 0
    
    # Game loop for human testing
    running = True
    while running:
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # 30 FPS

    pygame.quit()