
# Generated: 2025-08-28T02:44:10.852543
# Source Brief: brief_04550.md
# Brief Index: 4550

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities
class Tower:
    def __init__(self, pos, spec, spec_idx):
        self.pos = pygame.Vector2(pos)
        self.spec = spec
        self.spec_idx = spec_idx
        self.cooldown = 0
        self.target = None

class Enemy:
    def __init__(self, pos, health, speed, gold_value):
        self.pos = pygame.Vector2(pos)
        self.health = health
        self.max_health = health
        self.speed = speed
        self.gold_value = gold_value
        self.path_idx = 1
        self.is_alive = True

class Projectile:
    def __init__(self, start_pos, target_enemy, damage, speed):
        self.pos = pygame.Vector2(start_pos)
        self.target = target_enemy
        self.damage = damage
        self.speed = speed
        direction = (self.target.pos - self.pos).normalize()
        self.velocity = direction * speed

class Particle:
    def __init__(self, pos, color, life, size, velocity_spread):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, velocity_spread)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to select a build location. Press Shift to cycle through tower types. "
        "Press Space to build the selected tower."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of enemies. "
        "Survive all waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3000
    MAX_WAVES = 20

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_BORDER = (0, 200, 80)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_ENEMY_BORDER = (255, 100, 100)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GOLD = (255, 215, 0)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_GRID_VALID = (100, 100, 100, 100)
    COLOR_GRID_INVALID = (150, 50, 50, 100)
    
    TOWER_SPECS = [
        {"name": "Cannon", "cost": 100, "damage": 10, "range": 80, "fire_rate": 0.8, "color": (50, 150, 255)}, # Blue Triangle
        {"name": "Artillery", "cost": 150, "damage": 35, "range": 120, "fire_rate": 2.0, "color": (255, 200, 50)}, # Yellow Square
        {"name": "Sniper", "cost": 200, "damage": 25, "range": 200, "fire_rate": 1.5, "color": (200, 50, 255)}, # Purple Pentagon
    ]
    
    PATH_WAYPOINTS = [
        (-20, 200), (80, 200), (80, 80), (240, 80), (240, 320),
        (400, 320), (400, 150), (560, 150), (560, 250), (700, 250)
    ]
    
    TOWER_GRID_DIMS = (6, 4)
    TOWER_SLOTS = [
        (160, 40), (160, 120), (160, 200), (160, 280),
        (320, 40), (320, 120), (320, 200), (320, 280),
        (480, 40), (480, 120), (480, 240), (480, 320),
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 18)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.gold = 150
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.inter_wave_timer = 3 * self.FPS # 3 seconds until first wave
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_idx = 0
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.input_cooldown = 0

        self.occupied_slots = [False] * len(self.TOWER_SLOTS)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.FPS)
        self.steps += 1
        
        reward = -0.01  # Time penalty

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- Update Game State ---
        update_rewards = self._update_game_state()
        reward += update_rewards

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
            else:
                reward += -100

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Cooldown for movement to make it less sensitive
        if self.input_cooldown > 0:
            self.input_cooldown -= 1
        
        if movement != 0 and self.input_cooldown == 0:
            self.input_cooldown = 5 # frames
            row, col = self.cursor_idx // 4, self.cursor_idx % 4
            if movement == 1: # Up
                row = max(0, row - 1)
            elif movement == 2: # Down
                row = min(2, row + 1)
            elif movement == 3: # Left
                col = max(0, col - 1)
            elif movement == 4: # Right
                col = min(3, col + 1)
            
            # This logic is a bit complex to map a 1D index to a non-uniform grid
            # Let's simplify and just move the index
            if movement == 1: # Up
                self.cursor_idx = max(0, self.cursor_idx - 4) if self.cursor_idx >= 4 else self.cursor_idx
            elif movement == 2: # Down
                self.cursor_idx = min(len(self.TOWER_SLOTS) - 1, self.cursor_idx + 4) if self.cursor_idx < 8 else self.cursor_idx
            elif movement == 3: # Left
                self.cursor_idx = max(0, self.cursor_idx - 1)
            elif movement == 4: # Right
                self.cursor_idx = min(len(self.TOWER_SLOTS) - 1, self.cursor_idx + 1)

        # Cycle tower type on shift press (rising edge)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: ui_switch
        
        # Place tower on space press (rising edge)
        if space_held and not self.prev_space_held:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.gold >= spec["cost"] and not self.occupied_slots[self.cursor_idx]:
                pos = self.TOWER_SLOTS[self.cursor_idx]
                self.towers.append(Tower(pos, spec, self.selected_tower_type))
                self.gold -= spec["cost"]
                self.occupied_slots[self.cursor_idx] = True
                # sfx: build_tower
            else:
                # sfx: build_fail
                pass

    def _update_game_state(self):
        reward = 0
        reward += self._update_waves()
        self._update_towers()
        self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        return reward

    def _update_waves(self):
        reward = 0
        if not self.wave_in_progress:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                self.wave_number += 1
                if self.wave_number > self.MAX_WAVES:
                    self.win = True
                else:
                    self._spawn_wave()
                    self.wave_in_progress = True
        elif self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            self.inter_wave_timer = 5 * self.FPS # 5 seconds between waves
            reward += 1.0 # Wave clear bonus
            # sfx: wave_cleared
        return reward

    def _spawn_wave(self):
        num_enemies = 3 + self.wave_number * 2
        base_health = 20 * (1.1 ** self.wave_number)
        base_speed = 0.8 * (1.05 ** self.wave_number)
        gold_value = 5 + self.wave_number
        
        for i in range(num_enemies):
            spawn_pos = (self.PATH_WAYPOINTS[0][0] - i * 25, self.PATH_WAYPOINTS[0][1])
            self.enemies.append(Enemy(spawn_pos, base_health, base_speed, gold_value))
        # sfx: wave_start

    def _update_towers(self):
        for tower in self.towers:
            if tower.cooldown > 0:
                tower.cooldown -= 1
                continue
            
            if tower.target and (not tower.target.is_alive or tower.pos.distance_to(tower.target.pos) > tower.spec["range"]):
                tower.target = None

            if not tower.target:
                for enemy in self.enemies:
                    if tower.pos.distance_to(enemy.pos) <= tower.spec["range"]:
                        tower.target = enemy
                        break
            
            if tower.target and tower.cooldown <= 0:
                self.projectiles.append(Projectile(tower.pos, tower.target, tower.spec["damage"], 8))
                tower.cooldown = tower.spec["fire_rate"] * self.FPS
                # sfx: tower_shoot

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj.pos += proj.velocity
            if proj.pos.distance_to(proj.target.pos) < 10:
                proj.target.health -= proj.damage
                for _ in range(5): # Hit particle effect
                    self.particles.append(Particle(proj.pos, (255, 255, 200), 10, 2, 1.5))
                self.projectiles.remove(proj)
                # sfx: projectile_hit
            elif not (0 <= proj.pos.x < self.SCREEN_WIDTH and 0 <= proj.pos.y < self.SCREEN_HEIGHT):
                self.projectiles.remove(proj)

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy.health <= 0:
                enemy.is_alive = False
                self.gold += enemy.gold_value
                reward += 0.1 # Kill reward
                self.enemies.remove(enemy)
                for _ in range(15): # Death particle effect
                    self.particles.append(Particle(enemy.pos, self.COLOR_ENEMY, 20, 3, 2.5))
                # sfx: enemy_death
                continue

            if enemy.path_idx >= len(self.PATH_WAYPOINTS):
                self.base_health -= 10
                enemy.is_alive = False
                self.enemies.remove(enemy)
                # sfx: base_damage
                continue

            target_pos = pygame.Vector2(self.PATH_WAYPOINTS[enemy.path_idx])
            direction = (target_pos - enemy.pos).normalize()
            enemy.pos += direction * enemy.speed

            if enemy.pos.distance_to(target_pos) < enemy.speed:
                enemy.path_idx += 1
        return reward
    
    def _update_particles(self):
        for p in self.particles[:]:
            p.pos += p.velocity
            p.life -= 1
            if p.life <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.win = False
            return True
        if self.win:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH_WAYPOINTS, 35)
        
        # Render Tower Slots & Cursor
        for i, pos in enumerate(self.TOWER_SLOTS):
            is_occupied = self.occupied_slots[i]
            is_cursor_on = (i == self.cursor_idx)
            
            rect = pygame.Rect(pos[0] - 20, pos[1] - 20, 40, 40)
            color = self.COLOR_GRID_INVALID if is_occupied else self.COLOR_GRID_VALID
            
            # Use a separate surface for transparency
            s = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), border_radius=4)
            self.screen.blit(s, rect.topleft)

            if is_cursor_on:
                flash_alpha = 128 + 127 * math.sin(self.steps * 0.3)
                pygame.draw.rect(self.screen, (*self.COLOR_CURSOR, flash_alpha), rect, 2, border_radius=4)
    
        # Render Base
        base_rect = pygame.Rect(self.PATH_WAYPOINTS[-1][0]-15, self.PATH_WAYPOINTS[-1][1]-15, 30, 30)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE_BORDER, base_rect, 2, border_radius=5)

        # Render Towers
        for tower in self.towers:
            self._render_tower(tower.pos, tower.spec_idx, tower.spec["color"])

        # Render Enemies
        for enemy in self.enemies:
            pos_int = (int(enemy.pos.x), int(enemy.pos.y))
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, *pos_int, 8, self.COLOR_ENEMY_BORDER)
            # Health bar
            health_pct = max(0, enemy.health / enemy.max_health)
            bar_w = 16
            bar_h = 3
            bar_x = enemy.pos.x - bar_w / 2
            bar_y = enemy.pos.y - 15
            pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0,200,0), (bar_x, bar_y, bar_w * health_pct, bar_h))

        # Render Projectiles
        for proj in self.projectiles:
            start = (int(proj.pos.x), int(proj.pos.y))
            end_vec = proj.pos - proj.velocity * 0.5
            end = (int(end_vec.x), int(end_vec.y))
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, start, end, 2)

        # Render Particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            pos_int = (int(p.pos.x), int(p.pos.y))
            size = int(p.size * (p.life / p.max_life))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (pos_int[0] - size, pos_int[1] - size))

    def _render_tower(self, pos, spec_idx, color):
        size = 12
        if spec_idx == 0: # Triangle
            points = [
                (pos[0], pos[1] - size),
                (pos[0] - size * 0.866, pos[1] + size * 0.5),
                (pos[0] + size * 0.866, pos[1] + size * 0.5),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif spec_idx == 1: # Square
            rect = pygame.Rect(pos[0] - size*0.8, pos[1] - size*0.8, size*1.6, size*1.6)
            pygame.draw.rect(self.screen, color, rect)
        elif spec_idx == 2: # Pentagon
            points = []
            for i in range(5):
                angle_rad = math.pi / 2 + (2 * math.pi * i / 5)
                x = pos[0] + size * math.cos(angle_rad)
                y = pos[1] + size * math.sin(angle_rad)
                points.append((x, y))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        # Top Left: Wave Info
        wave_text = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        if not self.wave_in_progress and not self.win:
            next_wave_in = self.inter_wave_timer / self.FPS
            wave_text += f" (Next in {next_wave_in:.1f}s)"
        self._draw_text(wave_text, (10, 10), self.font_main, self.COLOR_TEXT)
        
        # Top Right: Gold
        gold_text = f"Gold: {self.gold}"
        self._draw_text(gold_text, (self.SCREEN_WIDTH - 10, 10), self.font_main, self.COLOR_GOLD, align="topright")
        
        # Bottom Left: Base Health
        health_text = f"Base Health: {max(0, self.base_health)}/{self.max_base_health}"
        self._draw_text(health_text, (10, self.SCREEN_HEIGHT - 10), self.font_main, self.COLOR_TEXT, align="bottomleft")
        
        # Bottom Center: Selected Tower Info
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_info_text = f"Selected: {spec['name']} | Cost: {spec['cost']}"
        self._draw_text(tower_info_text, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10), self.font_main, spec['color'], align="midbottom")
        
        # Game Over / Win Screen
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            self._draw_text(message, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), self.font_large, color, align="center")

    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        elif align == "bottomleft":
            text_rect.bottomleft = pos
        elif align == "bottomright":
            text_rect.bottomright = pos
        elif align == "center":
            text_rect.center = pos
        elif align == "midbottom":
            text_rect.midbottom = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "gold": self.gold,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' or 'x11' or 'dummy'

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
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
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

    env.close()