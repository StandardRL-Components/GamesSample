
# Generated: 2025-08-28T05:51:09.830695
# Source Brief: brief_02745.md
# Brief Index: 2745

        
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


# Helper classes for game entities
class Particle:
    """A simple class for creating visual particle effects."""
    def __init__(self, pos, vel, lifespan, color, radius):
        self.pos = list(pos)
        self.vel = list(vel)
        self.lifespan = lifespan
        self.color = color
        self.radius = radius

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1
        self.radius = max(0, self.radius - 0.1)

    def draw(self, surface):
        if self.lifespan > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), self.color)

class Projectile:
    """Represents a projectile fired by a tower."""
    def __init__(self, start_pos, target_enemy, damage):
        self.pos = list(start_pos)
        self.target = target_enemy
        self.damage = damage
        self.speed = 10
        self.is_active = True

    def update(self):
        if not self.target or self.target.health <= 0:
            self.is_active = False
            return None

        direction = np.array(self.target.pos) - np.array(self.pos)
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.is_active = False
            return self.target.take_damage(self.damage)
        
        # Using np.linalg.norm for division by zero safety
        if distance > 0:
            direction = direction / distance
            self.pos += direction * self.speed
        else: # Projectile is already at target
             self.is_active = False
             return self.target.take_damage(self.damage)
        return None

    def draw(self, surface):
        # Draw a bright, visible line for the projectile
        pygame.draw.line(surface, (255, 255, 0), self.pos, self.pos + (np.array(self.pos) - np.array(self.target.pos)) * 0.1, 3)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), 3, (255, 255, 255))

class Enemy:
    """Represents an enemy moving along the path."""
    def __init__(self, path_pixels, health, speed, value):
        self.path_pixels = path_pixels
        self.path_index = 0
        self.pos = list(self.path_pixels[0])
        self.max_health = health
        self.health = health
        self.speed = speed
        self.value = value
        self.is_active = True
        self.radius = 10

    def update(self):
        if self.path_index >= len(self.path_pixels) - 1:
            self.is_active = False
            return "reached_base"

        target_pos = self.path_pixels[self.path_index + 1]
        direction = np.array(target_pos) - np.array(self.pos)
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = list(target_pos)
            self.path_index += 1
        elif distance > 0:
            direction = direction / distance
            self.pos += direction * self.speed
        
        return None

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_active = False
            return "killed"
        return "hit"

    def draw(self, surface):
        # 3D-like effect with a shadow and a top surface
        body_color = (220, 50, 50)
        shadow_color = (150, 30, 30)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, shadow_color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1] - 2), self.radius, body_color)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1] - 2), self.radius, (255, 100, 100))

        # Health bar above the enemy
        if self.health < self.max_health:
            bar_width = 20
            bar_height = 4
            health_pct = self.health / self.max_health
            bar_x = self.pos[0] - bar_width / 2
            bar_y = self.pos[1] - self.radius - 12

            pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(surface, (50, 200, 50), (bar_x, bar_y, bar_width * health_pct, bar_height))

class Tower:
    """Represents a player-built tower."""
    def __init__(self, grid_pos, screen_pos):
        self.grid_pos = grid_pos
        self.screen_pos = screen_pos
        self.range = 100
        self.damage = 10
        self.fire_rate = 30 # frames per shot
        self.cooldown = 0
        self.cost = 50

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        
        target = self.find_target(enemies)
        if target:
            self.cooldown = self.fire_rate
            # Sound effect placeholder: Tower fire
            return Projectile(self.screen_pos, target, self.damage)
        return None

    def find_target(self, enemies):
        for enemy in enemies:
            distance = np.linalg.norm(np.array(self.screen_pos) - np.array(enemy.pos))
            if distance <= self.range:
                return enemy
        return None

    def draw(self, surface):
        x, y = int(self.screen_pos[0]), int(self.screen_pos[1])
        base_color = (40, 180, 200)
        top_color = (80, 220, 240)
        
        # Isometric base
        base_points = [(x, y), (x + 12, y + 6), (x, y + 12), (x - 12, y + 6)]
        pygame.gfxdraw.filled_polygon(surface, base_points, (30, 140, 160))
        pygame.gfxdraw.aapolygon(surface, base_points, base_color)
        
        # Isometric top
        top_points = [(x, y - 10), (x + 10, y - 4), (x, y + 2), (x - 10, y - 4)]
        pygame.gfxdraw.filled_polygon(surface, top_points, base_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)
        
        # Barrel
        pygame.draw.line(surface, top_color, (x, y-10), (x, y-15), 3)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to select a build location. Press Space to build a tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this isometric tower defense game."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        self.font_s = pygame.font.SysFont("Arial", 16)
        self.font_m = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_l = pygame.font.SysFont("Arial", 30, bold=True)

        # Colors
        self.COLOR_BG = (30, 40, 50)
        self.COLOR_PATH = (60, 70, 80)
        self.COLOR_GRID = (50, 60, 70, 100)
        self.COLOR_BASE = (0, 255, 200)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_MSG_GOOD = (100, 255, 100)
        self.COLOR_MSG_BAD = (255, 100, 100)

        # Game constants
        self.MAX_STEPS = 30 * 60 * 3 # 3 minutes at 30fps
        self.WIN_WAVE = 10
        self.INITIAL_MONEY = 100
        self.INITIAL_BASE_HEALTH = 100
        self.WAVE_PREP_TIME = 300 # frames
        
        # Isometric projection helpers
        self.tile_w, self.tile_h = 32, 16
        self.origin_x, self.origin_y = self.width // 2, 80
        
        # Define path and tower spots
        self._define_layout()
        
        self.reset()

        # CRITICAL: Validate implementation
        self.validate_implementation()

    def _define_layout(self):
        # Logical path for enemies to follow
        path_grid = [
            (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (6, 5), (6, 6), (6, 7),
            (5, 7), (4, 7), (3, 7), (2, 7), (2, 8), (2, 9), (2, 10), (3, 10),
            (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (8, 9), (8, 8), (8, 7),
            (8, 6), (9, 6), (10, 6), (11, 6), (12, 6), (13, 6), (14, 6), (15, 6),
            (15, 5) # Base location
        ]
        self.path_pixels = [self._iso_to_screen(x, y) for x, y in path_grid]
        self.base_pos = self.path_pixels[-1]
        
        # Pre-defined spots for tower placement
        self.tower_spots_grid = [
            (2, 3), (4, 3), (6, 3), (5, 5), (4, 6), (3, 6), (2, 6), (1, 8),
            (3, 9), (5, 9), (7, 9), (9, 9), (7, 7), (9, 7), (10, 7), (12, 7),
            (14, 7), (14, 5)
        ]
        self.tower_spots_screen = [self._iso_to_screen(x, y) for x, y in self.tower_spots_grid]

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.tile_w / 2
        screen_y = self.origin_y + (x + y) * self.tile_h / 2
        return (screen_x, screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.money = self.INITIAL_MONEY
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.time_to_next_wave = self.WAVE_PREP_TIME // 2
        
        self.enemies_to_spawn = 0
        self.time_to_next_spawn = 0
        self.enemy_stats_multiplier = 1.0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.messages = []
        
        self.cursor_index = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01 # Small penalty for time passing to encourage efficiency
        
        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_press)

        self._update_wave_manager()
        self._spawn_enemies()
        
        new_projectiles = [proj for tower in self.towers if (proj := tower.update(self.enemies)) is not None]
        self.projectiles.extend(new_projectiles)
        
        enemies_reached_base = 0
        for enemy in self.enemies[:]:
            if enemy.update() == "reached_base":
                enemies_reached_base += 1
                self.enemies.remove(enemy)
        if enemies_reached_base > 0:
            self.base_health = max(0, self.base_health - 10 * enemies_reached_base)
            self._add_message(f"-{10 * enemies_reached_base} HP", self.COLOR_MSG_BAD, 60)
            # Sound effect placeholder: Base damage
        
        for proj in self.projectiles[:]:
            result = proj.update()
            if not proj.is_active:
                self.projectiles.remove(proj)
                if result: # Hit an enemy
                    self._create_particles(proj.pos, (255, 255, 150), 5)
                    # Sound effect placeholder: Enemy hit
                    if result == "killed":
                        self.money += proj.target.value
                        reward += 0.1
                        self.score += 1
                        self.enemies.remove(proj.target)
                        self._create_particles(proj.pos, (255, 80, 80), 15)
                        # Sound effect placeholder: Enemy death

        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles: p.update()
        self.messages = [m for m in self.messages if m['lifespan'] > 0]
        for m in self.messages: m['lifespan'] -= 1

        if self.wave_in_progress and not self.enemies and self.enemies_to_spawn == 0:
            self.wave_in_progress = False
            self.time_to_next_wave = self.WAVE_PREP_TIME
            wave_bonus = 25 + self.wave_number * 5
            self.money += wave_bonus
            reward += 1.0
            self.score += 10
            self._add_message(f"Wave {self.wave_number} Cleared! +${wave_bonus}", self.COLOR_MSG_GOOD, 120)
        
        self.steps += 1
        
        terminated = False
        if self.base_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 100
        elif self.wave_number > self.WIN_WAVE:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_press):
        if movement != 0:
            current_pos = self.tower_spots_grid[self.cursor_index]
            candidates = []
            for i, spot in enumerate(self.tower_spots_grid):
                if i == self.cursor_index: continue
                dx, dy = spot[0] - current_pos[0], spot[1] - current_pos[1]
                
                if movement == 1 and dy < 0 and abs(dx) <= abs(dy): candidates.append((i, abs(dy) + abs(dx)*0.5)) # Up
                elif movement == 2 and dy > 0 and abs(dx) <= abs(dy): candidates.append((i, dy + abs(dx)*0.5)) # Down
                elif movement == 3 and dx < 0 and abs(dy) <= abs(dx): candidates.append((i, abs(dx) + abs(dy)*0.5)) # Left
                elif movement == 4 and dx > 0 and abs(dy) <= abs(dx): candidates.append((i, dx + abs(dy)*0.5)) # Right
            
            if candidates:
                candidates.sort(key=lambda c: c[1])
                self.cursor_index = candidates[0][0]

        if space_press:
            grid_pos = self.tower_spots_grid[self.cursor_index]
            is_occupied = any(t.grid_pos == grid_pos for t in self.towers)
            if is_occupied:
                self._add_message("Location occupied!", self.COLOR_MSG_BAD, 60)
                return

            tower_cost = 50
            if self.money >= tower_cost:
                self.money -= tower_cost
                screen_pos = self.tower_spots_screen[self.cursor_index]
                self.towers.append(Tower(grid_pos, screen_pos))
                self._add_message(f"Tower built! (-${tower_cost})", self.COLOR_MSG_GOOD, 60)
                # Sound effect placeholder: Build tower
            else:
                self._add_message(f"Not enough money! (${tower_cost})", self.COLOR_MSG_BAD, 60)

    def _update_wave_manager(self):
        if not self.wave_in_progress and not self.game_over:
            self.time_to_next_wave -= 1
            if self.time_to_next_wave <= 0 and self.wave_number < self.WIN_WAVE:
                self.wave_number += 1
                self.wave_in_progress = True
                self.enemies_to_spawn = 3 + self.wave_number * 2
                self.time_to_next_spawn = 0
                self.enemy_stats_multiplier = 1.0 + (self.wave_number - 1) * 0.05
                self._add_message(f"Wave {self.wave_number} starting!", self.COLOR_UI_TEXT, 120)
            elif self.time_to_next_wave <= 0 and self.wave_number == self.WIN_WAVE:
                self.wave_number += 1 # Trigger win condition

    def _spawn_enemies(self):
        if self.wave_in_progress and self.enemies_to_spawn > 0:
            self.time_to_next_spawn -= 1
            if self.time_to_next_spawn <= 0:
                health = 50 * self.enemy_stats_multiplier
                speed = 1.5 * self.enemy_stats_multiplier
                value = 5
                self.enemies.append(Enemy(self.path_pixels, health, speed, value))
                self.enemies_to_spawn -= 1
                self.time_to_next_spawn = 45

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        if len(self.path_pixels) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixels, 20)

        for pos in self.tower_spots_screen:
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 5, self.COLOR_GRID)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 5, self.COLOR_GRID)

        base_x, base_y = int(self.base_pos[0]), int(self.base_pos[1])
        glow_size = int(20 + 5 * math.sin(self.steps * 0.1))
        glow_color = (*self.COLOR_BASE, 50)
        pygame.gfxdraw.filled_circle(self.screen, base_x, base_y, glow_size, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, base_x, base_y, 20, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, base_x, base_y, 20, (255, 255, 255))

        for tower in self.towers: tower.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)
        
        if not self.game_over:
            cursor_pos = self.tower_spots_screen[self.cursor_index]
            cx, cy = int(cursor_pos[0]), int(cursor_pos[1])
            size = int(15 + 3 * math.sin(self.steps * 0.2))
            alpha = int(150 + 100 * math.sin(self.steps * 0.2))
            pygame.gfxdraw.aacircle(self.screen, cx, cy, size, (*self.COLOR_CURSOR, alpha))
            pygame.gfxdraw.aacircle(self.screen, cx, cy, size - 1, (*self.COLOR_CURSOR, alpha))
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx - 5, cy), (cx + 5, cy))
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx, cy - 5), (cx, cy + 5))

    def _render_ui(self):
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.width, 40))
        
        health_text = self.font_m.render(f"Base HP: {int(self.base_health)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))
        money_text = self.font_m.render(f"Money: ${self.money}", True, self.COLOR_UI_TEXT)
        self.screen.blit(money_text, (180, 10))
        wave_text = self.font_m.render(f"Wave: {min(self.wave_number, self.WIN_WAVE)}/{self.WIN_WAVE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (340, 10))
        score_text = self.font_m.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (500, 10))

        if not self.wave_in_progress and not self.game_over and self.wave_number < self.WIN_WAVE:
            timer_text = self.font_s.render(f"Next wave in: {self.time_to_next_wave / 30:.1f}s", True, self.COLOR_UI_TEXT)
            self.screen.blit(timer_text, (self.width/2 - timer_text.get_width()/2, 45))

        for i, msg in enumerate(self.messages):
            alpha = max(0, min(255, msg['lifespan'] * 4))
            msg_surf = self.font_s.render(msg['text'], True, msg['color'])
            msg_surf.set_alpha(alpha)
            self.screen.blit(msg_surf, (self.width / 2 - msg_surf.get_width() / 2, self.height - 30 - i * 20))

        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text_str = "VICTORY!" if self.win else "GAME OVER"
            end_text_color = (100, 255, 100) if self.win else (255, 100, 100)
            end_surf = self.font_l.render(end_text_str, True, end_text_color)
            self.screen.blit(end_surf, (self.width/2 - end_surf.get_width()/2, self.height/2 - 50))
            final_score_surf = self.font_m.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            self.screen.blit(final_score_surf, (self.width/2 - final_score_surf.get_width()/2, self.height/2 + 10))
    
    def _add_message(self, text, color, lifespan):
        self.messages.insert(0, {'text': text, 'color': color, 'lifespan': lifespan})
        if len(self.messages) > 3:
            self.messages.pop()

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            vel = [self.np_random.uniform(-1, 1) * 2, self.np_random.uniform(-1, 1) * 2]
            lifespan = self.np_random.integers(20, 41)
            radius = self.np_random.uniform(2, 5)
            self.particles.append(Particle(pos, vel, lifespan, color, radius))

    def _get_info(self):
        return { "score": self.score, "steps": self.steps, "money": self.money, "wave": self.wave_number, "base_health": self.base_health }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly for testing and demonstration
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space = 0 # 0=released, 1=pressed
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_q:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, 0] # shift is unused
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # The observation is already a rendered frame, we just need to display it.
        # Pygame uses (width, height), but our obs is (height, width, 3). So we transpose.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()