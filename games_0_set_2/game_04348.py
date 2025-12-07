
# Generated: 2025-08-28T02:09:20.150994
# Source Brief: brief_04348.md
# Brief Index: 4348

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities to encapsulate logic and state
class Zombie:
    """Represents a single enemy unit."""
    def __init__(self, x, y, speed, health=3):
        self.x = x
        self.y = y
        self.speed = speed
        self.base_speed = speed
        self.health = health
        self.max_health = health
        self.size = 12
        self.slow_timer = 0

    def update(self, base_pos, towers):
        if self.slow_timer > 0:
            self.speed = self.base_speed * 0.5
            self.slow_timer -= 1
        else:
            self.speed = self.base_speed

        # Primary movement vector towards the base
        dx = base_pos[0] - self.x
        dy = base_pos[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            dx /= dist
            dy /= dist
        
        # Secondary repulsion vector to avoid towers
        repulsion_x, repulsion_y = 0, 0
        for tower in towers:
            rep_dx = self.x - tower.x
            rep_dy = self.y - tower.y
            rep_dist = math.hypot(rep_dx, rep_dy)
            if 0 < rep_dist < 40:
                force = (40 - rep_dist) / 40 
                repulsion_x += (rep_dx / rep_dist) * force * 0.5
                repulsion_y += (rep_dy / rep_dist) * force * 0.5

        # Combine vectors and normalize for final movement
        final_dx = dx + repulsion_x
        final_dy = dy + repulsion_y
        final_dist = math.hypot(final_dx, final_dy)
        if final_dist > 0:
            final_dx /= final_dist
            final_dy /= final_dist

        self.x += final_dx * self.speed
        self.y += final_dy * self.speed

    def draw(self, screen, color_enemy):
        # Body
        body_rect = (int(self.x - self.size / 2), int(self.y - self.size / 2), self.size, self.size)
        pygame.draw.rect(screen, color_enemy, body_rect)
        # Health bar
        if self.health < self.max_health:
            health_pct = self.health / self.max_health
            bar_y = body_rect[1] - 5
            pygame.draw.rect(screen, (255, 0, 0), (body_rect[0], bar_y, self.size, 3))
            pygame.draw.rect(screen, (0, 255, 0), (body_rect[0], bar_y, int(self.size * health_pct), 3))

class Tower:
    """Represents a player-placed defensive tower."""
    def __init__(self, x, y, tower_type):
        self.x = x
        self.y = y
        self.type = tower_type  # 'slow' or 'damage'
        self.range = 80 if tower_type == 'slow' else 100
        self.cooldown = 0
        self.cooldown_max = 10 if tower_type == 'slow' else 20
        self.size = 15

    def update(self, zombies, projectiles):
        self.cooldown = max(0, self.cooldown - 1)
        if self.cooldown == 0:
            if self.type == 'slow':
                # Slow towers affect all zombies in range simultaneously
                slowed_one = False
                for z in zombies:
                    if math.hypot(self.x - z.x, self.y - z.y) < self.range:
                        z.slow_timer = 30  # Slow for 1 second at 30fps
                        slowed_one = True
                if slowed_one:
                    self.cooldown = self.cooldown_max
                    # Sound: sfx_slow_tower_pulse
            elif self.type == 'damage':
                # Damage towers find the closest target and fire a projectile
                target, min_dist = None, self.range
                for z in zombies:
                    dist = math.hypot(self.x - z.x, self.y - z.y)
                    if dist < min_dist:
                        min_dist, target = dist, z
                if target:
                    projectiles.append(Projectile(self.x, self.y, target))
                    self.cooldown = self.cooldown_max
                    # Sound: sfx_damage_tower_shoot
    
    def draw(self, screen, color_slow, color_damage):
        color = color_slow if self.type == 'slow' else color_damage
        points = [
            (self.x, self.y - self.size / 1.5),
            (self.x - self.size / 2, self.y + self.size / 3),
            (self.x + self.size / 2, self.y + self.size / 3)
        ]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(screen, int_points, color)
        pygame.gfxdraw.filled_polygon(screen, int_points, color)

class Projectile:
    """Represents a projectile fired by a damage tower."""
    def __init__(self, x, y, target):
        self.x, self.y = x, y
        self.speed, self.size, self.damage = 10, 3, 1
        dx, dy = target.x - self.x, target.y - self.y
        dist = math.hypot(dx, dy)
        self.vx = (dx / dist) * self.speed if dist > 0 else 0
        self.vy = (dy / dist) * self.speed if dist > 0 else self.speed

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, screen, color):
        pygame.gfxdraw.aacircle(screen, int(self.x), int(self.y), self.size, color)
        pygame.gfxdraw.filled_circle(screen, int(self.x), int(self.y), self.size, color)

class Particle:
    """Represents a short-lived particle for visual effects."""
    def __init__(self, x, y, color, life):
        self.x, self.y = x, y
        self.vx, self.vy = random.uniform(-1, 1), random.uniform(-1, 1)
        self.life, self.max_life = life, life
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, screen):
        # Fade out particle over its lifetime
        alpha = int(255 * (self.life / self.max_life))
        size = int(3 * (self.life / self.max_life))
        if size > 0:
            # Use a temporary surface to draw with alpha transparency
            temp_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.color + (alpha,), (size, size), size)
            screen.blit(temp_surf, (int(self.x - size), int(self.y - size)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑↓←→ to move the placement cursor. "
        "Hold Shift to select a Damage Tower (yellow), release for a Slow Tower (blue). "
        "Press Space to build the selected tower."
    )

    game_description = (
        "Defend your base from waves of zombies by strategically placing defensive towers. "
        "Survive 10 waves to win."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PLAYER = (0, 255, 100)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_SLOW_TOWER = (50, 150, 255)
        self.COLOR_DAMAGE_TOWER = (255, 200, 50)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_ZONE = (50, 60, 70)
        self.COLOR_UI = (200, 200, 200)

        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps
        self.MAX_WAVES = 10
        self.BASE_MAX_HEALTH = 50
        self.WAVE_COOLDOWN = 30 * 5 # 5 seconds

        self.GRID_COLS, self.GRID_ROWS = 8, 4
        self.placement_zones = []
        x_margin, y_margin = 100, 80
        x_spacing = (self.WIDTH - 2 * x_margin) / max(1, self.GRID_COLS - 1)
        y_spacing = (self.HEIGHT - 2 * y_margin) / max(1, self.GRID_ROWS - 1)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self.placement_zones.append((x_margin + c * x_spacing, y_margin + r * y_spacing))

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_pos = (self.WIDTH / 2, self.HEIGHT - 30)
        self.base_health = self.BASE_MAX_HEALTH
        
        self.zombies, self.towers, self.projectiles, self.particles = [], [], [], []
        
        self.current_wave = 0
        self.wave_timer = self.WAVE_COOLDOWN // 2

        self.cursor_idx = 0
        self.selected_tower_type = 'slow'

        self.zone_occupied = [False] * len(self.placement_zones)

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            self.game_won = True
            return

        num_zombies = 5 + (self.current_wave - 1) * 2
        zombie_speed = 0.8 + (self.current_wave - 1) * 0.1
        zombie_health = 3 + int((self.current_wave - 1) / 2)

        for _ in range(num_zombies):
            side = self.np_random.integers(3)
            if side == 0: x, y = self.np_random.uniform(0, self.WIDTH), -20
            elif side == 1: x, y = -20, self.np_random.uniform(0, self.HEIGHT)
            else: x, y = self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)
            self.zombies.append(Zombie(x, y, zombie_speed, zombie_health))
        
        self.wave_timer = 0
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # 1. Handle Player Action
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1

        self.selected_tower_type = 'damage' if shift_held else 'slow'

        if movement == 1: self.cursor_idx = (self.cursor_idx - self.GRID_COLS) % len(self.placement_zones)
        elif movement == 2: self.cursor_idx = (self.cursor_idx + self.GRID_COLS) % len(self.placement_zones)
        elif movement == 3: self.cursor_idx = (self.cursor_idx - 1) % len(self.placement_zones)
        elif movement == 4: self.cursor_idx = (self.cursor_idx + 1) % len(self.placement_zones)

        if space_pressed and not self.zone_occupied[self.cursor_idx]:
            x, y = self.placement_zones[self.cursor_idx]
            self.towers.append(Tower(x, y, self.selected_tower_type))
            self.zone_occupied[self.cursor_idx] = True
            color = self.COLOR_DAMAGE_TOWER if self.selected_tower_type == 'damage' else self.COLOR_SLOW_TOWER
            for _ in range(10): self.particles.append(Particle(x, y, color, 20))
            # Sound: sfx_tower_place

        # 2. Update Game Logic
        for tower in self.towers: tower.update(self.zombies, self.projectiles)

        projectiles_to_remove = []
        for p in self.projectiles:
            p.update()
            if not (0 < p.x < self.WIDTH and 0 < p.y < self.HEIGHT):
                projectiles_to_remove.append(p)
                continue
            for z in self.zombies:
                if math.hypot(p.x - z.x, p.y - z.y) < z.size / 2 + p.size:
                    z.health -= p.damage
                    projectiles_to_remove.append(p)
                    for _ in range(3): self.particles.append(Particle(p.x, p.y, self.COLOR_PROJECTILE, 10))
                    # Sound: sfx_zombie_hit
                    if z.health <= 0:
                        reward += 0.1
                        self.score += 1
                        for _ in range(15): self.particles.append(Particle(z.x, z.y, self.COLOR_ENEMY, 25))
                        # Sound: sfx_zombie_die
                    break
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        self.zombies = [z for z in self.zombies if z.health > 0]

        zombies_to_remove = []
        for z in self.zombies:
            z.update(self.base_pos, self.towers)
            if math.hypot(z.x - self.base_pos[0], z.y - self.base_pos[1]) < 20:
                self.base_health -= 1
                reward -= 0.01
                zombies_to_remove.append(z)
                for _ in range(10): self.particles.append(Particle(self.base_pos[0], self.base_pos[1], self.COLOR_PLAYER, 20))
                # Sound: sfx_base_damage
        self.zombies = [z for z in self.zombies if z not in zombies_to_remove]

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles: p.update()

        if not self.zombies and self.wave_timer == 0 and self.current_wave <= self.MAX_WAVES:
            self.wave_timer = 1
            if self.current_wave > 0: reward += 1

        if self.wave_timer > 0:
            self.wave_timer += 1
            if self.wave_timer > self.WAVE_COOLDOWN: self._start_next_wave()

        # 3. Check Termination
        terminated = False
        if self.base_health <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over, terminated, reward = True, True, -50
        elif self.game_won:
            self.game_over, terminated, reward = True, True, 50
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _render_game(self):
        for i, pos in enumerate(self.placement_zones):
            color = (40, 45, 55) if self.zone_occupied[i] else self.COLOR_ZONE
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 18, color)
        
        base_size = 40
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (int(self.base_pos[0] - base_size/2), int(self.base_pos[1] - base_size/2), base_size, base_size))
        
        for tower in self.towers: tower.draw(self.screen, self.COLOR_SLOW_TOWER, self.COLOR_DAMAGE_TOWER)
        for zombie in self.zombies: zombie.draw(self.screen, self.COLOR_ENEMY)
        for p in self.projectiles: p.draw(self.screen, self.COLOR_PROJECTILE)
        for p in self.particles: p.draw(self.screen)

        cursor_pos = self.placement_zones[self.cursor_idx]
        cursor_color = self.COLOR_DAMAGE_TOWER if self.selected_tower_type == 'damage' else self.COLOR_SLOW_TOWER
        if self.zone_occupied[self.cursor_idx]: cursor_color = (128, 128, 128)
        pygame.draw.circle(self.screen, cursor_color, (int(cursor_pos[0]), int(cursor_pos[1])), 22, 3)

    def _render_ui(self):
        health_pct = max(0, self.base_health / self.BASE_MAX_HEALTH)
        bar_x, bar_y, bar_w, bar_h = self.base_pos[0] - 50, self.base_pos[1] - 40, 100, 10
        pygame.draw.rect(self.screen, (255,0,0), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, bar_w * health_pct, bar_h))

        wave_text = self.font.render(f"Wave: {min(self.current_wave, self.MAX_WAVES)}/{self.MAX_WAVES}", True, self.COLOR_UI)
        self.screen.blit(wave_text, (10, 10))

        score_text = self.font.render(f"Kills: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        if self.game_over:
            msg, color = ("YOU WON!", self.COLOR_PLAYER) if self.game_won else ("GAME OVER", self.COLOR_ENEMY)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2)))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score, "steps": self.steps, "wave": self.current_wave,
            "base_health": self.base_health, "zombies_remaining": len(self.zombies),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(reward, (int, float))
        assert isinstance(term, bool) and trunc is False and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    # Set the appropriate video driver for your OS ('x11', 'windows', 'dummy', etc.)
    # Use 'dummy' for headless execution.
    if "SDL_VIDEODRIVER" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running, total_reward = True, 0
    
    while running:
        movement_action, space_action, shift_action = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Simple input mapping for human play
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Kills: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)

    env.close()