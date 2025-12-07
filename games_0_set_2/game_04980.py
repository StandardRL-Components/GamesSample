
# Generated: 2025-08-28T03:37:14.053003
# Source Brief: brief_04980.md
# Brief Index: 4980

        
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
class Enemy:
    def __init__(self, health, speed, path, size=10):
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[0], dtype=float)
        self.max_health = health
        self.health = health
        self.speed = speed
        self.size = size
        self.slow_timer = 0
        self.value = 10 # Resources gained on kill

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        target_pos = np.array(self.path[self.path_index + 1], dtype=float)
        direction = target_pos - self.pos
        distance = np.linalg.norm(direction)
        
        current_speed = self.speed
        if self.slow_timer > 0:
            current_speed *= 0.5
            self.slow_timer -= 1

        if distance < current_speed:
            self.pos = target_pos
            self.path_index += 1
        else:
            self.pos += (direction / distance) * current_speed
        return False

class Tower:
    def __init__(self, grid_pos, tower_type, cell_size):
        self.grid_pos = grid_pos
        self.pixel_pos = (
            (grid_pos[0] + 0.5) * cell_size,
            (grid_pos[1] + 0.5) * cell_size,
        )
        self.type = tower_type
        self.cooldown_timer = 0
        
        if tower_type == "arrow":
            self.range = 100
            self.damage = 5
            self.cooldown = 20 # fast
            self.color = (0, 255, 128)
            self.cost = 25
        elif tower_type == "cannon":
            self.range = 120
            self.damage = 25
            self.cooldown = 80 # slow
            self.color = (255, 128, 0)
            self.cost = 75
        elif tower_type == "slow":
            self.range = 80
            self.damage = 0 # No damage
            self.cooldown = 10 # Pulses
            self.color = (0, 192, 255)
            self.cost = 50

    def update(self):
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

    def find_target(self, enemies):
        if self.cooldown_timer > 0:
            return None
        
        for enemy in enemies:
            dist = np.linalg.norm(np.array(self.pixel_pos) - enemy.pos)
            if dist <= self.range:
                self.cooldown_timer = self.cooldown
                return enemy
        return None

class Projectile:
    def __init__(self, start_pos, target, damage, speed, p_type):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.type = p_type
        
    def move(self):
        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)
        if distance < self.speed:
            return True # Hit
        else:
            self.pos += (direction / distance) * self.speed
            return False

class Particle:
    def __init__(self, pos, vel, size, color, lifespan):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.size = size
        self.color = color
        self.lifespan = lifespan
        self.life = lifespan

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.size = max(0, self.size * 0.98)
        return self.life <= 0


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to build an Arrow Tower, "
        "shift for a Cannon Tower, and space+shift for a Slow Tower."
    )

    game_description = (
        "A top-down tower defense game. Place towers on the grid to defend your base "
        "from waves of incoming enemies. Survive all 10 waves to win."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 40
        self.GRID_W, self.GRID_H = self.WIDTH // self.CELL_SIZE, self.HEIGHT // self.CELL_SIZE

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PATH = (30, 35, 55)
        self.COLOR_BASE = (0, 100, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        
        self.MAX_STEPS = 30 * 180 # 3 minutes at 30fps

        self.reset()
        
        # This is for internal use, not part of the Gym API
        self.action_cooldown = 0
        self.placement_cooldown = 0

        self.validate_implementation()

    def _define_path(self):
        self.path = []
        path_points = [
            (-1, 5), (2, 5), (2, 2), (6, 2), (6, 8), 
            (10, 8), (10, 1), (14, 1), (14, 6), (self.GRID_W, 6)
        ]
        for i in range(len(path_points) - 1):
            p1 = np.array(path_points[i]) * self.CELL_SIZE + self.CELL_SIZE / 2
            p2 = np.array(path_points[i+1]) * self.CELL_SIZE + self.CELL_SIZE / 2
            self.path.append(tuple(p1))
        self.path.append(tuple(p2))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.base_health = 100
        self.resources = 100
        self.wave_number = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.grid = [[None for _ in range(self.GRID_H)] for _ in range(self.GRID_W)]
        self._define_path()
        self.base_pos = (self.GRID_W - 1, 6)
        self.grid[self.base_pos[0]][self.base_pos[1]] = "base"

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        self.game_phase = "placement"
        self.inter_wave_timer = 30 * 5 # 5 seconds

        return self._get_observation(), self._get_info()

    def _start_wave(self):
        self.wave_number += 1
        self.game_phase = "wave"
        num_enemies = 2 + self.wave_number
        base_health = 20 + (self.wave_number - 1) * 10
        base_speed = 1.0 + (self.wave_number - 1) * 0.05

        for i in range(num_enemies):
            offset = self.np_random.uniform(-self.CELL_SIZE/3, self.CELL_SIZE/3, size=2)
            offset_path = [(p[0] + offset[0], p[1] + offset[1]) for p in self.path]
            
            health = base_health * self.np_random.uniform(0.9, 1.1)
            speed = base_speed * self.np_random.uniform(0.9, 1.1)
            
            enemy = Enemy(health, speed, offset_path)
            self.enemies.append(enemy)
            # Stagger spawn
            enemy.pos[0] -= i * self.CELL_SIZE * 2

    def _create_particles(self, pos, num, color, max_speed=2, min_size=1, max_size=3):
        for _ in range(num):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(min_size, max_size)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos, vel, size, color, lifespan))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(30)
        reward = 0
        
        # --- Handle player input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        if self.action_cooldown > 0: self.action_cooldown -= 1
        if self.placement_cooldown > 0: self.placement_cooldown -= 1

        if self.game_phase == "placement":
            # Move cursor
            if self.action_cooldown == 0:
                if movement == 1: self.cursor_pos[1] -= 1
                elif movement == 2: self.cursor_pos[1] += 1
                elif movement == 3: self.cursor_pos[0] -= 1
                elif movement == 4: self.cursor_pos[0] += 1
                if movement != 0: self.action_cooldown = 5 # Cooldown for cursor move
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

            # Place tower
            if self.placement_cooldown == 0:
                tower_type = None
                if space_held and shift_held: tower_type = "slow"
                elif space_held: tower_type = "arrow"
                elif shift_held: tower_type = "cannon"
                
                if tower_type:
                    cost_map = {"arrow": 25, "cannon": 75, "slow": 50}
                    if self.resources >= cost_map[tower_type] and self.grid[self.cursor_pos[0]][self.cursor_pos[1]] is None:
                        new_tower = Tower(tuple(self.cursor_pos), tower_type, self.CELL_SIZE)
                        self.towers.append(new_tower)
                        self.grid[self.cursor_pos[0]][self.cursor_pos[1]] = new_tower
                        self.resources -= cost_map[tower_type]
                        self.placement_cooldown = 10 # Cooldown for placing
                        # sfx: place_tower.wav
            
            # Wave timer
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                self._start_wave()

        # --- Game Logic Update ---
        if self.game_phase == "wave":
            # Update towers & create projectiles
            for tower in self.towers:
                tower.update()
                target = tower.find_target(self.enemies)
                if target:
                    if tower.type == "slow":
                        # sfx: slow_pulse.wav
                        self._create_particles(tower.pixel_pos, 15, tower.color, max_speed=tower.range/tower.cooldown, max_size=4)
                        for enemy in self.enemies:
                            dist = np.linalg.norm(np.array(tower.pixel_pos) - enemy.pos)
                            if dist <= tower.range:
                                enemy.slow_timer = max(enemy.slow_timer, 30) # Apply slow for 1 sec
                    else:
                        # sfx: shoot_arrow.wav or shoot_cannon.wav
                        speed = 15 if tower.type == "arrow" else 8
                        self.projectiles.append(Projectile(tower.pixel_pos, target, tower.damage, speed, tower.type))

            # Update projectiles
            new_projectiles = []
            for p in self.projectiles:
                hit = p.move()
                if hit:
                    reward += 0.1 # Reward for hitting
                    # sfx: hit_enemy.wav
                    p.target.health -= p.damage
                    self._create_particles(p.target.pos, 5, (255, 255, 100), max_speed=3)
                    if p.type == "cannon":
                        # sfx: explosion.wav
                        self._create_particles(p.target.pos, 20, (255, 150, 50), max_speed=5, max_size=5)
                        for enemy in self.enemies:
                            if enemy is not p.target:
                                dist = np.linalg.norm(enemy.pos - p.target.pos)
                                if dist < self.CELL_SIZE * 1.5: # Splash damage
                                    enemy.health -= p.damage * 0.5
                else:
                    new_projectiles.append(p)
            self.projectiles = new_projectiles

            # Update enemies
            surviving_enemies = []
            for enemy in self.enemies:
                if enemy.health <= 0:
                    reward += 1
                    self.score += 1
                    self.resources += enemy.value
                    # sfx: enemy_die.wav
                    self._create_particles(enemy.pos, 30, self.COLOR_ENEMY, max_speed=2, max_size=4)
                    continue
                
                if enemy.move():
                    self.base_health -= 10
                    reward -= 10
                    # sfx: base_damage.wav
                    self._create_particles(enemy.pos, 30, self.COLOR_BASE, max_speed=4)
                else:
                    surviving_enemies.append(enemy)
            self.enemies = surviving_enemies
            
            # Check for wave end
            if not self.enemies:
                self.game_phase = "placement"
                self.inter_wave_timer = 30 * 8 # 8 seconds between waves
                self.resources += 50 + self.wave_number * 10 # End of wave bonus
                if self.wave_number >= 10:
                    self.victory = True

        # Update particles
        self.particles = [p for p in self.particles if not p.update()]

        # --- Termination ---
        self.steps += 1
        terminated = self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.victory
        if terminated and not self.game_over:
            self.game_over = True
            if self.victory:
                reward += 100
            else:
                reward -= 100

        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Draw grid and path
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        for i in range(len(self.path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path[i], self.path[i+1], self.CELL_SIZE)

        # Draw base
        bx, by = self.base_pos
        base_rect = pygame.Rect(bx * self.CELL_SIZE, by * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.gfxdraw.rectangle(self.screen, base_rect, (*self.COLOR_BASE, 150))

        # Draw towers
        for tower in self.towers:
            pos = (int(tower.pixel_pos[0]), int(tower.pixel_pos[1]))
            if tower.type == "arrow":
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, tower.color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, tower.color)
            elif tower.type == "cannon":
                pygame.gfxdraw.box(self.screen, (pos[0]-10, pos[1]-10, 20, 20), tower.color)
            elif tower.type == "slow":
                points = [
                    (pos[0], pos[1] - 12), (pos[0] + 12, pos[1] + 8), (pos[0] - 12, pos[1] + 8)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, tower.color)
                pygame.gfxdraw.filled_polygon(self.screen, points, tower.color)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy.pos[0]), int(enemy.pos[1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy.size, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy.size, self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy.health / enemy.max_health
            bar_w = enemy.size * 2
            pygame.draw.rect(self.screen, (80,0,0), (pos[0]-bar_w/2, pos[1]-18, bar_w, 4))
            pygame.draw.rect(self.screen, (0,255,0), (pos[0]-bar_w/2, pos[1]-18, bar_w*health_pct, 4))
            if enemy.slow_timer > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy.size+2, (0, 192, 255))

        # Draw projectiles
        for p in self.projectiles:
            color = (0, 255, 128) if p.type == "arrow" else (255, 128, 0)
            pygame.draw.line(self.screen, color, p.pos, p.pos + (p.target.pos - p.pos) * 0.2, 3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.lifespan))
            color = (*p.color, alpha)
            pos = (int(p.pos[0]), int(p.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p.size), color)
        
        # Draw cursor
        if self.game_phase == "placement":
            c_rect = (self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, c_rect, 2)

    def _render_ui(self):
        # Top bar
        wave_text = self.font_main.render(f"Wave: {self.wave_number}/10", True, self.COLOR_TEXT)
        resource_text = self.font_main.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 5))
        self.screen.blit(resource_text, (self.WIDTH - resource_text.get_width() - 10, 5))
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 5))

        # Base health bar
        health_bar_rect = pygame.Rect(10, self.HEIGHT - 25, self.WIDTH - 20, 15)
        health_pct = max(0, self.base_health / 100)
        current_health_w = int(health_bar_rect.width * health_pct)
        pygame.draw.rect(self.screen, (80, 0, 0), health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (health_bar_rect.x, health_bar_rect.y, current_health_w, health_bar_rect.height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, health_bar_rect, 2)
        health_text = self.font_main.render(f"Base Health: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (health_bar_rect.centerx - health_text.get_width()//2, health_bar_rect.y - 25))

        # Game phase text
        if self.game_phase == "placement" and not self.game_over:
            seconds_left = self.inter_wave_timer // 30
            msg = f"Wave {self.wave_number + 1} starting in {seconds_left}s"
            phase_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(phase_text, (self.WIDTH//2 - phase_text.get_width()//2, self.HEIGHT//2 - phase_text.get_height()//2))

        # Game over text
        if self.game_over:
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 50, 50)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - end_text.get_height()//2 - 20))


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
            "wave": self.wave_number,
            "base_health": self.base_health,
            "resources": self.resources,
            "enemies_left": len(self.enemies)
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    done = False
    total_reward = 0
    
    while not done:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
    env.close()