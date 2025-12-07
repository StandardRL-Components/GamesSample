
# Generated: 2025-08-28T03:46:11.195793
# Source Brief: brief_05030.md
# Brief Index: 5030

        
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


# Helper classes for game objects
class Enemy:
    def __init__(self, health, speed, path, wave_num):
        self.pos = np.array(path[0], dtype=float)
        self.max_health = health
        self.health = health
        self.speed = speed
        self.path = path
        self.waypoint_index = 1
        self.radius = 8 + min(wave_num, 7) # Cap size
        self.is_alive = True
        self.value = 5 # Money awarded on kill

    def move(self):
        if self.waypoint_index >= len(self.path):
            return True # Reached the end

        target_pos = np.array(self.path[self.waypoint_index], dtype=float)
        direction = target_pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = target_pos
            self.waypoint_index += 1
        else:
            self.pos += (direction / distance) * self.speed
        return False

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_alive = False
            return True
        return False

class Tower:
    def __init__(self, pos):
        self.pos = pos
        self.range = 100
        self.damage = 10
        self.fire_rate = 1.0  # seconds per shot
        self.cooldown = 0.0
        self.target = None
        self.size = 12

    def update(self, dt, enemies):
        self.cooldown = max(0.0, self.cooldown - dt)
        if self.target and (not self.target.is_alive or np.linalg.norm(np.array(self.pos) - self.target.pos) > self.range):
            self.target = None

        if not self.target:
            self.find_target(enemies)

    def find_target(self, enemies):
        closest_enemy = None
        min_dist = self.range + 1
        for enemy in enemies:
            dist = np.linalg.norm(np.array(self.pos) - enemy.pos)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        self.target = closest_enemy

    def can_fire(self):
        return self.cooldown == 0.0 and self.target

    def fire(self):
        self.cooldown = self.fire_rate
        # sfx: Laser pew

class Projectile:
    def __init__(self, start_pos, target, damage):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target
        self.damage = damage
        self.speed = 10.0
        self.radius = 3

    def move(self):
        if not self.target.is_alive:
            return True # Target is gone

        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.target.take_damage(self.damage)
            return True
        else:
            self.pos += (direction / distance) * self.speed
        return False

class Particle:
    def __init__(self, x, y, color, life, size_range, speed_range):
        self.pos = np.array([x, y], dtype=float)
        self.color = color
        self.life = life
        self.max_life = life
        self.size = random.uniform(size_range[0], size_range[1])
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(speed_range[0], speed_range[1])
        self.vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])

    def update(self):
        self.pos += self.vel
        self.life -= 1
        return self.life <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to select tower spot. Space to place a tower."
    )

    game_description = (
        "A streamlined tower defense game. Place towers to defend your base from waves of enemies."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 30 * 60 # 60 seconds
    MAX_WAVES = 10

    COLOR_BG = (30, 35, 40)
    COLOR_PATH = (60, 65, 70)
    COLOR_BASE = (60, 180, 75)
    COLOR_ENEMY = (230, 25, 75)
    COLOR_TOWER = (0, 130, 200)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_UI_TEXT = (245, 245, 245)
    COLOR_PLACEMENT_SPOT = (90, 95, 100)
    COLOR_PLACEMENT_SELECTED = (255, 255, 25)

    TOWER_COST = 40
    BASE_STARTING_HEALTH = 100
    STARTING_MONEY = 80

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        self.path = [
            (50, -20), (50, 150), (250, 150), (250, 50),
            (450, 50), (450, 300), (150, 300), (150, 220), (590, 220)
        ]
        self.tower_spots = [
            (150, 100), (350, 100), (350, 200), (50, 250), (250, 260), (520, 150), (520, 280)
        ]

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.BASE_STARTING_HEALTH
        self.money = self.STARTING_MONEY
        
        self.current_wave = 0
        self.wave_timer = 5 * self.FPS # Time until first wave
        self.wave_state = "INTERMISSION" # INTERMISSION, SPAWNING
        self.enemies_to_spawn = 0

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.selected_spot_idx = 0
        self.space_was_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cycle through tower spots
        if movement in [1, 3]: # Up, Left
            self.selected_spot_idx = (self.selected_spot_idx - 1) % len(self.tower_spots)
        elif movement in [2, 4]: # Down, Right
            self.selected_spot_idx = (self.selected_spot_idx + 1) % len(self.tower_spots)

        # Place tower on space press
        if space_held and not self.space_was_held:
            spot_pos = self.tower_spots[self.selected_spot_idx]
            is_occupied = any(t.pos == spot_pos for t in self.towers)
            if not is_occupied and self.money >= self.TOWER_COST:
                self.money -= self.TOWER_COST
                self.towers.append(Tower(spot_pos))
                self.particles.extend([Particle(spot_pos[0], spot_pos[1], self.COLOR_TOWER, 20, (2, 5), (1, 3)) for _ in range(30)])
                # sfx: Build tower

        self.space_was_held = space_held
        
        # --- Game Logic ---
        dt = 1.0 / self.FPS

        # Wave Management
        if self.wave_state == "INTERMISSION":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                if self.current_wave > self.MAX_WAVES:
                    terminated = True
                    reward += 100 # Win bonus
                else:
                    self.wave_state = "SPAWNING"
                    self.enemies_to_spawn = 2 + self.current_wave
                    self.spawn_timer = 0
        
        elif self.wave_state == "SPAWNING":
            self.spawn_timer -= 1
            if self.enemies_to_spawn > 0 and self.spawn_timer <= 0:
                health = 10 + (self.current_wave - 1) * 5
                speed = 1.0 + (self.current_wave - 1) * 0.05
                self.enemies.append(Enemy(health, speed, self.path, self.current_wave))
                self.enemies_to_spawn -= 1
                self.spawn_timer = int(0.7 * self.FPS)
            
            if self.enemies_to_spawn == 0 and not self.enemies:
                self.wave_state = "INTERMISSION"
                self.wave_timer = 5 * self.FPS
                reward += 1 # Wave clear bonus

        # Update Towers
        for tower in self.towers:
            tower.update(dt, self.enemies)
            if tower.can_fire():
                self.projectiles.append(Projectile(tower.pos, tower.target, tower.damage))
                tower.fire()

        # Update Projectiles
        dead_projectiles = []
        for proj in self.projectiles:
            if proj.move():
                dead_projectiles.append(proj)
                if proj.target.is_alive: # Check if it was a hit, not just target dying
                    self.particles.extend([Particle(proj.target.pos[0], proj.target.pos[1], self.COLOR_ENEMY, 10, (1, 3), (0.5, 2)) for _ in range(5)])
                    # sfx: Enemy hit
        self.projectiles = [p for p in self.projectiles if p not in dead_projectiles]

        # Update Enemies
        surviving_enemies = []
        for enemy in self.enemies:
            if enemy.is_alive:
                if enemy.move():
                    self.base_health -= 10
                    self.particles.extend([Particle(self.WIDTH-10, self.HEIGHT/2, self.COLOR_BASE, 30, (3, 8), (1, 4)) for _ in range(50)])
                    # sfx: Base hit
                else:
                    surviving_enemies.append(enemy)
            else: # Enemy was killed
                reward += 0.1
                self.money += enemy.value
                self.particles.extend([Particle(enemy.pos[0], enemy.pos[1], self.COLOR_ENEMY, 25, (2, 4), (1, 3)) for _ in range(20)])
                # sfx: Enemy explosion
        self.enemies = surviving_enemies

        # Update Particles
        self.particles = [p for p in self.particles if not p.update()]
        
        # --- Termination Checks ---
        self.steps += 1
        if self.base_health <= 0:
            self.base_health = 0
            terminated = True
            reward -= 100 # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "money": self.money, "wave": self.current_wave}

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)

        # Draw base
        base_rect = pygame.Rect(self.WIDTH - 20, self.path[-1][1] - 15, 20, 30)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw tower placement spots
        for i, spot in enumerate(self.tower_spots):
            is_occupied = any(t.pos == spot for t in self.towers)
            color = self.COLOR_TOWER if is_occupied else self.COLOR_PLACEMENT_SPOT
            pygame.gfxdraw.filled_circle(self.screen, int(spot[0]), int(spot[1]), 15, color)
            pygame.gfxdraw.aacircle(self.screen, int(spot[0]), int(spot[1]), 15, color)
        
        # Draw selected spot indicator
        sel_spot = self.tower_spots[self.selected_spot_idx]
        is_occupied = any(t.pos == sel_spot for t in self.towers)
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        
        if is_occupied:
            color = tuple(int(c * 0.7) for c in self.COLOR_TOWER)
        elif self.money < self.TOWER_COST:
            color = tuple(int(128 + c * pulse) for c in (127,0,0)) # Pulsing red
        else:
            color = tuple(int(c * (0.6 + 0.4 * pulse)) for c in self.COLOR_PLACEMENT_SELECTED)
        
        pygame.draw.circle(self.screen, color, sel_spot, 18, 3)

        # Draw towers
        for tower in self.towers:
            pos = (int(tower.pos[0]), int(tower.pos[1]))
            rect = pygame.Rect(pos[0] - tower.size, pos[1] - tower.size, tower.size * 2, tower.size * 2)
            pygame.draw.rect(self.screen, self.COLOR_TOWER, rect, border_radius=3)
            # Draw range indicator if selected
            if tower.pos == self.tower_spots[self.selected_spot_idx]:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], tower.range, (255, 255, 255, 50))


        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy.pos[0]), int(enemy.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(enemy.radius), self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(enemy.radius), self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy.health / enemy.max_health
            bar_width = enemy.radius * 2
            pygame.draw.rect(self.screen, (50, 0, 0), (pos[0] - bar_width/2, pos[1] - enemy.radius - 8, bar_width, 5))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0] - bar_width/2, pos[1] - enemy.radius - 8, bar_width * health_pct, 5))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj.pos[0]), int(proj.pos[1]))
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, pos, (pos[0]-proj.vel[0]/2, pos[1]-proj.vel[1]/2), 3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = p.color + (alpha,)
            size = int(p.size * (p.life / p.max_life))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), size, color)

    def _render_ui(self):
        # Base Health Bar
        health_pct = self.base_health / self.BASE_STARTING_HEALTH
        bar_width = 200
        health_color = (int(230 * (1-health_pct)), int(180 * health_pct), 75)
        pygame.draw.rect(self.screen, (50, 50, 50), ((self.WIDTH - bar_width) / 2 - 2, 8, bar_width + 4, 24))
        pygame.draw.rect(self.screen, health_color, ((self.WIDTH - bar_width) / 2, 10, bar_width * health_pct, 20))
        
        # Text info
        wave_text = f"Wave: {min(self.current_wave, self.MAX_WAVES)}/{self.MAX_WAVES}"
        money_text = f"Money: ${self.money}"
        
        wave_surf = self.font_main.render(wave_text, True, self.COLOR_UI_TEXT)
        money_surf = self.font_main.render(money_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(wave_surf, (10, 10))
        self.screen.blit(money_surf, (10, 35))
        
        # Intermission text
        if self.wave_state == "INTERMISSION" and self.current_wave < self.MAX_WAVES:
            time_left = self.wave_timer / self.FPS
            intermission_text = f"Next wave in {time_left:.1f}s"
            inter_surf = self.font_main.render(intermission_text, True, self.COLOR_UI_TEXT)
            self.screen.blit(inter_surf, (self.WIDTH - inter_surf.get_width() - 10, 10))

        # Controls text
        controls_surf = self.font_small.render(self.user_guide, True, self.COLOR_UI_TEXT)
        self.screen.blit(controls_surf, (10, self.HEIGHT - controls_surf.get_height() - 5))

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()