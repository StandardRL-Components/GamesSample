import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Tower:
    def __init__(self, pos, tower_type, properties):
        self.pos = pos
        self.type = tower_type
        self.props = properties[tower_type]
        self.color = self.props['color']
        self.range = self.props['range']
        self.damage = self.props['damage']
        self.cooldown_time = self.props['cooldown']
        self.cooldown = 0

    def update(self, enemies, projectiles):
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        target = self.find_target(enemies)
        if target:
            # SFX: Tower fire sound
            projectiles.append(Projectile(self.pos, target, self.damage, self.props.get('proj_speed', 8), self.props.get('proj_color', (255, 255, 255))))
            self.cooldown = self.cooldown_time

    def find_target(self, enemies):
        for enemy in enemies:
            dist = math.hypot(self.pos[0] - enemy.pos[0], self.pos[1] - enemy.pos[1])
            if dist <= self.range:
                return enemy
        return None

class Enemy:
    def __init__(self, path, health, speed, value, np_random):
        self.path = path
        self.path_index = 0
        self.pos = list(path[0])
        self.max_health = health
        self.health = health
        self.speed = speed
        self.value = value
        self.np_random = np_random
        self.radius = 8

    def update(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        target_pos = self.path[self.path_index + 1]
        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.path_index += 1
            self.pos = list(self.path[self.path_index])
        else:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed
        return False

class Projectile:
    def __init__(self, start_pos, target, damage, speed, color):
        self.pos = list(start_pos)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.color = color
        self.radius = 3

    def update(self):
        if self.target.health <= 0:
            return True, False # Target is dead, projectile fizzles

        dx = self.target.pos[0] - self.pos[0]
        dy = self.target.pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.pos = list(self.target.pos)
            return True, True # Hit target
        
        self.pos[0] += (dx / dist) * self.speed
        self.pos[1] += (dy / dist) * self.speed
        return False, False # In transit

class Particle:
    def __init__(self, pos, vel, color, lifetime, size):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1
        return self.lifetime <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) or SHIFT to select a tower type. Press SPACE to place the selected tower on an available grid spot."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of enemies. Survive as long as you can!"
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 6000
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Arial', 16, bold=True)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_GRID = (255, 255, 255, 20)
        self.COLOR_BASE = (0, 150, 136)
        self.COLOR_ENEMY = (211, 47, 47)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BG = (10, 10, 10)
        self.COLOR_HEALTH_FG = (76, 175, 80)
        
        # Game constants
        self.INITIAL_RESOURCES = 250
        self.INITIAL_BASE_HEALTH = 100
        self.BASE_POS = (self.WIDTH - 40, self.HEIGHT / 2)
        self.BASE_SIZE = 30
        
        # Define enemy path
        self.path = [
            (-20, 50), (80, 50), (80, 200), (250, 200), (250, 120),
            (450, 120), (450, 300), (self.WIDTH - 40, 300), 
            (self.WIDTH - 40, self.HEIGHT / 2)
        ]

        # Define tower placement grid
        self.grid_locations = []
        for y in range(30, self.HEIGHT - 30, 50):
            for x in range(30, self.WIDTH - 80, 50):
                self.grid_locations.append((x, y))
        self.occupied_locations = set()

        # Tower properties
        self.TOWER_PROPS = [
            {'cost': 50, 'range': 80, 'damage': 2, 'cooldown': 30, 'color': (3, 169, 244)}, # Blue: Basic
            {'cost': 80, 'range': 120, 'damage': 1, 'cooldown': 15, 'color': (255, 235, 59)}, # Yellow: Fast
            {'cost': 120, 'range': 60, 'damage': 5, 'cooldown': 45, 'color': (156, 39, 176)}, # Purple: Heavy
            {'cost': 100, 'range': 150, 'damage': 2, 'cooldown': 35, 'color': (255, 152, 0)}, # Orange: Long Range
            {'cost': 200, 'range': 100, 'damage': 0.5, 'cooldown': 5, 'color': (0, 188, 212), 'proj_color': (0, 229, 255), 'proj_speed': 12} # Cyan: Machine Gun
        ]
        
        self.base_health = 0 # Initialize to be set in reset()
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.towers = []
        self.enemies = deque()
        self.projectiles = deque()
        self.particles = deque()
        
        self.occupied_locations.clear()
        
        self.selected_tower_type = -1
        
        self.base_enemy_health = 10
        self.enemy_spawn_interval = 30 # steps
        self.next_spawn_step = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle Player Input
        self.selected_tower_type = -1
        if shift_held:
            self.selected_tower_type = 4
        elif 1 <= movement <= 4:
            self.selected_tower_type = movement - 1
            
        if space_held and self.selected_tower_type != -1:
            cost = self.TOWER_PROPS[self.selected_tower_type]['cost']
            if self.resources >= cost:
                placement_reward = self._place_tower(self.selected_tower_type)
                reward += placement_reward

        # 2. Update Game Logic
        self.steps += 1
        
        # Update towers (find targets, fire)
        for tower in self.towers:
            tower.update(self.enemies, self.projectiles)

        # Update projectiles
        projectiles_to_remove = []
        for proj in self.projectiles:
            fizzled, hit = proj.update()
            if hit:
                # SFX: Enemy hit
                proj.target.health -= proj.damage
                reward += 0.1 # Reward for damaging
                self._create_particles(proj.pos, 5, proj.color, 1.5)
            if fizzled:
                projectiles_to_remove.append(proj)
        for proj in projectiles_to_remove: self.projectiles.remove(proj)

        # Update enemies (move, check for death or goal)
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy.update(): # Reached base
                self.base_health -= 10
                # SFX: Base damage explosion
                self._create_particles(self.BASE_POS, 20, self.COLOR_ENEMY, 3)
                enemies_to_remove.append(enemy)
            elif enemy.health <= 0:
                # SFX: Enemy defeated
                reward += 1.0 # Reward for defeating
                self.resources += enemy.value
                self.score += 10
                self._create_particles(enemy.pos, 15, self.COLOR_ENEMY, 2)
                enemies_to_remove.append(enemy)
        for enemy in enemies_to_remove: self.enemies.remove(enemy)

        # Spawn new enemies
        if self.steps >= self.next_spawn_step:
            self._spawn_enemy()
            self.next_spawn_step = self.steps + self.enemy_spawn_interval

        # Update difficulty
        if self.steps > 0:
            if self.steps % 500 == 0:
                self.base_enemy_health += 1
            if self.steps % 1000 == 0:
                self.enemy_spawn_interval = max(5, self.enemy_spawn_interval * 0.95)

        # Update particles
        particles_to_remove = []
        for p in self.particles:
            if p.update():
                particles_to_remove.append(p)
        for p in particles_to_remove: self.particles.remove(p)

        # 3. Check Termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.base_health > 0 and not truncated: # Win condition
                reward += 100
                self.score += 1000
            elif self.base_health <= 0: # Lose condition
                reward -= 100
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _place_tower(self, tower_type_idx):
        cost = self.TOWER_PROPS[tower_type_idx]['cost']
        
        available_spots = [loc for loc in self.grid_locations if loc not in self.occupied_locations]
        if not available_spots:
            return 0 # No reward/penalty if no space

        # Use a stable placement strategy for reproducibility if needed, but random is fine for now
        pos = self.np_random.choice(available_spots)
        pos = tuple(pos)


        self.towers.append(Tower(pos, tower_type_idx, self.TOWER_PROPS))
        self.occupied_locations.add(pos)
        self.resources -= cost
        self.score -= cost
        # SFX: Tower placement sound
        self._create_particles(pos, 10, self.TOWER_PROPS[tower_type_idx]['color'], 1)
        return -0.01 * cost # Small penalty for spending resources

    def _spawn_enemy(self):
        speed = self.np_random.uniform(1.0, 1.5)
        value = 10 + int(self.steps / 500)
        self.enemies.append(Enemy(self.path, self.base_enemy_health, speed, value, self.np_random))

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = self.np_random.integers(15, 30)
            size = self.np_random.uniform(1, 4)
            self.particles.append(Particle(pos, vel, color, lifetime, size))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_path()
        self._render_grid()
        self._render_base()
        
        for p in self.particles: self._render_particle(p)
        for t in self.towers: self._render_tower(t)
        for e in self.enemies: self._render_enemy(e)
        for p in self.projectiles: self._render_projectile(p)
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_path(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 35)
        pygame.draw.lines(self.screen, (self.COLOR_PATH[0]+10, self.COLOR_PATH[1]+10, self.COLOR_PATH[2]+10), False, self.path, 33)

    def _render_grid(self):
        for loc in self.grid_locations:
            if loc not in self.occupied_locations:
                rect = pygame.Rect(loc[0] - 15, loc[1] - 15, 30, 30)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_base(self):
        x, y = int(self.BASE_POS[0]), int(self.BASE_POS[1])
        s = int(self.BASE_SIZE)
        rect = pygame.Rect(x - s / 2, y - s / 2, s, s)
        pygame.draw.rect(self.screen, self.COLOR_BASE, rect)
        pygame.draw.rect(self.screen, tuple(min(255, c+30) for c in self.COLOR_BASE), rect, 3)

    def _render_tower(self, tower):
        x, y = int(tower.pos[0]), int(tower.pos[1])
        # Range indicator
        pygame.gfxdraw.aacircle(self.screen, x, y, int(tower.range), (tower.color[0], tower.color[1], tower.color[2], 50))
        # Base
        pygame.gfxdraw.filled_circle(self.screen, x, y, 12, (30,30,30))
        pygame.gfxdraw.aacircle(self.screen, x, y, 12, (50,50,50))
        # Top
        pygame.gfxdraw.filled_circle(self.screen, x, y, 10, tower.color)
        pygame.gfxdraw.aacircle(self.screen, x, y, 10, tuple(min(255, c+50) for c in tower.color))

    def _render_enemy(self, enemy):
        x, y = int(enemy.pos[0]), int(enemy.pos[1])
        r = int(enemy.radius)
        pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, x, y, r, tuple(min(255, c+50) for c in self.COLOR_ENEMY))
        # Health bar
        if enemy.health < enemy.max_health:
            bar_w = 20
            bar_h = 4
            health_pct = max(0, enemy.health / enemy.max_health)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x - bar_w/2, y - r - 10, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (x - bar_w/2, y - r - 10, bar_w * health_pct, bar_h))

    def _render_projectile(self, proj):
        x, y = int(proj.pos[0]), int(proj.pos[1])
        r = int(proj.radius)
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, x, y, r + 2, (proj.color[0], proj.color[1], proj.color[2], 100))
        pygame.gfxdraw.filled_circle(self.screen, x, y, r, proj.color)

    def _render_particle(self, p):
        alpha = int(255 * (p.lifetime / p.max_lifetime))
        color = (p.color[0], p.color[1], p.color[2], alpha)
        if alpha > 0:
            surf = pygame.Surface((int(p.size*2), int(p.size*2)), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (p.size, p.size), p.size)
            self.screen.blit(surf, (p.pos[0] - p.size, p.pos[1] - p.size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Base Health Bar
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, 200, 20))
        health_pct = max(0, self.base_health / self.INITIAL_BASE_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, 200 * health_pct, 20))
        health_text = self.font_small.render(f"BASE: {int(self.base_health)}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Resources
        res_text = self.font_large.render(f"${int(self.resources)}", True, (255, 235, 59))
        self.screen.blit(res_text, (230, 8))

        # Score & Time
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - 120, 10))
        time_text = self.font_small.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - 120, 30))

        # Tower Selection UI
        for i, props in enumerate(self.TOWER_PROPS):
            x = 15 + i * 50
            y = self.HEIGHT - 35
            rect = pygame.Rect(x, y, 40, 40)
            pygame.draw.rect(self.screen, props['color'], rect, 0, 5)
            cost_text = self.font_small.render(f"${props['cost']}", True, (0,0,0))
            self.screen.blit(cost_text, (x + 5, y + 22))
            
            if self.selected_tower_type == i:
                pygame.draw.rect(self.screen, (255,255,255), rect, 3, 5)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            msg = "VICTORY" if self.base_health > 0 else "GAME OVER"
            color = (100, 255, 100) if self.base_health > 0 else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "base_health": self.base_health,
            "enemies_on_screen": len(self.enemies),
        }

    def _check_termination(self):
        return self.base_health <= 0

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        # Test game logic assertions
        self.reset()
        assert self.base_health <= self.INITIAL_BASE_HEALTH, "Base health exceeded max"
        assert self.resources >= 0, "Resources went below zero"
        
        # print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set a non-dummy driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode='rgb_array')

    # --- To play with keyboard ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        if not terminated and not truncated:
            # Map keyboard to MultiDiscrete action space
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_1]: movement = 1
            elif keys[pygame.K_2]: movement = 2
            elif keys[pygame.K_3]: movement = 3
            elif keys[pygame.K_4]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_5] else 0 # Use '5' for the 5th tower
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Control the speed of the game when playing manually
        else:
             # Keep displaying the final frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()


    env.close()