
# Generated: 2025-08-27T17:08:40.038553
# Source Brief: brief_01438.md
# Brief Index: 1438

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game objects
class Enemy:
    def __init__(self, pos, health, speed, value, path):
        self.pos = pygame.math.Vector2(pos)
        self.max_health = health
        self.health = health
        self.speed = speed
        self.value = value
        self.path = path
        self.path_index = 0
        self.alive = True
        self.slow_timer = 0

    def update(self):
        if not self.alive:
            return

        current_speed = self.speed * 0.5 if self.slow_timer > 0 else self.speed
        self.slow_timer = max(0, self.slow_timer - 1)

        if self.path_index < len(self.path):
            target_pos = self.path[self.path_index]
            direction = (target_pos - self.pos)
            if direction.length() < current_speed:
                self.pos = target_pos
                self.path_index += 1
            else:
                self.pos += direction.normalize() * current_speed
        else:
            self.alive = False # Reached the end

    def draw(self, screen):
        if not self.alive:
            return
        x, y = int(self.pos.x), int(self.pos.y)
        size = 12
        color = (255, 80, 80) if self.slow_timer == 0 else (255, 150, 150)
        
        pygame.draw.rect(screen, color, (x - size // 2, y - size // 2, size, size))
        
        # Health bar
        bar_width = 20
        health_pct = self.health / self.max_health
        pygame.draw.rect(screen, (255, 0, 0), (x - bar_width // 2, y - size, bar_width, 3))
        pygame.draw.rect(screen, (0, 255, 0), (x - bar_width // 2, y - size, int(bar_width * health_pct), 3))

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.alive = False


class Tower:
    def __init__(self, grid_pos, tower_def, cell_size):
        self.grid_pos = grid_pos
        self.pos = pygame.math.Vector2(
            (grid_pos[0] + 0.5) * cell_size,
            (grid_pos[1] + 0.5) * cell_size
        )
        self.type_def = tower_def
        self.range = tower_def['range']
        self.damage = tower_def['damage']
        self.fire_rate = tower_def['fire_rate']
        self.effect = tower_def.get('effect', None)
        self.cooldown = 0
        self.target = None

    def update(self, enemies, projectiles):
        self.cooldown = max(0, self.cooldown - 1)
        
        # Find new target if current is dead or out of range
        if self.target and (not self.target.alive or self.pos.distance_to(self.target.pos) > self.range):
            self.target = None

        if not self.target:
            # Find closest enemy in range that is furthest along the path
            valid_targets = [e for e in enemies if self.pos.distance_to(e.pos) <= self.range]
            if valid_targets:
                self.target = max(valid_targets, key=lambda e: e.path_index + e.pos.distance_to(e.path[e.path_index-1]) if e.path_index > 0 else 0)

        # Fire if ready and has a target
        if self.target and self.cooldown == 0:
            # # sound: self.type_def['sound']
            projectiles.append(Projectile(self.pos, self.target, self.damage, self.effect))
            self.cooldown = 60 / self.fire_rate # Cooldown in frames (assuming 60fps logic, but it's step-based)

    def draw(self, screen):
        pygame.gfxdraw.filled_circle(screen, int(self.pos.x), int(self.pos.y), 12, self.type_def['color'])
        pygame.gfxdraw.aacircle(screen, int(self.pos.x), int(self.pos.y), 12, (200, 200, 200))
        # Draw inner symbol for type
        if self.type_def['name'] == "Gatling":
            pygame.draw.circle(screen, (255,255,255), self.pos, 3)
        elif self.type_def['name'] == "Cannon":
            pygame.draw.rect(screen, (255,255,255), (self.pos.x-3, self.pos.y-3, 6, 6))
        elif self.type_def['name'] == "Slow Tower":
            pygame.gfxdraw.filled_trigon(screen, int(self.pos.x), int(self.pos.y-5), int(self.pos.x-5), int(self.pos.y+3), int(self.pos.x+5), int(self.pos.y+3), (255,255,255))


class Projectile:
    def __init__(self, pos, target, damage, effect):
        self.pos = pygame.math.Vector2(pos)
        self.target = target
        self.damage = damage
        self.effect = effect
        self.speed = 10
        self.alive = True

    def update(self):
        if not self.alive:
            return
        if not self.target.alive:
            self.alive = False
            return
        
        direction = (self.target.pos - self.pos)
        if direction.length() < self.speed:
            self.pos = self.target.pos
            self.alive = False # Hit target
        else:
            self.pos += direction.normalize() * self.speed

    def draw(self, screen):
        if self.alive:
            pygame.gfxdraw.filled_circle(screen, int(self.pos.x), int(self.pos.y), 3, (100, 180, 255))
            pygame.gfxdraw.aacircle(screen, int(self.pos.x), int(self.pos.y), 3, (200, 220, 255))

class Particle:
    def __init__(self, pos, color):
        self.pos = pygame.math.Vector2(pos)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        self.lifespan = random.randint(10, 20)
        self.color = color

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.lifespan -= 1

    def draw(self, screen):
        if self.lifespan > 0:
            size = int(self.lifespan / 4)
            pygame.draw.circle(screen, self.color, self.pos, max(0, size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    user_guide = "Controls: Arrows to move cursor, Space to place tower, Shift to cycle tower type."
    game_description = "Defend your base from waves of enemies by placing strategic defensive towers."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 32, 20
    CELL_SIZE = WIDTH // GRID_W
    MAX_STEPS = 15000
    TOTAL_WAVES = 20

    COLOR_BG = (20, 25, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_GRID = (30, 40, 50)
    COLOR_BASE = (80, 200, 120)
    COLOR_TEXT = (220, 220, 220)
    
    TOWER_DEFS = [
        {"name": "Gatling", "cost": 100, "damage": 5, "range": 80, "fire_rate": 5, "color": (150, 150, 160), "sound": "gatling_fire"},
        {"name": "Cannon", "cost": 250, "damage": 40, "range": 120, "fire_rate": 0.75, "color": (100, 100, 110), "sound": "cannon_fire"},
        {"name": "Slow Tower", "cost": 150, "damage": 1, "range": 70, "fire_rate": 2, "effect": {"type": "slow", "duration": 60}, "color": (120, 150, 180), "sound": "slow_fire"},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("sans", 16)
        self.font_m = pygame.font.SysFont("sans", 20)
        self.font_l = pygame.font.SysFont("sans", 48)

        # State variables are initialized in reset()
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        # Resources & Health
        self.base_health = 100
        self.max_base_health = 100
        self.resources = 300
        
        # Wave management
        self.wave_number = 0
        self.time_to_next_wave = 180 # 3 seconds at 60fps logic
        self.enemies_in_wave = 0
        self.enemies_spawned_this_wave = 0
        self.wave_spawn_timer = 0
        
        # Game objects
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        # Path
        self._generate_path()
        self.base_pos = self.path[-1]

        # Player state
        self.cursor_pos = [self.GRID_W // 4, self.GRID_H // 2]
        self.selected_tower_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # --- Handle player actions ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update game logic ---
        # Wave management
        if not self.enemies and self.enemies_spawned_this_wave == self.enemies_in_wave:
            self.time_to_next_wave -= 1
            if self.time_to_next_wave <= 0 and self.wave_number < self.TOTAL_WAVES:
                self.wave_number += 1
                reward += 1.0 # Wave complete reward
                self._start_wave()

        # Spawning
        self._spawn_enemies()
        
        # Update objects
        for t in self.towers: t.update(self.enemies, self.projectiles)
        for p in self.projectiles: p.update()
        for e in self.enemies: e.update()
        for p in self.particles: p.update()
        
        # Collisions and state changes
        reward += self._handle_collisions()
        
        # Cleanup
        self.enemies = [e for e in self.enemies if e.alive]
        self.projectiles = [p for p in self.projectiles if p.alive]
        self.particles = [p for p in self.particles if p.lifespan > 0]
        
        # --- Termination checks ---
        self.steps += 1
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            reward = -100.0
            self.game_over = True
            terminated = True
        elif self.wave_number >= self.TOTAL_WAVES and not self.enemies:
            reward = 100.0
            self.game_over = True
            self.game_won = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.score += reward
        
        # Update previous action states
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Movement (one-shot, not continuous hold)
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Cycle tower type on press
        if shift_held and not self.prev_shift_held:
            # # sound: "ui_cycle"
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_DEFS)

        # Place tower on press
        if space_held and not self.prev_space_held:
            self._try_place_tower()

    def _try_place_tower(self):
        tower_def = self.TOWER_DEFS[self.selected_tower_idx]
        if self.resources >= tower_def['cost']:
            is_valid_placement = True
            # Cannot place on path
            if self.grid[self.cursor_pos[1]][self.cursor_pos[0]] == 1:
                is_valid_placement = False
            # Cannot place on other towers
            for t in self.towers:
                if t.grid_pos == self.cursor_pos:
                    is_valid_placement = False
                    break
            
            if is_valid_placement:
                # # sound: "place_tower"
                self.resources -= tower_def['cost']
                self.towers.append(Tower(tuple(self.cursor_pos), tower_def, self.CELL_SIZE))

    def _handle_collisions(self):
        reward = 0
        
        # Projectile hits
        for p in self.projectiles:
            if not p.alive: # Hit on its update step
                # # sound: "hit_enemy"
                p.target.take_damage(p.damage)
                for _ in range(5): self.particles.append(Particle(p.pos, (100, 180, 255)))
                
                if p.effect and p.effect['type'] == 'slow':
                    p.target.slow_timer = p.effect['duration']

                if not p.target.alive:
                    # # sound: "enemy_die"
                    reward += 0.1
                    self.resources += p.target.value
                    for _ in range(20): self.particles.append(Particle(p.target.pos, (255, 80, 80)))
        
        # Enemies reaching base
        for e in self.enemies:
            if not e.alive and e.path_index >= len(e.path):
                # # sound: "base_damage"
                self.base_health -= 10
                reward -= 1.0
                for _ in range(20): self.particles.append(Particle(e.pos, self.COLOR_BASE))
        
        self.base_health = max(0, self.base_health)
        return reward

    def _start_wave(self):
        self.enemies_in_wave = 3 + self.wave_number * 2
        self.enemies_spawned_this_wave = 0
        self.wave_spawn_timer = 0
        self.time_to_next_wave = 180 + self.wave_number * 15

    def _spawn_enemies(self):
        if self.enemies_spawned_this_wave < self.enemies_in_wave:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0:
                health = 50 * (1.05 ** self.wave_number)
                speed = 1.0 * (1.02 ** self.wave_number)
                value = 10 + self.wave_number
                self.enemies.append(Enemy(self.path[0], health, speed, value, self.path))
                self.enemies_spawned_this_wave += 1
                self.wave_spawn_timer = 30 # Time between enemies in a wave

    def _generate_path(self):
        self.grid = np.zeros((self.GRID_H, self.GRID_W), dtype=int)
        path_coords = []
        
        sy = random.randint(3, self.GRID_H - 4)
        curr = [0, sy]
        path_coords.append(curr)
        self.grid[curr[1], curr[0]] = 1

        while curr[0] < self.GRID_W - 1:
            possible_moves = []
            # Strong bias to move right
            for _ in range(5): possible_moves.append((1, 0))
            possible_moves.extend([(1, -1), (1, 1), (0, 1), (0, -1)])
            
            random.shuffle(possible_moves)
            moved = False
            for move in possible_moves:
                nx, ny = curr[0] + move[0], curr[1] + move[1]
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid[ny, nx] == 0:
                    # Check neighbors to prevent path from touching itself
                    neighbors = [(nx+1, ny), (nx-1, ny), (nx, ny+1), (nx, ny-1)]
                    neighbor_path_cells = 0
                    for neighbor in neighbors:
                        if 0 <= neighbor[0] < self.GRID_W and 0 <= neighbor[1] < self.GRID_H and self.grid[neighbor[1], neighbor[0]] == 1:
                            neighbor_path_cells += 1
                    
                    if neighbor_path_cells <= 1:
                        curr = [nx, ny]
                        path_coords.append(curr)
                        self.grid[ny, nx] = 1
                        moved = True
                        break
            if not moved: # Stuck, restart generation
                self._generate_path()
                return

        self.path = [pygame.math.Vector2((x + 0.5) * self.CELL_SIZE, (y + 0.5) * self.CELL_SIZE) for x, y in path_coords]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_H):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, r * self.CELL_SIZE), (self.WIDTH, r * self.CELL_SIZE))
        for c in range(self.GRID_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (c * self.CELL_SIZE, 0), (c * self.CELL_SIZE, self.HEIGHT))

        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, width=self.CELL_SIZE)
        
        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.base_pos.x - self.CELL_SIZE/2, self.base_pos.y - self.CELL_SIZE/2, self.CELL_SIZE, self.CELL_SIZE))

        # Draw game objects
        for p in self.particles: p.draw(self.screen)
        for t in self.towers: t.draw(self.screen)
        for e in self.enemies: e.draw(self.screen)
        for p in self.projectiles: p.draw(self.screen)
        
        # Draw cursor and tower range
        self._render_cursor()

    def _render_cursor(self):
        cursor_def = self.TOWER_DEFS[self.selected_tower_idx]
        cx = (self.cursor_pos[0] + 0.5) * self.CELL_SIZE
        cy = (self.cursor_pos[1] + 0.5) * self.CELL_SIZE
        
        is_valid = self.grid[self.cursor_pos[1]][self.cursor_pos[0]] == 0 and all(t.grid_pos != tuple(self.cursor_pos) for t in self.towers)
        can_afford = self.resources >= cursor_def['cost']

        color = (0, 255, 0)
        if not is_valid: color = (255, 0, 0)
        elif not can_afford: color = (255, 255, 0)
        
        # Draw range indicator
        pygame.gfxdraw.aacircle(self.screen, int(cx), int(cy), cursor_def['range'], (*color, 60))
        
        # Draw cursor box
        rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, (*color, 150), rect, 2)

    def _render_ui(self):
        # Top-left: Wave info
        wave_text = f"Wave: {self.wave_number}/{self.TOTAL_WAVES}"
        enemies_text = f"Enemies: {len(self.enemies)}/{self.enemies_in_wave}"
        self._draw_text(wave_text, (10, 10), self.font_m)
        self._draw_text(enemies_text, (10, 35), self.font_m)

        # Top-right: Stats
        health_text = f"Base HP: {self.base_health}/{self.max_base_health}"
        resources_text = f"Gold: {self.resources}"
        self._draw_text(health_text, (self.WIDTH - 10, 10), self.font_m, align="right")
        self._draw_text(resources_text, (self.WIDTH - 10, 35), self.font_m, align="right")

        # Bottom-center: Selected tower
        tower_def = self.TOWER_DEFS[self.selected_tower_idx]
        tower_info = f"Selected: {tower_def['name']} (Cost: {tower_def['cost']})"
        self._draw_text(tower_info, (self.WIDTH // 2, self.HEIGHT - 25), self.font_m, align="center")

        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "YOU WON!" if self.game_won else "GAME OVER"
            color = (100, 255, 150) if self.game_won else (255, 100, 100)
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 30), self.font_l, color, "center")

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, align="left"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "right": text_rect.topright = pos
        elif align == "center": text_rect.midtop = pos
        else: text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
            "towers_placed": len(self.towers),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11" or "windows" or "macOS"
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
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

        # Draw the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Survived {info['wave']-1} waves in {info['steps']} steps.")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Limit human play to 30 FPS
        
    env.close()