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



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to place a block. Shift to cycle block type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your fortress from waves of enemies by strategically placing blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 50)
    COLOR_FORTRESS = (0, 150, 200)
    COLOR_FORTRESS_DMG = (200, 100, 0)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_BAR_BG = (70, 70, 70)
    COLOR_HEALTH_BAR = (50, 200, 50)

    # Game Params
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 16
    BUILD_AREA_Y_START = 4
    ISO_TILE_WIDTH, ISO_TILE_HEIGHT = 40, 20
    MAX_STEPS = 1000
    MAX_WAVES = 20
    INITIAL_FORTRESS_HEALTH = 100

    # Block Types
    BLOCK_TYPES = {
        "WALL": {"color": (100, 120, 140), "health": 200, "cost": 1},
        "TURRET": {"color": (180, 180, 50), "health": 50, "cost": 3, "range": 4, "damage": 10, "cooldown": 5},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        # Pygame must run headless.
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.grid_origin = (0,0)
        self.enemies = []
        self.blocks = {}
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0,0]
        self.selected_block_type_idx = 0
        self.available_block_types = list(self.BLOCK_TYPES.keys())
        self.fortress_health = 0
        self.wave = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.turret_cooldowns = {}

        # self.reset() is called here to ensure np_random is initialized before validation
        self.reset()
        # self.validate_implementation() # This is a helper for development, not needed in final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fortress_health = self.INITIAL_FORTRESS_HEALTH
        self.wave = 0
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.BUILD_AREA_Y_START]
        self.selected_block_type_idx = 0

        self.enemies.clear()
        self.blocks.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.turret_cooldowns.clear()

        # Center the grid
        grid_pixel_width = (self.GRID_WIDTH + self.GRID_HEIGHT) * self.ISO_TILE_WIDTH / 2
        self.grid_origin = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - grid_pixel_width / 4 + 50)
        
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, place_action, cycle_action = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        step_reward = 0

        # 1. Player Action Phase
        if cycle_action:
            self.selected_block_type_idx = (self.selected_block_type_idx + 1) % len(self.available_block_types)
        
        self._move_cursor(movement)

        if place_action:
            if self._is_valid_build_pos(self.cursor_pos) and tuple(self.cursor_pos) not in self.blocks:
                block_type_name = self.available_block_types[self.selected_block_type_idx]
                block_info = self.BLOCK_TYPES[block_type_name]
                self.blocks[tuple(self.cursor_pos)] = {
                    "type": block_type_name,
                    "health": block_info["health"],
                }
                if block_type_name == "TURRET":
                    self.turret_cooldowns[tuple(self.cursor_pos)] = 0
                # self.score += 1 # Optional score for placing blocks

        # 2. Enemy and Turret AI Phase
        step_reward += self._update_turrets()
        step_reward += self._update_enemies()
        self._update_projectiles()
        self._update_particles()

        # 3. Wave Management
        if not self.enemies and not self.game_over:
            step_reward += 1  # Wave survived reward
            self.score += 10 * self.wave
            self.wave += 1
            if self.wave <= self.MAX_WAVES:
                self._spawn_wave()

        # 4. Termination Check
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            if self.fortress_health <= 0:
                step_reward = -100  # Fortress destroyed
            elif self.wave > self.MAX_WAVES:
                step_reward = 100  # Victory
                self.score += 1000
            self.game_over = True
        
        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "health": self.fortress_health}

    # --- Helper Methods ---
    
    def _world_to_iso(self, x, y):
        iso_x = self.grid_origin[0] + (x - y) * (self.ISO_TILE_WIDTH / 2)
        iso_y = self.grid_origin[1] + (x + y) * (self.ISO_TILE_HEIGHT / 2)
        return int(iso_x), int(iso_y)

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _is_valid_build_pos(self, pos):
        return self.BUILD_AREA_Y_START <= pos[1] < self.GRID_HEIGHT

    def _spawn_wave(self):
        num_enemies = self.wave + 1
        for _ in range(num_enemies):
            spawn_x = self.np_random.integers(0, self.GRID_WIDTH)
            spawn_y = self.np_random.integers(-3, 0)
            speed = 0.05 + self.wave * 0.005
            self.enemies.append(Enemy(spawn_x, spawn_y, speed, self))

    def _update_turrets(self):
        reward = 0
        turrets_to_fire = []

        for pos, block in self.blocks.items():
            if block["type"] == "TURRET":
                self.turret_cooldowns[pos] = max(0, self.turret_cooldowns[pos] - 1)
                if self.turret_cooldowns[pos] == 0:
                    turrets_to_fire.append(pos)
        
        if not self.enemies:
            return 0

        for pos in turrets_to_fire:
            turret_info = self.BLOCK_TYPES["TURRET"]
            # Find closest enemy in range
            target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = math.hypot(enemy.pos[0] - pos[0], enemy.pos[1] - pos[1])
                if dist <= turret_info["range"] and dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                start_pos = self._world_to_iso(pos[0], pos[1])
                start_pos = (start_pos[0], start_pos[1] - self.ISO_TILE_HEIGHT) # Fire from top
                self.projectiles.append(Projectile(start_pos, target, turret_info["damage"], self))
                self.turret_cooldowns[pos] = turret_info["cooldown"]

        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if enemy.update():
                enemies_to_remove.append(i)
                self.fortress_health -= 10
                self._create_particles(enemy.pixel_pos, self.COLOR_FORTRESS_DMG, 20)
                # Sound: fortress_hit.wav
        
        # Remove enemies that reached the fortress
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
        
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        enemies_to_remove = []

        for i, p in enumerate(self.projectiles):
            if p.update():
                projectiles_to_remove.append(i)
                # Check for hit
                if p.target in self.enemies and not p.target.is_dead:
                    hit = p.target.take_damage(p.damage)
                    self._create_particles(p.pos, self.COLOR_ENEMY, 10)
                    if hit: # if enemy was killed
                        reward += 0.1
                        self.score += 5
                        enemies_to_remove.append(p.target)
        
        # Remove projectiles and killed enemies
        for i in sorted(projectiles_to_remove, reverse=True): del self.projectiles[i]
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return reward

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            if p.update():
                particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _check_termination(self):
        return self.fortress_health <= 0 or self.wave > self.MAX_WAVES

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos, color, self.np_random))

    # --- Rendering Methods ---

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._world_to_iso(x, y)
                p2 = self._world_to_iso(x + 1, y)
                p3 = self._world_to_iso(x + 1, y + 1)
                p4 = self._world_to_iso(x, y + 1)
                color = self.COLOR_GRID if self._is_valid_build_pos((x,y)) else self.COLOR_BG
                pygame.draw.line(self.screen, color, p1, p2)
                pygame.draw.line(self.screen, color, p2, p3)

        # Draw fortress area
        fortress_y = self.BUILD_AREA_Y_START
        p1 = self._world_to_iso(0, fortress_y)
        p2 = self._world_to_iso(self.GRID_WIDTH, fortress_y)
        pygame.draw.line(self.screen, self.COLOR_FORTRESS, p1, p2, 3)

        # Draw blocks
        for pos, block in self.blocks.items():
            self._render_iso_cube(pos, self.BLOCK_TYPES[block["type"]]["color"], 1.0)
            if block["health"] < self.BLOCK_TYPES[block["type"]]["health"]:
                px, py = self._world_to_iso(pos[0], pos[1])
                health_pct = block["health"] / self.BLOCK_TYPES[block["type"]]["health"]
                bar_width = self.ISO_TILE_WIDTH * 0.8
                pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (px - bar_width/2, py - 25, bar_width, 5))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (px - bar_width/2, py - 25, bar_width * health_pct, 5))

        # Draw enemies
        for enemy in self.enemies:
            enemy.render(self.screen)

        # Draw projectiles and particles
        for p in self.projectiles: p.render(self.screen)
        for p in self.particles: p.render(self.screen)

        # Draw cursor
        if self._is_valid_build_pos(self.cursor_pos):
            block_type = self.available_block_types[self.selected_block_type_idx]
            color = self.BLOCK_TYPES[block_type]["color"]
            self._render_iso_cube(self.cursor_pos, color, 0.5, True)
        else:
            px, py = self._world_to_iso(self.cursor_pos[0], self.cursor_pos[1])
            pygame.draw.circle(self.screen, (255,0,0,100), (px, py), 10, 2)


    def _render_iso_cube(self, pos, color, alpha=1.0, wireframe=False):
        x, y = pos
        px, py = self._world_to_iso(x, y)
        
        tile_w = self.ISO_TILE_WIDTH
        tile_h = self.ISO_TILE_HEIGHT
        
        top_points = [
            (px, py - tile_h / 2),
            (px + tile_w / 2, py),
            (px, py + tile_h / 2),
            (px - tile_w / 2, py),
        ]
        
        left_side = [
            (px - tile_w / 2, py),
            (px, py + tile_h / 2),
            (px, py + tile_h / 2 + tile_h),
            (px - tile_w / 2, py + tile_h),
        ]

        right_side = [
            (px + tile_w / 2, py),
            (px, py + tile_h / 2),
            (px, py + tile_h / 2 + tile_h),
            (px + tile_w / 2, py + tile_h),
        ]
        
        darker_color = tuple(max(0, c - 40) for c in color)
        darkest_color = tuple(max(0, c - 60) for c in color)
        
        if wireframe:
            pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, top_points, 2)
            pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, left_side, 2)
            pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, right_side, 2)
        else:
            # Draw with alpha
            surf = pygame.Surface((int(tile_w) + 2, int(tile_h * 2) + 2), pygame.SRCALPHA)
            
            # Adjust points to be local to the surface
            offset_x = px - tile_w / 2
            offset_y = py - tile_h / 2
            local_top = [(p[0] - offset_x, p[1] - offset_y) for p in top_points]
            local_left = [(p[0] - offset_x, p[1] - offset_y) for p in left_side]
            local_right = [(p[0] - offset_x, p[1] - offset_y) for p in right_side]

            pygame.gfxdraw.aapolygon(surf, local_top, color)
            pygame.gfxdraw.filled_polygon(surf, local_top, color)
            pygame.gfxdraw.aapolygon(surf, local_left, darker_color)
            pygame.gfxdraw.filled_polygon(surf, local_left, darker_color)
            pygame.gfxdraw.aapolygon(surf, local_right, darkest_color)
            pygame.gfxdraw.filled_polygon(surf, local_right, darkest_color)
            
            if alpha < 1.0:
                surf.set_alpha(int(alpha * 255))
            
            self.screen.blit(surf, (offset_x, offset_y))


    def _render_ui(self):
        # Fortress Health
        health_text = self.font_small.render(f"Fortress Health: {self.fortress_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        health_pct = max(0, self.fortress_health / self.INITIAL_FORTRESS_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 35, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 35, 200 * health_pct, 20))

        # Wave and Score
        wave_text = self.font_small.render(f"Wave: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - 150, 10))
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 150, 35))

        # Selected Block
        block_type_name = self.available_block_types[self.selected_block_type_idx]
        block_text = self.font_small.render(f"Selected: {block_type_name}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (10, self.SCREEN_HEIGHT - 30))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "VICTORY" if self.wave > self.MAX_WAVES else "GAME OVER"
            text_surf = self.font_large.render(win_text, True, self.COLOR_CURSOR)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)


class Enemy:
    def __init__(self, x, y, speed, env):
        self.pos = [float(x), float(y)]
        self.speed = speed
        self.env = env
        self.health = 10 + env.wave * 2
        self.is_dead = False

    @property
    def grid_pos(self):
        return (int(round(self.pos[0])), int(round(self.pos[1])))
    
    @property
    def pixel_pos(self):
        return self.env._world_to_iso(self.pos[0], self.pos[1])

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0 and not self.is_dead:
            self.is_dead = True
            self.env._create_particles(self.pixel_pos, GameEnv.COLOR_ENEMY, 30)
            # Sound: enemy_explode.wav
            return True
        return False

    def update(self):
        if self.is_dead: return False
        
        # Simple pathfinding: move towards the fortress line
        target_y = self.env.BUILD_AREA_Y_START
        
        # Move on grid if not blocked
        next_grid_pos = list(self.grid_pos)
        if self.pos[1] < target_y:
            next_grid_pos[1] += 1
        
        if tuple(next_grid_pos) in self.env.blocks:
            # Attack block
            block = self.env.blocks[tuple(next_grid_pos)]
            block['health'] -= 1
            self.env._create_particles(self.env._world_to_iso(next_grid_pos[0], next_grid_pos[1]), (150,150,150), 3)
            # Sound: block_hit.wav
            if block['health'] <= 0:
                del self.env.blocks[tuple(next_grid_pos)]
                if block['type'] == 'TURRET':
                    del self.env.turret_cooldowns[tuple(next_grid_pos)]
                self.env.score -= 1 # Penalty for losing a block
        else:
            self.pos[1] += self.speed

        # Reached fortress
        if self.pos[1] >= target_y:
            return True
        return False
        
    def render(self, screen):
        if self.is_dead: return
        self.env._render_iso_cube(self.pos, GameEnv.COLOR_ENEMY, 0.8)

class Projectile:
    def __init__(self, start_pos, target_enemy, damage, env):
        self.pos = list(start_pos)
        self.target = target_enemy
        self.damage = damage
        self.env = env
        self.speed = 8

    def update(self):
        if self.target.is_dead: return True
        
        target_pos = self.target.pixel_pos
        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.pos = target_pos
            return True
        
        self.pos[0] += (dx / dist) * self.speed
        self.pos[1] += (dy / dist) * self.speed
        return False

    def render(self, screen):
        pygame.draw.circle(screen, (255, 255, 100), (int(self.pos[0]), int(self.pos[1])), 3)

class Particle:
    def __init__(self, pos, color, rng):
        self.pos = list(pos)
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.color = color
        self.lifespan = rng.integers(10, 20)
        self.size = rng.integers(2, 5)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.1 # Gravity
        self.lifespan -= 1
        return self.lifespan <= 0

    def render(self, screen):
        pygame.draw.rect(screen, self.color, (int(self.pos[0]), int(self.pos[1]), self.size, self.size))

if __name__ == '__main__':
    # This block allows you to play the game manually
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "dummy" for headless, "x11" or "windows" for visible
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override screen to be the display screen
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Game loop
    while not terminated:
        movement = 0
        place_action = 0
        cycle_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    place_action = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    cycle_action = 1
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        action = (movement, place_action, cycle_action)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        
        clock.tick(30) # Limit frame rate for human playability
        
    pygame.quit()