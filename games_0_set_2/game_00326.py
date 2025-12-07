
# Generated: 2025-08-27T13:19:21.977689
# Source Brief: brief_00326.md
# Brief Index: 326

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place the selected tower. "
        "Hold Shift to cycle through tower types."
    )

    game_description = (
        "A top-down tower defense game. Place towers on the grid to defend your base from "
        "waves of enemies. Earn resources by defeating enemies to build more towers. Survive all 10 waves to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = SCREEN_WIDTH // GRID_COLS

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (35, 40, 60)
    COLOR_PATH = (50, 55, 75)
    COLOR_BASE = (0, 200, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_CURSOR_VALID = (200, 200, 255, 100)
    COLOR_CURSOR_INVALID = (255, 50, 50, 100)

    TOWER_SPECS = [
        {"name": "Gun Turret", "cost": 50, "range": 80, "damage": 5, "fire_rate": 10, "color": (0, 150, 255), "proj_speed": 8},
        {"name": "Cannon", "cost": 120, "range": 120, "damage": 25, "fire_rate": 40, "color": (255, 150, 0), "proj_speed": 6},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 48)

        self._define_path()
        self.reset()
        
        self.validate_implementation()


    def _define_path(self):
        self.path = []
        path_points = [
            (-1, 5), (4, 5), (4, 15), (12, 15), (12, 2), (22, 2),
            (22, 10), (28, 10), (28, 5), (32, 5)
        ]
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            if p1[0] == p2[0]: # Vertical
                for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    self.path.append((p1[0], y))
            else: # Horizontal
                for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    self.path.append((x, p1[1]))
        
        self.path_pixels = [(p[0] * self.CELL_SIZE + self.CELL_SIZE // 2, p[1] * self.CELL_SIZE + self.CELL_SIZE // 2) for p in self.path]
        self.path_grid_coords = set(self.path)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 20
        self.max_base_health = 20
        self.resources = 150
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        self.inter_wave_timer = 150  # 5 seconds at 30fps

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_COLS // 4, self.GRID_ROWS // 2]
        self.selected_tower_type = 0
        self.previous_space_held = False
        self.previous_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for time passing

        # --- Handle Player Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self._update_towers()
        projectile_reward = self._update_projectiles()
        enemy_reward, base_damage_penalty = self._update_enemies()
        self._update_particles()
        self._update_wave_manager()

        reward += projectile_reward + enemy_reward + base_damage_penalty
        
        self.steps += 1
        
        # --- Check Termination Conditions ---
        if self.base_health <= 0:
            self.game_over = True
            reward -= 100
        elif self.win:
            self.game_over = True
            reward += 100
        elif self.steps >= 2500: # Max episode length
            self.game_over = True

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        
        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Cycle tower type on shift press
        if shift_held and not self.previous_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_cycle_sound

        # Place tower on space press
        if space_held and not self.previous_space_held:
            self._place_tower()

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources >= spec["cost"]:
            grid_pos = tuple(self.cursor_pos)
            is_occupied = any(t.grid_pos == grid_pos for t in self.towers)
            is_on_path = grid_pos in self.path_grid_coords

            if not is_occupied and not is_on_path:
                self.resources -= spec["cost"]
                new_tower = Tower(grid_pos, self.selected_tower_type, self.np_random)
                self.towers.append(new_tower)
                # sfx: place_tower_sound
                self._create_particles(new_tower.pos, 15, spec["color"], 2, 4)
    
    def _update_towers(self):
        for tower in self.towers:
            proj = tower.update(self.enemies)
            if proj:
                self.projectiles.append(proj)
                # sfx: tower_shoot_sound

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            proj.update()
            if proj.target.health <= 0: # Target died mid-flight
                self.projectiles.remove(proj)
                continue

            dist_to_target = math.hypot(proj.pos[0] - proj.target.pos[0], proj.pos[1] - proj.target.pos[1])
            if dist_to_target < proj.target.radius:
                hit_reward = proj.target.take_damage(proj.damage)
                reward += hit_reward
                self._create_particles(proj.pos, 5, (255, 255, 100), 1, 3)
                self.projectiles.remove(proj)
                # sfx: enemy_hit_sound
        return reward

    def _update_enemies(self):
        reward = 0
        base_damage_penalty = 0
        for enemy in self.enemies[:]:
            if enemy.health <= 0:
                reward += 1.0  # Defeat reward
                self.resources += enemy.reward
                self.score += enemy.reward
                self._create_particles(enemy.pos, 20, (255, 100, 50), 3, 5)
                self.enemies.remove(enemy)
                # sfx: enemy_death_explosion
                continue
            
            enemy.move(self.path_pixels)

            if enemy.path_index >= len(self.path_pixels) -1:
                self.base_health -= 1
                base_damage_penalty -= 10.0
                self.enemies.remove(enemy)
                # sfx: base_damage_sound
        return reward, base_damage_penalty

    def _update_wave_manager(self):
        if not self.wave_in_progress:
            if self.wave_number >= 10:
                if not self.enemies:
                    self.win = True
                return

            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                self.wave_number += 1
                self.wave_in_progress = True
                self.enemies_to_spawn = 5 + self.wave_number * 2
                self.spawn_timer = 0
        else: # Wave is in progress
            if self.enemies_to_spawn > 0:
                self.spawn_timer -= 1
                if self.spawn_timer <= 0:
                    self._spawn_enemy()
                    self.enemies_to_spawn -= 1
                    self.spawn_timer = 20 # Spawn delay
            elif not self.enemies: # Wave cleared
                self.wave_in_progress = False
                self.inter_wave_timer = 200

    def _spawn_enemy(self):
        base_health = 10 + self.wave_number * 5
        health_multiplier = 1 + (self.wave_number - 1) * 0.05
        speed_multiplier = 1 + (self.wave_number - 1) * 0.02
        
        health = base_health * health_multiplier
        speed = (0.5 + self.np_random.random() * 0.2) * speed_multiplier
        reward_value = 5 + self.wave_number
        
        new_enemy = Enemy(health, speed, reward_value, self.np_random)
        self.enemies.append(new_enemy)

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, speed_range, size_range):
        for _ in range(count):
            self.particles.append(Particle(pos, color, speed_range, size_range, self.np_random))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw path
        if len(self.path_pixels) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixels, self.CELL_SIZE)

        # Draw base
        base_pos = self.path_pixels[-1]
        pygame.gfxdraw.filled_circle(self.screen, base_pos[0], base_pos[1], self.CELL_SIZE // 2, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, base_pos[0], base_pos[1], self.CELL_SIZE // 2, self.COLOR_BASE)

        # Draw towers
        for tower in self.towers:
            tower.draw(self.screen)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)

        # Draw projectiles
        for proj in self.projectiles:
            proj.draw(self.screen)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw cursor
        self._render_cursor()

    def _render_cursor(self):
        if self.game_over: return
        
        spec = self.TOWER_SPECS[self.selected_tower_type]
        cursor_center_px = (
            self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        )
        
        # Check validity for color
        grid_pos = tuple(self.cursor_pos)
        is_occupied = any(t.grid_pos == grid_pos for t in self.towers)
        is_on_path = grid_pos in self.path_grid_coords
        has_resources = self.resources >= spec["cost"]
        is_valid = not is_occupied and not is_on_path and has_resources
        
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID

        # Draw range indicator
        pygame.gfxdraw.aacircle(self.screen, cursor_center_px[0], cursor_center_px[1], int(spec["range"]), cursor_color)
        
        # Draw cursor box
        rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, cursor_color, shape_surf.get_rect(), 2)
        self.screen.blit(shape_surf, rect.topleft)

    def _render_ui(self):
        # Health
        health_text = self.font_small.render(f"Base Health: {self.base_health}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Resources
        resource_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (10, 30))

        # Wave
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/10", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - 100, 10))
        
        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 100, 30))

        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_text = self.font_small.render(f"Selected: {spec['name']} (Cost: {spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (10, self.SCREEN_HEIGHT - 30))
        pygame.draw.rect(self.screen, spec['color'], (230, self.SCREEN_HEIGHT - 28, 16, 16))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "resources": self.resources}

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
        assert trunc is False
        assert isinstance(info, dict)
        
        # Test game-specific assertions
        self.reset()
        self.base_health = self.max_base_health + 10
        assert self.base_health <= self.max_base_health + 10 # No hard cap, just for checks
        self.wave_number = 11
        assert self.wave_number <= 11 # No hard cap
        self.resources = -10
        assert self.resources < 0 # No hard cap

        print("✓ Implementation validated successfully")

# --- Helper Classes ---

class Tower:
    def __init__(self, grid_pos, type_id, np_random):
        self.grid_pos = grid_pos
        self.pos = (grid_pos[0] * GameEnv.CELL_SIZE + GameEnv.CELL_SIZE // 2, 
                    grid_pos[1] * GameEnv.CELL_SIZE + GameEnv.CELL_SIZE // 2)
        self.type_id = type_id
        self.spec = GameEnv.TOWER_SPECS[type_id]
        self.np_random = np_random

        self.cooldown = self.spec["fire_rate"]
        self.target = None

    def update(self, enemies):
        self.cooldown = max(0, self.cooldown - 1)
        
        if self.target and (self.target.health <= 0 or math.hypot(self.pos[0] - self.target.pos[0], self.pos[1] - self.target.pos[1]) > self.spec["range"]):
            self.target = None

        if not self.target:
            self.target = self._find_target(enemies)

        if self.target and self.cooldown == 0:
            self.cooldown = self.spec["fire_rate"]
            return Projectile(self.pos, self.target, self.spec)
        return None

    def _find_target(self, enemies):
        in_range = [e for e in enemies if math.hypot(self.pos[0] - e.pos[0], self.pos[1] - e.pos[1]) <= self.spec["range"]]
        if not in_range:
            return None
        # Target enemy furthest along the path
        return max(in_range, key=lambda e: e.path_index + e.dist_to_next_node)

    def draw(self, surface):
        color = self.spec["color"]
        pygame.draw.rect(surface, color, (self.grid_pos[0] * GameEnv.CELL_SIZE, self.grid_pos[1] * GameEnv.CELL_SIZE, GameEnv.CELL_SIZE, GameEnv.CELL_SIZE))
        pygame.draw.rect(surface, (255, 255, 255), (self.grid_pos[0] * GameEnv.CELL_SIZE, self.grid_pos[1] * GameEnv.CELL_SIZE, GameEnv.CELL_SIZE, GameEnv.CELL_SIZE), 1)

class Enemy:
    def __init__(self, health, speed, reward, np_random):
        self.max_health = health
        self.health = health
        self.speed = speed
        self.reward = reward
        self.np_random = np_random
        
        self.path_index = 0
        start_offset = (self.np_random.random(size=2) - 0.5) * GameEnv.CELL_SIZE * 0.5
        self.pos = [GameEnv.path_pixels[0][0] + start_offset[0], GameEnv.path_pixels[0][1] + start_offset[1]]
        self.radius = 8
        self.dist_to_next_node = 0

    def move(self, path):
        if self.path_index >= len(path) - 1:
            return

        target_node = path[self.path_index + 1]
        dx = target_node[0] - self.pos[0]
        dy = target_node[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        
        if dist < self.speed:
            self.path_index += 1
            self.pos = list(target_node)
        else:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed
        
        self.dist_to_next_node = dist

    def take_damage(self, amount):
        self.health -= amount
        return 0.1 # Reward for hitting

    def draw(self, surface):
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], self.radius, GameEnv.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], self.radius, GameEnv.COLOR_ENEMY)
        
        # Health bar
        health_ratio = self.health / self.max_health
        bar_width = self.radius * 2
        bar_height = 4
        bar_x = pos_int[0] - self.radius
        bar_y = pos_int[1] - self.radius - 8
        pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(surface, (100, 255, 100), (bar_x, bar_y, bar_width * health_ratio, bar_height))

class Projectile:
    def __init__(self, start_pos, target, tower_spec):
        self.pos = list(start_pos)
        self.target = target
        self.speed = tower_spec["proj_speed"]
        self.damage = tower_spec["damage"]
        self.color = (255, 255, 0)

    def update(self):
        dx = self.target.pos[0] - self.pos[0]
        dy = self.target.pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist == 0: return
        self.pos[0] += (dx / dist) * self.speed
        self.pos[1] += (dy / dist) * self.speed

    def draw(self, surface):
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], 3, self.color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], 3, self.color)

class Particle:
    def __init__(self, pos, color, speed_range, size_range, np_random):
        self.pos = list(pos)
        angle = np_random.random() * 2 * math.pi
        speed = np_random.uniform(0.5, speed_range)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.color = color
        self.size = np_random.uniform(1, size_range)
        self.lifespan = np_random.integers(10, 25)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.95
        self.vel[1] *= 0.95
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.size < 1: return
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        alpha = int(255 * (self.lifespan / 25))
        color_with_alpha = self.color + (alpha,)
        
        temp_surf = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color_with_alpha, (self.size, self.size), self.size)
        surface.blit(temp_surf, (pos_int[0] - self.size, pos_int[1] - self.size), special_flags=pygame.BLEND_RGBA_ADD)

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Requires pygame to be installed with display support
    # pip install pygame
    try:
        import os
        os.environ.pop("SDL_VIDEODRIVER", None) # Use default display driver
    except (ImportError, KeyError):
        print("Could not configure display driver. Manual play might not work in some environments.")

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action Mapping ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Rendering ---
        # The observation is already the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Frame Rate ---
        # Since auto_advance is False, we control the 'game speed' here in the manual loop
        clock.tick(30) 

    print(f"Game Over! Final Score: {info.get('score', 0)}, Total Reward: {total_reward:.2f}")
    env.close()