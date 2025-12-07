
# Generated: 2025-08-28T00:46:12.209794
# Source Brief: brief_03890.md
# Brief Index: 3890

        
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
class Tower:
    """Represents a defensive tower."""
    def __init__(self, grid_pos, tower_type):
        self.grid_pos = grid_pos
        self.pixel_pos = (grid_pos[0] * 40 + 20, grid_pos[1] * 40 + 20)
        self.type = tower_type
        if self.type == 1: # Blue: Fast, low damage
            self.range = 2.5 * 40
            self.damage = 1
            self.fire_rate = 30 # ticks per shot
            self.color = (50, 150, 255)
        else: # Yellow: Slow, high damage
            self.range = 4.0 * 40
            self.damage = 2
            self.fire_rate = 60 # ticks per shot
            self.color = (255, 220, 50)
        self.cooldown = 0
        self.target = None

    def update(self, enemies, projectiles):
        """Updates tower logic: finds a target and fires if ready."""
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        # Find new target if current is out of range or dead
        if self.target and (not self.target.is_alive or math.hypot(self.pixel_pos[0] - self.target.pos[0], self.pixel_pos[1] - self.target.pos[1]) > self.range):
            self.target = None

        if not self.target:
            # Find the enemy that is furthest along the path
            best_target = None
            max_path_dist = -1
            for enemy in enemies:
                dist = math.hypot(self.pixel_pos[0] - enemy.pos[0], self.pixel_pos[1] - enemy.pos[1])
                if dist <= self.range:
                    path_dist = enemy.path_index + (1 - np.linalg.norm(np.array(enemy.path[enemy.path_index+1]) - enemy.pos) / np.linalg.norm(np.array(enemy.path[enemy.path_index+1]) - np.array(enemy.path[enemy.path_index])))
                    if path_dist > max_path_dist:
                        max_path_dist = path_dist
                        best_target = enemy
            self.target = best_target
        
        if self.target and self.cooldown <= 0:
            self.fire(projectiles)
            self.cooldown = self.fire_rate

    def fire(self, projectiles):
        # sfx: tower_shoot.wav
        projectiles.append(Projectile(self.pixel_pos, self.target, self.damage))

class Enemy:
    """Represents an enemy unit moving along a path."""
    def __init__(self, path, health, speed):
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[0], dtype=float)
        self.max_health = health
        self.health = health
        self.speed = speed
        self.is_alive = True

    def update(self):
        """Moves the enemy along its path."""
        if not self.is_alive or self.path_index >= len(self.path) - 1:
            return False # Reached the end

        target_node = self.path[self.path_index + 1]
        direction = np.array(target_node) - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = np.array(target_node, dtype=float)
            self.path_index += 1
        else:
            self.pos += (direction / distance) * self.speed
        
        return self.path_index >= len(self.path) - 1

class Projectile:
    """Represents a projectile fired from a tower."""
    def __init__(self, start_pos, target, damage):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target
        self.damage = damage
        self.speed = 8.0

    def update(self):
        """Moves the projectile towards its target."""
        if not self.target or not self.target.is_alive:
            return True # Remove projectile if target is gone

        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)
        
        if distance < self.speed:
            return True # Hit target
        else:
            self.pos += (direction / distance) * self.speed
            return False # Still moving

class Particle:
    """Represents a visual effect particle."""
    def __init__(self, x, y, color, life, size, velocity):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        self.vx, self.vy = velocity

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life <= 0

    def draw(self, surface):
        alpha = int(255 * (self.life / self.max_life))
        color = (*self.color, alpha)
        r = int(self.size * (self.life / self.max_life))
        if r > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), r, color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the cursor. Press space for a blue tower (fast, low damage) "
        "or shift for a yellow tower (slow, high damage). Press nothing to confirm placement and start the wave."
    )

    game_description = (
        "A top-down tower defense game. Place towers on the grid to defend your base from waves of enemies. "
        "Survive 10 waves to win."
    )

    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    TOTAL_WAVES = 10
    
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 55)
    COLOR_PATH = (50, 50, 70)
    COLOR_BASE = (50, 200, 50)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_CURSOR = (255, 255, 255)
    
    PHASE_PLACEMENT = 0
    PHASE_WAVE = 1
    
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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.path_coords = []
        self._generate_path()
        self.path_pixels = [(gx * self.CELL_SIZE + self.CELL_SIZE // 2, gy * self.CELL_SIZE + self.CELL_SIZE // 2) for gx, gy in self.path_coords]
        self.base_pos_grid = self.path_coords[-1]
        
        self.reset()
        
    def _generate_path(self):
        self.path_coords = [(0,5), (1,5), (2,5), (3,5), (3,4), (3,3), (3,2), (4,2), (5,2), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (7,7), (8,7), (9,7), (10,7), (10,6), (10,5), (11,5), (12,5), (13,5), (14,5), (15,5)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.max_base_health = 100
        
        self.wave_number = 0
        self.enemies_to_spawn = []
        self.active_enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.game_phase = self.PHASE_PLACEMENT
        self.placements_allowed = 1
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 1 # Default to type 1

        self.occupied_cells = set(self.path_coords)
        self.occupied_cells.add(self.base_pos_grid)

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        self.game_phase = self.PHASE_WAVE
        
        num_enemies = 3 + (self.wave_number - 1)
        enemy_health = 5 + (self.wave_number - 1) // 2
        enemy_speed = 1.0 + 0.05 * (self.wave_number - 1)
        
        spawn_delay = 60 # ticks between enemies
        self.enemies_to_spawn = []
        for i in range(num_enemies):
            self.enemies_to_spawn.append({
                "delay": i * spawn_delay,
                "health": enemy_health,
                "speed": enemy_speed
            })
        
        self.active_enemies = []
        
        return 0 # No reward just for starting

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small time penalty
        
        # --- Handle player input based on game phase ---
        if self.game_phase == self.PHASE_PLACEMENT:
            # Move cursor
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
            
            # Select tower type
            if space_held: self.selected_tower_type = 1
            if shift_held: self.selected_tower_type = 2
                
            # Confirm placement
            if movement == 0 and self.placements_allowed > 0:
                pos_tuple = tuple(self.cursor_pos)
                if pos_tuple not in self.occupied_cells:
                    # sfx: place_tower.wav
                    self.towers.append(Tower(pos_tuple, self.selected_tower_type))
                    self.occupied_cells.add(pos_tuple)
                    self.placements_allowed -= 1
                    self._start_next_wave()
        
        # --- Update game logic if wave is active ---
        if self.game_phase == self.PHASE_WAVE:
            # Spawn enemies
            spawn_list = self.enemies_to_spawn[:]
            self.enemies_to_spawn = []
            for enemy_data in spawn_list:
                if enemy_data["delay"] <= 0:
                    self.active_enemies.append(Enemy(self.path_pixels, enemy_data["health"], enemy_data["speed"]))
                else:
                    enemy_data["delay"] -= 1
                    self.enemies_to_spawn.append(enemy_data)

            # Update towers
            for tower in self.towers:
                tower.update(self.active_enemies, self.projectiles)
            
            # Update projectiles
            new_projectiles = []
            for proj in self.projectiles:
                if proj.update(): # True if hit or target gone
                    if proj.target and proj.target.is_alive and math.hypot(proj.pos[0] - proj.target.pos[0], proj.pos[1] - proj.target.pos[1]) < proj.speed * 1.5:
                        # sfx: enemy_hit.wav
                        proj.target.health -= proj.damage
                        reward += 0.1
                        self.score += 1
                        for _ in range(5): # Hit particles
                            angle = self.np_random.uniform(0, 2 * math.pi)
                            speed = self.np_random.uniform(1, 3)
                            self.particles.append(Particle(proj.pos[0], proj.pos[1], (255,255,255), 10, 2, (math.cos(angle) * speed, math.sin(angle) * speed)))
                else:
                    new_projectiles.append(proj)
            self.projectiles = new_projectiles
            
            # Update enemies
            new_active_enemies = []
            for enemy in self.active_enemies:
                if enemy.health <= 0 and enemy.is_alive:
                    # sfx: enemy_death.wav
                    enemy.is_alive = False
                    reward += 1
                    self.score += 10
                    for _ in range(20): # Death particles
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        speed = self.np_random.uniform(1, 4)
                        self.particles.append(Particle(enemy.pos[0], enemy.pos[1], self.COLOR_ENEMY, 20, 3, (math.cos(angle) * speed, math.sin(angle) * speed)))
                
                if enemy.is_alive:
                    if enemy.update(): # True if reached base
                        # sfx: base_damage.wav
                        self.base_health -= enemy.max_health # Damage based on enemy strength
                        reward -= 5
                        self.score -= 50
                    else:
                        new_active_enemies.append(enemy)
            self.active_enemies = new_active_enemies
            
            # Check for wave end
            if not self.active_enemies and not self.enemies_to_spawn:
                reward += 10 # Wave clear bonus
                self.score += 100
                if self.wave_number >= self.TOTAL_WAVES:
                    self.game_over = True # Victory
                    reward += 100
                    self.score += 1000
                else:
                    self.game_phase = self.PHASE_PLACEMENT
                    self.placements_allowed = 1

        # Update particles
        self.particles = [p for p in self.particles if not p.update()]
        
        self.steps += 1
        terminated = self._check_termination()
        if terminated and not self.game_over: # Loss condition
            reward -= 100
            self.score -= 1000
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.base_health <= 0:
            return True
        if self.wave_number >= self.TOTAL_WAVES and not self.active_enemies and not self.enemies_to_spawn:
            return True
        if self.steps >= 5000: # Max steps
            return True
        return False
        
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
        for gx, gy in self.path_coords:
            pygame.draw.rect(self.screen, self.COLOR_PATH, (gx * self.CELL_SIZE, gy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Draw base
        base_px, base_py = self.base_pos_grid[0] * self.CELL_SIZE, self.base_pos_grid[1] * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_px, base_py, self.CELL_SIZE, self.CELL_SIZE))
        pygame.gfxdraw.rectangle(self.screen, (base_px, base_py, self.CELL_SIZE, self.CELL_SIZE), (200, 255, 200))

        # Draw towers
        for tower in self.towers:
            px, py = tower.pixel_pos
            if tower.type == 1:
                pygame.gfxdraw.box(self.screen, (px-12, py-12, 24, 24), tower.color)
            else:
                points = [(px, py - 14), (px - 14, py + 10), (px + 14, py + 10)]
                pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), tower.color)
        
        # Draw projectiles
        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj.pos[0]), int(proj.pos[1]), 3, (255, 255, 255))
        
        # Draw enemies
        for enemy in self.active_enemies:
            if enemy.is_alive:
                px, py = int(enemy.pos[0]), int(enemy.pos[1])
                pygame.gfxdraw.filled_circle(self.screen, px, py, 10, self.COLOR_ENEMY)
                pygame.gfxdraw.aacircle(self.screen, px, py, 10, (255,100,100))
                # Health bar
                health_ratio = max(0, enemy.health / enemy.max_health)
                bar_width = 20
                pygame.draw.rect(self.screen, (50,0,0), (px - bar_width/2, py - 20, bar_width, 5))
                pygame.draw.rect(self.screen, (0,200,0), (px - bar_width/2, py - 20, bar_width * health_ratio, 5))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw placement UI
        if self.game_phase == self.PHASE_PLACEMENT:
            cx, cy = self.cursor_pos
            px, py = cx * self.CELL_SIZE, cy * self.CELL_SIZE
            
            # Cursor
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (px, py, self.CELL_SIZE, self.CELL_SIZE), 2)
            
            is_valid = tuple(self.cursor_pos) not in self.occupied_cells and self.placements_allowed > 0
            tower_preview_color = Tower(self.cursor_pos, self.selected_tower_type).color
            tower_range = Tower(self.cursor_pos, self.selected_tower_type).range
            alpha = 100 if is_valid else 40
            
            pygame.gfxdraw.filled_circle(self.screen, px + self.CELL_SIZE//2, py + self.CELL_SIZE//2, int(tower_range), (*tower_preview_color, alpha//2))
            pygame.gfxdraw.aacircle(self.screen, px + self.CELL_SIZE//2, py + self.CELL_SIZE//2, int(tower_range), (*tower_preview_color, alpha))
            
            if self.selected_tower_type == 1:
                pygame.gfxdraw.box(self.screen, (px+8, py+8, 24, 24), (*tower_preview_color, alpha+50))
            else:
                points = [(px+20, py+6), (px+6, py+30), (px+34, py+30)]
                pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], (*tower_preview_color, alpha+50))


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        # Wave
        wave_text = self.font_large.render(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 5))
        
        # Base Health
        base_px, base_py = self.base_pos_grid[0] * self.CELL_SIZE, self.base_pos_grid[1] * self.CELL_SIZE
        health_ratio = max(0, self.base_health / self.max_base_health)
        bar_width = self.CELL_SIZE * 2
        bar_height = 10
        pygame.draw.rect(self.screen, (50,0,0), (base_px + self.CELL_SIZE/2 - bar_width/2, base_py - bar_height - 5, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_px + self.CELL_SIZE/2 - bar_width/2, base_py - bar_height - 5, bar_width * health_ratio, bar_height))
        
        # Game Phase Text
        if self.game_phase == self.PHASE_PLACEMENT and self.wave_number < self.TOTAL_WAVES:
            if self.placements_allowed > 0:
                phase_text = self.font_small.render("PLACE TOWER. NO-OP TO START WAVE.", True, self.COLOR_TEXT)
            else:
                phase_text = self.font_small.render("NO MORE PLACEMENTS. NO-OP TO START WAVE.", True, self.COLOR_TEXT)
            self.screen.blit(phase_text, (self.SCREEN_WIDTH/2 - phase_text.get_width()/2, self.SCREEN_HEIGHT - 20))
        
        # Game Over Text
        if self.game_over:
            won = self.base_health > 0
            end_text = "VICTORY!" if won else "GAME OVER"
            end_color = (100, 255, 100) if won else (255, 100, 100)
            text_surf = self.font_large.render(end_text, True, end_color)
            self.screen.blit(text_surf, (self.SCREEN_WIDTH/2 - text_surf.get_width()/2, self.SCREEN_HEIGHT/2 - text_surf.get_height()/2))
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "enemies_left": len(self.active_enemies) + len(self.enemies_to_spawn)
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # Start with no-op

    while not done:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            # To make it turn-based for human play, we only register one action per key press
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                else: movement = 0 # Any other key is a no-op / confirm

                action = [movement, space, shift]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        # During waves, we need to auto-step
        if env.game_phase == env.PHASE_WAVE and not done:
            obs, reward, terminated, truncated, info = env.step([0,0,0])
            done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS for human play

    # Keep window open for a bit after game over
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
         for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
         env.clock.tick(30)

    env.close()