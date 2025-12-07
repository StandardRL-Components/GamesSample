
# Generated: 2025-08-27T23:20:51.261550
# Source Brief: brief_03430.md
# Brief Index: 3430

        
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


# --- Helper Classes for Game Entities ---

class Tower:
    def __init__(self, grid_pos):
        self.grid_pos = grid_pos
        self.range = 120  # pixels
        self.damage = 15
        self.fire_rate = 0.8  # seconds per shot
        self.cooldown = 0
        self.target = None

    def update(self, dt, enemies, projectiles):
        self.cooldown = max(0, self.cooldown - dt)
        if self.cooldown == 0:
            self.find_target(enemies)
            if self.target:
                projectiles.append(Projectile(self.grid_pos, self.target, self.damage))
                self.cooldown = self.fire_rate
                # sfx: Tower shoot sound

    def find_target(self, enemies):
        # Target the enemy closest to the end of the path
        self.target = None
        farthest_dist_on_path = -1
        
        # Pre-calculate tower's screen position once
        tower_screen_pos_x = 320 + (self.grid_pos[0] - self.grid_pos[1]) * 20
        tower_screen_pos_y = 120 + (self.grid_pos[0] + self.grid_pos[1]) * 10

        for enemy in enemies:
            dist_sq = (enemy.pos[0] - tower_screen_pos_x)**2 + (enemy.pos[1] - tower_screen_pos_y)**2
            if dist_sq <= self.range**2:
                # Enemy is in range, check if it's the most advanced one
                enemy_path_dist = enemy.path_index + enemy.progress_to_next
                if enemy_path_dist > farthest_dist_on_path:
                    farthest_dist_on_path = enemy_path_dist
                    self.target = enemy

class Enemy:
    def __init__(self, path, health, speed):
        self.path = path
        self.path_index = 0
        self.pos = list(path[0])
        self.health = health
        self.max_health = health
        self.speed = speed
        self.progress_to_next = 0.0
        self.is_alive = True

    def update(self, dt):
        if not self.is_alive or self.path_index >= len(self.path) - 1:
            return False  # Not moving or reached end

        start_node = self.path[self.path_index]
        end_node = self.path[self.path_index + 1]
        
        distance_to_travel = self.speed * dt
        
        while distance_to_travel > 0 and self.path_index < len(self.path) - 1:
            start_node = self.path[self.path_index]
            end_node = self.path[self.path_index + 1]
            
            vec = (end_node[0] - self.pos[0], end_node[1] - self.pos[1])
            node_dist = math.sqrt(vec[0]**2 + vec[1]**2)

            if node_dist == 0: # Should not happen, but a safeguard
                self.path_index += 1
                continue

            if distance_to_travel >= node_dist:
                self.pos = list(end_node)
                self.path_index += 1
                distance_to_travel -= node_dist
            else:
                move_fraction = distance_to_travel / node_dist
                self.pos[0] += vec[0] * move_fraction
                self.pos[1] += vec[1] * move_fraction
                distance_to_travel = 0

        # For targeting logic
        if self.path_index < len(self.path) - 1:
            start_node = self.path[self.path_index]
            end_node = self.path[self.path_index + 1]
            total_dist = math.hypot(end_node[0] - start_node[0], end_node[1] - start_node[1])
            current_dist = math.hypot(self.pos[0] - start_node[0], self.pos[1] - start_node[1])
            self.progress_to_next = current_dist / total_dist if total_dist > 0 else 0
        else:
            self.progress_to_next = 1.0


        return self.path_index >= len(self.path) - 1

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_alive = False
            return True # Died
        return False # Survived

class Projectile:
    def __init__(self, tower_grid_pos, target, damage):
        self.start_pos = (320 + (tower_grid_pos[0] - tower_grid_pos[1]) * 20, 120 + (tower_grid_pos[0] + tower_grid_pos[1]) * 10)
        self.pos = list(self.start_pos)
        self.target = target
        self.damage = damage
        self.speed = 400  # pixels per second

    def update(self, dt):
        if not self.target.is_alive:
            return True, False # Reached target (sort of), no damage

        vec = (self.target.pos[0] - self.pos[0], self.target.pos[1] - self.pos[1])
        dist = math.hypot(vec[0], vec[1])
        
        if dist < self.speed * dt:
            return True, True # Reached target, deal damage
        
        self.pos[0] += (vec[0] / dist) * self.speed * dt
        self.pos[1] += (vec[1] / dist) * self.speed * dt
        return False, False # Still moving

class Particle:
    def __init__(self, pos, color, life, size_range, vel_range):
        self.pos = list(pos)
        self.color = color
        self.life = life
        self.max_life = life
        self.size = random.uniform(size_range[0], size_range[1])
        self.vel = [random.uniform(vel_range[0], vel_range[1]), random.uniform(vel_range[0], vel_range[1])]

    def update(self, dt):
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.life -= dt
        return self.life <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selection cursor. Press space to build a tower on the selected tile."
    )

    game_description = (
        "Isometric tower defense. Place towers on the grid to stop waves of enemies from reaching the end of the path."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.dt = 1/30.0

        # Visuals
        self.font_s = pygame.font.SysFont("Consolas", 16)
        self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_PATH = (40, 80, 120)
        self.COLOR_TOWER = (0, 255, 128)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_HEALTH_BG = (80, 0, 0)
        self.COLOR_HEALTH_FG = (0, 200, 0)

        # Game constants
        self.GRID_W, self.GRID_H = 10, 10
        self.TILE_W, self.TILE_H = 40, 20
        self.GRID_ORIGIN = (320, 120)
        self.MAX_STEPS = 2000
        self.MAX_WAVES = 5
        self.WAVE_DEFINITIONS = [
            {'count': 10, 'health': 100, 'speed': 30, 'interval': 1.0},
            {'count': 15, 'health': 110, 'speed': 33, 'interval': 0.8},
            {'count': 20, 'health': 120, 'speed': 36, 'interval': 0.6},
            {'count': 25, 'health': 135, 'speed': 40, 'interval': 0.5},
            {'count': 30, 'health': 150, 'speed': 45, 'interval': 0.4},
        ]

        self._define_path()
        self.reset()
        
        # self.validate_implementation() # Commented out for final submission

    def _define_path(self):
        self.path_grid_coords = [
            (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (4, 5), (4, 6),
            (5, 6), (6, 6), (6, 5), (6, 4), (6, 3), (6, 2),
            (7, 2), (8, 2), (9, 2)
        ]
        self.path_waypoints = [self._grid_to_iso(p) for p in self.path_grid_coords]
        self.valid_tower_tiles = set()
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                if (c, r) not in self.path_grid_coords:
                    self.valid_tower_tiles.add((c, r))

    def _grid_to_iso(self, grid_pos):
        x, y = grid_pos
        iso_x = self.GRID_ORIGIN[0] + (x - y) * (self.TILE_W / 2)
        iso_y = self.GRID_ORIGIN[1] + (x + y) * (self.TILE_H / 2)
        return (iso_x, iso_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.last_space_held = False
        self.last_shift_held = False # Not used but good practice

        self.wave_number = 0
        self.wave_spawning = False
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        self.inter_wave_timer = 3.0 # Initial delay

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01 # Small penalty for existing
        damaged_enemy_this_step = False

        # --- Handle player actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)

        if space_held and not self.last_space_held:
            pos = tuple(self.cursor_pos)
            is_occupied = any(t.grid_pos == pos for t in self.towers)
            if pos in self.valid_tower_tiles and not is_occupied:
                self.towers.append(Tower(pos))
                # sfx: Tower placement sound
                for _ in range(20):
                    self.particles.append(Particle(self._grid_to_iso(pos), self.COLOR_TOWER, 0.5, (1, 4), (-80, 80)))

        self.last_space_held = space_held
        
        # --- Update game state ---
        
        # Wave management
        if not self.wave_spawning and not self.enemies:
            self.inter_wave_timer -= self.dt
            if self.inter_wave_timer <= 0:
                if self.wave_number > 0:
                    reward += 5 # Survived a wave
                
                if self.wave_number >= self.MAX_WAVES:
                    self.win = True
                else:
                    self._start_wave(self.wave_number)
                    self.wave_number += 1

        if self.wave_spawning:
            self.spawn_timer -= self.dt
            if self.spawn_timer <= 0 and self.enemies_to_spawn > 0:
                wave_data = self.WAVE_DEFINITIONS[self.wave_number-1]
                self.enemies.append(Enemy(self.path_waypoints, wave_data['health'], wave_data['speed']))
                self.enemies_to_spawn -= 1
                self.spawn_timer = wave_data['interval']
            if self.enemies_to_spawn == 0:
                self.wave_spawning = False

        # Update towers
        for tower in self.towers:
            tower.update(self.dt, self.enemies, self.projectiles)
            
        # Update projectiles
        for p in self.projectiles[:]:
            hit, dealt_damage = p.update(self.dt)
            if hit:
                self.projectiles.remove(p)
                if dealt_damage and p.target.is_alive:
                    # sfx: Projectile hit sound
                    reward += 0.1
                    damaged_enemy_this_step = True
                    for _ in range(10):
                        self.particles.append(Particle(p.pos, self.COLOR_PROJECTILE, 0.3, (1, 3), (-50, 50)))
                    if p.target.take_damage(p.damage):
                        # sfx: Enemy death sound
                        reward += 1
                        self.score += 10
                        for _ in range(30):
                            self.particles.append(Particle(p.target.pos, self.COLOR_ENEMY, 0.6, (2, 5), (-100, 100)))

        # Update enemies
        for enemy in self.enemies[:]:
            if not enemy.is_alive:
                self.enemies.remove(enemy)
                continue
            if enemy.update(self.dt):
                self.game_over = True # Reached end
                reward -= 50
                # sfx: Player loses sound

        # Update particles
        for particle in self.particles[:]:
            if particle.update(self.dt):
                self.particles.remove(particle)
        
        self.steps += 1
        
        # --- Check termination conditions ---
        if self.win:
            self.game_over = True
            reward += 50
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _start_wave(self, wave_idx):
        if wave_idx < len(self.WAVE_DEFINITIONS):
            wave_data = self.WAVE_DEFINITIONS[wave_idx]
            self.enemies_to_spawn = wave_data['count']
            self.spawn_timer = 0
            self.wave_spawning = True
            self.inter_wave_timer = 5.0 # Reset for next inter-wave period

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid and path
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                pos = (c, r)
                iso_pos = self._grid_to_iso(pos)
                tile_points = [
                    (iso_pos[0], iso_pos[1] - self.TILE_H / 2),
                    (iso_pos[0] + self.TILE_W / 2, iso_pos[1]),
                    (iso_pos[0], iso_pos[1] + self.TILE_H / 2),
                    (iso_pos[0] - self.TILE_W / 2, iso_pos[1])
                ]
                color = self.COLOR_PATH if pos in self.path_grid_coords else self.COLOR_GRID
                pygame.gfxdraw.aapolygon(self.screen, tile_points, color)
                if pos in self.path_grid_coords:
                     pygame.gfxdraw.filled_polygon(self.screen, tile_points, color)


        # Render cursor
        cursor_iso_pos = self._grid_to_iso(self.cursor_pos)
        cursor_points = [
            (cursor_iso_pos[0], cursor_iso_pos[1] - self.TILE_H / 2),
            (cursor_iso_pos[0] + self.TILE_W / 2, cursor_iso_pos[1]),
            (cursor_iso_pos[0], cursor_iso_pos[1] + self.TILE_H / 2),
            (cursor_iso_pos[0] - self.TILE_W / 2, cursor_iso_pos[1])
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, cursor_points, 2)

        # Render towers
        for tower in self.towers:
            pos = self._grid_to_iso(tower.grid_pos)
            pygame.draw.circle(self.screen, self.COLOR_TOWER, (int(pos[0]), int(pos[1] - 5)), 8)
            pygame.draw.rect(self.screen, self.COLOR_TOWER, (int(pos[0]-4), int(pos[1]-5), 8, 10))

        # Render enemies
        for enemy in self.enemies:
            pos = (int(enemy.pos[0]), int(enemy.pos[1]))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 6)
            # Health bar
            health_pct = max(0, enemy.health / enemy.max_health)
            bar_len = 12
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (pos[0] - bar_len/2, pos[1] - 12, bar_len, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (pos[0] - bar_len/2, pos[1] - 12, bar_len * health_pct, 3))

        # Render projectiles
        for p in self.projectiles:
            pos = (int(p.pos[0]), int(p.pos[1]))
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, pos, 3)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE)

        # Render particles
        for particle in self.particles:
            alpha = int(255 * (particle.life / particle.max_life))
            color = (*particle.color, alpha)
            temp_surf = pygame.Surface((particle.size*2, particle.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(particle.size), int(particle.size)), int(particle.size))
            self.screen.blit(temp_surf, (int(particle.pos[0] - particle.size), int(particle.pos[1] - particle.size)))

    def _render_ui(self):
        score_text = self.font_s.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        wave_text_str = f"WAVE: {self.wave_number}/{self.MAX_WAVES}"
        if not self.enemies and not self.wave_spawning and not self.win:
             wave_text_str += f" (Next in {max(0, self.inter_wave_timer):.1f}s)"
        wave_text = self.font_s.render(wave_text_str, True, self.COLOR_UI)
        self.screen.blit(wave_text, (10, 30))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_TOWER if self.win else self.COLOR_ENEMY
            end_text = self.font_l.render(msg, True, color)
            text_rect = end_text.get_rect(center=(320, 200))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "enemies_left": len(self.enemies) + self.enemies_to_spawn,
            "towers_built": len(self.towers),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    env.reset()
    
    # To display the game, you need a different setup
    # This is for headless execution as per spec
    
    # Run a few random steps
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Episode finished.")
            env.reset()
            
    env.close()
    print("Environment test run completed.")

    # --- Interactive Play Example ---
    # To run this, you might need to install pygame if you haven't already.
    # `pip install pygame`
    print("\nStarting interactive play mode...")
    
    interactive_env = GameEnv()
    obs, info = interactive_env.reset()
    
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            # Simple delay after game over before reset
            pygame.time.wait(2000)
            obs, info = interactive_env.reset()
            terminated = False

        # --- Event Handling for Human Player ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0 # not used
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = interactive_env.step(action)

        # --- Render to Screen ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Match env's framerate

    interactive_env.close()