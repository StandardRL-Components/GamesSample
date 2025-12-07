
# Generated: 2025-08-28T04:59:25.437765
# Source Brief: brief_05430.md
# Brief Index: 5430

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place the selected block. "
        "Hold Shift to cycle between block types (Obstacle, Turret)."
    )

    game_description = (
        "Defend your isometric fortress from waves of enemies by strategically placing defensive blocks. "
        "Survive 10 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 14
        self.MAX_STEPS = 2500
        self.MAX_WAVES = 10
        self.FORTRESS_HEALTH_MAX = 100
        self.WAVE_DELAY = 150 # steps between waves

        # Visual constants
        self.TILE_WIDTH_ISO = 32
        self.TILE_HEIGHT_ISO = 16
        self.TILE_DEPTH = 12
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_FORTRESS = (60, 180, 75)
        self.COLOR_FORTRESS_SIDE = (50, 150, 65)
        self.COLOR_ENEMY = (230, 25, 75)
        self.COLOR_OBSTACLE = (0, 130, 200)
        self.COLOR_OBSTACLE_SIDE = (0, 110, 170)
        self.COLOR_TURRET = (245, 130, 48)
        self.COLOR_TURRET_SIDE = (225, 110, 28)
        self.COLOR_PROJECTILE = (255, 225, 25)
        self.COLOR_CURSOR_VALID = (255, 255, 255, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (70, 240, 240)
        self.COLOR_HEALTH_BAR_BG = (128, 0, 0)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.np_random = None
        self.reset()

        # Validate implementation after initialization
        self.validate_implementation()

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH_ISO / 2
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT_ISO / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, grid_x, grid_y, color_top, color_side, depth):
        x, y = self._iso_to_screen(grid_x, grid_y)
        w, h = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO
        
        points_top = [
            (x, y),
            (x + w / 2, y + h / 2),
            (x, y + h),
            (x - w / 2, y + h / 2)
        ]
        points_left = [
            (x - w / 2, y + h / 2),
            (x, y + h),
            (x, y + h + depth),
            (x - w / 2, y + h / 2 + depth)
        ]
        points_right = [
            (x + w / 2, y + h / 2),
            (x, y + h),
            (x, y + h + depth),
            (x + w / 2, y + h / 2 + depth)
        ]

        pygame.gfxdraw.filled_polygon(surface, points_left, color_side)
        pygame.gfxdraw.aapolygon(surface, points_left, color_side)
        pygame.gfxdraw.filled_polygon(surface, points_right, color_side)
        pygame.gfxdraw.aapolygon(surface, points_right, color_side)
        pygame.gfxdraw.filled_polygon(surface, points_top, color_top)
        pygame.gfxdraw.aapolygon(surface, points_top, color_top)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.fortress_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.fortress_health = self.FORTRESS_HEALTH_MAX
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.grid[self.fortress_pos] = 1 # 1: Fortress

        self.enemies = []
        self.blocks = []
        self.projectiles = []
        self.particles = []

        self.wave_number = 0
        self.wave_in_progress = False
        self.time_until_next_wave = self.WAVE_DELAY // 2

        self.cursor_pos = [self.fortress_pos[0], self.fortress_pos[1] - 3]
        self.selected_block_type = 0
        self.block_types = ["Obstacle", "Turret"]
        self.block_costs = {"Obstacle": 0, "Turret": 0} # No cost for simplicity
        self.block_configs = {
            "Turret": {"range": 4, "cooldown": 45, "damage": 1}
        }

        self.last_shift_state = 0
        self.last_space_state = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage efficiency
        self.steps += 1

        # 1. Handle player input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held, shift_held)

        # 2. Update game state
        enemy_killed_reward = self._update_enemies()
        projectile_hit_reward = self._update_turrets_and_projectiles()
        self._update_particles()
        reward += enemy_killed_reward + projectile_hit_reward

        # 3. Wave Management
        wave_cleared_reward = self._manage_waves()
        reward += wave_cleared_reward

        # 4. Check for termination
        terminated = False
        if self.fortress_health <= 0:
            self.game_over = True
            self.win = False
            reward = -100
        elif self.wave_number > self.MAX_WAVES and not self.wave_in_progress:
            self.game_over = True
            self.win = True
            reward = 50
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            reward = -50 # Penalty for timeout
        
        if self.game_over:
            terminated = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Cycle block type on button press (not hold)
        if shift_held and not self.last_shift_state:
            self.selected_block_type = (self.selected_block_type + 1) % len(self.block_types)
        self.last_shift_state = shift_held

        # Place block on button press
        if space_held and not self.last_space_state:
            cx, cy = self.cursor_pos
            if self.grid[cx, cy] == 0: # Can only place on empty tiles
                block_type = self.block_types[self.selected_block_type]
                new_block = {'pos': (cx, cy), 'type': block_type, 'health': 100}
                if block_type == "Turret":
                    new_block['cooldown_timer'] = 0
                self.blocks.append(new_block)
                self.grid[cx, cy] = 2 # 2: Block
        self.last_space_state = space_held

    def _manage_waves(self):
        reward = 0
        if not self.wave_in_progress:
            self.time_until_next_wave -= 1
            if self.time_until_next_wave <= 0 and self.wave_number <= self.MAX_WAVES:
                self._start_next_wave()
        elif len(self.enemies) == 0:
            self.wave_in_progress = False
            self.time_until_next_wave = self.WAVE_DELAY
            if self.wave_number > 0: # Don't reward for clearing wave 0
                reward += 5
        return reward

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return

        self.wave_in_progress = True
        num_enemies = 3 + (self.wave_number - 1)
        enemy_speed = 0.02 + (self.wave_number - 1) * 0.005
        enemy_health = 1 + self.wave_number // 3

        spawn_points = [(0, 0), (self.GRID_WIDTH - 1, 0), (0, self.GRID_HEIGHT - 1), (self.GRID_WIDTH - 1, self.GRID_HEIGHT - 1)]
        for i in range(num_enemies):
            start_pos = random.choice(spawn_points)
            sx, sy = self._iso_to_screen(start_pos[0], start_pos[1])
            self.enemies.append({
                'grid_pos': list(start_pos),
                'screen_pos': [sx, sy],
                'health': enemy_health,
                'speed': enemy_speed,
                'path_timer': 0
            })

    def _update_enemies(self):
        reward = 0
        fx, fy = self.fortress_pos
        for enemy in self.enemies[:]:
            enemy['path_timer'] -= 1
            if enemy['path_timer'] <= 0:
                enemy['path_timer'] = 15 # Recalculate path every 15 frames
                ex, ey = int(round(enemy['grid_pos'][0])), int(round(enemy['grid_pos'][1]))
                
                # Simple greedy pathfinding
                neighbors = [(ex + 1, ey), (ex - 1, ey), (ex, ey + 1), (ex, ey - 1)]
                best_move = None
                min_dist = float('inf')

                for nx, ny in neighbors:
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] != 2:
                        dist = abs(nx - fx) + abs(ny - fy)
                        if dist < min_dist:
                            min_dist = dist
                            best_move = (nx, ny)
                
                if best_move:
                    enemy['target_grid_pos'] = best_move
                else:
                    enemy['target_grid_pos'] = (ex, ey) # Stay put if blocked
            
            # Interpolate screen position for smooth movement
            target_screen_pos = self._iso_to_screen(enemy['target_grid_pos'][0], enemy['target_grid_pos'][1])
            direction = [target_screen_pos[0] - enemy['screen_pos'][0], target_screen_pos[1] - enemy['screen_pos'][1]]
            dist = math.hypot(*direction)
            if dist > 1:
                direction = [d / dist for d in direction]
                enemy['screen_pos'][0] += direction[0] * enemy['speed'] * self.TILE_WIDTH_ISO
                enemy['screen_pos'][1] += direction[1] * enemy['speed'] * self.TILE_HEIGHT_ISO
                
                # Update grid pos based on which tile center we are closest to
                enemy['grid_pos'][0] += direction[0] * enemy['speed']
                enemy['grid_pos'][1] += direction[1] * enemy['speed']

            # Check for reaching fortress
            if abs(enemy['grid_pos'][0] - fx) < 0.5 and abs(enemy['grid_pos'][1] - fy) < 0.5:
                self.fortress_health -= 10
                self._create_particles(self._iso_to_screen(fx, fy), 20, self.COLOR_ENEMY)
                self.enemies.remove(enemy)
                continue

            if enemy['health'] <= 0:
                reward += 1
                self._create_particles(enemy['screen_pos'], 15, self.COLOR_ENEMY)
                self.enemies.remove(enemy)

        return reward

    def _update_turrets_and_projectiles(self):
        reward = 0
        turret_config = self.block_configs["Turret"]
        
        # Turrets find targets and fire
        for block in self.blocks:
            if block['type'] == 'Turret':
                block['cooldown_timer'] -= 1
                if block['cooldown_timer'] <= 0:
                    target = None
                    min_dist = turret_config['range']
                    for enemy in self.enemies:
                        dist = math.hypot(enemy['grid_pos'][0] - block['pos'][0], enemy['grid_pos'][1] - block['pos'][1])
                        if dist < min_dist:
                            min_dist = dist
                            target = enemy
                    
                    if target:
                        block['cooldown_timer'] = turret_config['cooldown']
                        start_pos = self._iso_to_screen(block['pos'][0], block['pos'][1])
                        self.projectiles.append({
                            'pos': list(start_pos),
                            'target_enemy': target,
                            'speed': 5
                        })
                        self._create_particles(start_pos, 3, self.COLOR_PROJECTILE, 0.5)

        # Update projectiles
        for proj in self.projectiles[:]:
            if proj['target_enemy'] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj['target_enemy']['screen_pos']
            direction = [target_pos[0] - proj['pos'][0], target_pos[1] - proj['pos'][1]]
            dist = math.hypot(*direction)
            if dist < proj['speed']:
                # Hit
                proj['target_enemy']['health'] -= turret_config['damage']
                reward += 0.1
                self._create_particles(proj['pos'], 5, self.COLOR_PROJECTILE)
                self.projectiles.remove(proj)
            else:
                # Move
                direction = [d / dist for d in direction]
                proj['pos'][0] += direction[0] * proj['speed']
                proj['pos'][1] += direction[1] * proj['speed']
        
        return reward

    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': random.randint(10, 20), 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # Render blocks
        for block in self.blocks:
            color, side_color = (self.COLOR_TURRET, self.COLOR_TURRET_SIDE) if block['type'] == 'Turret' else (self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_SIDE)
            self._draw_iso_cube(self.screen, block['pos'][0], block['pos'][1], color, side_color, self.TILE_DEPTH)

        # Render fortress
        self._draw_iso_cube(self.screen, self.fortress_pos[0], self.fortress_pos[1], self.COLOR_FORTRESS, self.COLOR_FORTRESS_SIDE, self.TILE_DEPTH)
        
        # Render cursor
        cx, cy = self.cursor_pos
        is_valid = self.grid[cx, cy] == 0
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        x, y = self._iso_to_screen(cx, cy)
        w, h = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO
        points = [(x, y), (x + w / 2, y + h / 2), (x, y + h), (x - w / 2, y + h / 2)]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Render enemies
        for enemy in self.enemies:
            x, y = enemy['screen_pos']
            pygame.gfxdraw.filled_trigon(self.screen, int(x), int(y - 8), int(x - 6), int(y + 4), int(x + 6), int(y + 4), self.COLOR_ENEMY)
            pygame.gfxdraw.aatrigon(self.screen, int(x), int(y - 8), int(x - 6), int(y + 4), int(x + 6), int(y + 4), self.COLOR_ENEMY)

        # Render projectiles
        for proj in self.projectiles:
            x, y = proj['pos']
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(x), int(y)), 3)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _render_ui(self):
        # Health Bar
        fx, fy = self._iso_to_screen(self.fortress_pos[0], self.fortress_pos[1])
        health_pct = self.fortress_health / self.FORTRESS_HEALTH_MAX
        bar_width = 40
        bar_height = 5
        bar_x = fx - bar_width // 2
        bar_y = fy - self.TILE_HEIGHT_ISO // 2 - 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_pct, bar_height))

        # Text info
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 35))

        block_text = self.font_small.render(f"BLOCK: {self.block_types[self.selected_block_type]}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (self.WIDTH - block_text.get_width() - 10, 10))

        if self.game_over:
            status_text = "VICTORY!" if self.win else "GAME OVER"
            status_render = self.font_large.render(status_text, True, self.COLOR_HEALTH_BAR if self.win else self.COLOR_ENEMY)
            text_rect = status_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(status_render, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "fortress_health": self.fortress_health,
            "enemies_remaining": len(self.enemies),
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Isometric Fortress Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    done = False
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        env.clock.tick(30) # Limit to 30 FPS

    env.close()