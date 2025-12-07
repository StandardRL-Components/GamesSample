
# Generated: 2025-08-28T07:10:17.330751
# Source Brief: brief_03159.md
# Brief Index: 3159

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import heapq
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to place a wall block."
    )

    game_description = (
        "Defend your fortress core from waves of enemies by strategically placing walls to redirect them."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE

        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_FORTRESS = (100, 100, 120)
        self.COLOR_FORTRESS_GLOW = (150, 150, 180)
        self.COLOR_BLOCK = (0, 200, 150)
        self.COLOR_BLOCK_GLOW = (100, 255, 220)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 150, 150)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_CURSOR_INVALID = (200, 0, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 50, 50)

        # Game Parameters
        self.MAX_STEPS = 3000 # Increased from 1000 to allow for 10 waves at 30fps
        self.MAX_WAVES = 10
        self.INITIAL_FORTRESS_HEALTH = 100
        
        # State variables are initialized in reset()
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fortress_health = self.INITIAL_FORTRESS_HEALTH
        self.wave_num = 0
        self.wave_cleared_this_step = False

        # Game Objects
        self.fortress_pos_grid = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.blocks = set()
        self.enemies = []
        self.particles = []
        
        # Player state
        self.cursor_pos_grid = [self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2]
        self.last_space_held = False
        
        # Wave management
        self.wave_spawn_timer = 90  # 3 seconds at 30fps before first wave

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(30)
        self.steps += 1
        reward = 0.0
        self.wave_cleared_this_step = False
        
        # 1. Handle Player Input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held)
        
        block_placed = self._handle_block_placement(space_held)
        if block_placed:
            reward -= 0.01
            # When a block is placed, all enemies must recalculate their paths
            for enemy in self.enemies:
                enemy['path'] = self._find_path(enemy['grid_pos'])

        # 2. Update Game Logic
        self._update_waves()
        self._update_enemies()
        self._update_particles()
        
        # 3. Calculate Rewards
        if self.wave_cleared_this_step:
            reward += 1.0

        # 4. Check Termination Conditions
        terminated = False
        if self.fortress_health <= 0:
            self.game_over = True
            terminated = True
            reward = -100.0
        elif self.wave_num > self.MAX_WAVES:
            self.game_over = True
            terminated = True
            reward = 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            # No terminal reward for timeout, score reflects performance
        
        # 5. Return 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 1: self.cursor_pos_grid[1] -= 1  # Up
        elif movement == 2: self.cursor_pos_grid[1] += 1  # Down
        elif movement == 3: self.cursor_pos_grid[0] -= 1  # Left
        elif movement == 4: self.cursor_pos_grid[0] += 1  # Right
        
        self.cursor_pos_grid[0] = np.clip(self.cursor_pos_grid[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos_grid[1] = np.clip(self.cursor_pos_grid[1], 0, self.GRID_HEIGHT - 1)

    def _is_valid_placement(self, grid_pos):
        if tuple(grid_pos) in self.blocks: return False
        if tuple(grid_pos) == self.fortress_pos_grid: return False
        # Prevent blocking spawn edges
        if grid_pos[0] == 0 or grid_pos[0] == self.GRID_WIDTH - 1: return False
        if grid_pos[1] == 0 or grid_pos[1] == self.GRID_HEIGHT - 1: return False
        return True

    def _handle_block_placement(self, space_held):
        placed = False
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            if self._is_valid_placement(self.cursor_pos_grid):
                self.blocks.add(tuple(self.cursor_pos_grid))
                self._spawn_particles(self.cursor_pos_grid[0] * self.GRID_SIZE + self.GRID_SIZE//2,
                                      self.cursor_pos_grid[1] * self.GRID_SIZE + self.GRID_SIZE//2,
                                      self.COLOR_BLOCK_GLOW, 20)
                placed = True
        self.last_space_held = space_held
        return placed

    def _update_waves(self):
        if self.wave_spawn_timer > 0:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer == 0:
                self.wave_num += 1
                self._spawn_wave()
        elif not self.enemies and self.wave_num <= self.MAX_WAVES:
            self.wave_cleared_this_step = True
            self.score += 10 * self.wave_num
            self.wave_spawn_timer = 150 # 5 seconds between waves

    def _spawn_wave(self):
        num_enemies = 2 + self.wave_num
        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0: # Top
                grid_pos = [self.np_random.integers(1, self.GRID_WIDTH - 1), 0]
            elif side == 1: # Bottom
                grid_pos = [self.np_random.integers(1, self.GRID_WIDTH - 1), self.GRID_HEIGHT - 1]
            elif side == 2: # Left
                grid_pos = [0, self.np_random.integers(1, self.GRID_HEIGHT - 1)]
            else: # Right
                grid_pos = [self.GRID_WIDTH - 1, self.np_random.integers(1, self.GRID_HEIGHT - 1)]

            enemy = {
                'pos': [grid_pos[0] * self.GRID_SIZE, grid_pos[1] * self.GRID_SIZE],
                'grid_pos': grid_pos,
                'path': self._find_path(grid_pos),
                'speed': 0.5 + self.wave_num * 0.05 + self.np_random.uniform(-0.1, 0.1)
            }
            self.enemies.append(enemy)

    def _update_enemies(self):
        fortress_center_px = (
            self.fortress_pos_grid[0] * self.GRID_SIZE + self.GRID_SIZE // 2,
            self.fortress_pos_grid[1] * self.GRID_SIZE + self.GRID_SIZE // 2
        )
        
        for enemy in self.enemies[:]:
            # Recalculate path periodically or if it's empty
            if not enemy['path'] or self.steps % 30 == 0:
                enemy['path'] = self._find_path(enemy['grid_pos'])

            if enemy['path']:
                target_grid_pos = enemy['path'][0]
                target_px = [
                    target_grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2,
                    target_grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2
                ]
                
                # Move towards target
                dx = target_px[0] - enemy['pos'][0]
                dy = target_px[1] - enemy['pos'][1]
                dist = math.hypot(dx, dy)
                
                if dist < enemy['speed']:
                    enemy['pos'] = [target_px[0], target_px[1]]
                    enemy['grid_pos'] = list(target_grid_pos)
                    enemy['path'].pop(0)
                else:
                    enemy['pos'][0] += (dx / dist) * enemy['speed']
                    enemy['pos'][1] += (dy / dist) * enemy['speed']
            
            # Check for collision with fortress
            dist_to_fortress = math.hypot(enemy['pos'][0] - fortress_center_px[0], enemy['pos'][1] - fortress_center_px[1])
            if dist_to_fortress < self.GRID_SIZE * 0.8:
                self.fortress_health -= 10
                self.score -= 50
                self._spawn_particles(enemy['pos'][0], enemy['pos'][1], self.COLOR_ENEMY_GLOW, 30)
                self.enemies.remove(enemy)
                # Sound: fortress_hit.wav

    def _find_path(self, start_grid_pos):
        # A* Pathfinding
        start_node = tuple(start_grid_pos)
        goal_node = self.fortress_pos_grid
        
        frontier = [(0, start_node)] # (priority, node)
        came_from = {start_node: None}
        cost_so_far = {start_node: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal_node:
                break

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.GRID_WIDTH and 0 <= neighbor[1] < self.GRID_HEIGHT:
                    if neighbor in self.blocks:
                        continue
                    
                    new_cost = cost_so_far[current] + 1
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + self._heuristic(neighbor, goal_node)
                        heapq.heappush(frontier, (priority, neighbor))
                        came_from[neighbor] = current
        else: # No path found
            return []

        # Reconstruct path
        path = []
        current = goal_node
        while current != start_node:
            path.append(current)
            if current not in came_from: return [] # Should not happen if path was found
            current = came_from[current]
        path.reverse()
        return path

    def _heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _spawn_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

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
            "wave": self.wave_num,
            "health": self.fortress_health,
            "enemies": len(self.enemies),
            "blocks": len(self.blocks)
        }

    def _render_game(self):
        # Grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Fortress
        fx, fy = self.fortress_pos_grid
        fortress_rect = pygame.Rect(fx * self.GRID_SIZE, fy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        pygame.gfxdraw.box(self.screen, fortress_rect, self.COLOR_FORTRESS)
        pygame.gfxdraw.filled_circle(self.screen, fortress_rect.centerx, fortress_rect.centery, self.GRID_SIZE, (*self.COLOR_FORTRESS_GLOW, 30))
        pygame.gfxdraw.aacircle(self.screen, fortress_rect.centerx, fortress_rect.centery, self.GRID_SIZE, (*self.COLOR_FORTRESS_GLOW, 50))

        # Blocks
        for bx, by in self.blocks:
            block_rect = pygame.Rect(bx * self.GRID_SIZE, by * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.gfxdraw.box(self.screen, block_rect, self.COLOR_BLOCK)
            pygame.gfxdraw.rectangle(self.screen, block_rect, self.COLOR_BLOCK_GLOW)
        
        # Enemies
        for enemy in self.enemies:
            ex, ey = int(enemy['pos'][0]), int(enemy['pos'][1])
            radius = self.GRID_SIZE // 3
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, ex, ey, radius, self.COLOR_ENEMY)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, radius + 3, (*self.COLOR_ENEMY_GLOW, 30))
            pygame.gfxdraw.aacircle(self.screen, ex, ey, radius + 3, (*self.COLOR_ENEMY_GLOW, 50))

        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius'] * (p['lifespan'] / 30)), p['color'])

        # Cursor
        cursor_rect = pygame.Rect(
            self.cursor_pos_grid[0] * self.GRID_SIZE, 
            self.cursor_pos_grid[1] * self.GRID_SIZE, 
            self.GRID_SIZE, 
            self.GRID_SIZE
        )
        is_valid = self._is_valid_placement(self.cursor_pos_grid)
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        pygame.draw.rect(self.screen, cursor_color, cursor_rect, 2)
        # Add a pulsating glow
        glow_alpha = 50 + 30 * math.sin(self.steps * 0.2)
        glow_surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        glow_surface.fill((*cursor_color, glow_alpha))
        self.screen.blit(glow_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.fortress_health / self.INITIAL_FORTRESS_HEALTH)
        bar_width = (self.SCREEN_WIDTH - 20)
        bar_height = 15
        bg_rect = pygame.Rect(10, 10, bar_width, bar_height)
        fill_rect = pygame.Rect(10, 10, int(bar_width * health_ratio), bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, bg_rect, 1)

        # Wave Text
        wave_text = f"WAVE: {self.wave_num if self.wave_num <= self.MAX_WAVES else 'CLEAR'}"
        text_surf = self.font_small.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (15, 30))

        # Score Text
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 15, 30))
        self.screen.blit(text_surf, text_rect)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.wave_num > self.MAX_WAVES:
                msg = "VICTORY"
            else:
                msg = "GAME OVER"
                
            text_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Simple human player loop
    running = True
    while running:
        # Default action is no-op
        action = [0, 0, 0] # movement=none, space=released, shift=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment handles its own rendering to the observation
        # For human playback, we need a separate display
        if 'display' not in locals():
            pygame.display.set_caption("Block Fortress Defense")
            display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        
        # Create a surface from the observation and blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over. Final Info: {info}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

    env.close()