
# Generated: 2025-08-27T15:17:05.900892
# Source Brief: brief_00944.md
# Brief Index: 944

        
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
        "Controls: Use arrow keys (↑↓←→) to navigate the maze. Collect gems to score points. Grab the red star for a score multiplier!"
    )

    game_description = (
        "A fast-paced arcade maze game. Grab as many gems as you can before the 60-second timer runs out. Use the bonus star wisely for a high score!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TILE_SIZE = 20
        self.GRID_W, self.GRID_H = self.WIDTH // self.TILE_SIZE, self.HEIGHT // self.TILE_SIZE
        self.MAZE_W, self.MAZE_H = 15, 9 # Must be odd
        
        self.MAX_TIME = 60 * self.FPS
        self.WIN_SCORE = 50
        self.NUM_GEMS = 10
        self.NUM_BONUS = 1
        
        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_WALL = (40, 40, 80)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_BONUS = (255, 50, 50)
        self.GEM_COLORS = [
            (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.time_remaining = 0
        self.player_pos = [0, 0]
        self.maze = []
        self.gems = []
        self.bonus_items = []
        self.bonus_active = False
        self.particles = []
        self.grid_offset_x = 0
        self.grid_offset_y = 0

        self.reset()
        
        # Run validation at the end of init
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.time_remaining = self.MAX_TIME
        self.bonus_active = False
        self.particles.clear()

        # --- Maze and World Generation ---
        self.maze = self._generate_maze(self.MAZE_W, self.MAZE_H)
        self.grid_offset_x = (self.GRID_W - (self.MAZE_W * 2 + 1)) // 2
        self.grid_offset_y = (self.GRID_H - (self.MAZE_H * 2 + 1)) // 2

        open_tiles = self._get_open_tiles()
        self.np_random.shuffle(open_tiles)

        # Place player
        self.player_pos = list(open_tiles.pop())

        # Place gems and bonus items
        self.gems = [tuple(open_tiles.pop()) for _ in range(self.NUM_GEMS)]
        self.bonus_items = [tuple(open_tiles.pop()) for _ in range(self.NUM_BONUS)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        self.time_remaining -= 1
        
        self._update_particles()

        if not self.game_over:
            # --- Player Movement ---
            dist_before = self._find_nearest_gem_dist()

            px, py = self.player_pos
            nx, ny = px, py

            if movement == 1: ny -= 1  # Up
            elif movement == 2: ny += 1  # Down
            elif movement == 3: nx -= 1  # Left
            elif movement == 4: nx += 1  # Right
            
            if not self._is_wall(nx, ny):
                self.player_pos = [nx, ny]

            # --- Movement Reward ---
            dist_after = self._find_nearest_gem_dist()
            if dist_after < dist_before:
                reward += 0.1 # Moved closer
            elif dist_after > dist_before:
                reward -= 0.01 # Moved further

            # --- Item Collection ---
            player_pos_tuple = tuple(self.player_pos)
            
            if player_pos_tuple in self.gems:
                gem_reward = 5 if self.bonus_active else 1
                reward += gem_reward
                self.score += gem_reward
                self.bonus_active = False
                
                self.gems.remove(player_pos_tuple)
                self._create_particles(player_pos_tuple, self.GEM_COLORS[self.gems.__len__() % len(self.GEM_COLORS)])
                # Add new gem
                open_tiles = self._get_open_tiles()
                if open_tiles:
                    new_gem_pos = self.np_random.choice(open_tiles, axis=0)
                    self.gems.append(tuple(new_gem_pos))
                # Sound: gem_collect.wav

            if player_pos_tuple in self.bonus_items:
                self.bonus_active = True
                self.bonus_items.remove(player_pos_tuple)
                self._create_particles(player_pos_tuple, self.COLOR_BONUS)
                # Sound: bonus_collect.wav

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.win_condition_met:
                reward += 100 # Win bonus
                # Sound: win.wav
            else:
                reward -= 100 # Loss penalty
                # Sound: lose.wav

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Rendering Methods ---

    def _render_game(self):
        # Draw maze
        for y in range(self.MAZE_H * 2 + 1):
            for x in range(self.MAZE_W * 2 + 1):
                if self.maze[y][x] == 1:
                    px, py = (x + self.grid_offset_x) * self.TILE_SIZE, (y + self.grid_offset_y) * self.TILE_SIZE
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (px, py, self.TILE_SIZE, self.TILE_SIZE))

        # Draw gems
        flash_alpha = 180 + 75 * math.sin(self.steps * 0.2)
        for gx, gy in self.gems:
            px, py = gx * self.TILE_SIZE, gy * self.TILE_SIZE
            gem_color = self.GEM_COLORS[self.gems.index((gx, gy)) % len(self.GEM_COLORS)]
            pygame.draw.rect(self.screen, gem_color, (px + 4, py + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8))
            
            # Glow effect
            s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s, (*gem_color, flash_alpha), (self.TILE_SIZE//2, self.TILE_SIZE//2), self.TILE_SIZE//2 - 2)
            self.screen.blit(s, (px, py), special_flags=pygame.BLEND_RGBA_ADD)


        # Draw bonus item
        pulse = 1 + 0.2 * math.sin(self.steps * 0.15)
        for bx, by in self.bonus_items:
            px, py = bx * self.TILE_SIZE, by * self.TILE_SIZE
            center_x, center_y = px + self.TILE_SIZE // 2, py + self.TILE_SIZE // 2
            
            points = []
            for i in range(5):
                angle = math.radians(90 + i * 72)
                outer_radius = self.TILE_SIZE // 2 * pulse
                inner_radius = outer_radius / 2.5
                points.append((center_x + outer_radius * math.cos(angle), center_y - outer_radius * math.sin(angle)))
                angle += math.radians(36)
                points.append((center_x + inner_radius * math.cos(angle), center_y - inner_radius * math.sin(angle)))
            
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BONUS)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BONUS)

        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            p_color = (p['color'][0], p['color'][1], p['color'][2], int(255 * life_ratio))
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, p_color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['pos'][0] - p['size'], p['pos'][1] - p['size']), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw player
        px, py = self.player_pos[0] * self.TILE_SIZE, self.player_pos[1] * self.TILE_SIZE
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px + 2, py + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4))
        
        # Player glow
        s = pygame.Surface((self.TILE_SIZE*2, self.TILE_SIZE*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER, 100), (self.TILE_SIZE, self.TILE_SIZE), self.TILE_SIZE - 4)
        self.screen.blit(s, (px - self.TILE_SIZE//2, py - self.TILE_SIZE//2), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_str = f"TIME: {self.time_remaining // self.FPS:02d}"
        time_color = (255, 100, 100) if self.time_remaining < 10 * self.FPS else self.COLOR_TEXT
        time_text = self.font_medium.render(time_str, True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Bonus Active
        if self.bonus_active:
            bonus_text = self.font_medium.render("5x", True, self.COLOR_BONUS)
            self.screen.blit(bonus_text, (self.WIDTH // 2 - bonus_text.get_width()//2, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_condition_met else "TIME'S UP!"
            color = (100, 255, 100) if self.win_condition_met else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    # --- Helper Methods ---

    def _generate_maze(self, width, height):
        # Maze grid is (2*width+1) x (2*height+1)
        w, h = width * 2 + 1, height * 2 + 1
        maze = [[1] * w for _ in range(h)]
        
        # Carve path using recursive backtracking
        stack = [(self.np_random.integers(width), self.np_random.integers(height))]
        maze[stack[0][1]*2+1][stack[0][0]*2+1] = 0
        
        visited = {stack[0]}

        while stack:
            cx, cy = stack[-1]
            
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                nx, ny, dx, dy = self.np_random.choice(neighbors, axis=0)
                nx, ny = int(nx), int(ny) # np.choice returns np.int64
                
                maze[cy*2+1+dy][cx*2+1+dx] = 0
                maze[ny*2+1][nx*2+1] = 0
                
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        # Break some walls to create loops
        for _ in range(width * height // 4):
            x = self.np_random.integers(1, w - 1)
            y = self.np_random.integers(1, h - 1)
            if maze[y][x] == 1 and ((maze[y-1][x] == 0 and maze[y+1][x] == 0) or (maze[y][x-1] == 0 and maze[y][x+1] == 0)):
                maze[y][x] = 0
        
        return maze

    def _is_wall(self, grid_x, grid_y):
        if not (0 <= grid_x < self.GRID_W and 0 <= grid_y < self.GRID_H):
            return True # Out of bounds
        
        maze_x = grid_x - self.grid_offset_x
        maze_y = grid_y - self.grid_offset_y

        if not (0 <= maze_x < self.MAZE_W*2+1 and 0 <= maze_y < self.MAZE_H*2+1):
            return True # Outside maze area
            
        return self.maze[maze_y][maze_x] == 1

    def _get_open_tiles(self):
        occupied = {tuple(self.player_pos)} | set(self.gems) | set(self.bonus_items)
        open_tiles = []
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if not self._is_wall(x, y) and (x, y) not in occupied:
                    open_tiles.append([x, y])
        return open_tiles

    def _find_nearest_gem_dist(self):
        if not self.gems:
            return 0
        px, py = self.player_pos
        min_dist = float('inf')
        for gx, gy in self.gems:
            dist = abs(px - gx) + abs(py - gy) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _create_particles(self, pos, color):
        px, py = pos[0] * self.TILE_SIZE + self.TILE_SIZE//2, pos[1] * self.TILE_SIZE + self.TILE_SIZE//2
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.win_condition_met = True
            return True
        if self.time_remaining <= 0:
            return True
        return False

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for display
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Maze")
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            print("--- Game Reset ---")

        env.clock.tick(env.FPS)

    env.close()
    pygame.quit()