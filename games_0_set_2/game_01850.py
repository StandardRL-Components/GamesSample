
# Generated: 2025-08-27T18:30:10.688105
# Source Brief: brief_01850.md
# Brief Index: 1850

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Collect all the gold keys to unlock the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down puzzle game. Navigate a procedurally generated maze, collect all the keys, and reach the exit before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Colors and Visuals ---
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (45, 55, 75)
    COLOR_PATH = (30, 35, 50)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PLAYER_OUTLINE = (255, 255, 255)
    COLOR_KEY = (255, 215, 0)
    COLOR_EXIT_LOCKED = (100, 0, 0)
    COLOR_EXIT_UNLOCKED = (0, 255, 120)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # --- Game Constants ---
    MAZE_W_CELLS = 15  # Playable area width (must be odd)
    MAZE_H_CELLS = 9   # Playable area height (must be odd)
    SCREEN_W, SCREEN_H = 640, 400
    MAX_STEPS = 1000
    INITIAL_MOVES = 50
    NUM_KEYS = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.Font(None, 64)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
        except pygame.error:
            # Fallback if default font is not found (e.g., in minimal containers)
            self.font_large = pygame.font.SysFont("sans-serif", 64)
            self.font_medium = pygame.font.SysFont("sans-serif", 32)
            self.font_small = pygame.font.SysFont("sans-serif", 24)

        # Calculate maze rendering dimensions
        self.cell_size = min(
            (self.SCREEN_W - 40) // self.MAZE_W_CELLS,
            (self.SCREEN_H - 80) // self.MAZE_H_CELLS
        )
        self.maze_pixel_w = self.MAZE_W_CELLS * self.cell_size
        self.maze_pixel_h = self.MAZE_H_CELLS * self.cell_size
        self.maze_offset_x = (self.SCREEN_W - self.maze_pixel_w) // 2
        self.maze_offset_y = (self.SCREEN_H - self.maze_pixel_h) // 2 + 20

        # These will be initialized in reset()
        self.maze = None
        self.player_pos = None
        self.keys_pos = None
        self.exit_pos = None
        self.moves_left = 0
        self.keys_collected = 0
        self.total_keys = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.animation_tick = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.INITIAL_MOVES
        self.keys_collected = 0
        self.total_keys = self.NUM_KEYS
        self.particles = []
        self.animation_tick = 0

        self._generate_maze()
        
        return self._get_observation(), self._get_info()
    
    def _generate_maze(self):
        # 0 = path, 1 = wall
        self.maze = np.ones((self.MAZE_H_CELLS, self.MAZE_W_CELLS), dtype=np.uint8)
        
        # Recursive backtracking
        def carve(x, y):
            self.maze[y, x] = 0
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            self.np_random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx * 2, y + dy * 2
                if 0 <= nx < self.MAZE_W_CELLS and 0 <= ny < self.MAZE_H_CELLS and self.maze[ny, nx] == 1:
                    self.maze[y + dy, x + dx] = 0
                    carve(nx, ny)

        start_x, start_y = self.np_random.integers(0, (self.MAZE_W_CELLS // 2)) * 2 + 1, \
                           self.np_random.integers(0, (self.MAZE_H_CELLS // 2)) * 2 + 1
        carve(start_x, start_y)
        
        # Get all valid spawn points
        path_cells = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(path_cells)
        
        # Place player, keys, and exit
        self.player_pos = tuple(path_cells.pop())
        self.keys_pos = [tuple(pos) for pos in path_cells[:self.NUM_KEYS]]
        # Ensure exit is far from player start
        self.exit_pos = tuple(path_cells[-1])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        moved = False
        if movement != 0: # 0 is no-op
            px, py = self.player_pos
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            nx, ny = px + dx, py + dy
            
            if 0 <= nx < self.MAZE_W_CELLS and 0 <= ny < self.MAZE_H_CELLS and self.maze[ny, nx] == 0:
                self.player_pos = (nx, ny)
                self.moves_left -= 1
                reward -= 0.1 # Cost of moving
                moved = True
        
        # Check for key collection
        if self.player_pos in self.keys_pos:
            self.keys_pos.remove(self.player_pos)
            self.keys_collected += 1
            reward += 5.0
            self.score += 5
            self._create_particles(self.player_pos, self.COLOR_KEY, 20)
            # SFX: Key collect sound

        # Check for win condition
        if self.player_pos == self.exit_pos and self.keys_collected == self.total_keys:
            self.game_over = True
            self.win = True
            reward += 50.0
            self.score += 50
            self._create_particles(self.player_pos, self.COLOR_EXIT_UNLOCKED, 50)
            # SFX: Level complete sound
        
        self.steps += 1
        
        # Check for termination conditions
        terminated = self.game_over or self.moves_left <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True # Lost due to moves/steps
            self.win = False

        self._update_animations()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_animations(self):
        self.animation_tick += 1
        # Update particles
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.05
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _create_particles(self, pos_grid, color, count):
        px, py = self._grid_to_pixel(pos_grid)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': 1.0,
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _grid_to_pixel(self, grid_pos):
        gx, gy = grid_pos
        px = self.maze_offset_x + gx * self.cell_size + self.cell_size // 2
        py = self.maze_offset_y + gy * self.cell_size + self.cell_size // 2
        return px, py

    def _render_game(self):
        # Draw maze background and walls
        for y in range(self.MAZE_H_CELLS):
            for x in range(self.MAZE_W_CELLS):
                rect = pygame.Rect(
                    self.maze_offset_x + x * self.cell_size,
                    self.maze_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

        # Draw exit
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        all_keys_collected = self.keys_collected == self.total_keys
        exit_color = self.COLOR_EXIT_UNLOCKED if all_keys_collected else self.COLOR_EXIT_LOCKED
        
        s = self.cell_size * 0.6
        if all_keys_collected:
            glow_size = s + 8 + abs(math.sin(self.animation_tick * 0.1)) * 4
            glow_color = (*exit_color, 60)
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf, (exit_px - glow_size, exit_py - glow_size), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, exit_color, (exit_px - s/2, exit_py - s/2, s, s), border_radius=3)

        # Draw keys
        key_size = self.cell_size * 0.5
        for k_pos in self.keys_pos:
            key_px, key_py = self._grid_to_pixel(k_pos)
            
            # Pulsing glow effect
            glow_size = key_size/2 + 4 + abs(math.sin(self.animation_tick * 0.05 + key_px)) * 2
            glow_color = (*self.COLOR_KEY, 50)
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf, (key_px - glow_size, key_py - glow_size), special_flags=pygame.BLEND_RGBA_ADD)

            # Key square
            pygame.draw.rect(self.screen, self.COLOR_KEY, (key_px - key_size/2, key_py - key_size/2, key_size, key_size), border_radius=2)
        
        # Draw particles
        for p in self.particles:
            size = int(p['life'] * self.cell_size * 0.2)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

        # Draw player
        player_px, player_py = self._grid_to_pixel(self.player_pos)
        player_radius = int(self.cell_size * 0.35)
        
        # Glow
        glow_size = player_radius + 6
        glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 30), (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (player_px - glow_size, player_py - glow_size), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, player_px, player_py, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_px, player_py, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_px, player_py, player_radius, self.COLOR_PLAYER_OUTLINE)


    def _render_text(self, text, font, y_pos, color, center_x=None):
        if center_x is None:
            center_x = self.SCREEN_W // 2
        
        text_surf = font.render(text, True, color)
        text_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect(center=(center_x, y_pos))
        shadow_rect = text_shadow.get_rect(center=(center_x + 2, y_pos + 2))

        self.screen.blit(text_shadow, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Moves left
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_medium.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (20, 15))

        # Keys collected
        keys_text = f"Keys: {self.keys_collected} / {self.total_keys}"
        keys_surf = self.font_medium.render(keys_text, True, self.COLOR_TEXT)
        self.screen.blit(keys_surf, (self.SCREEN_W - keys_surf.get_width() - 20, 15))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            if self.win:
                self._render_text("YOU WIN!", self.font_large, self.SCREEN_H // 2, self.COLOR_EXIT_UNLOCKED)
            else:
                self._render_text("GAME OVER", self.font_large, self.SCREEN_H // 2, self.COLOR_EXIT_LOCKED)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "keys_collected": self.keys_collected,
        }
        
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
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires pygame to be installed with display support
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Display the game screen
        try:
            # Attempt to create a display window
            if 'display' not in locals():
                display = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
                pygame.display.set_caption("Maze Runner")

            # Transpose back for pygame display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display.blit(surf, (0, 0))
            pygame.display.flip()
        except pygame.error as e:
            if "No available video device" in str(e):
                print("No video device available. Cannot display game. Running in headless mode.")
                if not running: break # Exit if we were already trying to quit
                running = False # Stop the loop if display fails
            else:
                raise e

        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()