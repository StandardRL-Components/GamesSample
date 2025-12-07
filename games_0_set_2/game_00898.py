
# Generated: 2025-08-27T15:08:01.526598
# Source Brief: brief_00898.md
# Brief Index: 898

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a tile. Shift to flag a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic mine-sweeping puzzle game. Reveal all safe tiles while avoiding the hidden mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = (5, 5)
    NUM_MINES = 10
    MAX_STEPS = 1000

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID_LINE = (50, 50, 60)
    COLOR_TILE_HIDDEN = (70, 80, 90)
    COLOR_TILE_REVEALED = (110, 120, 130)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_MINE = (220, 50, 50)
    COLOR_FLAG = (255, 100, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_UI_TEXT = (200, 200, 220)
    NUMBER_COLORS = {
        1: (50, 150, 255),
        2: (50, 200, 50),
        3: (255, 50, 50),
        4: (150, 50, 255),
        5: (255, 150, 50),
        6: (50, 200, 200),
        7: (200, 200, 50),
        8: (200, 50, 150),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_tile = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Game state variables (initialized in reset)
        self.grid = None
        self.revealed_grid = None
        self.flagged_grid = None
        self.cursor_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.safe_tiles_revealed = None
        self.total_safe_tiles = None
        self.particles = []

        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.particles = []

        self._generate_grid()
        self.revealed_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.flagged_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.safe_tiles_revealed = 0
        self.total_safe_tiles = self.GRID_SIZE[0] * self.GRID_SIZE[1] - self.NUM_MINES

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = np.zeros(self.GRID_SIZE, dtype=int)
        
        # Place mines
        mine_indices = self.np_random.choice(self.GRID_SIZE[0] * self.GRID_SIZE[1], self.NUM_MINES, replace=False)
        mine_coords = [(i % self.GRID_SIZE[0], i // self.GRID_SIZE[0]) for i in mine_indices]
        for x, y in mine_coords:
            self.grid[x, y] = -1 # -1 represents a mine

        # Calculate numbers
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                if self.grid[x, y] == -1:
                    continue
                
                mine_count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1]:
                            if self.grid[nx, ny] == -1:
                                mine_count += 1
                self.grid[x, y] = mine_count

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Actions ---
        # 1. Cursor Movement
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_SIZE[1] - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_SIZE[0] - 1, self.cursor_pos[0] + 1)

        # 2. Flagging (Shift)
        if shift_pressed:
            x, y = self.cursor_pos
            if not self.revealed_grid[x, y]:
                self.flagged_grid[x, y] = not self.flagged_grid[x, y]
                # No reward for flagging

        # 3. Revealing (Space)
        if space_pressed:
            x, y = self.cursor_pos
            if self.flagged_grid[x, y]:
                reward -= 0.5 # Penalty for trying to reveal a flagged tile
            elif self.revealed_grid[x, y]:
                reward -= 0.1 # Penalty for revealing an already open tile
            else:
                # Reveal the tile
                if self.grid[x, y] == -1: # Hit a mine
                    self.game_over = True
                    self.win = False
                    reward = -100
                    # sound: explosion
                    self._create_explosion(x, y)
                else: # Hit a safe tile
                    revealed_count = self._reveal_tile(x, y)
                    self.safe_tiles_revealed += revealed_count
                    reward += revealed_count # +1 for each newly revealed tile
                    
                    # Check for win condition
                    if self.safe_tiles_revealed == self.total_safe_tiles:
                        self.game_over = True
                        self.win = True
                        reward = 100
                        # sound: win_jingle

        self.steps += 1
        self.score += reward
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reveal_tile(self, x, y):
        """Recursively reveals tiles, starting from (x, y). Returns number of tiles revealed."""
        tiles_to_reveal = [(x, y)]
        revealed_count = 0
        
        while tiles_to_reveal:
            cx, cy = tiles_to_reveal.pop()
            
            if not (0 <= cx < self.GRID_SIZE[0] and 0 <= cy < self.GRID_SIZE[1]):
                continue
            if self.revealed_grid[cx, cy] or self.flagged_grid[cx, cy]:
                continue
            
            self.revealed_grid[cx, cy] = True
            revealed_count += 1
            # sound: click
            
            # If it's a 0-tile, reveal its neighbors
            if self.grid[cx, cy] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        tiles_to_reveal.append((cx + dx, cy + dy))
        return revealed_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Calculate grid dimensions and offsets
        grid_pixel_width = self.SCREEN_WIDTH * 0.8
        grid_pixel_height = self.SCREEN_HEIGHT * 0.8
        self.tile_size = min(grid_pixel_width / self.GRID_SIZE[0], grid_pixel_height / self.GRID_SIZE[1])
        offset_x = (self.SCREEN_WIDTH - self.tile_size * self.GRID_SIZE[0]) / 2
        offset_y = (self.SCREEN_HEIGHT - self.tile_size * self.GRID_SIZE[1]) / 2 + 30

        # Draw grid and tiles
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                rect = pygame.Rect(
                    offset_x + x * self.tile_size,
                    offset_y + y * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                
                is_revealed = self.revealed_grid[x, y] or (self.game_over and self.grid[x, y] == -1)
                
                if is_revealed:
                    pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect)
                    value = self.grid[x, y]
                    if value == -1: # Mine
                        cx, cy = int(rect.centerx), int(rect.centery)
                        rad = int(self.tile_size * 0.3)
                        pygame.gfxdraw.filled_circle(self.screen, cx, cy, rad, self.COLOR_MINE)
                        pygame.gfxdraw.aacircle(self.screen, cx, cy, rad, self.COLOR_MINE)
                    elif value > 0: # Number
                        num_text = self.font_tile.render(str(value), True, self.NUMBER_COLORS.get(value, self.COLOR_TEXT))
                        text_rect = num_text.get_rect(center=rect.center)
                        self.screen.blit(num_text, text_rect)
                else: # Hidden
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect)
                    if self.flagged_grid[x, y]:
                        # Draw a flag
                        p1 = (rect.centerx, rect.top + self.tile_size * 0.2)
                        p2 = (rect.centerx, rect.bottom - self.tile_size * 0.2)
                        p3 = (rect.left + self.tile_size * 0.2, rect.centery - self.tile_size * 0.1)
                        pygame.draw.line(self.screen, self.COLOR_FLAG, p1, p2, 3)
                        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [p1, (rect.right - self.tile_size * 0.2, rect.centery - self.tile_size*0.2), p3])

                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            offset_x + self.cursor_pos[0] * self.tile_size,
            offset_y + self.cursor_pos[1] * self.tile_size,
            self.tile_size,
            self.tile_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)
        
        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Mines display
        flags_placed = np.sum(self.flagged_grid)
        mines_text = self.font_ui.render(f"Mines: {self.NUM_MINES - flags_placed}", True, self.COLOR_UI_TEXT)
        mines_rect = mines_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(mines_text, mines_rect)

        # Game over message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else self.COLOR_MINE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _create_explosion(self, grid_x, grid_y):
        offset_x = (self.SCREEN_WIDTH - self.tile_size * self.GRID_SIZE[0]) / 2
        offset_y = (self.SCREEN_HEIGHT - self.tile_size * self.GRID_SIZE[1]) / 2 + 30
        center_x = offset_x + (grid_x + 0.5) * self.tile_size
        center_y = offset_y + (grid_y + 0.5) * self.tile_size

        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.integers(2, 6)
            lifetime = self.np_random.integers(20, 40)
            color = random.choice([self.COLOR_MINE, (255, 150, 0), (100, 100, 100)])
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'size': size,
                'lifetime': lifetime,
                'color': color
            })

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], int(p['size']))
                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "win": self.win,
            "safe_tiles_revealed": self.safe_tiles_revealed,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test game-specific assertions
        self.reset()
        assert np.sum(self.grid == -1) == self.NUM_MINES
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be executed when the environment is imported by Gymnasium
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Minesweeper Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print(" Minesweeper Gym Environment")
    print("="*30)
    print(env.user_guide)
    print("Press R to reset, Q to quit.")
    print("="*30 + "\n")

    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                if not terminated:
                    if event.key == pygame.K_UP:
                        movement = 1
                    elif event.key == pygame.K_DOWN:
                        movement = 2
                    elif event.key == pygame.K_LEFT:
                        movement = 3
                    elif event.key == pygame.K_RIGHT:
                        movement = 4
                    elif event.key == pygame.K_SPACE:
                        space = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        shift = 1
        
        if not terminated:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}")
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Win: {info['win']}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    env.close()