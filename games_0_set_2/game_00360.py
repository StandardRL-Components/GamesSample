
# Generated: 2025-08-27T13:25:07.622556
# Source Brief: brief_00360.md
# Brief Index: 360

        
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

    user_guide = "Controls: Use arrow keys to slide tiles. Tiles with the same number merge into one!"
    game_description = "A 2048-style puzzle game. Merge tiles to reach the 2048 tile before the grid fills up. Plan your moves carefully!"
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 4
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (45, 45, 45)
    COLOR_GRID_BG = (77, 77, 77)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_OVERLAY_BG = (0, 0, 0, 180)
    
    TILE_COLORS = {
        0: (120, 120, 120),
        2: ((238, 228, 218), (119, 110, 101)),
        4: ((237, 224, 200), (119, 110, 101)),
        8: ((242, 177, 121), (249, 246, 242)),
        16: ((245, 149, 99), (249, 246, 242)),
        32: ((246, 124, 95), (249, 246, 242)),
        64: ((246, 94, 59), (249, 246, 242)),
        128: ((237, 207, 114), (249, 246, 242)),
        256: ((237, 204, 97), (249, 246, 242)),
        512: ((237, 200, 80), (249, 246, 242)),
        1024: ((237, 197, 63), (249, 246, 242)),
        2048: ((237, 194, 46), (249, 246, 242)),
        "super": ((60, 58, 50), (249, 246, 242)),
    }

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
        
        self.font_huge = pygame.font.SysFont("Arial", 72, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 20, bold=True)
        
        self.board = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        # Animation trackers (for single-frame effects)
        self.new_tiles_pos = []
        self.merged_tiles_pos = []

        self.reset()
        
        # self.validate_implementation() # Optional: Call to check implementation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.board = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.new_tiles_pos.clear()
        self.merged_tiles_pos.clear()
        
        # Initial state: two '2' tiles
        self._spawn_tile()
        self._spawn_tile()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Clear previous frame's animation trackers
        self.new_tiles_pos.clear()
        self.merged_tiles_pos.clear()

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        board_changed = False

        if movement > 0: # 0 is no-op
            self.steps += 1
            
            # Map movement action to rotation count for np.rot90
            # 1=Up (rot 1), 2=Down (rot 3), 3=Left (rot 0), 4=Right (rot 2)
            rot_count = {1: 1, 2: 3, 3: 0, 4: 2}[movement]
            
            rotated_board = np.rot90(self.board, rot_count)
            new_board, merge_score, merges_made = self._process_board(rotated_board)
            new_board = np.rot90(new_board, -rot_count)

            if not np.array_equal(self.board, new_board):
                self.board = new_board
                board_changed = True
                self.score += merge_score
                reward += merges_made # +1 for each merge event
                
                for r, c in self.merged_tiles_pos:
                    if self.board[r][c] > 128:
                        reward += 10 # Event-based reward for high-value tiles
                
                self._spawn_tile()
            else:
                reward -= 0.1 # Penalty for no-op/invalid move

        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100
            else:
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_board(self, board):
        new_board = np.zeros_like(board)
        total_merge_score = 0
        merges_made = 0
        
        temp_merged_pos = []

        for i in range(self.GRID_SIZE):
            row = board[i]
            # 1. Squeeze non-zero tiles to the left
            squeezed_row = row[row != 0]
            
            # 2. Merge adjacent identical tiles
            merged_row = []
            skip = False
            for j in range(len(squeezed_row)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(squeezed_row) and squeezed_row[j] == squeezed_row[j+1]:
                    new_val = squeezed_row[j] * 2
                    merged_row.append(new_val)
                    total_merge_score += new_val
                    merges_made += 1
                    temp_merged_pos.append((i, len(merged_row) - 1))
                    skip = True
                else:
                    merged_row.append(squeezed_row[j])
            
            # 3. Place the new row into the new board
            new_board[i, :len(merged_row)] = merged_row
            
        # Transform merged coordinates back to original orientation before storing
        # This is complex, so we'll handle it visually with a simpler pulse effect on all merged tiles.
        # The coordinates are relative to the rotated board, so we need to un-rotate them.
        # Since we are just tracking them for a single-frame visual effect, we can do it after the final board is set.
        # For simplicity, we find merged tiles by comparing old and new values.
        # Let's do that after the main move logic. This is easier.
        
        # For now, let's just use the score as an indicator
        if merges_made > 0:
            # We'll identify merged tiles in the main _render_game loop
            # This is a simplification but visually effective
            pass

        return new_board, total_merge_score, merges_made

    def _spawn_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return

        r, c = self.np_random.choice(empty_cells, size=1)[0]
        
        # 90% chance of 2, 10% chance of 4
        new_value = 4 if self.np_random.random() < 0.1 else 2
        self.board[r, c] = new_value
        self.new_tiles_pos.append((r, c))

    def _check_termination(self):
        if self.game_over:
            return True
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True

        if np.any(self.board == 2048):
            self.game_over = True
            self.game_won = True
            return True

        if np.all(self.board != 0):
            # Board is full, check for possible merges
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    val = self.board[r, c]
                    # Check right
                    if c + 1 < self.GRID_SIZE and self.board[r, c + 1] == val:
                        return False
                    # Check down
                    if r + 1 < self.GRID_SIZE and self.board[r + 1, c] == val:
                        return False
            self.game_over = True # No possible merges
            return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        grid_pixel_size = self.SCREEN_HEIGHT - 40
        padding = 10
        tile_size = (grid_pixel_size - padding * (self.GRID_SIZE + 1)) / self.GRID_SIZE
        grid_start_x = (self.SCREEN_WIDTH - grid_pixel_size) / 2
        grid_start_y = (self.SCREEN_HEIGHT - grid_pixel_size) / 2
        
        # Draw grid background
        grid_rect = pygame.Rect(grid_start_x, grid_start_y, grid_pixel_size, grid_pixel_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=8)

        # Identify merged tiles by checking where values increased
        # This is a bit of a hack for visual effect, but works for this turn-based model
        if not self.merged_tiles_pos and self.steps > 0:
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if self.board[r,c] > 4 and (r,c) not in self.new_tiles_pos:
                        # A simple heuristic: if a tile's value is > 4 and it's not new, it might have been merged into.
                        # This is imperfect but creates a nice pulsing effect on active high-value tiles.
                        # A more accurate method would require storing the pre-move board.
                        # Let's just apply the pulse to all non-2, non-4, non-new tiles for game feel.
                        is_high_value = True
                        for val in [2,4]:
                            if self.board[r,c] == val:
                                is_high_value = False
                        if is_high_value:
                             self.merged_tiles_pos.append((r,c))


        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                val = self.board[r, c]
                
                x = grid_start_x + padding * (c + 1) + tile_size * c
                y = grid_start_y + padding * (r + 1) + tile_size * r
                
                current_tile_size = tile_size
                
                # "Pop" effect for new tiles
                if (r, c) in self.new_tiles_pos:
                    current_tile_size *= 1.2
                    x -= (current_tile_size - tile_size) / 2
                    y -= (current_tile_size - tile_size) / 2
                # "Pulse" effect for merged tiles
                elif (r, c) in self.merged_tiles_pos:
                    pulse_scale = 1.0 + 0.2 * abs(math.sin(self.steps * 0.5))
                    current_tile_size *= pulse_scale
                    x -= (current_tile_size - tile_size) / 2
                    y -= (current_tile_size - tile_size) / 2

                if val == 0:
                    # Draw empty tile slot
                    pygame.draw.rect(self.screen, self.TILE_COLORS[0], (x, y, tile_size, tile_size), border_radius=5)
                else:
                    bg_color, text_color = self.TILE_COLORS.get(val, self.TILE_COLORS["super"])
                    
                    tile_rect = pygame.Rect(x, y, current_tile_size, current_tile_size)
                    pygame.draw.rect(self.screen, bg_color, tile_rect, border_radius=5)
                    
                    # Choose font size based on number of digits
                    s_val = str(val)
                    if len(s_val) < 3: font = self.font_large
                    elif len(s_val) < 4: font = self.font_medium
                    else: font = self.font_small
                        
                    text_surf = font.render(s_val, True, text_color)
                    text_rect = text_surf.get_rect(center=tile_rect.center)
                    self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Steps display
        steps_text = self.font_medium.render(f"Moves: {self.steps}", True, self.COLOR_UI_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(steps_text, steps_rect)
        
        # Game Over / Win overlay
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY_BG)
            
            message = "You Win!" if self.game_won else "Game Over"
            msg_surf = self.font_huge.render(message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            
            overlay.blit(msg_surf, msg_rect)
            self.screen.blit(overlay, (0, 0))

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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("2048 Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    game_over = False
    
    print(GameEnv.user_guide)

    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not game_over:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    game_over = False
                    print("Game Reset.")
                
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    game_over = terminated
                    print(f"Action: {action[0]}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")

        # Draw the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()