
# Generated: 2025-08-27T17:57:54.545959
# Source Brief: brief_01688.md
# Brief Index: 1688

        
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
        "Controls: Use arrow keys (Up, Down, Left, Right) to slide all tiles in that direction. "
        "Tiles with the same number will merge."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Slide and merge numbered tiles on a 10x10 grid. "
        "Your goal is to create the elusive 2048 tile. Plan your moves carefully to avoid filling up the board."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 1000

        # Visuals
        self.COLOR_BG = (187, 173, 160)
        self.COLOR_GRID_BG = (205, 193, 180)
        self.TILE_COLORS = {
            0: self.COLOR_GRID_BG,
            2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
            16: (245, 149, 99), 32: (246, 124, 95), 64: (246, 94, 59),
            128: (237, 207, 114), 256: (237, 204, 97), 512: (237, 200, 80),
            1024: (237, 197, 63), 2048: (237, 194, 46),
        }
        self.TEXT_COLORS = {
            2: (119, 110, 101), 4: (119, 110, 101),
        }
        self.TEXT_COLOR_LIGHT = (249, 246, 242)
        self.SPAWN_GLOW_COLOR = (255, 255, 255)
        self.MERGE_GLOW_COLOR = (255, 255, 0)
        
        # Sizing
        self.GRID_PIXEL_SIZE = 374
        self.CELL_SIZE = 33
        self.GAP_SIZE = 4
        self.GRID_X = (self.WIDTH - self.GRID_PIXEL_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_PIXEL_SIZE) // 2
        self.TILE_RADIUS = 4
        
        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.FONT_LARGE = pygame.font.SysFont("Arial", 22, bold=True)
            self.FONT_MEDIUM = pygame.font.SysFont("Arial", 18, bold=True)
            self.FONT_SMALL = pygame.font.SysFont("Arial", 14, bold=True)
            self.FONT_UI = pygame.font.SysFont("Arial", 24, bold=True)
            self.FONT_MSG = pygame.font.SysFont("Arial", 48, bold=True)
        except pygame.error:
            self.FONT_LARGE = pygame.font.Font(None, 30)
            self.FONT_MEDIUM = pygame.font.Font(None, 24)
            self.FONT_SMALL = pygame.font.Font(None, 18)
            self.FONT_UI = pygame.font.Font(None, 32)
            self.FONT_MSG = pygame.font.Font(None, 60)

        # --- Game State ---
        self.board = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_move_effects = []
        
        # Initialize state variables
        self.reset()

        # Self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.board = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_move_effects = []
        
        self._spawn_tile()
        self._spawn_tile()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.last_move_effects = []
        
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        moved = False

        if movement in [1, 2, 3, 4]:
            board_before = self.board.copy()
            reward = self._move(movement)
            moved = not np.array_equal(board_before, self.board)

            if moved:
                self.score += reward
                self._spawn_tile()
                if not self.win and 2048 in self.board:
                    self.win = True
                    reward += 100 
            
        self.steps += 1
        terminated = self._check_termination(moved)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _move(self, direction):
        # 1=up, 2=down, 3=left, 4=right
        total_reward = 0
        
        # Use rotation to simplify logic to a single "left" slide
        rotations = {1: 1, 2: 3, 3: 0, 4: 2} # num of 90-degree counter-clockwise rotations
        num_rotations = rotations[direction]
        
        rotated_board = np.rot90(self.board, num_rotations)
        new_board = np.zeros_like(rotated_board)
        
        for i, row in enumerate(rotated_board):
            new_row, reward, merged_indices = self._process_line(row)
            new_board[i] = new_row
            total_reward += reward

            # Store merge effects with correct original coordinates
            for merged_idx in merged_indices:
                val = new_row[merged_idx]
                pos = self._unrotate_pos((i, merged_idx), num_rotations)
                self.last_move_effects.append({'type': 'merge', 'pos': pos, 'val': val})

        self.board = np.rot90(new_board, 4 - num_rotations)
        return total_reward

    def _process_line(self, line):
        tiles = line[line != 0]
        new_line = np.zeros_like(line)
        reward = 0
        merged_indices = []
        
        write_idx = 0
        read_idx = 0
        while read_idx < len(tiles):
            if read_idx + 1 < len(tiles) and tiles[read_idx] == tiles[read_idx + 1]:
                merged_val = tiles[read_idx] * 2
                new_line[write_idx] = merged_val
                reward += merged_val
                merged_indices.append(write_idx)
                read_idx += 2
            else:
                new_line[write_idx] = tiles[read_idx]
                read_idx += 1
            write_idx += 1
            
        return new_line, reward, merged_indices

    def _unrotate_pos(self, pos, num_rotations):
        r, c = pos
        s = self.GRID_SIZE - 1
        for _ in range(num_rotations):
            r, c = c, s - r
        return (r, c)

    def _spawn_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return

        r, c = random.choice(empty_cells)
        val = 4 if random.random() < 0.1 else 2
        self.board[r, c] = val
        self.last_move_effects.append({'type': 'spawn', 'pos': (r, c), 'val': val})

    def _check_termination(self, moved):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True

        if moved: # If we could move, we are not stuck
            return False

        # If we couldn't move, check if any moves are possible at all
        if np.all(self.board != 0): # Board is full
            # Check for possible merges
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    val = self.board[r, c]
                    # Check right
                    if c + 1 < self.GRID_SIZE and self.board[r, c+1] == val:
                        return False
                    # Check down
                    if r + 1 < self.GRID_SIZE and self.board[r+1, c] == val:
                        return False
            
            # Board is full and no merges are possible
            self.game_over = True
            return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.GRID_Y, self.GRID_PIXEL_SIZE, self.GRID_PIXEL_SIZE), border_radius=self.TILE_RADIUS)

        effects_map = {fx['pos']: fx for fx in self.last_move_effects}

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                val = self.board[r, c]
                
                tile_x = self.GRID_X + self.GAP_SIZE + c * (self.CELL_SIZE + self.GAP_SIZE)
                tile_y = self.GRID_Y + self.GAP_SIZE + r * (self.CELL_SIZE + self.GAP_SIZE)
                
                # Get base color
                color = self.TILE_COLORS.get(val, self.TILE_COLORS[2048])
                
                # Draw base tile
                pygame.draw.rect(self.screen, color, (tile_x, tile_y, self.CELL_SIZE, self.CELL_SIZE), border_radius=self.TILE_RADIUS)

                # Apply visual effects for spawn/merge
                effect = effects_map.get((r, c))
                if effect:
                    glow_color = self.SPAWN_GLOW_COLOR if effect['type'] == 'spawn' else self.MERGE_GLOW_COLOR
                    # Draw a slightly larger rect underneath as a glow
                    glow_size = self.CELL_SIZE + 4
                    glow_x = tile_x - 2
                    glow_y = tile_y - 2
                    pygame.draw.rect(self.screen, glow_color, (glow_x, glow_y, glow_size, glow_size), border_radius=self.TILE_RADIUS+2)
                    # Redraw the main tile on top
                    pygame.draw.rect(self.screen, color, (tile_x, tile_y, self.CELL_SIZE, self.CELL_SIZE), border_radius=self.TILE_RADIUS)


                if val != 0:
                    text_color = self.TEXT_COLORS.get(val, self.TEXT_COLOR_LIGHT)
                    
                    if val >= 1000:
                        font = self.FONT_SMALL
                    elif val >= 100:
                        font = self.FONT_MEDIUM
                    else:
                        font = self.FONT_LARGE

                    text_surface = font.render(str(val), True, text_color)
                    text_rect = text_surface.get_rect(center=(tile_x + self.CELL_SIZE / 2, tile_y + self.CELL_SIZE / 2))
                    self.screen.blit(text_surface, text_rect)

    def _render_ui(self):
        # Score display
        score_text = self.FONT_UI.render(f"SCORE: {self.score}", True, self.TEXT_COLOR_LIGHT)
        self.screen.blit(score_text, (self.GRID_X, 5))
        
        # Steps display
        steps_text = self.FONT_UI.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.TEXT_COLOR_LIGHT)
        steps_rect = steps_text.get_rect(right=self.GRID_X + self.GRID_PIXEL_SIZE, top=5)
        self.screen.blit(steps_text, steps_rect)

        # Game Over / Win Message
        if self.game_over or (self.win and not self.game_over):
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((238, 228, 218, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win and not self.game_over:
                msg_text = "YOU WIN!"
                msg_color = self.MERGE_GLOW_COLOR
            else:
                msg_text = "GAME OVER"
                msg_color = self.TEXT_COLORS[2]

            msg_surface = self.FONT_MSG.render(msg_text, True, msg_color)
            msg_rect = msg_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surface, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
            "max_tile": int(np.max(self.board)),
        }

    def close(self):
        pygame.font.quit()
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

if __name__ == "__main__":
    # Example of how to use the environment for human play
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame window for human play ---
    pygame.display.set_caption("2048 - 10x10")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op
    
    print(env.user_guide)

    while not done:
        # --- Human Controls ---
        human_action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                    human_action_taken = True
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                    human_action_taken = True
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                    human_action_taken = True
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                    human_action_taken = True
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    print(f"Game Reset. Initial Info: {info}")
                    action[0] = 0
                    continue

        if done:
            break

        if human_action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action[0]}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
            action[0] = 0 # Reset action to no-op after processing

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    # Show final screen for a moment
    if env.game_over:
        pygame.time.wait(2000)

    env.close()
    print("Game Over.")