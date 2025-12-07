
# Generated: 2025-08-28T06:08:39.817964
# Source Brief: brief_02825.md
# Brief Index: 2825

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Press space to hard drop."
    )

    game_description = (
        "Fast-paced, top-down puzzle game where players strategically place falling tetromino-like blocks to clear lines and achieve a high score."
    )

    auto_advance = False

    # Game constants
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BLOCK_SIZE = 20
    MAX_STEPS = 1000
    WIN_CONDITION_LINES = 100

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (45, 45, 60)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_WHITE = (255, 255, 255)
    
    # Tetromino shapes and colors
    TETROMINOES = {
        'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
        'O': [[[1, 1], [1, 1]]],
        'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
        'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
        'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]],
        'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
        'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]]
    }
    
    TETROMINO_COLORS = {
        'I': (0, 240, 240),   # Cyan
        'O': (240, 240, 0),   # Yellow
        'T': (160, 0, 240),   # Purple
        'J': (0, 0, 240),     # Blue
        'L': (240, 160, 0),   # Orange
        'S': (0, 240, 0),     # Green
        'Z': (240, 0, 0)      # Red
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.playfield_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.playfield_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2
        
        # State variables are initialized in reset()
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.piece_bag = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.line_clear_animation = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.line_clear_animation = None
        
        self._fill_piece_bag()
        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()

        if not self._is_valid_position(self.current_piece['x'], self.current_piece['y'], self.current_piece['shape'], self.current_piece['rotation']):
            self.game_over = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle line clear animation state
        if self.line_clear_animation and self.line_clear_animation['timer'] > 0:
            self.line_clear_animation['timer'] -= 1
            if self.line_clear_animation['timer'] == 0:
                self._execute_line_clear()

        # Process player input
        if not self.line_clear_animation or self.line_clear_animation['timer'] == 0:
            self._handle_input(movement)

        # Process game logic
        if space_held:
            reward += self._hard_drop()
        else:
            reward += self._normal_step()

        self.steps += 1
        
        # Check termination conditions
        terminated = self.game_over
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward += 100
            terminated = True
        if self.steps >= self.MAX_STEPS:
            terminated = True
        if self.game_over:
            reward -= 50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        # movement: 0=none, 1=up (rot_cw), 2=down (rot_ccw), 3=left, 4=right
        px, py, prot = self.current_piece['x'], self.current_piece['y'], self.current_piece['rotation']
        pshape = self.current_piece['shape']

        if movement == 3:  # Left
            if self._is_valid_position(px - 1, py, pshape, prot):
                self.current_piece['x'] -= 1
        elif movement == 4:  # Right
            if self._is_valid_position(px + 1, py, pshape, prot):
                self.current_piece['x'] += 1
        elif movement == 1:  # Rotate Clockwise
            new_rot = (prot + 1) % len(pshape)
            if self._is_valid_position(px, py, pshape, new_rot):
                self.current_piece['rotation'] = new_rot
        elif movement == 2:  # Rotate Counter-Clockwise
            new_rot = (prot - 1 + len(pshape)) % len(pshape)
            if self._is_valid_position(px, py, pshape, new_rot):
                self.current_piece['rotation'] = new_rot

    def _hard_drop(self):
        rows_dropped = 0
        while self._is_valid_position(self.current_piece['x'], self.current_piece['y'] + 1, self.current_piece['shape'], self.current_piece['rotation']):
            self.current_piece['y'] += 1
            rows_dropped += 1
        
        lock_reward = self._lock_piece()
        return rows_dropped * 0.1 + lock_reward

    def _normal_step(self):
        if self._is_valid_position(self.current_piece['x'], self.current_piece['y'] + 1, self.current_piece['shape'], self.current_piece['rotation']):
            self.current_piece['y'] += 1
            return 0.1  # Reward for falling one row
        else:
            return self._lock_piece()

    def _lock_piece(self):
        shape = self.current_piece['shape'][self.current_piece['rotation']]
        color_index = list(self.TETROMINOES.keys()).index(self.current_piece['type']) + 1
        
        initial_holes = self._count_holes()

        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_y, grid_x = self.current_piece['y'] + r, self.current_piece['x'] + c
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y][grid_x] = color_index
        
        final_holes = self._count_holes()
        hole_penalty = max(0, final_holes - initial_holes) * 0.1

        clear_reward = self._check_and_initiate_line_clear()

        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()

        if not self._is_valid_position(self.current_piece['x'], self.current_piece['y'], self.current_piece['shape'], self.current_piece['rotation']):
            self.game_over = True
            
        return clear_reward - hole_penalty

    def _check_and_initiate_line_clear(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r] > 0):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            self.line_clear_animation = {'lines': lines_to_clear, 'timer': 2} # 2 steps for flash
            # Reward is based on number of lines
            num_cleared = len(lines_to_clear)
            if num_cleared == 1: return 10
            if num_cleared == 2: return 30
            if num_cleared == 3: return 60
            if num_cleared >= 4: return 100
        return 0

    def _execute_line_clear(self):
        lines_to_clear = self.line_clear_animation['lines']
        for r in sorted(lines_to_clear, reverse=True):
            self.grid = np.delete(self.grid, r, axis=0)
            new_row = np.zeros((1, self.GRID_WIDTH), dtype=int)
            self.grid = np.vstack([new_row, self.grid])
        self.lines_cleared += len(lines_to_clear)
        self.score += [0, 100, 300, 500, 800][len(lines_to_clear)] * (self.lines_cleared // 10 + 1)
        self.line_clear_animation = None
        # sound effect placeholder: # sfx_line_clear.play()

    def _fill_piece_bag(self):
        self.piece_bag = list(self.TETROMINOES.keys())
        self.np_random.shuffle(self.piece_bag)

    def _new_piece(self):
        if not self.piece_bag:
            self._fill_piece_bag()
        
        piece_type = self.piece_bag.pop()
        shape = self.TETROMINOES[piece_type]
        
        return {
            'type': piece_type,
            'shape': shape,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - len(shape[0][0]) // 2,
            'y': 0
        }

    def _is_valid_position(self, x, y, shape, rotation):
        piece_matrix = shape[rotation]
        for r, row in enumerate(piece_matrix):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = x + c, y + r
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False  # Out of bounds
                    if self.grid[grid_y][grid_x] > 0:
                        return False  # Collision with another piece
        return True

    def _count_holes(self):
        holes = 0
        for c in range(self.GRID_WIDTH):
            found_block = False
            for r in range(self.GRID_HEIGHT):
                if self.grid[r][c] > 0:
                    found_block = True
                elif found_block and self.grid[r][c] == 0:
                    holes += 1
        return holes

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
            "lines_cleared": self.lines_cleared,
        }

    def _render_game(self):
        # Draw playfield border and grid
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.playfield_x - 5, self.playfield_y - 5, self.GRID_WIDTH * self.BLOCK_SIZE + 10, self.GRID_HEIGHT * self.BLOCK_SIZE + 10), 5, border_radius=3)
        for x in range(1, self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.playfield_x + x * self.BLOCK_SIZE, self.playfield_y), (self.playfield_x + x * self.BLOCK_SIZE, self.playfield_y + self.GRID_HEIGHT * self.BLOCK_SIZE))
        for y in range(1, self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.playfield_x, self.playfield_y + y * self.BLOCK_SIZE), (self.playfield_x + self.GRID_WIDTH * self.BLOCK_SIZE, self.playfield_y + y * self.BLOCK_SIZE))

        # Draw locked blocks
        color_list = list(self.TETROMINO_COLORS.values())
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] > 0:
                    color_index = int(self.grid[r][c] - 1)
                    self._draw_block(c, r, color_list[color_index])

        # Draw line clear animation
        if self.line_clear_animation and self.line_clear_animation['timer'] > 0:
            for r in self.line_clear_animation['lines']:
                rect = pygame.Rect(self.playfield_x, self.playfield_y + r * self.BLOCK_SIZE, self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_WHITE, rect)

        # Draw current piece (if not in animation pause)
        if not self.game_over and self.current_piece and (not self.line_clear_animation or self.line_clear_animation['timer'] == 0):
            color = self.TETROMINO_COLORS[self.current_piece['type']]
            
            # Draw ghost piece
            ghost_y = self.current_piece['y']
            while self._is_valid_position(self.current_piece['x'], ghost_y + 1, self.current_piece['shape'], self.current_piece['rotation']):
                ghost_y += 1
            self._draw_piece(self.current_piece, ghost_y, color, ghost=True)

            # Draw active piece
            self._draw_piece(self.current_piece, self.current_piece['y'], color, ghost=False)

    def _draw_piece(self, piece, y_offset, color, ghost=False):
        shape = piece['shape'][piece['rotation']]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(piece['x'] + c, y_offset + r, color, ghost)

    def _draw_block(self, grid_x, grid_y, color, ghost=False):
        screen_x = self.playfield_x + grid_x * self.BLOCK_SIZE
        screen_y = self.playfield_y + grid_y * self.BLOCK_SIZE
        rect = pygame.Rect(screen_x, screen_y, self.BLOCK_SIZE, self.BLOCK_SIZE)

        if ghost:
            # Draw a semi-transparent block
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, 60), s.get_rect(), border_radius=3)
            pygame.draw.rect(s, (*self.COLOR_WHITE, 80), s.get_rect(), 1, border_radius=3)
            self.screen.blit(s, (screen_x, screen_y))
        else:
            # Draw a block with a 3D effect
            light_color = tuple(min(255, c + 50) for c in color)
            dark_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, dark_color, rect, border_radius=3)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=2)
            # Highlight
            pygame.draw.line(self.screen, light_color, (screen_x + 2, screen_y + 2), (screen_x + self.BLOCK_SIZE - 3, screen_y + 2), 1)
            pygame.draw.line(self.screen, light_color, (screen_x + 2, screen_y + 2), (screen_x + 2, screen_y + self.BLOCK_SIZE - 3), 1)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_large.render(f"{self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (30, 30))
        self.screen.blit(score_val, (30, 60))
        
        # Lines display
        lines_text = self.font_large.render(f"LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_large.render(f"{self.lines_cleared}", True, self.COLOR_WHITE)
        self.screen.blit(lines_text, (30, 110))
        self.screen.blit(lines_val, (30, 140))

        # Next piece preview
        next_text = self.font_large.render(f"NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 150, 30))
        if self.next_piece:
            color = self.TETROMINO_COLORS[self.next_piece['type']]
            shape = self.next_piece['shape'][0]
            
            # Center the piece in the preview box
            piece_w = len(shape[0]) * self.BLOCK_SIZE
            piece_h = len(shape) * self.BLOCK_SIZE
            start_x = self.SCREEN_WIDTH - 150 + (120 - piece_w) // 2
            start_y = 80 + (80 - piece_h) // 2
            
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        rect = pygame.Rect(start_x + c * self.BLOCK_SIZE, start_y + r * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                        light_color = tuple(min(255, c + 50) for c in color)
                        dark_color = tuple(max(0, c - 50) for c in color)
                        pygame.draw.rect(self.screen, dark_color, rect, border_radius=3)
                        pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=2)
                        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "GAME OVER"
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                end_text = "YOU WIN!"
            
            text_surf = self.font_large.render(end_text, True, self.COLOR_WHITE)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Blitz")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated and running:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'r' to reset or escape to quit
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.key == pygame.K_ESCAPE:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        terminated = False
                        wait_for_reset = False

    env.close()