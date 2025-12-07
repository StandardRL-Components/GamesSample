
# Generated: 2025-08-27T16:05:31.364186
# Source Brief: brief_01118.md
# Brief Index: 1118

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the block, ↑↓ to rotate. Press space to drop the block instantly."
    )

    game_description = (
        "A fast-paced puzzle game where you place falling blocks to complete rows and columns. "
        "Complete 5 lines to win, but don't let the grid fill up!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    CELL_SIZE = 28
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2 + 20

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_WHITE = (255, 255, 255)
    COLOR_CLEAR_ANIM = (255, 255, 255)

    # Piece shapes and colors
    PIECE_SHAPES = {
        'T': [[(1, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (2, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (0, 1), (1, 1), (1, 2)]],
        'I': [[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 0), (2, 1), (2, 2), (2, 3)], [(0, 2), (1, 2), (2, 2), (3, 2)], [(1, 0), (1, 1), (1, 2), (1, 3)]],
        'L': [[(0, 1), (1, 1), (2, 1), (2, 0)], [(1, 0), (1, 1), (1, 2), (2, 2)], [(0, 1), (1, 1), (2, 1), (0, 2)], [(0, 0), (1, 0), (1, 1), (1, 2)]],
        'J': [[(0, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (2, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (2, 2)], [(1, 0), (1, 1), (0, 2), (1, 2)]],
        'S': [[(1, 0), (2, 0), (0, 1), (1, 1)], [(1, 0), (1, 1), (2, 1), (2, 2)], [(1, 1), (2, 1), (0, 2), (1, 2)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
        'Z': [[(0, 0), (1, 0), (1, 1), (2, 1)], [(2, 0), (1, 1), (2, 1), (1, 2)], [(0, 1), (1, 1), (1, 2), (2, 2)], [(1, 0), (0, 1), (1, 1), (0, 2)]],
        'O': [[(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)]]
    }
    PIECE_COLORS = {
        'T': (160, 0, 255), 'I': (0, 255, 255), 'L': (255, 160, 0), 'J': (0, 0, 255),
        'S': (0, 255, 0), 'Z': (255, 0, 0), 'O': (255, 255, 0)
    }
    PIECE_TYPES = list(PIECE_SHAPES.keys())

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.current_piece = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.completed_lines = 0
        self.fall_timer = 0.0
        self.fall_speed = 1.0  # seconds per drop
        self.prev_space_held = False
        self.clearing_animation = None # (type, index, timer)
        self.lines_to_clear = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.completed_lines = 0
        self.fall_speed = 1.0
        self.fall_timer = self.fall_speed
        self.prev_space_held = False
        self.clearing_animation = None
        self.lines_to_clear = []
        
        self._spawn_piece()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        if self.clearing_animation:
            self.clearing_animation['timer'] -= 1 / 30.0
            if self.clearing_animation['timer'] <= 0:
                self._perform_clear()
                self.clearing_animation = None
        elif not self.game_over:
            # --- Handle Input ---
            moved = False
            if movement == 3:  # Left
                if self._is_valid_position(self.current_piece, offset_x=-1):
                    self.current_piece['x'] -= 1
                    moved = True
            elif movement == 4:  # Right
                if self._is_valid_position(self.current_piece, offset_x=1):
                    self.current_piece['x'] += 1
                    moved = True
            elif movement == 1: # Up -> Rotate CW
                if self._is_valid_position(self.current_piece, rotation=1):
                    self.current_piece['rotation'] = (self.current_piece['rotation'] + 1) % 4
                    moved = True
            elif movement == 2: # Down -> Rotate CCW
                if self._is_valid_position(self.current_piece, rotation=-1):
                    self.current_piece['rotation'] = (self.current_piece['rotation'] - 1 + 4) % 4
                    moved = True

            # Hard drop on space press (not hold)
            if space_held and not self.prev_space_held:
                # sound: piece_drop
                while self._is_valid_position(self.current_piece, offset_y=1):
                    self.current_piece['y'] += 1
                self.fall_timer = 0 # Force placement
            
            self.prev_space_held = space_held

            # --- Update Game Logic ---
            self.fall_timer -= 1 / 30.0
            if self.fall_timer <= 0:
                if self._is_valid_position(self.current_piece, offset_y=1):
                    self.current_piece['y'] += 1
                    self.fall_timer = self.fall_speed
                else:
                    # Place piece and get placement rewards
                    reward += self._place_piece()
                    
                    # Check for line clears
                    clear_reward, lines_cleared = self._check_lines()
                    reward += clear_reward
                    
                    if lines_cleared > 0:
                        # sound: line_clear
                        self.completed_lines += lines_cleared
                    
                    # Spawn new piece
                    if not self.game_over:
                        self._spawn_piece()
                    
                    # Check win condition
                    if self.completed_lines >= 5:
                        self.win = True
                        self.game_over = True
                        reward += 100 # Win reward
                        
            # Update difficulty
            if self.steps > 0 and self.steps % 50 == 0:
                self.fall_speed = max(0.1, self.fall_speed - 0.05)

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.win:
            reward -= 100 # Lose reward

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_piece(self):
        shape_type = self.PIECE_TYPES[self.np_random.integers(0, len(self.PIECE_TYPES))]
        self.current_piece = {
            'type': shape_type,
            'color_idx': self.PIECE_TYPES.index(shape_type) + 1,
            'rotation': self.np_random.integers(0, 4),
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0
        }
        if not self._is_valid_position(self.current_piece):
            self.game_over = True # sound: game_over

    def _place_piece(self):
        shape = self.PIECE_SHAPES[self.current_piece['type']][self.current_piece['rotation']]
        reward = 0
        for x, y in shape:
            grid_x, grid_y = self.current_piece['x'] + x, self.current_piece['y'] + y
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                self.grid[grid_x, grid_y] = self.current_piece['color_idx']
                # Placement rewards
                if grid_y < self.GRID_HEIGHT - 1 and self.grid[grid_x, grid_y + 1] != 0: reward += 0.1 # Placed on another block
                if grid_y == self.GRID_HEIGHT - 1: reward += 0.1 # Placed on floor
                # Penalty for empty adjacent cells
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                        if self.grid[nx, ny] == 0:
                            reward -= 0.02

        self.current_piece = None
        return reward

    def _check_lines(self):
        reward = 0
        lines_cleared_count = 0
        
        # Check rows
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[:, y] != 0):
                self.lines_to_clear.append(('row', y))
                reward += 1.0
                lines_cleared_count += 1
        
        # Check columns
        for x in range(self.GRID_WIDTH):
            if np.all(self.grid[x, :] != 0):
                self.lines_to_clear.append(('col', x))
                reward += 1.0
                lines_cleared_count += 1
        
        if self.lines_to_clear:
            self.clearing_animation = {'timer': 0.2, 'lines': self.lines_to_clear}
        
        return reward, lines_cleared_count

    def _perform_clear(self):
        rows_cleared = sorted([line[1] for line in self.lines_to_clear if line[0] == 'row'], reverse=True)
        cols_cleared = sorted([line[1] for line in self.lines_to_clear if line[0] == 'col'], reverse=True)

        for y in rows_cleared:
            self.grid[:, y:] = np.roll(self.grid[:, y:], -1, axis=1)
            self.grid[:, -1] = 0

        for x in cols_cleared:
            self.grid[x:, :] = np.roll(self.grid[x:, :], -1, axis=0)
            self.grid[-1, :] = 0
        
        self.lines_to_clear = []

    def _is_valid_position(self, piece, offset_x=0, offset_y=0, rotation=0):
        new_rotation = (piece['rotation'] + rotation) % 4
        shape = self.PIECE_SHAPES[piece['type']][new_rotation]
        for x, y in shape:
            grid_x = piece['x'] + x + offset_x
            grid_y = piece['y'] + y + offset_y
            if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                return False
            if self.grid[grid_x, grid_y] != 0:
                return False
        return True

    def _check_termination(self):
        return self.game_over or self.steps >= 2000

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "completed_lines": self.completed_lines,
        }
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_block(self, surface, x, y, color, alpha=255):
        outer_rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        inner_rect = pygame.Rect(x + 3, y + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
        
        light_color = tuple(min(255, c + 60) for c in color)
        dark_color = tuple(max(0, c - 60) for c in color)

        if alpha < 255:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, (*dark_color, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE))
            pygame.draw.rect(s, (*color, alpha), (3, 3, self.CELL_SIZE-6, self.CELL_SIZE-6))
            pygame.draw.rect(s, (*light_color, alpha), (3, 3, self.CELL_SIZE-8, self.CELL_SIZE-8))
            surface.blit(s, (x, y))
        else:
            pygame.draw.rect(surface, dark_color, outer_rect)
            pygame.draw.rect(surface, color, inner_rect)
            pygame.draw.rect(surface, light_color, (x + 4, y + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8))
    
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        
        # Draw placed blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != 0:
                    color_idx = int(self.grid[x, y]) - 1
                    color = self.PIECE_COLORS[self.PIECE_TYPES[color_idx]]
                    self._draw_block(self.screen, self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, color)

        if self.current_piece:
            # Draw ghost piece
            ghost_y = self.current_piece['y']
            while self._is_valid_position(self.current_piece, offset_y=ghost_y - self.current_piece['y'] + 1):
                ghost_y += 1
            shape = self.PIECE_SHAPES[self.current_piece['type']][self.current_piece['rotation']]
            color = self.PIECE_COLORS[self.current_piece['type']]
            for x, y in shape:
                screen_x = self.GRID_OFFSET_X + (self.current_piece['x'] + x) * self.CELL_SIZE
                screen_y = self.GRID_OFFSET_Y + (ghost_y + y) * self.CELL_SIZE
                self._draw_block(self.screen, screen_x, screen_y, color, alpha=60)
            
            # Draw current piece
            for x, y in shape:
                screen_x = self.GRID_OFFSET_X + (self.current_piece['x'] + x) * self.CELL_SIZE
                screen_y = self.GRID_OFFSET_Y + (self.current_piece['y'] + y) * self.CELL_SIZE
                self._draw_block(self.screen, screen_x, screen_y, color)

        # Draw clearing animation
        if self.clearing_animation:
            for line_type, index in self.clearing_animation['lines']:
                if line_type == 'row':
                    rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y + index * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                else: # col
                    rect = pygame.Rect(self.GRID_OFFSET_X + index * self.CELL_SIZE, self.GRID_OFFSET_Y, self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_CLEAR_ANIM, rect)

        # Draw grid lines on top
        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y), (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE))


    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.GRID_OFFSET_X, 20))
        
        lines_text = self.font_main.render(f"LINES: {self.completed_lines} / 5", True, self.COLOR_UI_TEXT)
        lines_rect = lines_text.get_rect(topright=(self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, 20))
        self.screen.blit(lines_text, lines_rect)

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(end_text, end_rect)
            
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example usage to test the environment ---
if __name__ == '__main__':
    # Set Pygame to run headlessly if not rendering to screen
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 

    env = GameEnv()
    obs, info = env.reset()
    
    # For human play
    pygame.display.set_caption("Puzzle Game")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Action mapping for human play ---
    # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    # actions[1]: Space button (0=released, 1=held)
    # actions[2]: Shift button (0=released, 1=held)
    
    while running:
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        elif keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Match the environment's frame rate
        
    pygame.quit()