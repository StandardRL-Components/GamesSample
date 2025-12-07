
# Generated: 2025-08-28T05:41:00.201005
# Source Brief: brief_05662.md
# Brief Index: 5662

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Press space for hard drop and shift to hold a piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a 60-second onslaught of falling tetrominoes in this fast-paced, grid-based arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    
    PLAYFIELD_W = GRID_WIDTH * CELL_SIZE
    PLAYFIELD_H = GRID_HEIGHT * CELL_SIZE
    PLAYFIELD_X = (WIDTH - PLAYFIELD_W) // 2
    PLAYFIELD_Y = (HEIGHT - PLAYFIELD_H) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (220, 220, 240)
    COLOR_WHITE = (255, 255, 255)
    
    # Tetromino shapes and colors
    TETROMINOES = {
        'I': {'shape': [[1, 1, 1, 1]], 'color': (0, 240, 240)},
        'J': {'shape': [[1, 0, 0], [1, 1, 1]], 'color': (0, 0, 240)},
        'L': {'shape': [[0, 0, 1], [1, 1, 1]], 'color': (240, 160, 0)},
        'O': {'shape': [[1, 1], [1, 1]], 'color': (240, 240, 0)},
        'S': {'shape': [[0, 1, 1], [1, 1, 0]], 'color': (0, 240, 0)},
        'T': {'shape': [[0, 1, 0], [1, 1, 1]], 'color': (160, 0, 240)},
        'Z': {'shape': [[1, 1, 0], [0, 1, 1]], 'color': (240, 0, 0)},
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state variables
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.held_piece = None
        self.can_hold = True
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.fall_counter = 0.0
        self.initial_fall_speed = 0.02 # cells per frame
        self.fall_speed = self.initial_fall_speed
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.line_clear_animation = [] # list of (y_index, timer)
        
        self.total_time_seconds = 60
        self.time_elapsed = 0
        
        self.np_random = None

        self.reset()
        
        # Run self-check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT + 4), dtype=int) # Extra rows for spawning
        self.piece_bag = list(self.TETROMINOES.keys())
        random.shuffle(self.piece_bag)
        
        self.next_piece = self._create_piece(self._get_next_shape())
        self._spawn_new_piece()
        
        self.held_piece = None
        self.can_hold = True
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0
        
        self.fall_speed = self.initial_fall_speed
        self.fall_counter = 0.0
        
        self.last_space_held = True # Prevent action on first frame
        self.last_shift_held = True
        
        self.line_clear_animation = []
        
        return self._get_observation(), self._get_info()

    def _get_next_shape(self):
        if not self.piece_bag:
            self.piece_bag = list(self.TETROMINOES.keys())
            random.shuffle(self.piece_bag)
        return self.piece_bag.pop()

    def _create_piece(self, shape_key):
        shape_data = self.TETROMINOES[shape_key]
        piece = {
            'key': shape_key,
            'shape': shape_data['shape'],
            'color': shape_data['color'],
            'x': self.GRID_WIDTH // 2 - len(shape_data['shape'][0]) // 2,
            'y': 2, # Start above visible area
            'rotation': 0
        }
        return piece

    def _spawn_new_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self._create_piece(self._get_next_shape())
        self.can_hold = True
        if self._check_collision(self.current_piece, (0, 0)):
            self.game_over = True

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        self.game_over = self.game_over or (self.time_elapsed >= self.total_time_seconds * 60)
        if self.game_over:
            final_reward = 100 if self.time_elapsed >= self.total_time_seconds * 60 else -100
            return self._get_observation(), final_reward, True, False, self._get_info()

        self.steps += 1
        self.time_elapsed += 1
        
        self.fall_speed = self.initial_fall_speed + (self.time_elapsed / 3600) * 0.08 # Scale to ~5x speed over 60s

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle one-shot actions ---
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        if shift_pressed and self.can_hold:
            self._hold_piece()
            # sfx: hold_piece.wav
        elif space_pressed:
            reward += self._hard_drop()
            # sfx: hard_drop.wav
        else:
            # --- Handle continuous actions ---
            if movement == 3: # Left
                self._move(-1)
            elif movement == 4: # Right
                self._move(1)
            elif movement == 1: # Rotate Left (mapped from up)
                self._rotate(-1)
            elif movement == 2: # Rotate Right (mapped from down)
                self._rotate(1)
            
            # --- Game Physics ---
            current_fall_speed = self.fall_speed * 5 if movement == 2 else self.fall_speed
            self.fall_counter += current_fall_speed
            
            if self.fall_counter >= 1.0:
                self.fall_counter -= 1.0
                if not self._check_collision(self.current_piece, (0, 1)):
                    self.current_piece['y'] += 1
                else:
                    reward += self._lock_piece()
                    # sfx: piece_lock.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        clear_reward, lines_cleared = self._check_and_clear_lines()
        reward += clear_reward
        if lines_cleared > 0:
             # sfx: line_clear.wav
             pass
        
        terminated = self.game_over or (self.time_elapsed >= self.total_time_seconds * 60)
        if terminated:
            reward += 100 if self.time_elapsed >= self.total_time_seconds * 60 else -100
            if self.time_elapsed >= self.total_time_seconds * 60:
                # sfx: victory.wav
                pass
            else:
                # sfx: game_over.wav
                pass

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move(self, dx):
        if not self._check_collision(self.current_piece, (dx, 0)):
            self.current_piece['x'] += dx
            # sfx: move.wav
            return True
        return False
    
    def _rotate(self, direction):
        piece = self.current_piece
        original_shape = piece['shape']
        
        if direction == 1: # Clockwise
            new_shape = [list(row) for row in zip(*piece['shape'][::-1])]
        else: # Counter-clockwise
            new_shape = [list(row) for row in zip(*piece['shape'])][::-1]
        
        piece['shape'] = new_shape
        
        # Wall kick logic
        for dx in [0, 1, -1, 2, -2]: # Basic wall kick checks
            if not self._check_collision(piece, (dx, 0)):
                piece['x'] += dx
                # sfx: rotate.wav
                return True
        
        piece['shape'] = original_shape # Revert if no valid position found
        return False

    def _hold_piece(self):
        self.can_hold = False
        if self.held_piece:
            self.current_piece, self.held_piece = self.held_piece, self.current_piece
            self.current_piece['x'] = self.GRID_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
            self.current_piece['y'] = 2
        else:
            self.held_piece = self.current_piece
            self._spawn_new_piece()
    
    def _hard_drop(self):
        dy = 0
        while not self._check_collision(self.current_piece, (0, dy + 1)):
            dy += 1
        self.current_piece['y'] += dy
        return self._lock_piece()

    def _check_collision(self, piece, offset):
        dx, dy = offset
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = piece['x'] + c + dx, piece['y'] + r + dy
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT + 4):
                        return True
                    if self.grid[grid_x, grid_y] != 0:
                        return True
        return False

    def _lock_piece(self):
        placement_reward = 0
        piece_coords = []
        color_index = list(self.TETROMINOES.keys()).index(self.current_piece['key']) + 1
        
        for r, row in enumerate(self.current_piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece['x'] + c
                    grid_y = self.current_piece['y'] + r
                    if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT + 4:
                        self.grid[grid_x, grid_y] = color_index
                        piece_coords.append((grid_x, grid_y))

        placement_reward += self._calculate_placement_reward(piece_coords, color_index)
        self._spawn_new_piece()
        self.score += 10 # Base score for placing a piece
        return placement_reward

    def _calculate_placement_reward(self, piece_coords, color_index):
        reward = 0
        # Hole penalty
        for x, y in piece_coords:
            for r in range(y + 1, self.GRID_HEIGHT + 4):
                if self.grid[x, r] == 0:
                    reward -= 2 # Punish for creating a hole
                    break
        
        # Adjacency bonus
        for x, y in piece_coords:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT + 4:
                    if self.grid[nx, ny] == color_index:
                        reward += 5
        return reward

    def _check_and_clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT + 4):
            if all(self.grid[:, r]):
                lines_to_clear.append(r)
        
        if not lines_to_clear:
            return 0, 0
            
        for r in lines_to_clear:
            self.line_clear_animation.append([r, 5]) # 5 frames of animation
            self.grid[:, r] = 0

        # Shift blocks down
        lines_cleared = len(lines_to_clear)
        lines_to_clear.sort()
        
        for r in lines_to_clear:
            self.grid[:, 1:r+1] = self.grid[:, 0:r].copy()
            self.grid[:, 0] = 0

        self.score += (100 * lines_cleared) * lines_cleared # exponential reward for multi-line clears
        return (10 * lines_cleared), lines_cleared

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
            "time_remaining": max(0, self.total_time_seconds - (self.time_elapsed / 60)),
            "game_over": self.game_over
        }

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.PLAYFIELD_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.PLAYFIELD_Y), (px, self.PLAYFIELD_Y + self.PLAYFIELD_H))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.PLAYFIELD_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X, py), (self.PLAYFIELD_X + self.PLAYFIELD_W, py))
        
        # Draw locked blocks
        colors = list(self.TETROMINOES.values())
        for x in range(self.GRID_WIDTH):
            for y in range(4, self.GRID_HEIGHT + 4):
                if self.grid[x, y] != 0:
                    color = colors[self.grid[x, y] - 1]['color']
                    self._draw_block(x, y - 4, color)
        
        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_y_offset = 0
            while not self._check_collision(self.current_piece, (0, ghost_y_offset + 1)):
                ghost_y_offset += 1
            
            ghost_piece = self.current_piece.copy()
            ghost_piece['y'] += ghost_y_offset
            self._draw_piece(ghost_piece, is_ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            self._draw_piece(self.current_piece)
            
        # Draw line clear animation
        new_animations = []
        for anim in self.line_clear_animation:
            y, timer = anim
            rect = pygame.Rect(self.PLAYFIELD_X, self.PLAYFIELD_Y + (y - 4) * self.CELL_SIZE, self.PLAYFIELD_W, self.CELL_SIZE)
            flash_color = (255, 255, 255, timer * 50)
            flash_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
            flash_surface.fill(flash_color)
            self.screen.blit(flash_surface, rect.topleft)
            anim[1] -= 1
            if anim[1] > 0:
                new_animations.append(anim)
        self.line_clear_animation = new_animations

    def _draw_piece(self, piece, is_ghost=False):
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = piece['x'] + c
                    grid_y = piece['y'] + r - 4 # Adjust for hidden rows
                    if grid_y >= 0:
                        self._draw_block(grid_x, grid_y, piece['color'], is_ghost)

    def _draw_block(self, grid_x, grid_y, color, is_ghost=False):
        x = self.PLAYFIELD_X + grid_x * self.CELL_SIZE
        y = self.PLAYFIELD_Y + grid_y * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.gfxdraw.rectangle(self.screen, rect, (*color, 80))
        else:
            light_color = tuple(min(255, c + 60) for c in color)
            dark_color = tuple(max(0, c - 60) for c in color)
            pygame.draw.rect(self.screen, dark_color, rect)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect)
            pygame.draw.line(self.screen, light_color, inner_rect.topleft, inner_rect.topright)
            pygame.draw.line(self.screen, light_color, inner_rect.topleft, inner_rect.bottomleft)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 20, 20))

        # Time remaining bar
        time_ratio = (self.time_elapsed / (self.total_time_seconds * 60))
        bar_width = self.PLAYFIELD_W
        bar_x = self.PLAYFIELD_X
        bar_y = self.PLAYFIELD_Y - 20
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, 10))
        pygame.draw.rect(self.screen, self.TETROMINOES['I']['color'], (bar_x, bar_y, bar_width * time_ratio, 10))

        # Next Piece
        self._render_preview_box("NEXT", self.next_piece, self.PLAYFIELD_X + self.PLAYFIELD_W + 20, self.PLAYFIELD_Y)
        
        # Held Piece
        self._render_preview_box("HOLD", self.held_piece, self.PLAYFIELD_X - 120, self.PLAYFIELD_Y)
        
        # Game Over / Win message
        if self.game_over:
            message = "VICTORY!" if self.time_elapsed >= self.total_time_seconds * 60 else "GAME OVER"
            msg_surf = self.font_main.render(message, True, self.COLOR_WHITE)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            bg_rect = msg_rect.inflate(20, 20)
            pygame.draw.rect(self.screen, self.COLOR_BG, bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, bg_rect, 2)
            self.screen.blit(msg_surf, msg_rect)

    def _render_preview_box(self, title, piece, x, y):
        box_w, box_h = 100, 100
        pygame.draw.rect(self.screen, self.COLOR_GRID, (x, y, box_w, box_h), 2)
        title_surf = self.font_small.render(title, True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (x + (box_w - title_surf.get_width())//2, y + 5))
        
        if piece:
            shape = piece['shape']
            shape_w = len(shape[0]) * self.CELL_SIZE
            shape_h = len(shape) * self.CELL_SIZE
            start_x = x + (box_w - shape_w) // 2
            start_y = y + (box_h - shape_h) // 2 + 10

            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        px = start_x + c * self.CELL_SIZE
                        py = start_y + r * self.CELL_SIZE
                        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                        light_color = tuple(min(255, c + 60) for c in piece['color'])
                        dark_color = tuple(max(0, c - 60) for c in piece['color'])
                        pygame.draw.rect(self.screen, dark_color, rect)
                        inner_rect = rect.inflate(-4, -4)
                        pygame.draw.rect(self.screen, piece['color'], inner_rect)
                        pygame.draw.line(self.screen, light_color, inner_rect.topleft, inner_rect.topright)
                        pygame.draw.line(self.screen, light_color, inner_rect.topleft, inner_rect.bottomleft)

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
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy' # Run headless
    
    env = GameEnv()
    env.validate_implementation()
    
    # --- Example of running the environment ---
    # To visualize, you would need a different setup without the dummy driver.
    # For example, create a simple script:
    #
    # import pygame
    # import time
    # from game import GameEnv # Assuming you save the code as game.py
    #
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    #
    # screen = pygame.display.set_mode((640, 400))
    # pygame.display.set_caption("Tetris Arcade")
    # clock = pygame.time.Clock()
    #
    # done = False
    # while not done:
    #     movement = 0 # no-op
    #     space = 0
    #     shift = 0
    #
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]: movement = 3
    #     if keys[pygame.K_RIGHT]: movement = 4
    #     if keys[pygame.K_UP]: movement = 1
    #     if keys[pygame.K_DOWN]: movement = 2
    #     if keys[pygame.K_SPACE]: space = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
    #
    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         time.sleep(2)
    #         obs, info = env.reset()
    #
    #     # Render to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     clock.tick(60)
    #
    # env.close()