
# Generated: 2025-08-28T05:41:39.612491
# Source Brief: brief_02703.md
# Brief Index: 2703

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, ↓ to soft drop. "
        "Hold shift to rotate counter-clockwise. Press space for hard drop."
    )

    game_description = (
        "A fast-paced falling block puzzle. Strategically place pieces to clear lines "
        "and score points before the stack reaches the top. The game speeds up as you play!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2 - 100
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GHOST = (128, 128, 128, 100)
    
    BLOCK_COLORS = [
        (0, 0, 0),          # 0: Empty
        (0, 240, 240),      # 1: I (Cyan)
        (240, 240, 0),      # 2: O (Yellow)
        (160, 0, 240),      # 3: T (Purple)
        (0, 0, 240),        # 4: J (Blue)
        (240, 160, 0),      # 5: L (Orange)
        (0, 240, 0),        # 6: S (Green)
        (240, 0, 0),        # 7: Z (Red)
    ]

    # Tetromino shapes
    TETROMINOES = {
        1: [[1, 1, 1, 1]], # I
        2: [[1, 1], [1, 1]], # O
        3: [[0, 1, 0], [1, 1, 1]], # T
        4: [[1, 0, 0], [1, 1, 1]], # J
        5: [[0, 0, 1], [1, 1, 1]], # L
        6: [[0, 1, 1], [1, 1, 0]], # S
        7: [[1, 1, 0], [0, 1, 1]]  # Z
    }

    # Game parameters
    WIN_SCORE = 1000
    MAX_STEPS = 10000
    INITIAL_FALL_SPEED = 1.0 # cells per second
    SPEED_INCREMENT = 0.01

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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.grid = None
        self.current_block = None
        self.next_block = None
        self.fall_timer = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.blocks_placed = 0
        self.space_pressed_this_block = False
        self.sideways_move_penalty_applied = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.fall_timer = 0
        self.blocks_placed = 0
        
        self._spawn_new_block() # Spawns the 'next' block
        self._spawn_new_block() # Spawns the 'current' block, moving the previous 'next'
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        reward += self._handle_actions(action)
        
        self.fall_timer += self.fall_speed
        
        if self.fall_timer >= 30: # 30 FPS
            self.fall_timer = self.fall_timer % 30
            reward += self._move_block(1, 0)
        
        lines_cleared = 0
        placed_block = False
        
        if not self._is_valid_position(self.current_block['shape'], (self.current_block['y'] + 1, self.current_block['x'])):
            self._lock_block()
            placed_block = True
            lines_cleared = self._clear_lines()
            # sfx: block_lock.wav
            
            self.blocks_placed += 1
            if self.blocks_placed > 0 and self.blocks_placed % 100 == 0:
                self.fall_speed += self.SPEED_INCREMENT
                
            self._spawn_new_block()
            
            if not self._is_valid_position(self.current_block['shape'], (self.current_block['y'], self.current_block['x'])):
                self.game_over = True

        reward += self._calculate_reward(lines_cleared, placed_block)
        
        terminated = self._check_termination()
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100
            else:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        if movement == 1: # Up -> Rotate CW
            self._rotate_block(clockwise=True)
            # sfx: rotate.wav
        elif movement == 2: # Down -> Soft Drop
            self.fall_timer += 5
        elif movement == 3: # Left
            reward += self._move_block(0, -1)
            # sfx: move.wav
        elif movement == 4: # Right
            reward += self._move_block(0, 1)
            # sfx: move.wav

        if shift_held:
            self._rotate_block(clockwise=False)
            # sfx: rotate.wav

        if space_held and not self.space_pressed_this_block:
            self._hard_drop()
            self.space_pressed_this_block = True
            # sfx: hard_drop.wav
            
        return reward

    def _calculate_reward(self, lines_cleared, placed_block):
        reward = 0
        if placed_block:
            reward += 0.1
        
        score_map = {1: 100, 2: 300, 3: 500, 4: 800}
        reward_map = {1: 1, 2: 3, 3: 5, 4: 10}
        
        if lines_cleared in score_map:
            self.score += score_map[lines_cleared]
            reward += reward_map[lines_cleared]
            
        return reward

    def _check_termination(self):
        return self.game_over or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _spawn_new_block(self):
        if self.next_block is None:
            shape_id = self.np_random.integers(1, len(self.TETROMINOES) + 1)
            self.next_block = {
                'id': shape_id,
                'shape': self.TETROMINOES[shape_id],
                'color': self.BLOCK_COLORS[shape_id]
            }
        
        self.current_block = self.next_block
        self.current_block['x'] = self.GRID_WIDTH // 2 - len(self.current_block['shape'][0]) // 2
        self.current_block['y'] = 0
        
        shape_id = self.np_random.integers(1, len(self.TETROMINOES) + 1)
        self.next_block = {
            'id': shape_id,
            'shape': self.TETROMINOES[shape_id],
            'color': self.BLOCK_COLORS[shape_id]
        }
        
        self.space_pressed_this_block = False
        self.sideways_move_penalty_applied = False

    def _is_valid_position(self, shape, offset):
        off_y, off_x = offset
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_y, grid_x = y + off_y, x + off_x
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_y, grid_x] != 0:
                        return False
        return True

    def _move_block(self, dy, dx):
        new_y = self.current_block['y'] + dy
        new_x = self.current_block['x'] + dx
        if self._is_valid_position(self.current_block['shape'], (new_y, new_x)):
            self.current_block['y'] = new_y
            self.current_block['x'] = new_x
            if dx != 0 and not self.sideways_move_penalty_applied:
                self.sideways_move_penalty_applied = True
                return -0.02
        return 0

    def _rotate_block(self, clockwise=True):
        shape = self.current_block['shape']
        if clockwise:
            new_shape = [list(row) for row in zip(*shape[::-1])]
        else:
            new_shape = [list(row) for row in zip(*shape)][::-1]

        for dx in [0, 1, -1, 2, -2]:
            if self._is_valid_position(new_shape, (self.current_block['y'], self.current_block['x'] + dx)):
                self.current_block['shape'] = new_shape
                self.current_block['x'] += dx
                return
    
    def _hard_drop(self):
        while self._is_valid_position(self.current_block['shape'], (self.current_block['y'] + 1, self.current_block['x'])):
            self.current_block['y'] += 1
        self.fall_timer = 30

    def _lock_block(self):
        shape = self.current_block['shape']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_y = self.current_block['y'] + y
                    grid_x = self.current_block['x'] + x
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_block['id']

    def _clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.grid) if np.all(row != 0)]
        if not lines_to_clear:
            return 0
        
        # sfx: line_clear.wav
        for i in lines_to_clear:
            self.grid[i, :] = 8
        
        self._get_observation() 
        
        num_cleared = len(lines_to_clear)
        self.grid = np.delete(self.grid, lines_to_clear, axis=0)
        new_rows = np.zeros((num_cleared, self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack((new_rows, self.grid))
        
        return num_cleared

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, 
            (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))

        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, 
                (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y),
                (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG,
                (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE),
                (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y + i * self.CELL_SIZE))

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell_id = self.grid[y, x]
                if cell_id != 0:
                    color = self.BLOCK_COLORS[cell_id] if cell_id != 8 else (255, 255, 255)
                    self._draw_cell(x, y, color)

        if not self.game_over:
            self._render_ghost_block()
            self._render_falling_block()

    def _render_ghost_block(self):
        ghost_y = self.current_block['y']
        while self._is_valid_position(self.current_block['shape'], (ghost_y + 1, self.current_block['x'])):
            ghost_y += 1
        
        shape = self.current_block['shape']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    px = self.GRID_X + (self.current_block['x'] + x) * self.CELL_SIZE
                    py = self.GRID_Y + (ghost_y + y) * self.CELL_SIZE
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    s.fill(self.COLOR_GHOST)
                    self.screen.blit(s, (px, py))

    def _render_falling_block(self):
        shape = self.current_block['shape']
        color = self.current_block['color']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self._draw_cell(self.current_block['x'] + x, self.current_block['y'] + y, color)
    
    def _draw_cell(self, grid_x, grid_y, color):
        px, py = self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE
        pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE))
        
        light_color = tuple(min(255, c + 40) for c in color)
        dark_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.line(self.screen, light_color, (px, py), (px + self.CELL_SIZE - 1, py), 2)
        pygame.draw.line(self.screen, light_color, (px, py), (px, py + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, dark_color, (px + self.CELL_SIZE - 1, py), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, dark_color, (px, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 2)

    def _render_ui(self):
        ui_x = self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 40
        score_text = self.font_large.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, self.GRID_Y))
        self.screen.blit(score_val, (ui_x, self.GRID_Y + 30))

        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (ui_x, self.GRID_Y + 100))
        
        if self.next_block:
            shape = self.next_block['shape']
            color = self.next_block['color']
            start_x = ui_x + 10
            start_y = self.GRID_Y + 130
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        px, py = start_x + x * self.CELL_SIZE, start_y + y * self.CELL_SIZE
                        self._draw_cell(px / self.CELL_SIZE, py / self.CELL_SIZE, color)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "GAME OVER"
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
            
            game_over_text = self.font_large.render(msg, True, (255, 50, 50))
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    import os
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Falling Block Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)
        
    env.close()