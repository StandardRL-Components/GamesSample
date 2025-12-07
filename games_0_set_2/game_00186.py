
# Generated: 2025-08-27T12:51:52.989924
# Source Brief: brief_00186.md
# Brief Index: 186

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Space for hard drop, hold Shift for soft drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, falling block puzzle. Clear lines to score points before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    BLOCK_SIZE = 18
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 30, 40)
    COLOR_WHITE = (255, 255, 255)

    # Tetromino shapes and their colors
    TETROMINOES = {
        'I': [[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]],
        'O': [[[1, 1], [1, 1]]],
        'T': [[[0, 1, 0], [1, 1, 1], [0, 0, 0]], [[0, 1, 0], [0, 1, 1], [0, 1, 0]], [[0, 0, 0], [1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 0], [0, 1, 0]]],
        'S': [[[0, 1, 1], [1, 1, 0], [0, 0, 0]], [[0, 1, 0], [0, 1, 1], [0, 0, 1]]],
        'Z': [[[1, 1, 0], [0, 1, 1], [0, 0, 0]], [[0, 0, 1], [0, 1, 1], [0, 1, 0]]],
        'J': [[[0, 1, 0], [0, 1, 0], [1, 1, 0]], [[1, 0, 0], [1, 1, 1], [0, 0, 0]], [[0, 1, 1], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 1]]],
        'L': [[[0, 1, 0], [0, 1, 0], [0, 1, 1]], [[0, 0, 0], [1, 1, 1], [1, 0, 0]], [[1, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1], [0, 0, 0]]]
    }
    
    TETROMINO_COLORS = {
        'I': (66, 215, 245), 'O': (245, 227, 66), 'T': (188, 66, 245),
        'S': (66, 245, 114), 'Z': (245, 66, 66), 'J': (66, 114, 245), 'L': (245, 161, 66)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        self.grid_render_pos = (
            (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2,
            (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2
        )
        
        # Game state variables are initialized in reset()
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_time = 0
        self.base_fall_speed = 1000 # ms per grid cell
        self.fall_speed = 1000
        self.prev_action = np.array([0, 0, 0])
        self.line_clear_effects = []
        self.rng = None
        
        self.reset()
        self.validate_implementation()

    def _new_piece(self):
        shape_key = self.rng.choice(list(self.TETROMINOES.keys()))
        return {
            "shape_key": shape_key,
            "shape": self.TETROMINOES[shape_key],
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - len(self.TETROMINOES[shape_key][0][0]) // 2,
            "y": 0,
            "color": self.TETROMINO_COLORS[shape_key]
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.fall_time = 0
        self.base_fall_speed = 1000
        self.fall_speed = self.base_fall_speed
        self.prev_action = np.array([0, 0, 0])
        self.line_clear_effects = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle player input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Discrete actions (on key press)
        rotate_cw = movement == 1 and self.prev_action[0] != 1
        rotate_ccw = movement == 2 and self.prev_action[0] != 2
        hard_drop = space_held and not (self.prev_action[1] == 1)
        
        # Continuous actions (on key hold)
        move_left = movement == 3
        move_right = movement == 4
        soft_drop = shift_held

        if rotate_cw: self._rotate_piece(1)
        if rotate_ccw: self._rotate_piece(-1)
        
        if move_left:
            if self._move_piece(-1, 0): reward -= 0.02
        if move_right:
            if self._move_piece(1, 0): reward -= 0.02

        if hard_drop:
            # Sfx: Hard drop sound
            moved_rows = 0
            while self._move_piece(0, 1):
                moved_rows += 1
            reward += moved_rows * 0.1
            self._lock_piece()
        else:
            # --- Update game logic (auto-fall) ---
            self.fall_time += self.clock.get_time()
            current_fall_speed = self.fall_speed / 5 if soft_drop else self.fall_speed

            if self.fall_time > current_fall_speed:
                self.fall_time = 0
                if not self._move_piece(0, 1):
                    self._lock_piece()
                else:
                    reward += 0.1 # Reward for natural fall
        
        reward += self._calculate_reward()
        terminated = self._check_termination()

        if self.steps >= 5000:
            terminated = True
        
        self.prev_action = action
        self.clock.tick(30)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _rotate_piece(self, direction):
        if self.current_piece is None: return
        # Sfx: Rotate sound
        piece = self.current_piece
        original_rotation = piece['rotation']
        piece['rotation'] = (piece['rotation'] + direction) % len(piece['shape'])
        
        # Wall kick logic
        original_x = piece['x']
        for offset in [0, -1, 1, -2, 2]:
            piece['x'] = original_x + offset
            if not self._check_collision(piece):
                return
        
        # Rotation failed, revert
        piece['rotation'] = original_rotation
        piece['x'] = original_x

    def _move_piece(self, dx, dy):
        if self.current_piece is None: return False
        
        self.current_piece['x'] += dx
        self.current_piece['y'] += dy

        if self._check_collision(self.current_piece):
            self.current_piece['x'] -= dx
            self.current_piece['y'] -= dy
            return False
        return True

    def _check_collision(self, piece):
        shape = piece['shape'][piece['rotation']]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = piece['x'] + c
                    grid_y = piece['y'] + r
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True # Out of bounds
                    if self.grid[grid_y, grid_x] != 0:
                        return True # Collides with another block
        return False

    def _lock_piece(self):
        if self.current_piece is None: return
        # Sfx: Lock piece sound
        
        # Calculate risky/safe placement reward before locking
        reward = self._calculate_placement_reward()

        shape = self.current_piece['shape'][self.current_piece['rotation']]
        color_index = list(self.TETROMINO_COLORS.keys()).index(self.current_piece['shape_key']) + 1
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece['x'] + c
                    grid_y = self.current_piece['y'] + r
                    if 0 <= grid_y < self.GRID_HEIGHT:
                        self.grid[grid_y, grid_x] = color_index
        
        # Clear lines and get reward
        reward += self._clear_lines()

        # Spawn new piece
        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()

        # Check for game over
        if self._check_collision(self.current_piece):
            self.game_over = True
            # Sfx: Game over sound
            reward -= 100
        
        self._last_reward = reward

    def _clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            # Sfx: Line clear sound
            for r in lines_to_clear:
                self.grid[r, :] = 0
                # Add visual effect
                self.line_clear_effects.append({'y': r, 'timer': 5})

            # Shift rows down
            cleared_count = len(lines_to_clear)
            for r in sorted(lines_to_clear, reverse=False):
                for row_idx in range(r, 0, -1):
                    self.grid[row_idx, :] = self.grid[row_idx - 1, :]
                self.grid[0, :] = 0

            # Update score and speed
            self.lines_cleared += cleared_count
            score_map = {1: 40, 2: 100, 3: 300, 4: 1200}
            self.score += score_map.get(cleared_count, 0)
            
            # Difficulty scaling
            self.fall_speed = self.base_fall_speed - (self.score // 200) * 50
            self.fall_speed = max(100, self.fall_speed) # Cap speed
            
            return cleared_count * 1 # Reward for clearing lines
        return 0

    def _calculate_reward(self):
        # This function is used to apply rewards that were calculated in other functions
        # during the step, as the reward needs to be returned at the end.
        reward = getattr(self, '_last_reward', 0)
        self._last_reward = 0
        return reward

    def _calculate_placement_reward(self):
        if self.current_piece is None: return 0
        
        piece = self.current_piece
        shape = piece['shape'][piece['rotation']]
        adjacent_blocks = 0
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = piece['x'] + c, piece['y'] + r
                    # Check neighbors (left, right, down)
                    for dx, dy in [(-1, 0), (1, 0), (0, 1)]:
                        nx, ny = grid_x + dx, grid_y + dy
                        if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
                            adjacent_blocks += 1 # Walls count as adjacent
                            continue
                        if self.grid[ny, nx] != 0:
                            adjacent_blocks += 1
        
        if adjacent_blocks >= 3:
            return 2 # Risky placement
        if adjacent_blocks <= 1:
            return -0.2 # Safe placement
        return 0

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= 1000:
            self._last_reward = 100 # Win reward
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Update and draw line clear effects
        for effect in self.line_clear_effects[:]:
            effect['timer'] -= 1
            if effect['timer'] <= 0:
                self.line_clear_effects.remove(effect)
            else:
                gx, gy = self.grid_render_pos
                y_pos = gy + effect['y'] * self.BLOCK_SIZE
                width = self.GRID_WIDTH * self.BLOCK_SIZE
                alpha = 150 + (effect['timer'] * 20)
                flash_surf = pygame.Surface((width, self.BLOCK_SIZE), pygame.SRCALPHA)
                flash_surf.fill((255, 255, 255, alpha))
                self.screen.blit(flash_surf, (gx, y_pos))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        gx, gy = self.grid_render_pos
        
        # Draw grid background and lines
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (gx, gy, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE))
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (gx + x * self.BLOCK_SIZE, gy), (gx + x * self.BLOCK_SIZE, gy + self.GRID_HEIGHT * self.BLOCK_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (gx, gy + y * self.BLOCK_SIZE), (gx + self.GRID_WIDTH * self.BLOCK_SIZE, gy + y * self.BLOCK_SIZE))

        # Draw locked blocks
        color_keys = list(self.TETROMINO_COLORS.keys())
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    color = self.TETROMINO_COLORS[color_keys[int(self.grid[r, c]) - 1]]
                    self._draw_block(c, r, color)

        if self.current_piece and not self.game_over:
            # Draw ghost piece
            ghost_piece = self.current_piece.copy()
            while not self._check_collision(ghost_piece):
                ghost_piece['y'] += 1
            ghost_piece['y'] -= 1
            
            shape = ghost_piece['shape'][ghost_piece['rotation']]
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(ghost_piece['x'] + c, ghost_piece['y'] + r, ghost_piece['color'], is_ghost=True)

            # Draw current piece
            shape = self.current_piece['shape'][self.current_piece['rotation']]
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece['x'] + c, self.current_piece['y'] + r, self.current_piece['color'])

    def _draw_block(self, x, y, color, is_ghost=False):
        gx, gy = self.grid_render_pos
        px, py = gx + x * self.BLOCK_SIZE, gy + y * self.BLOCK_SIZE
        
        if is_ghost:
            rect = (px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.gfxdraw.box(self.screen, rect, (*color, 60))
            pygame.gfxdraw.rectangle(self.screen, rect, (*color, 120))
        else:
            main_rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
            
            # 3D effect
            light_color = tuple(min(255, c + 40) for c in color)
            dark_color = tuple(max(0, c - 40) for c in color)
            
            pygame.draw.rect(self.screen, dark_color, main_rect)
            inner_rect = pygame.Rect(px + 2, py + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4)
            pygame.draw.rect(self.screen, color, inner_rect)
            
            # Top and left highlights
            pygame.draw.line(self.screen, light_color, (px, py), (px + self.BLOCK_SIZE - 1, py))
            pygame.draw.line(self.screen, light_color, (px, py), (px, py + self.BLOCK_SIZE - 1))


    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_main.render(f"{self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(score_val, (20, 50))

        # Lines display
        lines_text = self.font_main.render(f"LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_main.render(f"{self.lines_cleared}", True, self.COLOR_WHITE)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - lines_text.get_width() - 20, 20))
        self.screen.blit(lines_val, (self.SCREEN_WIDTH - lines_val.get_width() - 20, 50))

        # Next piece preview
        next_text = self.font_main.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 120, 100))
        
        preview_box = pygame.Rect(self.SCREEN_WIDTH - 140, 130, 120, 100)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, preview_box)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box, 2)
        
        if self.next_piece:
            shape = self.next_piece['shape'][0]
            color = self.next_piece['color']
            
            shape_w = len(shape[0]) * self.BLOCK_SIZE
            shape_h = len(shape) * self.BLOCK_SIZE
            start_x = preview_box.centerx - shape_w // 2
            start_y = preview_box.centery - shape_h // 2
            
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        px = start_x + c * self.BLOCK_SIZE
                        py = start_y + r * self.BLOCK_SIZE
                        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
                        light_color = tuple(min(255, c + 40) for c in color)
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.line(self.screen, light_color, (px, py), (px + self.BLOCK_SIZE - 1, py))
                        pygame.draw.line(self.screen, light_color, (px, py), (px, py + self.BLOCK_SIZE - 1))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "LEVEL CLEAR" if self.score >= 1000 else "GAME OVER"
            text_surf = self.font_main.render(win_text, True, self.COLOR_WHITE)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "game_over": self.game_over
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Falling Block Puzzle")
    game_clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop for human play
    while not terminated:
        # --- Human Input ---
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = 0 # 0=released, 1=held
        shift_held = 0 # 0=released, 1=held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = np.array([movement, space_held, shift_held])
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        # The observation is already a rendered frame, just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before closing
            
        game_clock.tick(30)
        
    env.close()
    pygame.quit()