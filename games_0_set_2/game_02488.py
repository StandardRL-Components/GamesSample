
# Generated: 2025-08-27T20:31:12.671325
# Source Brief: brief_02488.md
# Brief Index: 2488

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, ↓ for soft drop. Space for hard drop, Shift for counter-clockwise rotation."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manipulate falling blocks to clear lines and achieve the highest score against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2 - 120
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        # Visuals
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        
        self._init_tetrominoes()

        # Game parameters
        self.MAX_STEPS = 1800 # 60 seconds at 30fps
        self.TIME_LIMIT_MS = 60 * 1000
        self.LINES_TO_WIN = 20
        
        # Initialize state variables (will be set in reset)
        self.grid = None
        self.score = None
        self.steps = None
        self.lines_cleared = None
        self.game_over = None
        self.start_time = None
        self.fall_speed_initial = 0.8
        self.fall_speed = None
        self.fall_counter = None
        self.rng = None
        self.current_piece = None
        self.next_piece = None
        self.line_clear_animation = []

        self.reset()
        self.validate_implementation()
    
    def _init_tetrominoes(self):
        self.TETROMINOES = {
            'I': [np.array([[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]]),
                  np.array([[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]])],
            'O': [np.array([[1,1], [1,1]])],
            'T': [np.array([[0,1,0], [1,1,1], [0,0,0]]),
                  np.array([[0,1,0], [0,1,1], [0,1,0]]),
                  np.array([[0,0,0], [1,1,1], [0,1,0]]),
                  np.array([[0,1,0], [1,1,0], [0,1,0]])],
            'J': [np.array([[1,0,0], [1,1,1], [0,0,0]]),
                  np.array([[0,1,1], [0,1,0], [0,1,0]]),
                  np.array([[0,0,0], [1,1,1], [0,0,1]]),
                  np.array([[0,1,0], [0,1,0], [1,1,0]])],
            'L': [np.array([[0,0,1], [1,1,1], [0,0,0]]),
                  np.array([[0,1,0], [0,1,0], [0,1,1]]),
                  np.array([[0,0,0], [1,1,1], [1,0,0]]),
                  np.array([[1,1,0], [0,1,0], [0,1,0]])],
            'S': [np.array([[0,1,1], [1,1,0], [0,0,0]]),
                  np.array([[0,1,0], [0,1,1], [0,0,1]])],
            'Z': [np.array([[1,1,0], [0,1,1], [0,0,0]]),
                  np.array([[0,0,1], [0,1,1], [0,1,0]])]
        }
        self.PIECE_TYPES = list(self.TETROMINOES.keys())
        self.COLORS = {
            'I': (66, 215, 245), 'O': (245, 227, 66), 'T': (176, 66, 245),
            'J': (66, 99, 245), 'L': (245, 161, 66), 'S': (66, 245, 114),
            'Z': (245, 66, 84)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.fall_speed = self.fall_speed_initial
        self.fall_counter = 0
        self.line_clear_animation = []
        
        self.start_time = pygame.time.get_ticks()
        
        # Populate current and next piece
        self.next_piece = self._create_new_piece()
        self._spawn_new_piece()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        self.game_over = self._check_termination()

        if self.game_over:
            if self.lines_cleared >= self.LINES_TO_WIN:
                reward = 50.0  # Win reward
            else:
                reward = -50.0 # Lose reward
        else:
            if not self.line_clear_animation:
                reward += self._handle_input(action)
                reward += self._update_fall()
        
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement Actions
        if movement == 3: # Left
            self._move(-1)
        elif movement == 4: # Right
            self._move(1)
        
        # Rotation
        if movement == 1: # Up for Rotate CW
            self._rotate(1)
        if shift_held: # Shift for Rotate CCW
            self._rotate(-1)
        
        # Drops
        if space_held: # Hard drop
            return self._hard_drop()
        
        if movement == 2: # Down for Soft drop
            if self._move_piece_down():
                self.fall_counter = 0 # Reset fall timer after soft drop
                return 0.1 # Reward for moving closer to bottom
        return 0.0
    
    def _update_fall(self):
        self.fall_counter += 1 / 30.0 # Time passed this frame (at 30fps)
        if self.fall_counter >= self.fall_speed:
            self.fall_counter = 0
            if not self._move_piece_down():
                # Piece has landed and can't move down
                return self._lock_piece()
            else:
                # Successful natural fall
                return 0.1
        return 0.0

    def _create_new_piece(self):
        piece_type = self.rng.choice(self.PIECE_TYPES)
        return {
            "type": piece_type,
            "shape": self.TETROMINOES[piece_type][0],
            "rotation": 0,
            "row": 0,
            "col": self.GRID_WIDTH // 2 - len(self.TETROMINOES[piece_type][0][0]) // 2,
            "color_idx": self.PIECE_TYPES.index(piece_type) + 1
        }

    def _spawn_new_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self._create_new_piece()
        # Game over if new piece spawns in an invalid position
        if not self._is_valid_position(self.current_piece):
            self.game_over = True

    def _move(self, dx):
        if self._is_valid_position(self.current_piece, offset_col=dx):
            self.current_piece["col"] += dx
            return True
        return False

    def _rotate(self, direction):
        rot = self.current_piece["rotation"]
        shape_list = self.TETROMINOES[self.current_piece["type"]]
        new_rot = (rot + direction) % len(shape_list)
        
        test_piece = self.current_piece.copy()
        test_piece["rotation"] = new_rot
        test_piece["shape"] = shape_list[new_rot]

        # Basic wall kick check
        kick_offsets = [(0, 0), (0, -1), (0, 1), (0, -2), (0, 2), (-1, 0)]
        for r_off, c_off in kick_offsets:
            if self._is_valid_position(test_piece, offset_row=r_off, offset_col=c_off):
                self.current_piece["rotation"] = new_rot
                self.current_piece["shape"] = shape_list[new_rot]
                self.current_piece["row"] += r_off
                self.current_piece["col"] += c_off
                return True
        return False

    def _move_piece_down(self):
        if self._is_valid_position(self.current_piece, offset_row=1):
            self.current_piece["row"] += 1
            return True
        return False

    def _hard_drop(self):
        downward_moves = 0
        while self._move_piece_down():
            downward_moves += 1
        return self._lock_piece() + (downward_moves * 0.1)

    def _lock_piece(self):
        shape = self.current_piece["shape"]
        row, col = self.current_piece["row"], self.current_piece["col"]
        color_idx = self.current_piece["color_idx"]
        
        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    grid_r, grid_c = row + r, col + c
                    if 0 <= grid_r < self.GRID_HEIGHT and 0 <= grid_c < self.GRID_WIDTH:
                        self.grid[grid_r, grid_c] = color_idx
        
        # sound placeholder: # sfx_lock_piece
        reward = self._clear_lines()
        self._spawn_new_piece()
        return reward

    def _clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            # sound placeholder: # sfx_line_clear
            self.line_clear_animation = [(r, 15) for r in lines_to_clear] # row, frames_left
            self.lines_cleared += len(lines_to_clear)
            self.score += [0, 100, 300, 500, 800][len(lines_to_clear)]
            
            # Difficulty scaling
            speed_increases = self.lines_cleared // 5
            self.fall_speed = max(0.1, self.fall_speed_initial - speed_increases * 0.05)
            
            # Reward
            return {1: 1, 2: 3, 3: 6, 4: 10}.get(len(lines_to_clear), 0)
        
        return -0.2 # Penalty for placing block without clearing a line

    def _is_valid_position(self, piece, offset_row=0, offset_col=0):
        shape = piece["shape"]
        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    grid_r = piece["row"] + r + offset_row
                    grid_c = piece["col"] + c + offset_col
                    if not (0 <= grid_c < self.GRID_WIDTH and 0 <= grid_r < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_r, grid_c] != 0:
                        return False
        return True

    def _check_termination(self):
        time_elapsed = pygame.time.get_ticks() - self.start_time
        if time_elapsed >= self.TIME_LIMIT_MS:
            return True
        if self.lines_cleared >= self.LINES_TO_WIN:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines": self.lines_cleared}

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    color_key = self.PIECE_TYPES[int(self.grid[r, c]) - 1]
                    self._draw_cell(c, r, self.COLORS[color_key])

        # Draw ghost piece
        if not self.game_over and self.current_piece:
            ghost_piece = self.current_piece.copy()
            while self._is_valid_position(ghost_piece, offset_row=1):
                ghost_piece["row"] += 1
            self._draw_piece(ghost_piece, ghost=True)

        # Draw current piece
        if not self.game_over and self.current_piece:
            self._draw_piece(self.current_piece)
        
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE), (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE))
        for c in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X + c * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + self.HEIGHT))
        
        # Handle line clear animation
        if self.line_clear_animation:
            next_animation_state = []
            for r, frames in self.line_clear_animation:
                alpha = int(255 * (frames / 15.0))
                flash_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                flash_surface.fill((255, 255, 255, alpha))
                self.screen.blit(flash_surface, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE))
                if frames > 1:
                    next_animation_state.append((r, frames - 1))
            self.line_clear_animation = next_animation_state
            if not self.line_clear_animation: # Animation finished
                # Actually remove rows from grid data
                cleared_rows = [r for r, f in self.line_clear_animation]
                new_grid = np.zeros_like(self.grid)
                new_row = self.GRID_HEIGHT - 1
                for r in range(self.GRID_HEIGHT - 1, -1, -1):
                    if not np.all(self.grid[r,:] != 0):
                        new_grid[new_row,:] = self.grid[r,:]
                        new_row -= 1
                self.grid = new_grid

    def _draw_piece(self, piece, ghost=False):
        shape = piece["shape"]
        color_key = piece["type"]
        color = self.COLORS[color_key]
        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    self._draw_cell(piece["col"] + c, piece["row"] + r, color, ghost)

    def _draw_cell(self, c, r, color, ghost=False):
        x = self.GRID_X + c * self.CELL_SIZE
        y = self.GRID_Y + r * self.CELL_SIZE
        if ghost:
            pygame.draw.rect(self.screen, color, (x, y, self.CELL_SIZE, self.CELL_SIZE), 1)
        else:
            inner_color = tuple(min(255, val + 50) for val in color)
            pygame.draw.rect(self.screen, color, (x, y, self.CELL_SIZE, self.CELL_SIZE))
            pygame.draw.rect(self.screen, inner_color, (x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_large.render(f"{self.score:06d}", True, (255,255,255))
        self.screen.blit(score_text, (30, 30))
        self.screen.blit(score_val, (30, 60))

        # Lines
        lines_text = self.font_large.render(f"LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_large.render(f"{self.lines_cleared}/{self.LINES_TO_WIN}", True, (255,255,255))
        self.screen.blit(lines_text, (30, 110))
        self.screen.blit(lines_val, (30, 140))

        # Timer
        time_elapsed = pygame.time.get_ticks() - self.start_time
        time_left_ratio = max(0, (self.TIME_LIMIT_MS - time_elapsed) / self.TIME_LIMIT_MS)
        timer_bar_width = 180
        
        timer_color = (int(255 * (1 - time_left_ratio)), int(255 * time_left_ratio), 0)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.WIDTH - 210, 30, timer_bar_width, 20))
        pygame.draw.rect(self.screen, timer_color, (self.WIDTH - 210, 30, timer_bar_width * time_left_ratio, 20))
        
        timer_text = self.font_medium.render("TIME", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - 210 - timer_text.get_width() - 10, 30))

        # Next Piece
        next_box_x, next_box_y = self.WIDTH - 180, 80
        next_text = self.font_medium.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (next_box_x, next_box_y))
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (next_box_x, next_box_y + 30, 4 * self.CELL_SIZE, 4 * self.CELL_SIZE))
        if self.next_piece:
            shape = self.next_piece["shape"]
            color = self.COLORS[self.next_piece["type"]]
            for r, row_data in enumerate(shape):
                for c, cell in enumerate(row_data):
                    if cell:
                        x = next_box_x + c * self.CELL_SIZE
                        y = next_box_y + 30 + r * self.CELL_SIZE
                        inner_color = tuple(min(255, val + 50) for val in color)
                        pygame.draw.rect(self.screen, color, (x, y, self.CELL_SIZE, self.CELL_SIZE))
                        pygame.draw.rect(self.screen, inner_color, (x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.lines_cleared >= self.LINES_TO_WIN else "GAME OVER"
            msg_render = self.font_large.render(msg, True, (255, 255, 255))
            self.screen.blit(msg_render, (self.WIDTH // 2 - msg_render.get_width() // 2, self.HEIGHT // 2 - msg_render.get_height() // 2))

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Puzzle Game")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()