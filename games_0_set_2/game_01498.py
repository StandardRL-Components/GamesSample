
# Generated: 2025-08-27T17:20:21.567865
# Source Brief: brief_01498.md
# Brief Index: 1498

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: ←→ to move, ↑ to rotate clockwise, SHIFT to rotate counter-clockwise, "
        "↓ to soft drop, SPACE to hard drop."
    )

    game_description = (
        "A rhythm-based puzzle game. Match falling blocks to clear lines and "
        "reach the target score before time runs out. Blocks pulse to an internal beat."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.FPS = 30
        self.GAME_TIME_LIMIT = 60.0  # seconds
        self.LINE_CLEAR_TARGET = 20
        self.BEAT_INTERVAL = 0.5 # seconds per drop

        # --- Action Mapping ---
        # MultiDiscrete([5, 2, 2])
        # action[0]: 0=none, 1=up(rot_cw), 2=down(soft_drop), 3=left, 4=right
        # action[1]: 0=released, 1=space(hard_drop)
        # action[2]: 0=released, 1=shift(rot_ccw)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WHITE = (255, 255, 255)
        self.PIECE_COLORS = [
            (0, 0, 0),  # 0 is empty
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 80, 255),   # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
            (255, 160, 80),  # Orange
        ]

        # --- Tetromino Shapes ---
        # Shape, Rotations, Color Index
        self.TETROMINOES = {
            'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
            'O': [[[1, 1], [1, 1]]],
            'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
            'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]],
            'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
            'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
            'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]]
        }
        self.PIECE_KEYS = list(self.TETROMINOES.keys())

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = None
        self.lines_cleared = None
        self.time_remaining = None
        self.game_over = None
        self.steps = None
        self.beat_timer = None
        self.pulse_timer = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.line_clear_animation = None
        self.rng = None

        self.grid_render_pos = (
            (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2,
            (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        )

        self.reset()
        self.validate_implementation()

    def _create_new_piece(self):
        shape_key = self.rng.choice(self.PIECE_KEYS)
        return {
            "key": shape_key,
            "shape": self.TETROMINOES[shape_key],
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 2,
            "y": 0,
            "color_idx": self.rng.integers(1, len(self.PIECE_COLORS))
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.current_piece = self._create_new_piece()
        self.next_piece = self._create_new_piece()
        self.score = 0
        self.lines_cleared = 0
        self.time_remaining = self.GAME_TIME_LIMIT
        self.game_over = False
        self.steps = 0
        self.beat_timer = 0.0
        self.pulse_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.line_clear_animation = []

        if not self._is_valid_position(self.current_piece):
            self.game_over = True

        return self._get_observation(), self._get_info()

    def _is_valid_position(self, piece, offset_x=0, offset_y=0):
        shape = piece["shape"][piece["rotation"]]
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = piece["x"] + x + offset_x
                    grid_y = piece["y"] + y + offset_y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False
                    if grid_y >= 0 and self.grid[grid_y, grid_x] != 0:
                        return False
        return True

    def _handle_input(self, movement, space_held, shift_held):
        # --- Movement and Soft Drop ---
        if movement == 3:  # Left
            if self._is_valid_position(self.current_piece, offset_x=-1):
                self.current_piece["x"] -= 1
        elif movement == 4:  # Right
            if self._is_valid_position(self.current_piece, offset_x=1):
                self.current_piece["x"] += 1
        elif movement == 2:  # Down (Soft Drop)
            if self._is_valid_position(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
            else: # If soft drop causes lock, process it immediately
                return self._lock_piece()

        # --- Rotation ---
        rot_cw_pressed = movement == 1
        rot_ccw_pressed = shift_held and not self.prev_shift_held

        if rot_cw_pressed or rot_ccw_pressed:
            original_rotation = self.current_piece["rotation"]
            if rot_cw_pressed:
                self.current_piece["rotation"] = (original_rotation + 1) % len(self.current_piece["shape"])
            elif rot_ccw_pressed:
                self.current_piece["rotation"] = (original_rotation - 1 + len(self.current_piece["shape"])) % len(self.current_piece["shape"])

            if not self._is_valid_position(self.current_piece):
                # Wall kick: try moving left/right
                if self._is_valid_position(self.current_piece, offset_x=1):
                    self.current_piece["x"] += 1
                elif self._is_valid_position(self.current_piece, offset_x=-1):
                    self.current_piece["x"] -= 1
                else: # Kick failed, revert rotation
                    self.current_piece["rotation"] = original_rotation

        # --- Hard Drop ---
        hard_drop_pressed = space_held and not self.prev_space_held
        if hard_drop_pressed:
            # sfx: hard_drop_sound
            while self._is_valid_position(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
            return self._lock_piece()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return 0.0

    def _lock_piece(self):
        # sfx: piece_lock_sound
        shape = self.current_piece["shape"][self.current_piece["rotation"]]
        piece_y_coords = []
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece["x"] + x
                    grid_y = self.current_piece["y"] + y
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_piece["color_idx"]
                        piece_y_coords.append(grid_y)

        # --- Calculate Placement Reward ---
        reward = 0.1  # Base reward for placing a block
        
        # Risk/Safety reward
        max_height = 0
        if piece_y_coords:
            max_height = self.GRID_HEIGHT - min(piece_y_coords)
        if max_height > self.GRID_HEIGHT * 0.75:
            reward -= 0.2 # Risky placement
        
        empty_cols = sum(1 for c in range(self.GRID_WIDTH) if np.all(self.grid[:, c] == 0))
        if empty_cols >= 2:
            reward += 1.0 # Safe placement

        # --- Line Clearing ---
        cleared, num_cleared = self._check_and_clear_lines()
        if num_cleared > 0:
            # sfx: line_clear_sound
            self.lines_cleared += num_cleared
            reward += num_cleared * 1.0
            if num_cleared > 1:
                reward += 5.0 # Bonus for multi-line clear
            self.score += (10 * num_cleared) ** 2

        # --- Spawn Next Piece ---
        self.current_piece = self.next_piece
        self.next_piece = self._create_new_piece()

        # --- Check for Game Over ---
        if not self._is_valid_position(self.current_piece):
            self.game_over = True
            # sfx: game_over_sound

        return reward

    def _check_and_clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y, :] != 0):
                lines_to_clear.append(y)

        if not lines_to_clear:
            return False, 0

        for y in lines_to_clear:
            self.line_clear_animation.append({"y": y, "timer": 5})

        # Shift grid down
        new_grid = np.zeros_like(self.grid)
        new_y = self.GRID_HEIGHT - 1
        for y in range(self.GRID_HEIGHT - 1, -1, -1):
            if y not in lines_to_clear:
                new_grid[new_y, :] = self.grid[y, :]
                new_y -= 1
        self.grid = new_grid
        return True, len(lines_to_clear)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), -100.0, True, False, self._get_info()

        self.steps += 1
        self.pulse_timer += 1
        self.time_remaining -= 1.0 / self.FPS
        reward = 0.0

        # Update line clear animations
        self.line_clear_animation = [anim for anim in self.line_clear_animation if anim["timer"] > 0]
        for anim in self.line_clear_animation:
            anim["timer"] -= 1

        # --- Handle player input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held, shift_held)

        if self.game_over: # Hard drop or soft drop could end the game
            return self._get_observation(), reward - 100.0, True, False, self._get_info()

        # --- Automatic Drop on Beat ---
        self.beat_timer += 1.0 / self.FPS
        if self.beat_timer >= self.BEAT_INTERVAL:
            self.beat_timer = 0
            if self._is_valid_position(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
            else:
                reward += self._lock_piece()

        # --- Check Termination Conditions ---
        terminated = False
        if self.game_over:
            reward -= 100.0
            terminated = True
        elif self.lines_cleared >= self.LINE_CLEAR_TARGET:
            reward += 100.0
            terminated = True
            # sfx: victory_sound
        elif self.time_remaining <= 0:
            reward -= 100.0
            terminated = True
        elif self.steps >= 1800: # Max steps (60s * 30fps)
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        gx, gy = self.grid_render_pos
        cs = self.CELL_SIZE

        # Draw grid background and border
        pygame.draw.rect(self.screen, self.COLOR_GRID, (gx - 5, gy - 5, self.GRID_WIDTH * cs + 10, self.GRID_HEIGHT * cs + 10))
        pygame.draw.rect(self.screen, self.COLOR_BG, (gx, gy, self.GRID_WIDTH * cs, self.GRID_HEIGHT * cs))

        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (gx + x * cs, gy), (gx + x * cs, gy + self.GRID_HEIGHT * cs))
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (gx, gy + y * cs), (gx + self.GRID_WIDTH * cs, gy + y * cs))

        # Draw locked pieces
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    color = self.PIECE_COLORS[self.grid[y, x]]
                    pygame.draw.rect(self.screen, color, (gx + x * cs + 1, gy + y * cs + 1, cs - 2, cs - 2))

        # Draw current piece
        if self.current_piece and not self.game_over:
            shape = self.current_piece["shape"][self.current_piece["rotation"]]
            color = self.PIECE_COLORS[self.current_piece["color_idx"]]
            pulse_val = (math.sin(self.pulse_timer * 0.2) + 1) / 2 # 0 to 1
            outline_color = tuple(min(255, c + 80 * pulse_val) for c in color)
            
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        px = gx + (self.current_piece["x"] + x) * cs
                        py = gy + (self.current_piece["y"] + y) * cs
                        pygame.draw.rect(self.screen, outline_color, (px, py, cs, cs))
                        pygame.draw.rect(self.screen, color, (px + 2, py + 2, cs - 4, cs - 4))
        
        # Draw line clear animation
        for anim in self.line_clear_animation:
            flash_alpha = 150 + (anim['timer'] / 5) * 105
            flash_surface = pygame.Surface((self.GRID_WIDTH * cs, cs), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, flash_alpha))
            self.screen.blit(flash_surface, (gx, gy + anim['y'] * cs))

    def _render_ui(self):
        # --- Helper to draw text with shadow ---
        def draw_text(text, font, color, pos, shadow_color=(0,0,0)):
            text_surface = font.render(text, True, color)
            shadow_surface = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surface, pos)

        # --- Lines Cleared ---
        lines_text = f"LINES: {self.lines_cleared}/{self.LINE_CLEAR_TARGET}"
        draw_text(lines_text, self.font_main, self.COLOR_UI_TEXT, (20, 20))
        
        # --- Score ---
        score_text = f"SCORE: {self.score}"
        text_width = self.font_main.size(score_text)[0]
        draw_text(score_text, self.font_main, self.COLOR_UI_TEXT, (self.WIDTH - text_width - 20, 20))

        # --- Time Remaining ---
        time_text = f"TIME: {max(0, int(self.time_remaining))}"
        text_width = self.font_main.size(time_text)[0]
        draw_text(time_text, self.font_main, self.COLOR_UI_TEXT, ((self.WIDTH - text_width) // 2, self.HEIGHT - 40))
        
        # --- Next Piece ---
        next_box_x = self.grid_render_pos[0] + self.GRID_WIDTH * self.CELL_SIZE + 20
        next_box_y = self.grid_render_pos[1]
        draw_text("NEXT", self.font_ui, self.COLOR_UI_TEXT, (next_box_x, next_box_y))
        pygame.draw.rect(self.screen, self.COLOR_GRID, (next_box_x - 5, next_box_y + 25, 5 * self.CELL_SIZE, 5 * self.CELL_SIZE), border_radius=5)
        
        if self.next_piece:
            shape = self.next_piece["shape"][0] # Use default rotation
            color = self.PIECE_COLORS[self.next_piece["color_idx"]]
            w = len(shape[0])
            h = len(shape)
            start_x = next_box_x + (5 * self.CELL_SIZE - w * self.CELL_SIZE) / 2
            start_y = next_box_y + 25 + (5 * self.CELL_SIZE - h * self.CELL_SIZE) / 2
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(self.screen, color, (start_x + x * self.CELL_SIZE, start_y + y * self.CELL_SIZE, self.CELL_SIZE - 2, self.CELL_SIZE - 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "time_remaining": self.time_remaining,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Setup ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    # --- Game Loop for Manual Play ---
    while not terminated:
        # --- Action Mapping for Keyboard ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to Display ---
        # The observation is already a rendered frame, just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Handle Pygame Events and Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)

        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Optional: Add a delay before reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()