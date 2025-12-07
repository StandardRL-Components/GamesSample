import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Tetris-like puzzle game where the player drops space debris to clear lines.
    The goal is to clear 15 lines without letting the debris stack to the top.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to rotate. Hold space for soft drop, press shift for hard drop."
    )

    game_description = (
        "Rotate and drop falling space debris to fill lines and clear the orbit in this top-down puzzle game."
    )

    auto_advance = True

    # --- Game Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    CELL_SIZE = 18
    WIN_CONDITION_LINES = 15
    MAX_STEPS = 3000

    # --- Visuals ---
    SIDE_PANEL_WIDTH = 180
    GAME_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    GAME_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    # FIX: Set screen dimensions to match the required observation space
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 65)
    COLOR_PANEL = (20, 25, 45)
    COLOR_TEXT = (220, 230, 255)
    COLOR_DANGER = (180, 50, 50, 50) # RGBA for transparency
    
    PIECE_COLORS = [
        (0, 255, 255),  # I (Cyan)
        (0, 0, 255),    # J (Blue)
        (255, 165, 0),  # L (Orange)
        (255, 255, 0),  # O (Yellow)
        (0, 255, 0),    # S (Green)
    ]

    SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 0, 0], [1, 1, 1]],  # J
        [[0, 0, 1], [1, 1, 1]],  # L
        [[1, 1], [1, 1]],  # O
        [[0, 1, 1], [1, 1, 0]],  # S
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # FIX: Observation space now uses the corrected SCREEN_WIDTH and SCREEN_HEIGHT
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        self.game_area_x = (self.SCREEN_WIDTH - self.GAME_AREA_WIDTH - self.SIDE_PANEL_WIDTH) // 2
        self.game_area_y = (self.SCREEN_HEIGHT - self.GAME_AREA_HEIGHT) // 2

        # FIX: Define game_rect as an instance attribute to be accessible in render methods
        self.game_rect = pygame.Rect(self.game_area_x, self.game_area_y, self.GAME_AREA_WIDTH, self.GAME_AREA_HEIGHT)

        self.grid = None
        self.current_piece = None
        self.next_piece_shape_idx = None
        self.steps = None
        self.score = None
        self.lines_cleared = None
        self.game_over = None
        self.game_won = None
        self.fall_speed = None
        self.fall_progress = None
        
        self.line_clear_animation = None # {"rows": [], "timer": 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.game_won = False
        
        self.fall_speed = 0.5 # Cells per frame
        self.fall_progress = 0.0

        self.line_clear_animation = None
        
        self._spawn_new_piece()
        self._spawn_new_piece() # First call sets next, second sets current

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small penalty per step to encourage speed

        if self.line_clear_animation:
            self.line_clear_animation["timer"] -= 1
            if self.line_clear_animation["timer"] <= 0:
                self._finish_line_clear()
            # No other actions while clearing lines
        elif not self.game_over and not self.game_won:
            # --- Handle Actions ---
            # 1. Hard Drop (Shift) - overrides all other movement for the step
            if shift_held:
                # sfx: hard_drop_sound
                drop_distance = 0
                while not self._check_collision(self.current_piece, (0, drop_distance + 1)):
                    drop_distance += 1
                self.current_piece["y"] += drop_distance
                self._lock_piece()
                reward += 0.1 # Small reward for decisive action

            else:
                # 2. Rotation and Movement
                if movement == 1: # Up -> Rotate
                    self._rotate_piece()
                elif movement == 3: # Left
                    self._move_piece(-1)
                elif movement == 4: # Right
                    self._move_piece(1)
                
                # 3. Falling
                current_fall_speed = self.fall_speed * 3 if space_held else self.fall_speed
                self.fall_progress += current_fall_speed
                
                if self.fall_progress >= 1.0:
                    fall_steps = int(self.fall_progress)
                    self.fall_progress %= 1.0
                    
                    for _ in range(fall_steps):
                        if not self._check_collision(self.current_piece, (0, 1)):
                            self.current_piece["y"] += 1
                        else:
                            self._lock_piece()
                            break # Stop falling if locked

        self.steps += 1
        
        # --- Check Termination ---
        terminated = self.game_over or self.game_won
        truncated = self.steps >= self.MAX_STEPS
        
        if self.game_won:
            reward += 100
        elif self.game_over:
            reward -= 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_new_piece(self):
        if self.next_piece_shape_idx is None:
            self.next_piece_shape_idx = self.np_random.integers(0, len(self.SHAPES))

        shape_idx = self.next_piece_shape_idx
        shape = self.SHAPES[shape_idx]
        
        self.current_piece = {
            "shape_idx": shape_idx,
            "shape": shape,
            "x": self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            "y": 0,
            "color": self.PIECE_COLORS[shape_idx]
        }
        
        self.next_piece_shape_idx = self.np_random.integers(0, len(self.SHAPES))

        # Check for game over
        if self._check_collision(self.current_piece, (0, 0)):
            self.game_over = True
            # sfx: game_over_sound

    def _lock_piece(self):
        # sfx: piece_lock_sound
        shape = self.current_piece["shape"]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece["x"] + c
                    grid_y = self.current_piece["y"] + r
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_piece["shape_idx"] + 1 # Use 1-based index for color
        
        self._check_line_clears()
        if not self.line_clear_animation: # If no lines were cleared
            self._spawn_new_piece()

    def _check_line_clears(self):
        full_rows = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] > 0):
                full_rows.append(r)

        if full_rows:
            # sfx: line_clear_sound
            self.line_clear_animation = {"rows": full_rows, "timer": 10} # 10 frames of animation
            lines_cleared_this_turn = len(full_rows)
            self.lines_cleared += lines_cleared_this_turn
            self.score += (10 * lines_cleared_this_turn) ** 2 # Bonus for multi-line clears

            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                self.game_won = True
                # sfx: game_win_sound
    
    def _finish_line_clear(self):
        rows_to_clear = self.line_clear_animation["rows"]
        # Remove rows from bottom to top to preserve indices
        for row_idx in sorted(rows_to_clear, reverse=True):
            self.grid = np.delete(self.grid, row_idx, axis=0)
        
        # Add new empty rows at the top
        new_rows = np.zeros((len(rows_to_clear), self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack((new_rows, self.grid))

        # Update difficulty
        if self.lines_cleared // 3 > (self.lines_cleared - len(rows_to_clear)) // 3:
            self.fall_speed += 0.05
        
        self.line_clear_animation = None
        self._spawn_new_piece()

    def _check_collision(self, piece, offset):
        off_x, off_y = offset
        shape = piece["shape"]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = piece["x"] + c + off_x
                    grid_y = piece["y"] + r + off_y
                    
                    if not (0 <= grid_x < self.GRID_WIDTH): return True # Wall collision
                    if not (grid_y < self.GRID_HEIGHT): return True # Floor collision
                    if grid_y < 0: continue # Above screen is fine

                    if self.grid[grid_y, grid_x] > 0: return True # Debris collision
        return False

    def _move_piece(self, dx):
        if not self._check_collision(self.current_piece, (dx, 0)):
            self.current_piece["x"] += dx
            # sfx: move_sideways_sound

    def _rotate_piece(self):
        # sfx: rotate_sound
        original_shape = self.current_piece["shape"]
        rotated_shape = list(zip(*original_shape[::-1]))
        
        # Create a temporary piece to check for collision
        temp_piece = self.current_piece.copy()
        temp_piece["shape"] = rotated_shape

        if not self._check_collision(temp_piece, (0, 0)):
            self.current_piece["shape"] = rotated_shape
        else: # Try wall kicks
            for kick_x in [-1, 1, -2, 2]:
                if not self._check_collision(temp_piece, (kick_x, 0)):
                    self.current_piece["shape"] = rotated_shape
                    self.current_piece["x"] += kick_x
                    return

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "game_over": self.game_over,
            "game_won": self.game_won
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw game area border and grid lines
        # FIX: Use self.game_rect defined in __init__
        pygame.draw.rect(self.screen, self.COLOR_GRID, self.game_rect)
        for i in range(1, self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_BG, 
                             (self.game_area_x + i * self.CELL_SIZE, self.game_area_y), 
                             (self.game_area_x + i * self.CELL_SIZE, self.game_area_y + self.GAME_AREA_HEIGHT))
        for i in range(1, self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_BG, 
                             (self.game_area_x, self.game_area_y + i * self.CELL_SIZE), 
                             (self.game_area_x + self.GAME_AREA_WIDTH, self.game_area_y + i * self.CELL_SIZE))
        
        # Draw danger zone
        danger_surface = pygame.Surface((self.GAME_AREA_WIDTH, self.CELL_SIZE * 2), pygame.SRCALPHA)
        danger_surface.fill(self.COLOR_DANGER)
        self.screen.blit(danger_surface, (self.game_area_x, self.game_area_y))

        # Draw locked debris
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    color_idx = int(self.grid[r, c] - 1)
                    self._draw_cell(c, r, self.PIECE_COLORS[color_idx])

        # Draw line clear animation
        if self.line_clear_animation:
            flash_alpha = abs(5 - self.line_clear_animation["timer"]) * 50 # Pulsing effect
            flash_color = (255, 255, 255, flash_alpha)
            for r in self.line_clear_animation["rows"]:
                flash_surface = pygame.Surface((self.GAME_AREA_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
                flash_surface.fill(flash_color)
                self.screen.blit(flash_surface, (self.game_area_x, self.game_area_y + r * self.CELL_SIZE))
        else:
            # Draw current piece
            if self.current_piece and not self.game_over:
                shape = self.current_piece["shape"]
                for r, row in enumerate(shape):
                    for c, cell in enumerate(row):
                        if cell:
                            self._draw_cell(self.current_piece["x"] + c, self.current_piece["y"] + r, self.current_piece["color"])

    def _draw_cell(self, grid_x, grid_y, color):
        px = self.game_area_x + grid_x * self.CELL_SIZE
        py = self.game_area_y + grid_y * self.CELL_SIZE
        
        main_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        # Main color
        pygame.draw.rect(self.screen, color, main_rect)
        
        # 3D-ish effect
        highlight_color = tuple(min(255, c + 40) for c in color)
        shadow_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.line(self.screen, highlight_color, (px, py), (px + self.CELL_SIZE - 1, py), 2)
        pygame.draw.line(self.screen, highlight_color, (px, py), (px, py + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, shadow_color, (px + self.CELL_SIZE - 1, py), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, shadow_color, (px, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 2)

    def _render_ui(self):
        panel_x = self.game_area_x + self.GAME_AREA_WIDTH + 20
        
        # Score
        score_text = self.font_main.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (panel_x, self.game_area_y + 10))
        score_val = self.font_main.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (panel_x, self.game_area_y + 40))

        # Lines
        lines_text = self.font_main.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (panel_x, self.game_area_y + 90))
        lines_val = self.font_main.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_val, (panel_x, self.game_area_y + 120))

        # Next Piece Preview
        next_text = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (panel_x, self.game_area_y + 170))
        
        preview_box_rect = pygame.Rect(panel_x, self.game_area_y + 200, 4 * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box_rect)

        if self.next_piece_shape_idx is not None:
            shape = self.SHAPES[self.next_piece_shape_idx]
            color = self.PIECE_COLORS[self.next_piece_shape_idx]
            shape_w = len(shape[0]) * self.CELL_SIZE
            shape_h = len(shape) * self.CELL_SIZE
            
            start_x = preview_box_rect.centerx - shape_w // 2
            start_y = preview_box_rect.centery - shape_h // 2

            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        px = start_x + c * self.CELL_SIZE
                        py = start_y + r * self.CELL_SIZE
                        pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE-2, self.CELL_SIZE-2))

        # Game Over / Win Message
        if self.game_over or self.game_won:
            overlay = pygame.Surface((self.GAME_AREA_WIDTH, self.GAME_AREA_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (self.game_area_x, self.game_area_y))
            
            message = "MISSION COMPLETE" if self.game_won else "ORBITAL DEBRIS OVERLOAD"
            msg_render = self.font_main.render(message, True, (255, 255, 100))
            # FIX: Use self.game_rect to center the message
            msg_rect = msg_render.get_rect(center=self.game_rect.center)
            self.screen.blit(msg_render, msg_rect)


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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # --- Manual Play ---
    # Unset the dummy video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for display
    pygame.display.set_caption("Orbital Debris Cleaner")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        # No down action mapping
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        truncated = trunc
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds before closing

    env.close()