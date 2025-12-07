
# Generated: 2025-08-27T13:43:48.145510
# Source Brief: brief_00464.md
# Brief Index: 464

        
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

    user_guide = (
        "Controls: ←→ to move, ↑ to rotate, ↓ to soft drop. "
        "Space to hard drop. Shift to hold piece."
    )

    game_description = (
        "A fast-paced, grid-based puzzle game. Manipulate falling blocks to clear "
        "lines and achieve a target score before the blocks stack too high."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GHOST = (255, 255, 255, 50)
    COLOR_WHITE = (255, 255, 255)
    
    PIECE_COLORS = [
        (0, 0, 0),          # 0: Empty
        (0, 240, 240),      # 1: I (Cyan)
        (240, 240, 0),      # 2: O (Yellow)
        (160, 0, 240),      # 3: T (Purple)
        (0, 0, 240),        # 4: J (Blue)
        (240, 160, 0),      # 5: L (Orange)
        (0, 240, 0),        # 6: S (Green)
        (240, 0, 0),        # 7: Z (Red)
    ]

    PIECE_SHAPES = {
        1: [[1, 1, 1, 1]], # I
        2: [[1, 1], [1, 1]], # O
        3: [[0, 1, 0], [1, 1, 1]], # T
        4: [[1, 0, 0], [1, 1, 1]], # J
        5: [[0, 0, 1], [1, 1, 1]], # L
        6: [[0, 1, 1], [1, 1, 0]], # S
        7: [[1, 1, 0], [0, 1, 1]]  # Z
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.render_mode = render_mode
        self.game_over = False
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.held_piece = None
        self.can_hold = True
        self.score = 0
        self.lines_cleared = 0
        self.fall_speed = 0
        self.fall_progress = 0
        self.steps = 0
        self.max_steps = 10000
        self.reward_this_step = 0
        self.line_clear_animation = None
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.prev_up_held = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        self.piece_bag = list(range(1, 8))
        random.shuffle(self.piece_bag)

        self.current_piece = self._get_new_piece()
        self.next_piece = self._get_new_piece()
        self.held_piece = None
        
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.can_hold = True
        
        self.fall_speed = 1.0 / 30.0 # 1 cell per second at 30fps
        self.fall_progress = 0.0

        self.line_clear_animation = None

        self.prev_space_held = True
        self.prev_shift_held = True
        self.prev_up_held = True
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = -0.01 # Small penalty for time passing
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        up_held = movement == 1
        
        # --- Handle one-shot actions on rising edge ---
        if up_held and not self.prev_up_held:
            self._rotate_piece()
        if space_held and not self.prev_space_held:
            self._hard_drop()
        if shift_held and not self.prev_shift_held:
            self._hold_piece()

        # --- Handle continuous actions ---
        if movement == 3: # Left
            self._move_piece(-1, 0)
        elif movement == 4: # Right
            self._move_piece(1, 0)
        
        # --- Update game physics ---
        soft_drop_multiplier = 5.0 if movement == 2 else 1.0
        self.fall_progress += self.fall_speed * soft_drop_multiplier

        if self.fall_progress >= 1.0:
            self.fall_progress = 0
            if not self._move_piece(0, 1): # If move down fails, lock piece
                self._lock_piece()

        # --- Update previous action states for next step ---
        self.prev_up_held = up_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        self.steps += 1
        reward = self.reward_this_step
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "lines_cleared": self.lines_cleared, "steps": self.steps}

    def _check_termination(self):
        if self.game_over:
            self.reward_this_step -= 100
            return True
        if self.lines_cleared >= 10:
            self.reward_this_step += 100
            self.game_over = True
            return True
        if self.steps >= self.max_steps:
            return True
        return False

    # --- Game Logic ---
    
    def _get_new_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(1, 8))
            random.shuffle(self.piece_bag)
        shape_type = self.piece_bag.pop()
        shape = self.PIECE_SHAPES[shape_type]
        
        piece = {
            "type": shape_type,
            "shape": np.array(shape),
            "x": self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            "y": 0,
            "color": self.PIECE_COLORS[shape_type]
        }
        
        # If new piece spawns in an invalid position, game over
        if not self._is_valid_position(piece):
            self.game_over = True
        
        return piece

    def _move_piece(self, dx, dy):
        if self._is_valid_position(self.current_piece, offset=(dx, dy)):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def _rotate_piece(self):
        # # Sound effect placeholder
        # pygame.mixer.Sound.play(rotate_sound)
        
        original_shape = self.current_piece['shape']
        rotated = np.rot90(original_shape, k=-1) # Rotate clockwise
        
        # Wall kick logic
        test_offsets = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
        
        for dx, dy in test_offsets:
            temp_piece = self.current_piece.copy()
            temp_piece['shape'] = rotated
            if self._is_valid_position(temp_piece, offset=(dx, dy)):
                self.current_piece['shape'] = rotated
                self.current_piece['x'] += dx
                self.current_piece['y'] += dy
                return
    
    def _hard_drop(self):
        # # Sound effect placeholder
        # pygame.mixer.Sound.play(hard_drop_sound)
        
        dy = 0
        while self._is_valid_position(self.current_piece, offset=(0, dy + 1)):
            dy += 1
        self.current_piece['y'] += dy
        self.fall_progress = 1.0 # Force lock on next physics update
        self._lock_piece()

    def _hold_piece(self):
        if not self.can_hold:
            return
        
        # # Sound effect placeholder
        # pygame.mixer.Sound.play(hold_sound)
        
        if self.held_piece is None:
            self.held_piece = self.current_piece
            self.current_piece = self._get_new_piece()
        else:
            self.held_piece, self.current_piece = self.current_piece, self.held_piece
            self.current_piece['x'] = self.GRID_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
            self.current_piece['y'] = 0

        self.can_hold = False

    def _is_valid_position(self, piece, offset=(0, 0)):
        px, py = piece['x'] + offset[0], piece['y'] + offset[1]
        shape = piece['shape']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = px + x, py + y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_x, grid_y] != 0:
                        return False
        return True

    def _lock_piece(self):
        # # Sound effect placeholder
        # pygame.mixer.Sound.play(lock_sound)
        
        shape = self.current_piece['shape']
        px, py = self.current_piece['x'], self.current_piece['y']
        color_idx = self.current_piece['type']

        # Heuristic for bad placement: count holes created
        holes_created = 0
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = px + x, py + y
                    if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                        self.grid[grid_x, grid_y] = color_idx
                        # Check for empty space directly below
                        if grid_y + 1 < self.GRID_HEIGHT and self.grid[grid_x, grid_y + 1] == 0:
                            holes_created += 1
        
        if holes_created > 2: # Penalize creating significant gaps
             self.reward_this_step -= 0.2

        lines = self._clear_lines()
        if lines > 0:
            # # Sound effect placeholder
            # pygame.mixer.Sound.play(line_clear_sound)
            self.lines_cleared += lines
            line_rewards = {1: 1, 2: 2, 3: 4, 4: 8} # Exponential reward for multi-line clears
            self.reward_this_step += line_rewards.get(lines, 0)
            self.score += line_rewards.get(lines, 0) * 100
            
            # Increase fall speed every 2 lines
            self.fall_speed = (1.0 + 0.5 * (self.lines_cleared // 2)) / 30.0

        self.current_piece = self.next_piece
        self.next_piece = self._get_new_piece()
        self.can_hold = True
        self.fall_progress = 0

    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[:, y] != 0):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            self.line_clear_animation = {"rows": lines_to_clear, "timer": 5} # 5 frames of animation
            for y in lines_to_clear:
                self.grid[:, y] = 0 # Clear the line
            
            # Shift blocks down
            lines_cleared_count = len(lines_to_clear)
            for y in sorted(lines_to_clear, reverse=False):
                for row_y in range(y, 0, -1):
                    self.grid[:, row_y] = self.grid[:, row_y - 1]
                self.grid[:, 0] = 0

            return lines_cleared_count
        return 0

    # --- Rendering ---

    def _render_game(self):
        self._draw_grid()
        self._draw_static_blocks()
        if not self.game_over:
            self._draw_ghost_piece()
            self._draw_piece(self.current_piece)
        if self.line_clear_animation:
            self._draw_line_clear_animation()

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
    
    def _draw_block(self, screen_x, screen_y, color, alpha=255):
        rect = pygame.Rect(screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Create a beveled effect
        light_color = tuple(min(255, c + 40) for c in color)
        dark_color = tuple(max(0, c - 40) for c in color)

        if alpha < 255:
            surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            surface.fill((*color, alpha))
            pygame.draw.rect(surface, (*light_color, alpha), (0, 0, self.CELL_SIZE - 1, self.CELL_SIZE - 1), 2)
            self.screen.blit(surface, rect.topleft)
        else:
            pygame.draw.rect(self.screen, dark_color, rect)
            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))
            pygame.draw.rect(self.screen, light_color, rect.inflate(-6, -6))


    def _draw_static_blocks(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] != 0:
                    color = self.PIECE_COLORS[self.grid[x, y]]
                    screen_x = self.GRID_X_OFFSET + x * self.CELL_SIZE
                    screen_y = self.GRID_Y_OFFSET + y * self.CELL_SIZE
                    self._draw_block(screen_x, screen_y, color)

    def _draw_piece(self, piece, offset=(0, 0), is_ghost=False):
        if piece is None: return
        shape = piece['shape']
        px, py = piece['x'] + offset[0], piece['y'] + offset[1]
        color = piece['color']
        alpha = 60 if is_ghost else 255

        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    screen_x = self.GRID_X_OFFSET + (px + x) * self.CELL_SIZE
                    screen_y = self.GRID_Y_OFFSET + (py + y) * self.CELL_SIZE
                    self._draw_block(screen_x, screen_y, color, alpha)

    def _draw_ghost_piece(self):
        ghost_piece = self.current_piece.copy()
        dy = 0
        while self._is_valid_position(ghost_piece, offset=(0, dy + 1)):
            dy += 1
        self._draw_piece(ghost_piece, offset=(0, dy), is_ghost=True)

    def _draw_line_clear_animation(self):
        if self.line_clear_animation and self.line_clear_animation["timer"] > 0:
            for row_y in self.line_clear_animation["rows"]:
                rect = pygame.Rect(
                    self.GRID_X_OFFSET,
                    self.GRID_Y_OFFSET + row_y * self.CELL_SIZE,
                    self.GRID_WIDTH * self.CELL_SIZE,
                    self.CELL_SIZE
                )
                
                # Flash effect
                alpha = 255 * (self.line_clear_animation["timer"] / 5.0)
                flash_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
                flash_surface.fill((255, 255, 255, alpha))
                self.screen.blit(flash_surface, rect.topleft)

            self.line_clear_animation["timer"] -= 1
            if self.line_clear_animation["timer"] <= 0:
                self.line_clear_animation = None

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_large.render(f"{self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (40, 40))
        self.screen.blit(score_val, (40, 60))

        # Lines
        lines_text = self.font_small.render(f"LINES", True, self.COLOR_TEXT)
        lines_val = self.font_large.render(f"{self.lines_cleared} / 10", True, self.COLOR_WHITE)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - 120, 40))
        self.screen.blit(lines_val, (self.SCREEN_WIDTH - 120, 60))

        # Next Piece Preview
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE + 30, 120))
        if self.next_piece:
            self._draw_ui_piece(self.next_piece, self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE + 40, 150)
        
        # Held Piece Preview
        hold_text = self.font_small.render("HOLD", True, self.COLOR_TEXT)
        self.screen.blit(hold_text, (self.GRID_X_OFFSET - 100, 120))
        if self.held_piece:
            self._draw_ui_piece(self.held_piece, self.GRID_X_OFFSET - 90, 150)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self.lines_cleared >= 10 else "GAME OVER"
            text_surf = self.font_large.render(end_text, True, self.COLOR_WHITE)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_ui_piece(self, piece, screen_x, screen_y):
        shape = piece['shape']
        color = piece['color']
        ui_cell_size = 12
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(
                        screen_x + x * ui_cell_size,
                        screen_y + y * ui_cell_size,
                        ui_cell_size,
                        ui_cell_size
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Puzzle Blocks")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'r' to reset
            waiting_for_reset = True
            while waiting_for_reset:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        waiting_for_reset = False
                 clock.tick(30)

        clock.tick(30) # Run at 30 FPS

    env.close()