
# Generated: 2025-08-27T16:23:20.511977
# Source Brief: brief_01208.md
# Brief Index: 1208

        
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
    """
    A fast-paced, grid-based falling block puzzle game where strategic placement
    and risk-taking are rewarded. This environment is designed for visual quality
    and an engaging arcade gameplay experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↓ for soft drop. Space to rotate, Shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based falling block puzzle game. Clear lines to score points "
        "before the stack reaches the top."
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
        self.rng = np.random.default_rng()

        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (640 - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (400 - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.MAX_STEPS = 10000
        self.WIN_SCORE = 1000

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.PIECE_COLORS = [
            (0, 0, 0),         # 0 is empty
            (255, 87, 87),     # I (Red)
            (87, 138, 255),    # J (Blue)
            (255, 165, 0),     # L (Orange)
            (255, 255, 87),    # O (Yellow)
            (87, 255, 87),     # S (Green)
            (176, 87, 255),    # T (Purple)
            (87, 255, 255)     # Z (Cyan)
        ]

        # Tetromino shapes with rotations
        self.PIECES = [
            [[[1,1,1,1]], [[1],[1],[1],[1]]], # I
            [[[1,0,0],[1,1,1]], [[1,1],[1,0],[1,0]], [[1,1,1],[0,0,1]], [[0,1],[0,1],[1,1]]], # J
            [[[0,0,1],[1,1,1]], [[1,0],[1,0],[1,1]], [[1,1,1],[1,0,0]], [[1,1],[0,1],[0,1]]], # L
            [[[1,1],[1,1]]], # O
            [[[0,1,1],[1,1,0]], [[1,0],[1,1],[0,1]]], # S
            [[[0,1,0],[1,1,1]], [[1,0],[1,1],[1,0]], [[1,1,1],[0,1,0]], [[0,1],[1,1],[0,1]]], # T
            [[[1,1,0],[0,1,1]], [[0,1],[1,1],[1,0]]]  # Z
        ]

        # Font
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # State variables are initialized in reset()
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_time = 0
        self.fall_speed = 1000
        self.last_space_held = False
        self.last_shift_held = False
        self.line_clear_animation = []

        self.reset()
        self.validate_implementation()

    def _new_piece(self):
        piece_type = self.rng.integers(0, len(self.PIECES))
        return {
            "type": piece_type,
            "shape_idx": piece_type + 1,
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 2,
            "y": 0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_time = 0
        self.fall_speed = 1000  # ms per grid cell
        self.last_space_held = False
        self.last_shift_held = False
        self.line_clear_animation = []

        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()

        if self._check_collision(self.current_piece):
            self.game_over = True

        return self._get_observation(), self._get_info()

    def _get_shape(self, piece):
        shapes = self.PIECES[piece["type"]]
        return shapes[piece["rotation"] % len(shapes)]

    def _check_collision(self, piece):
        shape = self._get_shape(piece)
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    px, py = piece["x"] + x, piece["y"] + y
                    if not (0 <= px < self.GRID_WIDTH and py < self.GRID_HEIGHT):
                        return True
                    if py >= 0 and self.grid[px, py] != 0:
                        return True
        return False

    def _lock_piece(self):
        shape = self._get_shape(self.current_piece)
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    px, py = self.current_piece["x"] + x, self.current_piece["y"] + y
                    if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
                        self.grid[px, py] = self.current_piece["shape_idx"]
        # Sound placeholder: lock piece sfx

    def _clear_lines(self):
        lines_cleared = 0
        full_rows = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[:, y] != 0):
                full_rows.append(y)
                lines_cleared += 1

        if lines_cleared > 0:
            # Sound placeholder: line clear sfx
            for y in sorted(full_rows, reverse=True):
                self.grid[:, 1:y+1] = self.grid[:, 0:y]
                self.grid[:, 0] = 0
                self.line_clear_animation.append({"y": y, "timer": 10})

        return lines_cleared

    def step(self, action):
        reward = 0
        self.steps += 1
        delta_time = 33  # Assume ~30 FPS

        if self.game_over:
            return self._get_observation(), -100, True, False, self._get_info()

        # 1. Unpack actions and handle input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        rotate_action = space_held and not self.last_space_held
        hard_drop_action = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # 2. Process actions
        if hard_drop_action:
            # Sound placeholder: hard drop sfx
            while not self._check_collision({"x": self.current_piece["x"], "y": self.current_piece["y"] + 1, **self.current_piece}):
                self.current_piece["y"] += 1
            self.fall_time = self.fall_speed # Force lock
        else:
            if movement == 3:  # Left
                self.current_piece["x"] -= 1
                if self._check_collision(self.current_piece): self.current_piece["x"] += 1
            elif movement == 4:  # Right
                self.current_piece["x"] += 1
                if self._check_collision(self.current_piece): self.current_piece["x"] -= 1

            if rotate_action:
                # Sound placeholder: rotate sfx
                original_rotation = self.current_piece["rotation"]
                self.current_piece["rotation"] += 1
                if self._check_collision(self.current_piece): # Basic wall kick
                    original_x = self.current_piece["x"]
                    for offset in [-1, 1, -2, 2]:
                        self.current_piece["x"] = original_x + offset
                        if not self._check_collision(self.current_piece): break
                    else: # If no kick works, revert
                        self.current_piece["x"] = original_x
                        self.current_piece["rotation"] = original_rotation

        # 3. Apply gravity / soft drop
        self.fall_time += delta_time
        if movement == 2:  # Soft drop
            self.fall_time += 100

        if self.fall_time >= self.fall_speed:
            self.fall_time = 0
            self.current_piece["y"] += 1
            if self._check_collision(self.current_piece):
                self.current_piece["y"] -= 1
                self._lock_piece()
                
                # Check for game over if piece locked above screen
                if self.current_piece["y"] < 0:
                    self.game_over = True
                
                lines_cleared = self._clear_lines()
                
                reward += 0.1 # Place piece reward
                if lines_cleared == 0:
                    reward -= 0.2 # Penalty for not clearing lines

                line_rewards = {1: 1, 2: 3, 3: 6, 4: 10}
                line_scores = {1: 10, 2: 30, 3: 60, 4: 100}
                if lines_cleared > 0:
                    reward += line_rewards.get(lines_cleared, 10)
                    self.score += line_scores.get(lines_cleared, 100)

                # Update fall speed based on score (gets faster)
                self.fall_speed = max(200, 1000 - (self.score // 500) * 50)

                self.current_piece = self.next_piece
                self.next_piece = self._new_piece()

                if self._check_collision(self.current_piece):
                    self.game_over = True

        # 4. Check termination conditions
        if self.game_over:
            reward = -100
        elif self.score >= self.WIN_SCORE:
            reward = 100
            self.game_over = True
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Draw grid background and border
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, (0,0,0), grid_rect)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, 1)

        # Draw placed blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] != 0:
                    self._draw_block(x, y, self.PIECE_COLORS[int(self.grid[x, y])])

        # Draw ghost piece
        ghost_piece = self.current_piece.copy()
        while not self._check_collision({"y": ghost_piece["y"] + 1, **ghost_piece}):
            ghost_piece["y"] += 1
        shape = self._get_shape(ghost_piece)
        color = self.PIECE_COLORS[self.current_piece["shape_idx"]]
        for y_off, row in enumerate(shape):
            for x_off, cell in enumerate(row):
                if cell:
                    self._draw_block(ghost_piece["x"] + x_off, ghost_piece["y"] + y_off, color, is_ghost=True)

        # Draw current falling piece
        if not self.game_over:
            shape = self._get_shape(self.current_piece)
            color = self.PIECE_COLORS[self.current_piece["shape_idx"]]
            for y_off, row in enumerate(shape):
                for x_off, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece["x"] + x_off, self.current_piece["y"] + y_off, color)

        # Draw line clear animation
        active_animations = []
        for anim in self.line_clear_animation:
            y = anim["y"]
            intensity = anim["timer"] / 10.0
            color = (intensity * 255, intensity * 255, intensity * 255)
            rect = pygame.Rect(self.GRID_X, self.GRID_Y + y * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            anim["timer"] -= 1
            if anim["timer"] > 0:
                active_animations.append(anim)
        self.line_clear_animation = active_animations

    def _draw_block(self, grid_x, grid_y, color, is_ghost=False):
        if grid_y < 0: return
        px, py = self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE
        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 1)
        else:
            highlight = tuple(min(255, c + 60) for c in color)
            shadow = tuple(max(0, c - 60) for c in color)
            pygame.gfxdraw.box(self.screen, rect, color)
            pygame.draw.line(self.screen, highlight, (px, py), (px + self.CELL_SIZE - 1, py), 1)
            pygame.draw.line(self.screen, highlight, (px, py), (px, py + self.CELL_SIZE - 1), 1)
            pygame.draw.line(self.screen, shadow, (px + self.CELL_SIZE - 1, py + 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 1)
            pygame.draw.line(self.screen, shadow, (px + 1, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 1)

    def _render_ui(self):
        score_surf = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))

        next_label_surf = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        preview_x = self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20
        self.screen.blit(next_label_surf, (preview_x, 20))
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (preview_x - 5, 45, 4 * self.CELL_SIZE + 10, 4 * self.CELL_SIZE + 10), 1)

        if self.next_piece:
            shape = self._get_shape(self.next_piece)
            color = self.PIECE_COLORS[self.next_piece["shape_idx"]]
            start_x = preview_x + (4 * self.CELL_SIZE - len(shape[0]) * self.CELL_SIZE) // 2
            start_y = 50 + (4 * self.CELL_SIZE - len(shape) * self.CELL_SIZE) // 2
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        self._draw_block(x, y, color, is_ghost=False)
                        # Redraw on main screen relative coords
                        px = start_x + x * self.CELL_SIZE
                        py = start_y + y * self.CELL_SIZE
                        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                        highlight = tuple(min(255, c + 60) for c in color)
                        shadow = tuple(max(0, c - 60) for c in color)
                        pygame.gfxdraw.box(self.screen, rect, color)
                        pygame.draw.line(self.screen, highlight, (px, py), (px + self.CELL_SIZE - 1, py), 1)
                        pygame.draw.line(self.screen, highlight, (px, py), (px, py + self.CELL_SIZE - 1), 1)
                        pygame.draw.line(self.screen, shadow, (px + self.CELL_SIZE - 1, py + 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 1)
                        pygame.draw.line(self.screen, shadow, (px + 1, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 1)

        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            game_over_surf = self.font_large.render("GAME OVER", True, (255, 80, 80))
            text_rect = game_over_surf.get_rect(center=(320, 200))
            self.screen.blit(game_over_surf, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Falling Block Puzzle")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not terminated:
        # --- Human Controls ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame, so we just show it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit to 30 FPS for smooth play

    print(f"Game Over! Final Score: {info['score']}")
    env.close()