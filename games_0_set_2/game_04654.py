
# Generated: 2025-08-28T03:03:06.013232
# Source Brief: brief_04654.md
# Brief Index: 4654

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ to soft drop. Press space to hard drop on the beat."
    )

    game_description = (
        "A rhythmic puzzle game. Drop blocks on the beat to clear lines and hit the target score. Don't miss the rhythm!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # Game Constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.screen_width - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (self.screen_height - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.BPM = 60
        self.FPS = 30 # For smooth animations
        self.FRAMES_PER_BEAT = (60 / self.BPM) * self.FPS
        self.WIN_LINES = 20
        self.MAX_MISSES = 3
        self.MAX_STEPS = 180 * self.FPS # 180 seconds at 30fps

        # Visuals
        self.COLOR_BG = (20, 20, 35)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_ACCENT = (100, 255, 255)
        self.TETROMINOS = {
            'I': [[1, 1, 1, 1]],
            'O': [[1, 1], [1, 1]],
            'T': [[0, 1, 0], [1, 1, 1]],
            'S': [[0, 1, 1], [1, 1, 0]],
            'Z': [[1, 1, 0], [0, 1, 1]],
            'J': [[1, 0, 0], [1, 1, 1]],
            'L': [[0, 0, 1], [1, 1, 1]],
        }
        self.TETROMINO_COLORS = {
            'I': (0, 255, 255), 'O': (255, 255, 0), 'T': (128, 0, 128),
            'S': (0, 255, 0), 'Z': (255, 0, 0), 'J': (0, 0, 255), 'L': (255, 165, 0)
        }
        self.SHAPES = list(self.TETROMINOS.keys())

        # State variables will be initialized in reset()
        self.state_vars = [
            "steps", "score", "game_over", "grid", "lines_cleared", "misses",
            "current_piece_shape", "current_piece_rotation", "current_piece_x", "current_piece_y",
            "beat_timer", "last_action", "player_dropped_piece", "is_topped_out",
            "line_clear_animation", "reward"
        ]
        for var in self.state_vars:
            setattr(self, var, None)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.is_topped_out = False
        self.lines_cleared = 0
        self.misses = 0

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.beat_timer = 0
        self.last_action = np.array([0, 0, 0])
        self.line_clear_animation = [] # Stores (y_index, timer)

        self._new_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward = 0
        self.game_over = self._check_termination()

        if self.game_over:
            if self.lines_cleared >= self.WIN_LINES:
                self.reward = 100
            else:
                self.reward = -100
            return self._get_observation(), self.reward, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()
        self._update_animations()

        self.steps += 1
        self.last_action = action
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.lines_cleared >= self.WIN_LINES:
                self.reward += 100
            else:
                self.reward -= 100

        return self._get_observation(), self.reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        up_pressed = movement == 1 and self.last_action[0] != 1
        space_pressed = space_held and not (self.last_action[1] == 1)

        if movement == 3: # Left
            self._move_piece(dx=-1)
        elif movement == 4: # Right
            self._move_piece(dx=1)
        elif movement == 2: # Down (soft drop)
            if self._move_piece(dy=1):
                self.score += 0.01 # Small incentive
                self.reward += 0.01

        if up_pressed:
            self._rotate_piece()

        if space_pressed:
            self._hard_drop()

    def _update_game_state(self):
        self.beat_timer += 1
        if self.beat_timer >= self.FRAMES_PER_BEAT:
            self.beat_timer = 0
            # SFX: Beat Tick
            if not self._move_piece(dy=1):
                # Piece landed automatically
                if not self.player_dropped_piece:
                    self.misses += 1
                    # SFX: Missed Beat
                self._place_piece()
                self._clear_lines()
                self._new_piece()

    def _update_animations(self):
        self.line_clear_animation = [(y, t - 1) for y, t in self.line_clear_animation if t > 0]

    def _new_piece(self):
        self.current_piece_shape = self.SHAPES[self.np_random.integers(len(self.SHAPES))]
        self.current_piece_rotation = 0
        self.current_piece_x = self.GRID_WIDTH // 2 - 1
        self.current_piece_y = 0
        self.player_dropped_piece = False

        if not self._is_valid_position():
            self.is_topped_out = True
            self.game_over = True
            # SFX: Game Over

    def _move_piece(self, dx=0, dy=0):
        self.current_piece_x += dx
        self.current_piece_y += dy
        if not self._is_valid_position():
            self.current_piece_x -= dx
            self.current_piece_y -= dy
            return False
        # SFX: Move
        return True

    def _rotate_piece(self):
        original_rotation = self.current_piece_rotation
        self.current_piece_rotation = (self.current_piece_rotation + 1) % len(self._get_piece_rotations())
        if not self._is_valid_position():
            # Basic wall kick check
            original_x = self.current_piece_x
            for offset in [-1, 1, -2, 2]: # Try shifting
                self.current_piece_x = original_x + offset
                if self._is_valid_position():
                    # SFX: Rotate
                    return
            self.current_piece_x = original_x
            self.current_piece_rotation = original_rotation # Revert if no valid position found
        else:
            # SFX: Rotate
            pass

    def _hard_drop(self):
        # SFX: Hard Drop
        while self._is_valid_position():
            self.current_piece_y += 1
        self.current_piece_y -= 1

        self.player_dropped_piece = True
        self.reward += 1

        # Check for edge column penalty
        piece_coords = self._get_piece_coords()
        if any(x < 2 or x >= self.GRID_WIDTH - 2 for x, _ in piece_coords):
            self.reward -= 0.2
        
        self.score += 10
        self.beat_timer = self.FRAMES_PER_BEAT -1 # Trigger next piece immediately
        self._place_piece()
        self._clear_lines()
        self._new_piece()

    def _place_piece(self):
        # SFX: Piece Land
        piece_coords = self._get_piece_coords()
        color_index = self.SHAPES.index(self.current_piece_shape) + 1
        for x, y in piece_coords:
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = color_index

    def _clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.grid) if np.all(row > 0)]
        if not lines_to_clear:
            return

        # SFX: Line Clear
        for y in lines_to_clear:
            self.line_clear_animation.append((y, 10)) # Animate for 10 frames

        for y in sorted(lines_to_clear, reverse=True):
            self.grid = np.delete(self.grid, y, axis=0)
        
        new_rows = np.zeros((len(lines_to_clear), self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack((new_rows, self.grid))
        
        num_cleared = len(lines_to_clear)
        self.lines_cleared += num_cleared
        
        # Scoring and Rewards
        reward_map = {1: 5, 2: 10, 3: 20, 4: 50}
        score_map = {1: 100, 2: 300, 3: 500, 4: 800}
        self.reward += reward_map.get(num_cleared, 0)
        self.score += score_map.get(num_cleared, 0)

    def _get_piece_rotations(self):
        shape = self.TETROMINOS[self.current_piece_shape]
        rotations = [shape]
        for _ in range(3):
            shape = list(zip(*shape[::-1]))
            if shape not in rotations:
                rotations.append(shape)
        return rotations

    def _get_piece_coords(self, x_offset=0, y_offset=0):
        rotations = self._get_piece_rotations()
        piece_shape = rotations[self.current_piece_rotation % len(rotations)]
        coords = []
        for r, row in enumerate(piece_shape):
            for c, cell in enumerate(row):
                if cell:
                    coords.append((self.current_piece_x + c + x_offset, self.current_piece_y + r + y_offset))
        return coords

    def _is_valid_position(self, x_offset=0, y_offset=0):
        for x, y in self._get_piece_coords(x_offset, y_offset):
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False
            if y >= 0 and self.grid[y, x] > 0:
                return False
        return True

    def _check_termination(self):
        return (self.lines_cleared >= self.WIN_LINES or 
                self.misses >= self.MAX_MISSES or 
                self.is_topped_out or
                self.steps >= self.MAX_STEPS)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines": self.lines_cleared, "misses": self.misses}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw landed pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    color_index = int(self.grid[r, c] - 1)
                    color = self.TETROMINO_COLORS[self.SHAPES[color_index]]
                    self._draw_cell(c, r, color)

        # Draw ghost piece
        if not self.game_over:
            ghost_y = self.current_piece_y
            while self._is_valid_position(y_offset=ghost_y - self.current_piece_y + 1):
                ghost_y += 1
            original_y = self.current_piece_y
            self.current_piece_y = ghost_y
            color = self.TETROMINO_COLORS[self.current_piece_shape]
            self._draw_piece(tuple(c * 0.3 for c in color))
            self.current_piece_y = original_y
        
        # Draw falling piece
        if not self.game_over:
            color = self.TETROMINO_COLORS[self.current_piece_shape]
            self._draw_piece(color)

        # Draw line clear animation
        for y, timer in self.line_clear_animation:
            alpha = int(255 * (timer / 10))
            flash_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE))

    def _draw_piece(self, color):
        for x, y in self._get_piece_coords():
            if y >= 0:
                self._draw_cell(x, y, color)
    
    def _draw_cell(self, grid_x, grid_y, color):
        rect = pygame.Rect(
            self.GRID_X + grid_x * self.CELL_SIZE,
            self.GRID_Y + grid_y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, 1) # Border

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Lines
        lines_text = self.font_main.render(f"Lines: {self.lines_cleared} / {self.WIN_LINES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_text, (self.screen_width - lines_text.get_width() - 20, 20))
        
        # Misses
        miss_text = self.font_small.render(f"Misses: {'●' * self.misses}{'○' * (self.MAX_MISSES - self.misses)}", True, (255, 80, 80))
        self.screen.blit(miss_text, (20, 60))

        # Rhythm Bar
        bar_width = self.screen_width - 40
        bar_height = 15
        bar_y = self.screen_height - 30
        progress = self.beat_timer / self.FRAMES_PER_BEAT
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (20, bar_y, bar_width, bar_height), border_radius=5)
        fill_width = bar_width * progress
        if fill_width > 2: # Don't draw a tiny sliver
            pygame.draw.rect(self.screen, self.COLOR_ACCENT, (20, bar_y, fill_width, bar_height), border_radius=5)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            if self.lines_cleared >= self.WIN_LINES:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Obs dtype is {test_obs.dtype}"
        
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Rhythm Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-closing or allow reset
            pygame.time.wait(2000)
            running = False

        clock.tick(env.FPS)
        
    pygame.quit()