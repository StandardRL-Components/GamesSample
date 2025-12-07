
# Generated: 2025-08-28T04:42:39.941600
# Source Brief: brief_05332.md
# Brief Index: 5332

        
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
        "Controls: ←→ to move, ↓ for soft drop. Space for hard drop, Shift to rotate."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a 60-second onslaught of falling blocks in this fast-paced arcade puzzle game. Clear lines to score points and prevent the stack from reaching the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.SIDE_PANEL_WIDTH = 220
        self.GRID_OFFSET_X = (self.WIDTH - self.SIDE_PANEL_WIDTH - (self.GRID_WIDTH * self.CELL_SIZE)) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - (self.GRID_HEIGHT * self.CELL_SIZE)) // 2
        
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS

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
        
        # Fonts
        try:
            self.font_large = pygame.font.SysFont('Consolas', 36, bold=True)
            self.font_medium = pygame.font.SysFont('Consolas', 24)
            self.font_small = pygame.font.SysFont('Consolas', 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_PANEL = (25, 30, 45)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_DIM = (150, 150, 170)
        self.COLOR_WHITE = (255, 255, 255)
        
        # Tetromino shapes and colors
        self.TETROMINOES = {
            'I': {'shape': [(0, -1), (0, 0), (0, 1), (0, 2)], 'color': (0, 240, 240)},
            'O': {'shape': [(0, 0), (0, 1), (1, 0), (1, 1)], 'color': (240, 240, 0)},
            'T': {'shape': [(-1, 0), (0, 0), (1, 0), (0, -1)], 'color': (160, 0, 240)},
            'S': {'shape': [(-1, 0), (0, 0), (0, -1), (1, -1)], 'color': (0, 240, 0)},
            'Z': {'shape': [(-1, -1), (0, -1), (0, 0), (1, 0)], 'color': (240, 0, 0)},
            'J': {'shape': [(-1, 0), (0, 0), (1, 0), (1, -1)], 'color': (0, 0, 240)},
            'L': {'shape': [(-1, -1), (-1, 0), (0, 0), (1, 0)], 'color': (240, 160, 0)}
        }
        self.PIECE_KEYS = list(self.TETROMINOES.keys())
        self.COLOR_MAP = [self.COLOR_BG] + [data['color'] for data in self.TETROMINOES.values()]
        
        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece_key = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.fall_speed = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.line_clear_effects = []

        # This will be called by the gym wrapper, but good practice to have it
        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = self.FPS // 2  # 0.5 seconds per cell
        self.fall_timer = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        self.line_clear_effects = []
        
        self.next_piece_key = self.np_random.choice(self.PIECE_KEYS)
        self._spawn_new_piece()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time passing to encourage speed
        self.steps += 1
        
        if not self.game_over:
            reward += self._handle_input(action)
            reward += self._update_game_state()
        
        terminated = self._check_termination()
        
        if terminated and not self.game_over: # Survived the time limit
            reward = 50.0
        elif terminated and self.game_over: # Topped out
            reward = -100.0

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # Movement (Left/Right)
        move_dir = 0
        if movement == 3: move_dir = -1  # Left
        if movement == 4: move_dir = 1   # Right
        if move_dir != 0:
            self._move_piece(move_dir, 0)

        # Rotation (Shift) - on press
        if shift_action and not self.last_shift_held:
            self._rotate_piece()
        self.last_shift_held = shift_action

        # Hard Drop (Space) - on press
        if space_action and not self.last_space_held:
            reward += self._hard_drop()
        self.last_space_held = space_action

        # Soft Drop (Down)
        is_soft_dropping = movement == 2
        if is_soft_dropping:
            self.fall_timer += self.fall_speed // 2  # Speed up fall

        return reward

    def _update_game_state(self):
        # Update difficulty
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.fall_speed = max(5, self.fall_speed - 1) # Faster every 10s, min 5 frames/drop

        # Natural gravity
        self.fall_timer += 1
        if self.fall_timer >= self.fall_speed:
            self.fall_timer = 0
            if not self._move_piece(0, 1):
                # Piece has landed
                return self._lock_piece()
        return 0.0

    def _spawn_new_piece(self):
        self.current_piece = {
            'key': self.next_piece_key,
            'x': self.GRID_WIDTH // 2,
            'y': 1,
            'rotation': 0
        }
        self.next_piece_key = self.np_random.choice(self.PIECE_KEYS)
        
        if not self._is_valid_position(self.current_piece):
            self.game_over = True

    def _get_piece_coords(self, piece_data):
        key = piece_data['key']
        shape = self.TETROMINOES[key]['shape']
        coords = []
        for r, c in shape:
            for _ in range(piece_data['rotation'] % 4):
                r, c = c, -r
            coords.append((piece_data['y'] + c, piece_data['x'] + r))
        return coords

    def _is_valid_position(self, piece_data, offset_x=0, offset_y=0):
        temp_piece = piece_data.copy()
        temp_piece['x'] += offset_x
        temp_piece['y'] += offset_y
        coords = self._get_piece_coords(temp_piece)
        
        for r, c in coords:
            if not (0 <= c < self.GRID_WIDTH and 0 <= r < self.GRID_HEIGHT):
                return False
            if r >= 0 and self.grid[r, c] != 0:
                return False
        return True

    def _move_piece(self, dx, dy):
        if self._is_valid_position(self.current_piece, dx, dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def _rotate_piece(self):
        rotated_piece = self.current_piece.copy()
        rotated_piece['rotation'] += 1
        
        # Wall kick logic
        for offset_x in [0, 1, -1, 2, -2]: # Try to shift to make rotation work
            if self._is_valid_position(rotated_piece, offset_x, 0):
                self.current_piece['rotation'] += 1
                self.current_piece['x'] += offset_x
                # sfx: rotate
                return True
        # sfx: rotate_fail
        return False

    def _hard_drop(self):
        dy = 0
        while self._is_valid_position(self.current_piece, 0, dy + 1):
            dy += 1
        self._move_piece(0, dy)
        self.fall_timer = self.fall_speed # Force lock on next frame
        # sfx: hard_drop
        return self._lock_piece()

    def _lock_piece(self):
        coords = self._get_piece_coords(self.current_piece)
        color_index = self.PIECE_KEYS.index(self.current_piece['key']) + 1
        
        is_safe = True
        for r, c in coords:
            if r >= 0:
                self.grid[r, c] = color_index
                if r + 1 >= self.GRID_HEIGHT or self.grid[r + 1, c] == 0:
                    is_safe = False # At least one block has nothing below it
        
        # sfx: lock
        reward = -0.2 if is_safe else 2.0 # Reward for risky vs safe placement
        reward += self._check_and_clear_lines()
        
        self._spawn_new_piece()
        return reward

    def _check_and_clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            # sfx: line_clear
            for r in lines_to_clear:
                self.grid[r, :] = 0
                self.line_clear_effects.append({'row': r, 'timer': self.FPS // 4})
            
            # Shift rows down
            new_grid = np.zeros_like(self.grid)
            new_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if r not in lines_to_clear:
                    new_grid[new_row, :] = self.grid[r, :]
                    new_row -= 1
            self.grid = new_grid
            
            num_cleared = len(lines_to_clear)
            self.score += num_cleared * 100 * num_cleared # Bonus for multi-line clears
            return float(num_cleared) # Reward is 1.0 per line
        
        return 0.0

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS)),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background and lines
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y,
                                self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (grid_rect.left, y), (grid_rect.right, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (x, grid_rect.top), (x, grid_rect.bottom))

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_cell(c, r, self.COLOR_MAP[self.grid[r, c]])
        
        # Draw current piece and ghost piece if not game over
        if self.current_piece and not self.game_over:
            # Ghost piece
            ghost_piece = self.current_piece.copy()
            dy = 0
            while self._is_valid_position(ghost_piece, 0, dy + 1):
                dy += 1
            ghost_piece['y'] += dy
            ghost_coords = self._get_piece_coords(ghost_piece)
            ghost_color = self.TETROMINOES[ghost_piece['key']]['color']
            for r, c in ghost_coords:
                self._draw_cell(c, r, ghost_color, is_ghost=True)

            # Active piece
            active_coords = self._get_piece_coords(self.current_piece)
            active_color = self.TETROMINOES[self.current_piece['key']]['color']
            for r, c in active_coords:
                self._draw_cell(c, r, active_color)

        # Draw line clear effects
        for effect in self.line_clear_effects[:]:
            r = effect['row']
            alpha = int(255 * (effect['timer'] / (self.FPS // 4)))
            flash_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + r * self.CELL_SIZE))
            effect['timer'] -= 1
            if effect['timer'] <= 0:
                self.line_clear_effects.remove(effect)

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        if grid_y < 0: return # Don't draw above the grid
        
        x = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2) # Just the outline
        else:
            # Main fill
            pygame.draw.rect(self.screen, color, rect)
            # 3D effect
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright)
            pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft)
            pygame.draw.line(self.screen, shadow, rect.bottomleft, rect.bottomright)
            pygame.draw.line(self.screen, shadow, rect.topright, rect.bottomright)

    def _render_ui(self):
        panel_x = self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE + 20
        
        # Time remaining
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS))
        time_text = f"{int(time_left // 60):02}:{int(time_left % 60):02}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (20, 20))

        # Score
        score_label = self.font_medium.render("SCORE", True, self.COLOR_TEXT_DIM)
        score_val = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_label, (panel_x, 20))
        self.screen.blit(score_val, (panel_x, 50))
        
        # Next Piece
        next_label = self.font_medium.render("NEXT", True, self.COLOR_TEXT_DIM)
        self.screen.blit(next_label, (panel_x, 120))
        
        if self.next_piece_key:
            piece_data = self.TETROMINOES[self.next_piece_key]
            shape = piece_data['shape']
            color = piece_data['color']
            
            # Find center of shape to align it
            min_r = min(c[1] for c in shape)
            max_r = max(c[1] for c in shape)
            min_c = min(c[0] for c in shape)
            max_c = max(c[0] for c in shape)
            
            for r_off, c_off in shape:
                # Center the piece in the preview box
                draw_c = c_off - (min_c + max_c) / 2.0
                draw_r = r_off - (min_r + max_r) / 2.0
                
                x = panel_x + 60 + draw_c * self.CELL_SIZE
                y = 180 + draw_r * self.CELL_SIZE
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                highlight = tuple(min(255, c + 40) for c in color)
                shadow = tuple(max(0, c - 40) for c in color)
                pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright)
                pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft)
                pygame.draw.line(self.screen, shadow, rect.bottomleft, rect.bottomright)
                pygame.draw.line(self.screen, shadow, rect.topright, rect.bottomright)

        # Game Over / Win Text
        if self.game_over:
            self._render_overlay_text("GAME OVER", self.COLOR_WHITE)
        elif self.steps >= self.MAX_STEPS:
            self._render_overlay_text("YOU WIN!", self.COLOR_WHITE)

    def _render_overlay_text(self, text, color):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # Monkey-patch the render method for human playback
        def render(self):
            if not hasattr(self, 'human_screen'):
                pygame.display.init()
                self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption("Arcade Block Survivor")
            
            # Get the current frame
            frame = self._get_observation()
            # Pygame uses (width, height), numpy uses (height, width)
            frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.human_screen.blit(frame_surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.FPS)
        GameEnv.render = render

    env = GameEnv(render_mode="rgb_array") # Always init with rgb_array
    
    # Run a few episodes
    for episode in range(3):
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not terminated:
            # In a real scenario, an agent would provide this action
            # Here, we simulate random actions
            action = env.action_space.sample()
            
            # Or, for manual play:
            if render_mode == "human":
                keys = pygame.key.get_pressed()
                mov = 0 # none
                if keys[pygame.K_UP]: mov = 1 # unused in this mapping
                if keys[pygame.K_DOWN]: mov = 2
                if keys[pygame.K_LEFT]: mov = 3
                if keys[pygame.K_RIGHT]: mov = 4
                
                space = 1 if keys[pygame.K_SPACE] else 0
                shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
                action = np.array([mov, space, shift])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if render_mode == "human":
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        terminated = True
                        
        print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()