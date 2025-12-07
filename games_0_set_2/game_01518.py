
# Generated: 2025-08-27T17:24:01.426619
# Source Brief: brief_01518.md
# Brief Index: 1518

        
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
        "Controls: ←→ to move, ↓ for soft drop. Press Space to rotate and Shift to hard drop."
    )

    game_description = (
        "Block Blitz: A fast-paced puzzle game. Strategically drop falling blocks to clear lines for a high score."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 20
        self.BOARD_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.BOARD_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.BOARD_X = (self.WIDTH - self.BOARD_WIDTH) // 2
        self.BOARD_Y = (self.HEIGHT - self.BOARD_HEIGHT) // 2

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_BOARD_BG = (35, 40, 60)
        self.COLOR_GRID = (50, 55, 75)
        self.COLOR_DANGER = (80, 20, 30, 100) # RGBA for transparency
        self.COLOR_GHOST = (255, 255, 255, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_WHITE = (255, 255, 255)
        self.BLOCK_COLORS = [
            (0, 240, 240),  # I (Cyan)
            (240, 240, 0),  # O (Yellow)
            (160, 0, 240),  # T (Purple)
            (0, 0, 240),    # J (Blue)
            (240, 160, 0),  # L (Orange)
            (0, 240, 0),    # S (Green)
            (240, 0, 0),    # Z (Red)
            (255, 105, 180),# Dot (Pink)
            (128, 128, 128),# Small L (Gray)
            (0, 128, 0),    # U-shape (Dark Green)
        ]

        # --- Block Shapes (centered around a pivot) ---
        self.SHAPES = [
            [(-2, 0), (-1, 0), (0, 0), (1, 0)],  # I
            [(-1, 0), (0, 0), (-1, 1), (0, 1)],  # O
            [(-1, 0), (0, 0), (1, 0), (0, 1)],   # T
            [(-1, 1), (-1, 0), (0, 0), (1, 0)],  # J
            [(-1, 0), (0, 0), (1, 0), (1, 1)],   # L
            [(-1, 1), (0, 1), (0, 0), (1, 0)],   # S
            [(-1, 0), (0, 0), (0, 1), (1, 1)],   # Z
            [(0, 0)],                           # Dot
            [(-1,1), (-1,0), (0,0)],             # Small L
            [(-1,1), (1,1), (-1,0), (0,0), (1,0)]# U-Shape
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.current_block = None
        self.next_block = None
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.fall_timer = 0
        self.fall_speed = 0
        self.previous_actions = None
        self.line_clear_animation = None
        self.max_steps = 10000

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.fall_timer = 0.0
        self.fall_speed = 0.5  # Seconds per grid cell
        self.previous_actions = np.array([0, 0, 0])
        self.line_clear_animation = None

        self._spawn_block(first=True)
        self._spawn_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), -100.0, True, False, self._get_info()

        # --- Handle line clear animation delay ---
        if self.line_clear_animation:
            self.line_clear_animation['timer'] -= 1
            if self.line_clear_animation['timer'] <= 0:
                self._finish_line_clear()
            # Skip game logic during animation
            return self._get_observation(), reward, False, False, self._get_info()

        # --- Process Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        prev_space, prev_shift = self.previous_actions[1] == 1, self.previous_actions[2] == 1

        # Movement
        if movement == 3: # Left
            self._move_block(-1, 0)
        elif movement == 4: # Right
            self._move_block(1, 0)
        
        # Soft Drop
        if movement == 2: # Down
            self.fall_timer += self.fall_speed * 0.75 # Accelerate fall

        # Rotation (on key press)
        if space_held and not prev_space:
            self._rotate_block()
            # sfx: rotate_sound

        # Hard Drop (on key press)
        if shift_held and not prev_shift:
            drop_reward = self._hard_drop()
            reward += drop_reward
            self._lock_block()
            clear_reward, lines = self._check_lines()
            reward += clear_reward
            if not self.game_over:
                self._spawn_block()
            # sfx: hard_drop_sound
        else:
            # --- Normal Game Tick ---
            self.fall_timer += self.clock.get_time() / 1000.0
            if self.fall_timer >= self.fall_speed:
                self.fall_timer = 0
                if not self._move_block(0, 1):
                    # Block has landed
                    lock_reward = self._get_placement_reward()
                    reward += lock_reward
                    self._lock_block()
                    clear_reward, lines = self._check_lines()
                    reward += clear_reward
                    if not self.game_over:
                        self._spawn_block()
                    # sfx: block_land_sound

        self.previous_actions = action

        # --- Termination Conditions ---
        terminated = self.game_over
        if self.lines_cleared >= 20:
            reward += 100.0
            terminated = True
        if self.steps >= self.max_steps:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_block(self, first=False):
        if first:
            self.next_block = self._new_piece()
            return
        
        self.current_block = self.next_block
        self.next_block = self._new_piece()
        self.current_block['x'] = self.GRID_WIDTH // 2
        self.current_block['y'] = 0

        # Check for game over
        if self._check_collision(self.current_block['shape_idx'], self.current_block['rotation'], self.current_block['x'], self.current_block['y']):
            self.game_over = True
            # sfx: game_over_sound

    def _new_piece(self):
        shape_idx = self.np_random.integers(0, len(self.SHAPES))
        return {
            'shape_idx': shape_idx,
            'rotation': 0,
            'x': 0, 'y': 0,
            'color': self.BLOCK_COLORS[shape_idx]
        }
    
    def _get_piece_coords(self, shape_idx, rotation, x, y):
        shape = self.SHAPES[shape_idx]
        coords = []
        for sx, sy in shape:
            for _ in range(rotation):
                sx, sy = sy, -sx # 90 deg clockwise rotation
            coords.append((sx + x, sy + y))
        return coords

    def _check_collision(self, shape_idx, rotation, x, y):
        for bx, by in self._get_piece_coords(shape_idx, rotation, x, y):
            if not (0 <= bx < self.GRID_WIDTH and 0 <= by < self.GRID_HEIGHT):
                return True # Wall collision
            if self.grid[by, bx] != 0:
                return True # Grid collision
        return False

    def _move_block(self, dx, dy):
        if self.current_block is None: return False
        
        new_x = self.current_block['x'] + dx
        new_y = self.current_block['y'] + dy
        if not self._check_collision(self.current_block['shape_idx'], self.current_block['rotation'], new_x, new_y):
            self.current_block['x'] = new_x
            self.current_block['y'] = new_y
            return True
        return False

    def _rotate_block(self):
        if self.current_block is None: return
        
        old_rot = self.current_block['rotation']
        new_rot = (old_rot + 1) % 4
        
        if not self._check_collision(self.current_block['shape_idx'], new_rot, self.current_block['x'], self.current_block['y']):
            self.current_block['rotation'] = new_rot
        else: # Wall kick
            for dx in [-1, 1, -2, 2]:
                if not self._check_collision(self.current_block['shape_idx'], new_rot, self.current_block['x'] + dx, self.current_block['y']):
                    self.current_block['x'] += dx
                    self.current_block['rotation'] = new_rot
                    return

    def _hard_drop(self):
        if self.current_block is None: return 0.0
        
        y = self.current_block['y']
        while self._move_block(0, 1):
            pass
        
        return self._get_placement_reward()

    def _get_placement_reward(self):
        if self.current_block is None: return 0.0
        coords = self._get_piece_coords(self.current_block['shape_idx'], self.current_block['rotation'], self.current_block['x'], self.current_block['y'])
        max_y = max(by for bx, by in coords)
        
        if max_y <= 2: # Risky placement (top 3 rows are 0,1,2)
            return 2.0
        if max_y >= self.GRID_HEIGHT / 2: # Safe placement
            return -0.2
        return 0.0

    def _lock_block(self):
        if self.current_block is None: return
        
        coords = self._get_piece_coords(self.current_block['shape_idx'], self.current_block['rotation'], self.current_block['x'], self.current_block['y'])
        color_idx = self.current_block['shape_idx'] + 1
        
        for bx, by in coords:
            if 0 <= bx < self.GRID_WIDTH and 0 <= by < self.GRID_HEIGHT:
                self.grid[by, bx] = color_idx
        
        self.current_block = None
        # sfx: lock_sound

    def _check_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y, :] != 0):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            self.line_clear_animation = {'lines': lines_to_clear, 'timer': 10} # 10 frames animation
            # sfx: line_clear_start_sound
            return len(lines_to_clear), len(lines_to_clear) # Reward, num lines
        
        return 0.0, 0

    def _finish_line_clear(self):
        lines = self.line_clear_animation['lines']
        
        for y in sorted(lines, reverse=True):
            self.grid[1:y+1, :] = self.grid[0:y, :]
            self.grid[0, :] = 0
        
        self.lines_cleared += len(lines)
        self.score += len(lines) * 100 * len(lines) # Bonus for multi-line clears
        
        # Update fall speed
        speed_reduction = (self.lines_cleared // 5) * 0.02
        self.fall_speed = max(0.1, 0.5 - speed_reduction)
        
        self.line_clear_animation = None
        # sfx: line_clear_finish_sound

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw board background and border
        board_rect = pygame.Rect(self.BOARD_X, self.BOARD_Y, self.BOARD_WIDTH, self.BOARD_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BOARD_BG, board_rect)
        pygame.draw.rect(self.screen, self.COLOR_GRID, board_rect, 2)

        # Draw danger zone
        danger_surf = pygame.Surface((self.BOARD_WIDTH, self.CELL_SIZE * 3), pygame.SRCALPHA)
        danger_surf.fill(self.COLOR_DANGER)
        self.screen.blit(danger_surf, (self.BOARD_X, self.BOARD_Y))

        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = self.BOARD_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.BOARD_Y), (px, self.BOARD_Y + self.BOARD_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.BOARD_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X, py), (self.BOARD_X + self.BOARD_WIDTH, py))

        # Draw placed blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    color_idx = int(self.grid[y, x]) - 1
                    self._draw_cell(x, y, self.BLOCK_COLORS[color_idx])

        # Draw ghost piece
        if self.current_block and not self.line_clear_animation:
            ghost_x, ghost_y = self.current_block['x'], self.current_block['y']
            while not self._check_collision(self.current_block['shape_idx'], self.current_block['rotation'], ghost_x, ghost_y + 1):
                ghost_y += 1
            
            coords = self._get_piece_coords(self.current_block['shape_idx'], self.current_block['rotation'], ghost_x, ghost_y)
            for bx, by in coords:
                self._draw_cell(bx, by, self.COLOR_GHOST, is_ghost=True)

        # Draw current block
        if self.current_block and not self.line_clear_animation:
            coords = self._get_piece_coords(self.current_block['shape_idx'], self.current_block['rotation'], self.current_block['x'], self.current_block['y'])
            for bx, by in coords:
                self._draw_cell(bx, by, self.current_block['color'])
        
        # Draw line clear animation
        if self.line_clear_animation:
            alpha = 255 * (1 - abs(self.line_clear_animation['timer'] - 5) / 5) # Flashing effect
            for y in self.line_clear_animation['lines']:
                rect = pygame.Rect(self.BOARD_X, self.BOARD_Y + y * self.CELL_SIZE, self.BOARD_WIDTH, self.CELL_SIZE)
                flash_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                flash_surf.fill((255, 255, 255, alpha))
                self.screen.blit(flash_surf, rect.topleft)

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        rect = pygame.Rect(
            self.BOARD_X + grid_x * self.CELL_SIZE,
            self.BOARD_Y + grid_y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect)
        else:
            pygame.draw.rect(self.screen, color, rect)
            # Add a subtle 3D effect
            darker_color = tuple(max(0, c - 50) for c in color)
            lighter_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.line(self.screen, lighter_color, rect.topleft, rect.topright, 2)
            pygame.draw.line(self.screen, lighter_color, rect.topleft, rect.bottomleft, 2)
            pygame.draw.line(self.screen, darker_color, rect.bottomright, rect.topright, 2)
            pygame.draw.line(self.screen, darker_color, rect.bottomright, rect.bottomleft, 2)


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:08d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Lines
        lines_text = self.font_medium.render(f"LINES: {self.lines_cleared}/{20}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (20, 60))

        # Next Block
        next_text = self.font_medium.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.WIDTH - 120, 20))
        
        next_box = pygame.Rect(self.WIDTH - 140, 50, 120, 100)
        pygame.draw.rect(self.screen, self.COLOR_BOARD_BG, next_box)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_box, 2)

        if self.next_block:
            coords = self._get_piece_coords(self.next_block['shape_idx'], 0, 0, 0)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            
            block_width = (max(xs) - min(xs) + 1) * self.CELL_SIZE
            block_height = (max(ys) - min(ys) + 1) * self.CELL_SIZE
            
            center_x = next_box.centerx
            center_y = next_box.centery

            for bx, by in coords:
                px = center_x + bx * self.CELL_SIZE - block_width // 4
                py = center_y + by * self.CELL_SIZE - block_height // 4
                
                rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.next_block['color'], rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.lines_cleared >= 20 else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared
        }
    
    def close(self):
        pygame.quit()

    def render(self):
        # This method is not used by the environment but can be useful for human playing
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        
        frame = self._get_observation()
        # Pygame uses (width, height), numpy uses (height, width)
        frame = np.transpose(frame, (1, 0, 2))
        
        surf = pygame.surfarray.make_surface(frame)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(30) # For human playability

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # Use a window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Action mapping for human keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1 # unused in this game
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Control frame rate for human play

    env.close()