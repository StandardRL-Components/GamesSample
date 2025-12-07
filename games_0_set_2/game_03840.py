
# Generated: 2025-08-28T00:35:59.415386
# Source Brief: brief_03840.md
# Brief Index: 3840

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑/↓ to rotate. Space for hard drop, Shift for soft drop."
    )

    game_description = (
        "A fast-paced falling block puzzle. Rotate and place blocks to clear lines, score points, and prevent the stack from reaching the top. Clear 10 lines to win!"
    )

    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CELL_SIZE = 18
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2 - 100
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GHOST = (255, 255, 255, 50)
    PIECE_COLORS = [
        (0, 0, 0),  # 0: Empty
        (239, 131, 49),  # 1: I (Orange)
        (65, 169, 239),  # 2: J (Blue)
        (239, 212, 49),  # 3: L (Yellow)
        (134, 212, 42),  # 4: O (Green)
        (173, 94, 212),  # 5: S (Purple)
        (239, 65, 65),   # 6: T (Red)
        (49, 239, 185),  # 7: Z (Cyan)
    ]
    PLACED_BLOCK_DARKEN_FACTOR = 0.6

    # --- Piece Shapes (Rotations) ---
    TETROMINOES = {
        'I': [[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 0), (2, 1), (2, 2), (2, 3)], [(0, 2), (1, 2), (2, 2), (3, 2)], [(1, 0), (1, 1), (1, 2), (1, 3)]],
        'J': [[(0, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (2, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (2, 2)], [(1, 0), (1, 1), (0, 2), (1, 2)]],
        'L': [[(2, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (1, 2), (2, 2)], [(0, 1), (1, 1), (2, 1), (0, 2)], [(0, 0), (1, 0), (1, 1), (1, 2)]],
        'O': [[(1, 0), (2, 0), (1, 1), (2, 1)]] * 4,
        'S': [[(1, 0), (2, 0), (0, 1), (1, 1)], [(1, 0), (1, 1), (2, 1), (2, 2)], [(1, 1), (2, 1), (0, 2), (1, 2)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
        'T': [[(1, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (2, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (0, 1), (1, 1), (1, 2)]],
        'Z': [[(0, 0), (1, 0), (1, 1), (2, 1)], [(2, 0), (1, 1), (2, 1), (1, 2)], [(0, 1), (1, 1), (1, 2), (2, 2)], [(1, 0), (0, 1), (1, 1), (0, 2)]]
    }
    PIECE_IDS = {'I': 1, 'J': 2, 'L': 3, 'O': 4, 'S': 5, 'T': 6, 'Z': 7}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.game_over = False
        self.steps = 0
        self.score = 0
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.bag = list(self.TETROMINOES.keys())
        self.np_random.shuffle(self.bag)
        
        self.current_piece = None
        self.next_piece_shape = self._get_from_bag()
        self._new_piece()
        
        self.base_fall_time = 1.0  # seconds
        self.fall_speed_reduction_per_line = 0.05
        self.fall_timer = self.base_fall_time
        
        self.line_clear_animation = None
        self.prev_space_held = False
        self.reward_buffer = 0

        self.move_cooldown = 0
        self.rotate_cooldown = 0
        self.MOVE_COOLDOWN_FRAMES = 4
        self.ROTATE_COOLDOWN_FRAMES = 6
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Time penalty
        
        self._handle_input(action)
        self._update_game_state()
        
        reward += self.reward_buffer
        self.reward_buffer = 0

        terminated = self._check_termination()
        if terminated:
            if self.lines_cleared >= 10:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self.move_cooldown = max(0, self.move_cooldown - 1)
        self.rotate_cooldown = max(0, self.rotate_cooldown - 1)

        # Horizontal Movement
        if self.move_cooldown == 0:
            if movement == 3: # Left
                self._move(-1)
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
            elif movement == 4: # Right
                self._move(1)
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

        # Rotation
        if self.rotate_cooldown == 0:
            if movement == 1: # Up -> Rotate CW
                self._rotate(1)
                self.rotate_cooldown = self.ROTATE_COOLDOWN_FRAMES
            elif movement == 2: # Down -> Rotate CCW
                self._rotate(-1)
                self.rotate_cooldown = self.ROTATE_COOLDOWN_FRAMES

        # Hard Drop (on press)
        if space_held and not self.prev_space_held:
            self._hard_drop()
        self.prev_space_held = space_held

        # Soft Drop (while held)
        self.soft_drop_active = shift_held

    def _update_game_state(self):
        # If a line clear animation is active, just wait for it to finish
        if self.line_clear_animation:
            self.line_clear_animation['timer'] -= 1
            if self.line_clear_animation['timer'] <= 0:
                self._finish_line_clear()
            return
            
        # Update fall timer
        dt = self.clock.get_time() / 1000.0
        fall_multiplier = 0.1 if self.soft_drop_active else 1.0
        self.fall_timer -= dt * (1/fall_multiplier)

        if self.fall_timer <= 0:
            self._gravity_step()
            fall_time = max(0.1, self.base_fall_time - self.lines_cleared * self.fall_speed_reduction_per_line)
            self.fall_timer = fall_time

    def _gravity_step(self):
        if not self._check_collision(dx=0, dy=1):
            self.current_piece['y'] += 1
        else:
            self._lock_piece()

    def _get_from_bag(self):
        if not self.bag:
            self.bag = list(self.TETROMINOES.keys())
            self.np_random.shuffle(self.bag)
        return self.bag.pop()

    def _new_piece(self):
        self.current_piece = {
            'shape': self.next_piece_shape,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0,
            'id': self.PIECE_IDS[self.next_piece_shape]
        }
        self.next_piece_shape = self._get_from_bag()
        if self._check_collision():
            self.game_over = True

    def _check_collision(self, piece_data=None, dx=0, dy=0, drot=0):
        if piece_data is None:
            piece_data = self.current_piece
        
        shape_id = piece_data['shape']
        x, y = piece_data['x'] + dx, piece_data['y'] + dy
        rotation = (piece_data['rotation'] + drot) % len(self.TETROMINOES[shape_id])
        shape_coords = self.TETROMINOES[shape_id][rotation]

        for sx, sy in shape_coords:
            px, py = x + sx, y + sy
            if not (0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT):
                return True # Wall collision
            if py >= 0 and self.grid[py, px] != 0:
                return True # Block collision
        return False

    def _move(self, dx):
        if not self._check_collision(dx=dx):
            self.current_piece['x'] += dx
            # pygame.mixer.Sound.play(move_sound)

    def _rotate(self, drot):
        if self._check_collision(drot=drot):
            # Wall Kick Logic (simple version)
            for kick_x in [-1, 1, -2, 2]:
                if not self._check_collision(dx=kick_x, drot=drot):
                    self.current_piece['x'] += kick_x
                    self.current_piece['rotation'] = (self.current_piece['rotation'] + drot) % len(self.TETROMINOES[self.current_piece['shape']])
                    # pygame.mixer.Sound.play(rotate_sound)
                    return
        else:
            self.current_piece['rotation'] = (self.current_piece['rotation'] + drot) % len(self.TETROMINOES[self.current_piece['shape']])
            # pygame.mixer.Sound.play(rotate_sound)

    def _hard_drop(self):
        # Calculate holes before
        holes_before = self._count_holes(self.grid)

        while not self._check_collision(dy=1):
            self.current_piece['y'] += 1
        
        # Reward for safe edge placement
        is_at_edge = self.current_piece['x'] == 0 or self.current_piece['x'] + max(c[0] for c in self._get_current_shape_coords()) == self.GRID_WIDTH - 1
        if is_at_edge:
            self.reward_buffer -= 0.2

        self._lock_piece()
        # pygame.mixer.Sound.play(hard_drop_sound)

        # Calculate holes after and reward
        holes_after = self._count_holes(self.grid)
        if holes_after > holes_before:
            self.reward_buffer += 0.5 # Risky placement
        elif holes_after < holes_before:
            self.reward_buffer += 1.0 # Filled a gap

    def _lock_piece(self):
        shape_coords = self._get_current_shape_coords()
        for sx, sy in shape_coords:
            px, py = self.current_piece['x'] + sx, self.current_piece['y'] + sy
            if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
                self.grid[py, px] = self.current_piece['id']
        # pygame.mixer.Sound.play(lock_sound)
        self._check_for_line_clears()
        self._new_piece()

    def _check_for_line_clears(self):
        full_lines = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y, :] != 0):
                full_lines.append(y)
        
        if full_lines:
            self.line_clear_animation = {'lines': full_lines, 'timer': 6} # 6 frames animation
            
            # Calculate rewards for lines
            num_cleared = len(full_lines)
            self.reward_buffer += num_cleared * 1.0
            if num_cleared > 1:
                self.reward_buffer += 2.0
            
            # pygame.mixer.Sound.play(line_clear_sound)
        
    def _finish_line_clear(self):
        full_lines = self.line_clear_animation['lines']
        
        for y in sorted(full_lines, reverse=True):
            self.grid[1:y+1, :] = self.grid[0:y, :]
            self.grid[0, :] = 0
        
        self.lines_cleared += len(full_lines)
        self.score += len(full_lines) * 100 * (len(full_lines)) # Bonus for multi-line clears
        self.line_clear_animation = None


    def _count_holes(self, grid):
        holes = 0
        for x in range(self.GRID_WIDTH):
            block_found = False
            for y in range(self.GRID_HEIGHT):
                if grid[y, x] != 0:
                    block_found = True
                elif block_found and grid[y, x] == 0:
                    holes += 1
        return holes

    def _check_termination(self):
        return self.game_over or self.steps >= 1000 or self.lines_cleared >= 10

    def _get_current_shape_coords(self):
        shape_id = self.current_piece['shape']
        rotation = self.current_piece['rotation']
        return self.TETROMINOES[shape_id][rotation]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }
    
    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw placed blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell_id = self.grid[y, x]
                if cell_id != 0:
                    color = self.PIECE_COLORS[cell_id]
                    darker_color = tuple(c * self.PLACED_BLOCK_DARKEN_FACTOR for c in color)
                    self._draw_cell(x, y, darker_color)

        # Draw line clear animation
        if self.line_clear_animation:
            for y in self.line_clear_animation['lines']:
                rect = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_WHITE, rect)

        # Draw ghost and current piece
        if self.current_piece and not self.game_over:
            # Ghost piece
            ghost_y = self.current_piece['y']
            while not self._check_collision(dy=ghost_y - self.current_piece['y'] + 1):
                ghost_y += 1
            
            shape_coords = self._get_current_shape_coords()
            for sx, sy in shape_coords:
                self._draw_cell(self.current_piece['x'] + sx, ghost_y + sy, self.COLOR_GHOST, is_ghost=True)

            # Current piece
            color = self.PIECE_COLORS[self.current_piece['id']]
            for sx, sy in shape_coords:
                self._draw_cell(self.current_piece['x'] + sx, self.current_piece['y'] + sy, color)

        # Draw grid lines
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, 
                             (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE), 
                             (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE))
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, 
                             (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y), 
                             (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        px, py = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE, self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE
        rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2)
        else:
            pygame.draw.rect(self.screen, color, rect)
            # Add a slight 3D effect
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, highlight, (px, py), (px + self.CELL_SIZE - 1, py))
            pygame.draw.line(self.screen, highlight, (px, py), (px, py + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, shadow, (px + self.CELL_SIZE - 1, py), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, shadow, (px, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))

    def _render_ui(self):
        ui_x = self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE + 30
        
        # Score
        score_text = self.font_main.render(f"SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_main.render(f"{self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (ui_x, 50))
        self.screen.blit(score_val, (ui_x, 80))

        # Lines
        lines_text = self.font_main.render(f"LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_main.render(f"{self.lines_cleared} / 10", True, self.COLOR_WHITE)
        self.screen.blit(lines_text, (ui_x, 140))
        self.screen.blit(lines_val, (ui_x, 170))

        # Next Piece
        next_text = self.font_main.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (ui_x, 230))
        preview_box = (ui_x - 5, 260, 4 * self.CELL_SIZE + 10, 4 * self.CELL_SIZE + 10)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box)
        
        if self.next_piece_shape:
            shape_coords = self.TETROMINOES[self.next_piece_shape][0]
            color = self.PIECE_COLORS[self.PIECE_IDS[self.next_piece_shape]]
            min_x = min(c[0] for c in shape_coords)
            min_y = min(c[1] for c in shape_coords)
            for sx, sy in shape_coords:
                px = ui_x + (sx - min_x) * self.CELL_SIZE
                py = 270 + (sy - min_y) * self.CELL_SIZE
                rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.lines_cleared >= 10 else "GAME OVER"
            text_surface = self.font_main.render(status_text, True, self.COLOR_WHITE)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Gymnasium Block Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # Construct action from keyboard input
        keys = pygame.key.get_pressed()
        
        # Movement: 0=none, 1=up, 2=down, 3=left, 4=right
        move_action = 0
        if keys[pygame.K_UP]:
            move_action = 1
        elif keys[pygame.K_DOWN]:
            move_action = 2
        elif keys[pygame.K_LEFT]:
            move_action = 3
        elif keys[pygame.K_RIGHT]:
            move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != -0.1:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Lines: {info['lines_cleared']}, Terminated: {terminated}")
        
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                    print("--- Game Reset ---")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()