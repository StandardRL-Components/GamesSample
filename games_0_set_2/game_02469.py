
# Generated: 2025-08-28T04:57:57.745854
# Source Brief: brief_02469.md
# Brief Index: 2469

        
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
        "Controls: ←→ to move, ↑ to rotate CW, Shift to rotate CCW, ↓ for soft drop, Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Manipulate falling blocks to clear lines and reach the target score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT = 10, 20
    CELL_SIZE = 18
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_TIME_BAR = (100, 200, 255)
    COLOR_UI_TIME_BAR_BG = (50, 50, 70)
    COLOR_WHITE = (255, 255, 255)

    # Tetromino shapes and colors
    TETROMINOS = {
        'I': ([[1, 1, 1, 1]], (66, 217, 245)),
        'O': ([[1, 1], [1, 1]], (245, 225, 66)),
        'T': ([[0, 1, 0], [1, 1, 1]], (188, 66, 245)),
        'J': ([[1, 0, 0], [1, 1, 1]], (66, 81, 245)),
        'L': ([[0, 0, 1], [1, 1, 1]], (245, 158, 66)),
        'S': ([[0, 1, 1], [1, 1, 0]], (102, 245, 66)),
        'Z': ([[1, 1, 0], [0, 1, 1]], (245, 66, 66))
    }
    TETROMINO_KEYS = list(TETROMINOS.keys())

    # Game parameters
    MAX_TIME_SECONDS = 60
    FPS = 30 # For auto_advance logic
    MAX_STEPS = MAX_TIME_SECONDS * FPS
    TARGET_LINES = 10
    
    # Input handling (Delayed Auto Shift)
    DAS_DELAY = 5 # frames
    DAS_SPEED = 2 # frames

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.playfield_origin_x = (self.SCREEN_WIDTH - self.PLAYFIELD_WIDTH * self.CELL_SIZE) // 2
        self.playfield_origin_y = (self.SCREEN_HEIGHT - self.PLAYFIELD_HEIGHT * self.CELL_SIZE) // 2
        
        # State variables are initialized in reset()
        self.playfield = None
        self.current_block = None
        self.next_block = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over = False
        self.fall_counter = 0
        self.fall_frequency = 0
        self.lock_timer = 0
        self.line_clear_animation = None
        self.last_reward = 0
        self.lines_cleared_this_turn = 0

        # Input state
        self.move_dir = 0
        self.move_timer = 0
        self.rotate_cooldown = 0
        self.hard_drop_used = False

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.playfield = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.fall_counter = 0
        self.fall_frequency = self.FPS // 2 # Initial fall speed: 2 cells per second
        self.lock_timer = 0
        self.line_clear_animation = None
        self.last_reward = 0
        self.lines_cleared_this_turn = 0

        self.next_block = self._get_new_block()
        self._spawn_new_block()

        # Input state reset
        self.move_dir = 0
        self.move_timer = 0
        self.rotate_cooldown = 0
        self.hard_drop_used = False

        return self._get_observation(), self._get_info()
    
    def _get_new_block(self):
        shape_key = self.np_random.choice(self.TETROMINO_KEYS)
        shape, color = self.TETROMINOS[shape_key]
        return {
            'shape': np.array(shape),
            'color': color,
            'x': self.PLAYFIELD_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'key': shape_key
        }

    def _spawn_new_block(self):
        self.current_block = self.next_block
        self.next_block = self._get_new_block()
        self.fall_counter = 0
        self.lock_timer = 0
        self.hard_drop_used = False

        if not self._is_valid_position(self.current_block):
            self.game_over = True

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        self.steps += 1
        self.time_remaining -= 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle line clear animation delay
        if self.line_clear_animation is not None:
            self.line_clear_animation['timer'] -= 1
            if self.line_clear_animation['timer'] <= 0:
                self._finish_line_clear()
            # Don't process game logic during animation, but return the reward from the clear
            return self._get_observation(), self.last_reward, self.game_over, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Process actions
        reward_from_action = self._handle_input(movement, space_held, shift_held)
        reward += reward_from_action

        # If hard drop happened, lock and spawn next piece immediately
        if self.hard_drop_used:
            # Sound effect placeholder: # pygame.mixer.Sound('hard_drop.wav').play()
            self._lock_block()
            line_reward, hole_penalty = self._process_lock()
            reward += line_reward + hole_penalty
            if not self.game_over:
                self._spawn_new_block()
        else:
            # Automatic falling
            self.fall_counter += 1
            if self.fall_counter >= self.fall_frequency:
                self.fall_counter = 0
                self._move_block(0, 1)

            # Check for landing and locking
            if not self._is_valid_position(self.current_block, dy=1):
                self.lock_timer += 1
                if self.lock_timer > self.FPS * 0.5: # 0.5 second lock delay
                    self._lock_block()
                    line_reward, hole_penalty = self._process_lock()
                    reward += line_reward + hole_penalty
                    if not self.game_over:
                        self._spawn_new_block()
            else:
                self.lock_timer = 0
        
        # Update difficulty based on lines cleared in the last turn
        if self.lines_cleared_this_turn > 0:
            lines_before = self.lines_cleared - self.lines_cleared_this_turn
            if (self.lines_cleared // 2) > (lines_before // 2):
                self.fall_frequency = max(5, int(self.fall_frequency * 0.95))
            self.lines_cleared_this_turn = 0

        terminated = self.game_over or self.time_remaining <= 0 or self.lines_cleared >= self.TARGET_LINES
        if self.lines_cleared >= self.TARGET_LINES and not self.game_over:
            reward += 100 # Win condition reward

        self.last_reward = reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # Action priority: Hard Drop > Rotation > Movement
        if space_held and not self.hard_drop_used:
            # Hard drop is a one-shot action per piece
            reward += 0.1 * (self.PLAYFIELD_HEIGHT - self.current_block['y']) # Reward for quick placement
            while self._is_valid_position(self.current_block, dy=1):
                self.current_block['y'] += 1
            self.hard_drop_used = True
            return reward

        if self.rotate_cooldown > 0: self.rotate_cooldown -= 1

        if movement == 1 and self.rotate_cooldown == 0: # Rotate CW
            if self._rotate_block(1): self.rotate_cooldown = 5
        elif shift_held and self.rotate_cooldown == 0: # Rotate CCW
            if self._rotate_block(-1): self.rotate_cooldown = 5
        
        # Soft Drop
        if movement == 2:
            if self._move_block(0, 1):
                reward += 0.1 # Reward for soft drop
                self.fall_counter = 0 # Reset fall timer

        # Horizontal Movement (DAS)
        move_request = 0
        if movement == 3: move_request = -1
        if movement == 4: move_request = 1

        if move_request != 0:
            if self.move_dir != move_request: # New direction
                self.move_dir = move_request
                self.move_timer = 0
                self._move_block(self.move_dir, 0)
            else: # Direction held
                self.move_timer += 1
                if self.move_timer > self.DAS_DELAY and self.move_timer % self.DAS_SPEED == 0:
                    self._move_block(self.move_dir, 0)
        else:
            self.move_dir = 0
            self.move_timer = 0
        
        return reward

    def _move_block(self, dx, dy):
        if self._is_valid_position(self.current_block, dx, dy):
            self.current_block['x'] += dx
            self.current_block['y'] += dy
            if dx != 0 or dy != 0:
                self.lock_timer = 0
            return True
        return False

    def _rotate_block(self, direction):
        original_shape = self.current_block['shape']
        rotated_shape = np.rot90(original_shape, k=-direction)
        test_offsets = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
        
        for dx, dy in test_offsets:
            if self._is_valid_position({'shape': rotated_shape, 'x': self.current_block['x'], 'y': self.current_block['y']}, dx, dy):
                self.current_block['shape'] = rotated_shape
                self.current_block['x'] += dx
                self.current_block['y'] += dy
                self.lock_timer = 0
                return True
        return False

    def _is_valid_position(self, block, dx=0, dy=0):
        shape = block['shape']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = block['x'] + x + dx
                    new_y = block['y'] + y + dy
                    if not (0 <= new_x < self.PLAYFIELD_WIDTH and 0 <= new_y < self.PLAYFIELD_HEIGHT):
                        return False
                    if self.playfield[new_y, new_x] != 0:
                        return False
        return True

    def _lock_block(self):
        # Sound effect placeholder: # pygame.mixer.Sound('lock.wav').play()
        shape = self.current_block['shape']
        color_index = self.TETROMINO_KEYS.index(self.current_block['key']) + 1
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    playfield_x = int(self.current_block['x'] + x)
                    playfield_y = int(self.current_block['y'] + y)
                    if 0 <= playfield_y < self.PLAYFIELD_HEIGHT:
                        self.playfield[playfield_y, playfield_x] = color_index

    def _process_lock(self):
        full_lines = [y for y in range(self.PLAYFIELD_HEIGHT) if np.all(self.playfield[y, :] != 0)]
        num_cleared = len(full_lines)
        
        if num_cleared > 0:
            # Sound effect placeholder: # pygame.mixer.Sound('clear.wav').play()
            self.line_clear_animation = {'lines': full_lines, 'timer': 5}
            self.lines_cleared += num_cleared
            self.lines_cleared_this_turn = num_cleared
            self.score += [0, 100, 300, 700, 1500][num_cleared]
            reward = {1: 1, 2: 3, 3: 7, 4: 15}.get(num_cleared, 0)
        else:
            reward = 0

        # Risky placement penalty is calculated regardless of line clear
        penalty = self._calculate_placement_penalty()
        return reward, penalty

    def _finish_line_clear(self):
        lines_to_clear = self.line_clear_animation['lines']
        self.playfield = np.delete(self.playfield, lines_to_clear, axis=0)
        new_rows = np.zeros((len(lines_to_clear), self.PLAYFIELD_WIDTH), dtype=int)
        self.playfield = np.vstack((new_rows, self.playfield))
        self.line_clear_animation = None

    def _calculate_placement_penalty(self):
        # Simplified check for holes created under the piece
        holes = 0
        shape = self.current_block['shape']
        for y_s, row in enumerate(shape):
            for x_s, cell in enumerate(row):
                if cell:
                    px, py = int(self.current_block['x'] + x_s), int(self.current_block['y'] + y_s)
                    for y_check in range(py + 1, self.PLAYFIELD_HEIGHT):
                        if self.playfield[y_check, px] == 0:
                            holes += 1
                        else:
                            break
        return -0.2 * holes

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.PLAYFIELD_WIDTH + 1):
            start = (self.playfield_origin_x + x * self.CELL_SIZE, self.playfield_origin_y)
            end = (self.playfield_origin_x + x * self.CELL_SIZE, self.playfield_origin_y + self.PLAYFIELD_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.PLAYFIELD_HEIGHT + 1):
            start = (self.playfield_origin_x, self.playfield_origin_y + y * self.CELL_SIZE)
            end = (self.playfield_origin_x + self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.playfield_origin_y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw settled blocks
        for y in range(self.PLAYFIELD_HEIGHT):
            for x in range(self.PLAYFIELD_WIDTH):
                if self.playfield[y, x] != 0:
                    color_key = self.TETROMINO_KEYS[int(self.playfield[y, x] - 1)]
                    color = self.TETROMINOS[color_key][1]
                    self._draw_cell(x, y, color)
        
        # Draw line clear animation
        if self.line_clear_animation is not None:
            for y in self.line_clear_animation['lines']:
                rect = pygame.Rect(self.playfield_origin_x, self.playfield_origin_y + y * self.CELL_SIZE, self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                flash_color = list(self.COLOR_WHITE)
                flash_color.append(128 + (self.line_clear_animation['timer'] % 2) * 127) # Flicker
                s = pygame.Surface(rect.size, pygame.SRCALPHA)
                s.fill(flash_color)
                self.screen.blit(s, rect.topleft)

        # Draw current block
        if self.current_block and not self.game_over:
            self._draw_block(self.current_block, self.playfield_origin_x, self.playfield_origin_y)

    def _draw_block(self, block, origin_x, origin_y):
        shape = block['shape']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self._draw_cell(block['x'] + x, block['y'] + y, block['color'], origin_x, origin_y)
    
    def _draw_cell(self, grid_x, grid_y, color, origin_x=None, origin_y=None):
        if origin_x is None: origin_x = self.playfield_origin_x
        if origin_y is None: origin_y = self.playfield_origin_y

        px = origin_x + grid_x * self.CELL_SIZE
        py = origin_y + grid_y * self.CELL_SIZE
        rect = pygame.Rect(int(px), int(py), self.CELL_SIZE, self.CELL_SIZE)
        
        glow_color = tuple(min(255, c + 70) for c in color)
        glow_size = int(self.CELL_SIZE * 1.5)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color + (30,), (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surf, (rect.centerx - glow_size // 2, rect.centery - glow_size // 2), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        highlight_color = tuple(min(255, c + 40) for c in color)
        pygame.draw.rect(self.screen, highlight_color, rect.inflate(-self.CELL_SIZE*0.5, -self.CELL_SIZE*0.5), border_radius=2)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        lines_text = self.font_small.render(f"LINES: {self.lines_cleared} / {self.TARGET_LINES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_text, (20, 50))

        time_bar_width, time_bar_height = 200, 20
        time_bar_x, time_bar_y = (self.SCREEN_WIDTH - time_bar_width) / 2, 20
        time_ratio = max(0, self.time_remaining / self.MAX_STEPS)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_TIME_BAR_BG, (time_bar_x, time_bar_y, time_bar_width, time_bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_TIME_BAR, (time_bar_x, time_bar_y, time_bar_width * time_ratio, time_bar_height), border_radius=5)

        next_text = self.font_small.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 100, 20))
        preview_origin_x, preview_origin_y = self.SCREEN_WIDTH - 120, 50
        
        preview_box = pygame.Rect(preview_origin_x - 10, preview_origin_y - 10, 100, 80)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box, 2, border_radius=5)

        if self.next_block:
            preview_block = self.next_block.copy()
            shape = preview_block['shape']
            preview_block['x'] = (4 - len(shape[0])) / 2
            preview_block['y'] = (3 - len(shape)) / 2
            self._draw_block(preview_block, preview_origin_x, preview_origin_y)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.lines_cleared >= self.TARGET_LINES else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "time_remaining_steps": self.time_remaining,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Puzzle Blocks")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0)

    game_is_terminated = False

    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        # Only process input if the game is not over
        if not game_is_terminated:
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
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    game_is_terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not game_is_terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines_cleared']}")
                game_is_terminated = True

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()