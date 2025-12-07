
# Generated: 2025-08-27T21:37:57.563221
# Source Brief: brief_02856.md
# Brief Index: 2856

        
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
        "Controls: ←→ to move, ↓ for soft drop. ↑ to rotate clockwise, "
        "Shift to rotate counter-clockwise. Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based falling block puzzle game. Clear lines to score, "
        "but watch out for gaps! The game ends if the blocks reach the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    MAX_STEPS = 5000
    WIN_SCORE = 1000

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 30, 40)
    COLOR_UI_BORDER = (60, 60, 70)
    
    # Tetromino shapes and colors
    TETROMINOES = {
        'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
        'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
        'O': [[[1, 1], [1, 1]]],
        'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]],
        'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
        'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
        'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]]
    }
    
    BLOCK_COLORS = [
        (0, 0, 0),  # 0 is empty
        (199, 82, 224),   # T - Purple
        (82, 224, 221),   # I - Cyan
        (224, 221, 82),   # O - Yellow
        (224, 150, 82),   # L - Orange
        (82, 112, 224),   # J - Blue
        (112, 224, 82),   # S - Green
        (224, 82, 82),    # Z - Red
    ]

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
        
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2 - 80
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Fallback if seed is not provided
            if not hasattr(self, 'rng'):
                 self.rng = np.random.default_rng()

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.block_bag = list(self.TETROMINOES.keys())
        self.rng.shuffle(self.block_bag)

        self.next_block_shape = self._get_next_from_bag()
        self._spawn_block()

        self.fall_speed_initial = 0.8  # seconds per grid cell
        self.fall_speed_current = self.fall_speed_initial
        self.fall_timer = 0.0

        self.last_space_held = False
        self.last_shift_held = False
        self.last_up_held = False

        self.line_clear_animation = [] # list of (y_coord, timer)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.02  # Small penalty per step to encourage speed
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        up_held = movement == 1

        # --- Handle player input ---
        # Rising edge detection for single-press actions
        hard_drop = space_held and not self.last_space_held
        rotate_cw = up_held and not self.last_up_held
        rotate_ccw = shift_held and not self.last_shift_held

        if hard_drop:
            # sfx: hard_drop_sound
            reward += self._hard_drop()
            self._place_block_and_proceed()
        else:
            if rotate_cw:
                self._rotate_block(1) # sfx: rotate_sound
            if rotate_ccw:
                self._rotate_block(-1) # sfx: rotate_sound

            if movement == 3: # Left
                self._move_block(-1, 0)
            elif movement == 4: # Right
                self._move_block(1, 0)
            
            # --- Auto-fall logic ---
            time_delta = 1/30.0 # Assuming 30 FPS
            soft_drop_multiplier = 10.0 if movement == 2 else 1.0
            self.fall_timer += time_delta * soft_drop_multiplier

            if self.fall_timer >= self.fall_speed_current:
                self.fall_timer = 0
                if not self._move_block(0, 1):
                    # Block has landed
                    reward += self._place_block_and_proceed()
        
        # --- Update game state ---
        self._update_animations(1/30.0)
        self.fall_speed_current = self.fall_speed_initial - (self.steps // 200) * 0.05
        self.fall_speed_current = max(0.1, self.fall_speed_current) # Cap fall speed

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        self.last_up_held = up_held

        terminated = self._check_termination()
        if terminated and self.score >= self.WIN_SCORE:
            reward += 100 # Win bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_block_and_proceed(self):
        # sfx: block_land_sound
        hole_penalty = self._calculate_hole_penalty()
        self._place_block()
        lines_cleared, reward_lines = self._clear_lines()
        
        if lines_cleared > 0:
            # sfx: line_clear_sound
            self.score += (lines_cleared ** 2) * 10
        
        self._spawn_block()
        return hole_penalty + reward_lines

    def _check_termination(self):
        if self.game_over: return True
        if self.steps >= self.MAX_STEPS: return True
        if self.score >= self.WIN_SCORE: return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Game Logic Helpers ---

    def _get_next_from_bag(self):
        if not self.block_bag:
            self.block_bag = list(self.TETROMINOES.keys())
            self.rng.shuffle(self.block_bag)
        return self.block_bag.pop()

    def _spawn_block(self):
        self.current_block = {
            'type': self.next_block_shape,
            'rotation': 0,
            'pos': [self.GRID_WIDTH // 2 - 1, 0]
        }
        self.current_block['shape'] = self.TETROMINOES[self.current_block['type']][0]
        self.current_block['color_idx'] = list(self.TETROMINOES.keys()).index(self.current_block['type']) + 1
        
        self.next_block_shape = self._get_next_from_bag()

        if self._check_collision(self.current_block['shape'], self.current_block['pos']):
            self.game_over = True

    def _check_collision(self, shape, pos):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = pos[0] + x, pos[1] + y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True
                    if self.grid[grid_y, grid_x] != 0:
                        return True
        return False

    def _move_block(self, dx, dy):
        new_pos = [self.current_block['pos'][0] + dx, self.current_block['pos'][1] + dy]
        if not self._check_collision(self.current_block['shape'], new_pos):
            self.current_block['pos'] = new_pos
            return True
        return False

    def _rotate_block(self, direction):
        rotations = self.TETROMINOES[self.current_block['type']]
        num_rotations = len(rotations)
        new_rotation_idx = (self.current_block['rotation'] + direction) % num_rotations
        new_shape = rotations[new_rotation_idx]

        # Wall kick logic
        for kick_dx in [0, -1, 1, -2, 2]: # Standard kicks
            new_pos = [self.current_block['pos'][0] + kick_dx, self.current_block['pos'][1]]
            if not self._check_collision(new_shape, new_pos):
                self.current_block['rotation'] = new_rotation_idx
                self.current_block['shape'] = new_shape
                self.current_block['pos'] = new_pos
                return

    def _hard_drop(self):
        dy = 0
        while not self._check_collision(self.current_block['shape'], [self.current_block['pos'][0], self.current_block['pos'][1] + dy + 1]):
            dy += 1
        self.current_block['pos'][1] += dy
        return 0 # No direct reward for hard drop, reward comes from placement

    def _place_block(self):
        shape = self.current_block['shape']
        pos = self.current_block['pos']
        color_idx = self.current_block['color_idx']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = pos[0] + x, pos[1] + y
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = color_idx

    def _clear_lines(self):
        lines_to_clear = [y for y, row in enumerate(self.grid) if np.all(row != 0)]
        if not lines_to_clear:
            return 0, 0
        
        for y in lines_to_clear:
            self.line_clear_animation.append({'y': y, 'timer': 0.2})
        
        self.grid = np.delete(self.grid, lines_to_clear, axis=0)
        new_lines = np.zeros((len(lines_to_clear), self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack((new_lines, self.grid))
        
        num_cleared = len(lines_to_clear)
        reward = 1.0 if num_cleared == 1 else 2.0
        return num_cleared, reward

    def _calculate_hole_penalty(self):
        penalty = 0
        shape = self.current_block['shape']
        pos = self.current_block['pos']
        
        for y_local, row in enumerate(shape):
            for x_local, cell in enumerate(row):
                if cell:
                    grid_x = pos[0] + x_local
                    # Check for holes directly under this cell
                    for y_scan in range(pos[1] + y_local + 1, self.GRID_HEIGHT):
                        if self.grid[y_scan, grid_x] == 0:
                            penalty -= 0.2
                        else:
                            break # Hit another block, stop scanning down this column
        return penalty

    # --- Rendering Helpers ---

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.grid_offset_x, self.grid_offset_y, self.grid_pixel_width, self.grid_pixel_height))

        # Draw placed blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    color_idx = int(self.grid[y, x])
                    self._draw_cell(x, y, self.BLOCK_COLORS[color_idx])

        # Draw ghost piece
        if not self.game_over:
            ghost_pos = self.current_block['pos'][:]
            while not self._check_collision(self.current_block['shape'], [ghost_pos[0], ghost_pos[1] + 1]):
                ghost_pos[1] += 1
            self._draw_block(self.current_block, pos=ghost_pos, ghost=True)

        # Draw current falling block
        if not self.game_over:
            self._draw_block(self.current_block)

        # Draw grid lines on top
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.grid_offset_x, self.grid_offset_y + y * self.CELL_SIZE), (self.grid_offset_x + self.grid_pixel_width, self.grid_offset_y + y * self.CELL_SIZE))
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.grid_offset_x + x * self.CELL_SIZE, self.grid_offset_y), (self.grid_offset_x + x * self.CELL_SIZE, self.grid_offset_y + self.grid_pixel_height))

        # Draw line clear animation
        for anim in self.line_clear_animation:
            y_pos = self.grid_offset_y + anim['y'] * self.CELL_SIZE
            alpha = max(0, min(255, int(255 * (anim['timer'] / 0.2) * 2)))
            flash_surface = pygame.Surface((self.grid_pixel_width, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.grid_offset_x, y_pos))

    def _render_ui(self):
        # UI Panel for Score and Next Block
        ui_x = self.grid_offset_x + self.grid_pixel_width + 20
        ui_y = self.grid_offset_y
        ui_width = 160
        ui_height = 200
        
        # Score Box
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x, ui_y, ui_width, 80))
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (ui_x, ui_y, ui_width, 80), 2)
        score_text = self.font_main.render(f"{self.score}", True, self.COLOR_TEXT)
        score_label = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_label, (ui_x + (ui_width - score_label.get_width())//2, ui_y + 10))
        self.screen.blit(score_text, (ui_x + (ui_width - score_text.get_width())//2, ui_y + 40))

        # Next Block Box
        next_y = ui_y + 90
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x, next_y, ui_width, 120))
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (ui_x, next_y, ui_width, 120), 2)
        next_label = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_label, (ui_x + (ui_width - next_label.get_width())//2, next_y + 10))
        
        # Draw the next block inside its box
        next_block_type = self.next_block_shape
        next_shape = self.TETROMINOES[next_block_type][0]
        next_color_idx = list(self.TETROMINOES.keys()).index(next_block_type) + 1
        shape_w = len(next_shape[0]) * self.CELL_SIZE
        shape_h = len(next_shape) * self.CELL_SIZE
        
        for y, row in enumerate(next_shape):
            for x, cell in enumerate(row):
                if cell:
                    px = ui_x + (ui_width - shape_w) // 2 + x * self.CELL_SIZE
                    py = next_y + 35 + (60 - shape_h) // 2 + y * self.CELL_SIZE
                    self._draw_raw_cell(px, py, self.BLOCK_COLORS[next_color_idx])
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text_str = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_TEXT)
            self.screen.blit(end_text, ((self.SCREEN_WIDTH - end_text.get_width()) // 2, (self.SCREEN_HEIGHT - end_text.get_height()) // 2))

    def _draw_block(self, block, pos=None, ghost=False):
        shape = block['shape']
        render_pos = pos or block['pos']
        color = self.BLOCK_COLORS[block['color_idx']]
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self._draw_cell(render_pos[0] + x, render_pos[1] + y, color, ghost)

    def _draw_cell(self, grid_x, grid_y, color, ghost=False):
        px, py = self.grid_offset_x + grid_x * self.CELL_SIZE, self.grid_offset_y + grid_y * self.CELL_SIZE
        self._draw_raw_cell(px, py, color, ghost)

    def _draw_raw_cell(self, px, py, color, ghost=False):
        if ghost:
            pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE), 2)
        else:
            main_color = tuple(c * 0.7 for c in color)
            light_color = color
            dark_color = tuple(c * 0.5 for c in color)
            
            # Main face
            pygame.draw.rect(self.screen, main_color, (px + 2, py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))
            # Top and Left highlight
            pygame.draw.polygon(self.screen, light_color, [(px, py), (px + self.CELL_SIZE, py), (px + self.CELL_SIZE - 2, py + 2), (px + 2, py + 2), (px + 2, py + self.CELL_SIZE - 2), (px, py + self.CELL_SIZE)])
            # Bottom and Right shadow
            pygame.draw.polygon(self.screen, dark_color, [(px + self.CELL_SIZE, py + self.CELL_SIZE), (px, py + self.CELL_SIZE), (px + 2, py + self.CELL_SIZE - 2), (px + self.CELL_SIZE - 2, py + self.CELL_SIZE - 2), (px + self.CELL_SIZE - 2, py + 2), (px + self.CELL_SIZE, py)])

    def _update_animations(self, dt):
        # Line clear animation
        for anim in self.line_clear_animation[:]:
            anim['timer'] -= dt
            if anim['timer'] <= 0:
                self.line_clear_animation.remove(anim)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        ''' Call this at the end of __init__ to verify implementation. '''
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


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gymnasium Block Puzzle")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      HUMAN PLAYTHROUGH")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(observation, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS

    env.close()