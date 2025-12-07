import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:08:27.193498
# Source Brief: brief_00869.md
# Brief Index: 869
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a Tetris-like stacking game.
    The player must stack falling geometric shapes to reach a target height
    while avoiding bombs and clearing blocks by matching colors.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling shapes to reach the target height. Clear blocks by matching colors of three or more, but watch out for bombs!"
    )
    user_guide = (
        "Controls: ←→ to move the piece, ↑↓ to rotate. Press space for a hard drop."
    )
    auto_advance = True


    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAY_AREA_WIDTH = 240  # 12 blocks wide
    PLAY_AREA_HEIGHT = 400  # 20 blocks high
    BLOCK_SIZE = 20
    GRID_WIDTH = PLAY_AREA_WIDTH // BLOCK_SIZE
    GRID_HEIGHT = PLAY_AREA_HEIGHT // BLOCK_SIZE

    TARGET_HEIGHT_UNITS = 200  # In pixels, as per brief
    MAX_STEPS = 1000
    WIN_REWARD = 100
    LOSS_REWARD = -100
    BOMB_PENALTY = -20
    CLEAR_REWARD = 10

    # --- Visuals ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_BG = (30, 35, 50)
    COLOR_BORDER = (80, 90, 110)
    COLOR_TARGET_LINE = (255, 200, 0, 150)
    COLOR_GHOST = (255, 255, 255, 50)
    COLOR_BOMB = (255, 50, 50)
    SHAPE_COLORS = [
        (66, 135, 245),   # Blue
        (245, 66, 66),    # Red
        (66, 245, 72),    # Green
        (245, 227, 66),   # Yellow
        (168, 66, 245),   # Purple
        (245, 135, 66),   # Orange
    ]

    # --- Shape Definitions ---
    SHAPES = {
        'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
        'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]],
        'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
        'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]], [[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
        'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]], [[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]],
        'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]], [[1, 1, 1, 1]], [[1], [1], [1], [1]]],
        'O': [[[1, 1], [1, 1]]]
    }
    SHAPE_KEYS = list(SHAPES.keys())

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        self.play_area_x_offset = (self.SCREEN_WIDTH - self.PLAY_AREA_WIDTH) // 2

        self.grid = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.current_piece = None
        self.piece_count = None
        self.fall_speed = None
        self.max_height_px = None
        self.particles = []
        self.screen_shake = 0
        self.prev_space_held = False
        self.action_cooldown = 0
        self.last_height_reward_base = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.piece_count = 0
        self.fall_speed = 2.0  # Pixels per step
        self.max_height_px = 0
        self.last_height_reward_base = 0
        self.particles = []
        self.screen_shake = 0
        self.prev_space_held = False
        self.action_cooldown = 0

        self._spawn_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        self._handle_player_input(movement)
        self._handle_hard_drop(space_held)
        self.prev_space_held = space_held

        self.current_piece['y'] += self.fall_speed
        current_grid_y = int(self.current_piece['y'] / self.BLOCK_SIZE)

        if not self._is_valid_position(self.current_piece['x'], current_grid_y, self.current_piece['shape']):
            self.current_piece['y'] -= self.fall_speed
            
            event_reward = self._place_and_process_piece()
            reward += event_reward
            
            height_reward = self._calculate_height_reward()
            reward += height_reward

            self._spawn_piece()

        self._update_difficulty()
        self._update_effects()

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        truncated = self.steps >= self.MAX_STEPS
        terminated = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_player_input(self, movement):
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            return

        moved = False
        if movement == 3:  # Left
            moved = self._move_piece(-1, 0)
        elif movement == 4:  # Right
            moved = self._move_piece(1, 0)
        elif movement == 1:  # Up -> Rotate CW
            moved = self._rotate_piece(1)
        elif movement == 2:  # Down -> Rotate CCW
            moved = self._rotate_piece(-1)
        
        if moved:
            self.action_cooldown = 4 # frames

    def _handle_hard_drop(self, space_held):
        if space_held and not self.prev_space_held:
            # sfx: hard_drop.wav
            while self._is_valid_position(self.current_piece['x'], int((self.current_piece['y'] + self.BLOCK_SIZE) / self.BLOCK_SIZE), self.current_piece['shape']):
                self.current_piece['y'] += self.BLOCK_SIZE

    def _place_and_process_piece(self):
        reward = 0
        if self.current_piece['type'] == 'BOMB':
            # sfx: bomb_explode.wav
            self._detonate_bomb()
            self.score += self.BOMB_PENALTY
            reward += self.BOMB_PENALTY
        else:
            # sfx: place_block.wav
            num_cleared_groups = self._place_shape()
            if num_cleared_groups > 0:
                # sfx: clear_blocks.wav
                clear_bonus = self.CLEAR_REWARD * num_cleared_groups
                self.score += clear_bonus
                reward += clear_bonus
        return reward
    
    def _calculate_height_reward(self):
        height_reward_base = self.max_height_px * 0.1
        reward = height_reward_base - self.last_height_reward_base
        self.last_height_reward_base = height_reward_base
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.fall_speed += 0.5  # px per step

    def _update_effects(self):
        self._update_particles()
        if self.screen_shake > 0:
            self.screen_shake -= 1

    def _check_termination(self):
        reward = 0
        terminated = False
        if self.max_height_px >= self.TARGET_HEIGHT_UNITS:
            self.score += self.WIN_REWARD
            reward += self.WIN_REWARD
            terminated = True
        elif self.score < -50:
            reward += self.LOSS_REWARD
            terminated = True
        elif self.game_over:  # Topped out
            reward += self.LOSS_REWARD
            terminated = True
        
        if terminated:
            self.game_over = True
        return terminated, reward

    def _spawn_piece(self):
        self.piece_count += 1
        is_bomb = (self.piece_count % 5 == 0)

        if is_bomb:
            shape_type, shape_matrix, color_idx = 'BOMB', [[1]], -1
        else:
            shape_type = self.np_random.choice(self.SHAPE_KEYS)
            shape_matrix = self.SHAPES[shape_type][0]
            color_idx = self.np_random.integers(len(self.SHAPE_COLORS))

        start_x = self.GRID_WIDTH // 2 - len(shape_matrix[0]) // 2
        self.current_piece = {'type': shape_type, 'color_idx': color_idx, 'shape': shape_matrix,
                              'rotation': 0, 'x': start_x, 'y': 0.0}
        
        if not self._is_valid_position(self.current_piece['x'], 0, self.current_piece['shape']):
            self.game_over = True

    def _move_piece(self, dx, dy):
        if self._is_valid_position(self.current_piece['x'] + dx, int((self.current_piece['y'] / self.BLOCK_SIZE) + dy), self.current_piece['shape']):
            self.current_piece['x'] += dx
            return True
        return False

    def _rotate_piece(self, direction):
        if self.current_piece['type'] == 'BOMB': return False
        
        rotations = self.SHAPES[self.current_piece['type']]
        new_rot = (self.current_piece['rotation'] + direction) % len(rotations)
        new_shape = rotations[new_rot]
        
        if self._is_valid_position(self.current_piece['x'], int(self.current_piece['y'] / self.BLOCK_SIZE), new_shape):
            self.current_piece['rotation'] = new_rot
            self.current_piece['shape'] = new_shape
            return True
        return False

    def _is_valid_position(self, grid_x, grid_y, shape):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    x, y = grid_x + c, grid_y + r
                    if not (0 <= x < self.GRID_WIDTH and y < self.GRID_HEIGHT): return False
                    if y >= 0 and self.grid[y, x] > 0: return False
        return True

    def _place_shape(self):
        shape, grid_x = self.current_piece['shape'], self.current_piece['x']
        final_y = int(self.current_piece['y'] / self.BLOCK_SIZE)
        color_val = self.current_piece['color_idx'] + 1
        
        placed_coords = []
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    y, x = final_y + r, grid_x + c
                    if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                        self.grid[y, x] = color_val
                        placed_coords.append((y, x))
        
        cleared_groups = self._check_for_clears(placed_coords)
        self._apply_gravity_to_grid()
        self._update_max_height()
        return cleared_groups

    def _check_for_clears(self, start_coords):
        to_clear, checked, cleared_groups = set(), set(), 0
        for r_start, c_start in start_coords:
            if (r_start, c_start) in checked: continue
            color = self.grid[r_start, c_start]
            if color == 0: continue
            
            q, group = [(r_start, c_start)], set()
            head = 0
            while head < len(q):
                r, c = q[head]; head+=1
                if (r,c) in group: continue
                group.add((r,c))
                checked.add((r,c))
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH and self.grid[nr, nc] == color:
                        q.append((nr, nc))
            
            if len(group) >= 3:
                to_clear.update(group)
                cleared_groups += 1

        if to_clear:
            for r, c in to_clear:
                color_idx = self.grid[r, c] - 1
                self.grid[r, c] = 0
                px = self.play_area_x_offset + c * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
                py = r * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
                self._create_particles(px, py, self.SHAPE_COLORS[color_idx], 3)
        return cleared_groups

    def _apply_gravity_to_grid(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _detonate_bomb(self):
        self.screen_shake = 10
        grid_x, grid_y = self.current_piece['x'], int(self.current_piece['y'] / self.BLOCK_SIZE)
        
        px = self.play_area_x_offset + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        self._create_particles(px, py, self.COLOR_BOMB, 50, 5.0)

        for r_offset in range(-1, 2):
            for c_offset in range(-1, 2):
                y, x = grid_y + r_offset, grid_x + c_offset
                if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                    self.grid[y, x] = 0
        
        self._apply_gravity_to_grid()
        self._update_max_height()

    def _update_max_height(self):
        non_empty = np.any(self.grid > 0, axis=1)
        self.max_height_px = (self.GRID_HEIGHT - np.min(np.where(non_empty))) * self.BLOCK_SIZE if np.any(non_empty) else 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        ox, oy = (self.np_random.integers(-self.screen_shake, self.screen_shake + 1) if self.screen_shake > 0 else 0 for _ in range(2))
        self._render_game(ox, oy)
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (p['x'] + ox, p['y'] + oy), int(p['size']))
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, ox, oy):
        play_area_rect = pygame.Rect(self.play_area_x_offset + ox, oy, self.PLAY_AREA_WIDTH, self.PLAY_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, play_area_rect)

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    self._draw_block(c, r, self.SHAPE_COLORS[self.grid[r, c] - 1], ox, oy)
        
        if self.current_piece and not self.game_over:
            ghost_y = self.current_piece['y']
            while self._is_valid_position(self.current_piece['x'], int((ghost_y + self.BLOCK_SIZE) / self.BLOCK_SIZE), self.current_piece['shape']):
                ghost_y += self.BLOCK_SIZE
            self._draw_piece(self.current_piece, ghost_y, self.COLOR_GHOST, ox, oy)
            
            color = self.COLOR_BOMB if self.current_piece['type'] == 'BOMB' else self.SHAPE_COLORS[self.current_piece['color_idx']]
            self._draw_piece(self.current_piece, self.current_piece['y'], color, ox, oy, is_pixel_y=True)

        pygame.draw.rect(self.screen, self.COLOR_BORDER, play_area_rect, 3)
        target_y = self.SCREEN_HEIGHT - self.TARGET_HEIGHT_UNITS + oy
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (play_area_rect.left, target_y), (play_area_rect.right, target_y), 2)

    def _draw_block(self, grid_x, grid_y, color, ox, oy, y_is_pixel=False):
        px = self.play_area_x_offset + grid_x * self.BLOCK_SIZE + ox
        py = (grid_y if y_is_pixel else grid_y * self.BLOCK_SIZE) + oy
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, color, rect)
        darker_color = tuple(max(0, c - 50) for c in color)
        pygame.draw.rect(self.screen, darker_color, rect, 2)

    def _draw_piece(self, piece, y_pos, color, ox, oy, is_pixel_y=False):
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    if color == self.COLOR_GHOST:
                        px = self.play_area_x_offset + (piece['x'] + c) * self.BLOCK_SIZE + ox
                        py = (y_pos if is_pixel_y else y_pos) + r * self.BLOCK_SIZE + oy
                        pygame.gfxdraw.box(self.screen, pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE), color)
                    else:
                        self._draw_block(piece['x'] + c, y_pos + r * self.BLOCK_SIZE, color, ox, oy, y_is_pixel=True)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        bar_x, bar_h, bar_w = self.play_area_x_offset + self.PLAY_AREA_WIDTH + 20, self.PLAY_AREA_HEIGHT, 20
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (bar_x, 0, bar_w, bar_h))
        fill_h = min(bar_h, (self.max_height_px / bar_h) * bar_h)
        if fill_h > 0: pygame.draw.rect(self.screen, (100, 200, 255), (bar_x, bar_h - fill_h, bar_w, fill_h))
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (bar_x, 0, bar_w, bar_h), 2)
        target_y = bar_h - (self.TARGET_HEIGHT_UNITS / bar_h) * bar_h
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (bar_x, target_y), (bar_x + bar_w, target_y), 2)

        if self.game_over:
            win = self.max_height_px >= self.TARGET_HEIGHT_UNITS
            text, color = ("YOU WIN!", (100, 255, 100)) if win else ("GAME OVER", (255, 100, 100))
            surf = self.font_game_over.render(text, True, color)
            self.screen.blit(surf, (self.SCREEN_WIDTH // 2 - surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - surf.get_height() // 2))

    def _create_particles(self, x, y, color, count=10, max_speed=3.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, max_speed)
            self.particles.append({'x': x, 'y': y, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                                   'size': self.np_random.uniform(2, 5), 'lifespan': self.np_random.integers(20, 40), 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['size'] > 0.5]
        for p in self.particles:
            p['x'], p['y'], p['lifespan'], p['size'] = p['x'] + p['vx'], p['y'] + p['vy'], p['lifespan'] - 1, p['size'] * 0.97

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "max_height": self.max_height_px, "pieces": self.piece_count}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # The __main__ block is not part of the environment definition and is for local testing.
    # It will not be run by the evaluation server.
    env = GameEnv()
    obs, info = env.reset()
    
    # For local testing with display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()
    running, total_reward = True, 0.0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        # Step the environment
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle game over
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0.0

        # Control the frame rate
        clock.tick(30)
        
    env.close()
    pygame.quit()