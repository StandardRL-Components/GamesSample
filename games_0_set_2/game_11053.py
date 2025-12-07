import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:25:15.898066
# Source Brief: brief_01053.md
# Brief Index: 1053
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Entangled Tetris: A Gymnasium environment where the player manipulates falling
    quantum blocks, matches colors to generate energy, and triggers cascading
    portal effects to survive. This game prioritizes visual flair and satisfying
    game feel over strict Tetris simulation.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate falling quantum blocks, match colors to generate energy, and trigger portal effects to survive in this Tetris-inspired game."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate, ↓ to soft drop, and space to hard drop."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = self.HEIGHT - (self.GRID_HEIGHT * self.CELL_SIZE) - 10
        self.MAX_STEPS = 10000
        self.MAX_PARTICLES = 200

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self._define_colors_and_fonts()
        self._define_tetrominoes()

        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = None
        self.level = None
        self.lines_cleared = None
        self.game_over = None
        self.steps = None
        self.fall_timer = None
        self.fall_speed = None
        self.energy = None
        self.particles = None
        self.line_clear_animation = None
        self.bg_pulse = None
        self.prev_space_held = None
        self.prev_up_held = None
        self.reward_queue = None

        self.reset()

    def _define_colors_and_fonts(self):
        self.COLOR_BG = (15, 10, 35)
        self.COLOR_GRID = (30, 25, 60)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_UI_BAR = (50, 40, 90)
        self.COLOR_UI_BAR_FILL = (100, 200, 255)

        self.COLORS = {
            1: {'main': (255, 50, 50), 'glow': (120, 20, 20)},   # Red
            2: {'main': (50, 255, 50), 'glow': (20, 120, 20)},   # Green
            3: {'main': (50, 150, 255), 'glow': (20, 60, 120)},  # Blue
            4: {'main': (255, 255, 50), 'glow': (120, 120, 20)}, # Yellow (for a 4th color)
            5: {'main': (255, 50, 255), 'glow': (120, 20, 120)}, # Magenta (Portal)
        }
        self.COLOR_GHOST = (255, 255, 255, 40)
        self.FONT_UI = pygame.font.Font(None, 28)
        self.FONT_GAMEOVER = pygame.font.Font(None, 64)

    def _define_tetrominoes(self):
        self.TETROMINOES = {
            'I': [[1, 1, 1, 1]],
            'O': [[1, 1], [1, 1]],
            'T': [[0, 1, 0], [1, 1, 1]],
            'L': [[0, 0, 1], [1, 1, 1]],
            'J': [[1, 0, 0], [1, 1, 1]],
            'S': [[0, 1, 1], [1, 1, 0]],
            'Z': [[1, 1, 0], [0, 1, 1]],
        }
        self.PORTAL_PIECE = {'shape': [[1, 1], [1, 1]], 'color': 5, 'is_portal': True}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.game_over = False
        self.steps = 0
        self.fall_timer = 0
        self.fall_speed = 30 # Steps per grid cell fall
        self.energy = 0
        self.particles = deque()
        self.line_clear_animation = [] # list of (row_index, timer)
        self.bg_pulse = 0
        self.prev_space_held = False
        self.prev_up_held = False
        self.reward_queue = []

        self.next_piece = self._create_new_piece()
        self._spawn_new_piece()

        return self._get_observation(), self._get_info()

    def _create_new_piece(self):
        # Increase portal spawn chance with level
        portal_chance = min(0.3, 0.02 * (self.level // 5))
        if self.np_random.random() < portal_chance and self.level > 1:
            return self.PORTAL_PIECE.copy()

        shape_name = self.np_random.choice(list(self.TETROMINOES.keys()))
        return {
            'shape': self.TETROMINOES[shape_name],
            'color': self.np_random.integers(1, 5), # 1-4 are regular colors
            'x': self.GRID_WIDTH // 2 - len(self.TETROMINOES[shape_name][0]) // 2,
            'y': 0,
            'is_portal': False
        }

    def _spawn_new_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self._create_new_piece()
        self.current_piece['x'] = self.GRID_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
        self.current_piece['y'] = 0

        if not self._is_valid_position(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y']):
            self.game_over = True
            # sfx: game over
            self.reward_queue.append(-10.0) # Terminal penalty

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_queue.append(-0.01) # Small time penalty

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle player actions ---
        if movement == 1 and not self.prev_up_held: # Rotate (Up)
            self._rotate_piece()
        elif movement == 2: # Soft drop (Down)
            self._move_piece(0, 1)
            self.fall_timer = 0 # Reset gravity timer
        elif movement == 3: # Move Left
            self._move_piece(-1, 0)
        elif movement == 4: # Move Right
            self._move_piece(1, 0)

        if space_held and not self.prev_space_held: # Hard drop
            self._hard_drop()

        self.prev_space_held = space_held
        self.prev_up_held = (movement == 1)

        # --- Update game state (gravity) ---
        self.fall_timer += 1
        if self.fall_timer >= self.fall_speed:
            self.fall_timer = 0
            self._move_piece(0, 1)

        reward = self._calculate_reward()
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _move_piece(self, dx, dy):
        if self._is_valid_position(self.current_piece['shape'], self.current_piece['x'] + dx, self.current_piece['y'] + dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        elif dy > 0: # If move down failed, lock piece
            self._lock_piece()
        return False

    def _rotate_piece(self):
        if self.current_piece['is_portal']: return # Portals don't rotate
        # sfx: rotate
        original_shape = self.current_piece['shape']
        rotated_shape = list(zip(*original_shape[::-1]))
        
        # Wall kick logic
        for kick_dx in [0, 1, -1, 2, -2]:
            if self._is_valid_position(rotated_shape, self.current_piece['x'] + kick_dx, self.current_piece['y']):
                self.current_piece['shape'] = rotated_shape
                self.current_piece['x'] += kick_dx
                return

    def _hard_drop(self):
        # sfx: hard drop
        while self._move_piece(0, 1):
            pass # Keep moving down until it locks
        self.fall_timer = self.fall_speed # Force next step to be a lock check if somehow not locked

    def _lock_piece(self):
        # sfx: lock
        shape = self.current_piece['shape']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = self.current_piece['x'] + c, self.current_piece['y'] + r
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y][grid_x] = self.current_piece['color']

        self._post_lock_updates()
        self._spawn_new_piece()

    def _post_lock_updates(self):
        # Order of operations is important here
        if self.current_piece['is_portal']:
            self._handle_portal_activation()
        else:
            self._check_color_matches()
        
        self._check_line_clears()
        
        # Update level and speed
        new_level = (self.lines_cleared // 10) + 1
        if new_level > self.level:
            self.level = new_level
            # sfx: level up
            self.fall_speed = max(5, 30 - (self.level - 1) * 2) # Faster fall speed
            self.bg_pulse = 20 # visual feedback for level up

    def _check_color_matches(self):
        shape, x, y, color = self.current_piece['shape'], self.current_piece['x'], self.current_piece['y'], self.current_piece['color']
        matches = 0
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    px, py = x + c_idx, y + r_idx
                    # Check neighbors (up, down, left, right)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                            if self.grid[ny][nx] == color:
                                matches += 1
                                self.energy = min(100, self.energy + 5)
                                self._spawn_particles(px, py, color)
        if matches > 0:
            # sfx: color match
            self.reward_queue.append(matches * 0.1)

    def _handle_portal_activation(self):
        # sfx: portal activate
        self.reward_queue.append(2.0)
        self.bg_pulse = 30 # Big visual feedback
        y_pos = self.current_piece['y']
        
        # Shift a random row that the 2x2 portal block occupies
        row_to_shift = y_pos + self.np_random.integers(0, 2)
        if row_to_shift < self.GRID_HEIGHT:
            direction = self.np_random.choice([-1, 1]) # -1 for left, 1 for right
            self.grid[row_to_shift] = np.roll(self.grid[row_to_shift], direction)

    def _check_line_clears(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r] > 0):
                lines_to_clear.append(r)

        if lines_to_clear:
            # sfx: line clear (multiplied by number of lines)
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            self.score += [0, 100, 300, 500, 1000][num_cleared] * self.level
            self.reward_queue.append([0, 1, 3, 5, 10][num_cleared])
            self.bg_pulse = 15

            for r in lines_to_clear:
                self.line_clear_animation.append([r, 10]) # row index, timer
                self.grid[r] = 0 # Clear the line logically
            
            # Shift blocks down
            lines_cleared_sorted = sorted(lines_to_clear, reverse=True)
            for r in lines_cleared_sorted:
                for row_to_move in range(r - 1, -1, -1):
                    self.grid[row_to_move + 1] = self.grid[row_to_move]
                    self.grid[row_to_move] = 0

    def _is_valid_position(self, shape, grid_x, grid_y):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    x, y = grid_x + c, grid_y + r
                    if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT and self.grid[y][x] == 0):
                        return False
        return True
    
    def _calculate_reward(self):
        reward = sum(self.reward_queue)
        self.reward_queue.clear()
        return reward

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "lines": self.lines_cleared}

    def _get_observation(self):
        # --- Update Animations ---
        if self.bg_pulse > 0: self.bg_pulse -= 1
        
        # --- Render Background ---
        bg_color = self.COLOR_BG
        if self.bg_pulse > 0:
            pulse_alpha = self.bg_pulse / 30.0
            bg_color = tuple(int(c1 + (c2 - c1) * pulse_alpha) for c1, c2 in zip(self.COLOR_BG, self.COLOR_GRID))
        self.screen.fill(bg_color)

        # --- Render Game Elements ---
        self._render_grid()
        self._render_ghost_piece()
        self._render_falling_piece()
        self._render_particles()
        self._render_line_clear_animation()
        
        # --- Render UI ---
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE), 1)
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE), (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y + i * self.CELL_SIZE), 1)
        
        # Draw locked blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] > 0:
                    self._draw_block(c, r, self.grid[r][c])

    def _render_falling_piece(self):
        if self.current_piece and not self.game_over:
            shape, color = self.current_piece['shape'], self.current_piece['color']
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece['x'] + c, self.current_piece['y'] + r, color, is_falling=True)

    def _render_ghost_piece(self):
        if self.current_piece and not self.game_over:
            ghost_y = self.current_piece['y']
            while self._is_valid_position(self.current_piece['shape'], self.current_piece['x'], ghost_y + 1):
                ghost_y += 1
            
            shape = self.current_piece['shape']
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        px = self.GRID_X + (self.current_piece['x'] + c) * self.CELL_SIZE
                        py = self.GRID_Y + (ghost_y + r) * self.CELL_SIZE
                        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                        s.fill(self.COLOR_GHOST)
                        self.screen.blit(s, (px, py))

    def _draw_block(self, grid_c, grid_r, color_id, is_falling=False):
        px, py = self.GRID_X + grid_c * self.CELL_SIZE, self.GRID_Y + grid_r * self.CELL_SIZE
        color_map = self.COLORS[color_id]
        main_color, glow_color = color_map['main'], color_map['glow']
        
        # Glow effect for falling piece
        if is_falling:
            glow_radius = int(self.CELL_SIZE * 1.2)
            pygame.gfxdraw.filled_circle(self.screen, px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2, glow_radius, (*glow_color, 80))
            pygame.gfxdraw.filled_circle(self.screen, px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2, glow_radius - 3, (*main_color, 60))

        # Main block with border effect
        pygame.draw.rect(self.screen, glow_color, (px, py, self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.rect(self.screen, main_color, (px + 2, py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))
    
    def _render_line_clear_animation(self):
        for i in range(len(self.line_clear_animation) - 1, -1, -1):
            row_idx, timer = self.line_clear_animation[i]
            alpha = int(255 * (timer / 10.0))
            color = (255, 255, 255, alpha)
            
            rect = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            rect.fill(color)
            self.screen.blit(rect, (self.GRID_X, self.GRID_Y + row_idx * self.CELL_SIZE))
            
            self.line_clear_animation[i][1] -= 1
            if self.line_clear_animation[i][1] <= 0:
                self.line_clear_animation.pop(i)

    def _spawn_particles(self, grid_c, grid_r, color_id):
        if len(self.particles) > self.MAX_PARTICLES: return
        px = self.GRID_X + grid_c * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_Y + grid_r * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.COLORS[color_id]['main']
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _render_particles(self):
        for i in range(len(self.particles)):
            p = self.particles.popleft()
            p[0] += p[2] # update x
            p[1] += p[3] # update y
            p[4] -= 1    # update lifetime
            if p[4] > 0:
                size = int(max(0, (p[4] / 30.0) * 4))
                if size > 0:
                    pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), size)
                self.particles.append(p)

    def _render_ui(self):
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Level & Lines
        level_text = self.FONT_UI.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (20, 50))
        lines_text = self.FONT_UI.render(f"LINES: {self.lines_cleared}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_text, (20, 80))

        # Energy Bar
        energy_text = self.FONT_UI.render("ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (self.WIDTH - 170, 20))
        bar_x, bar_y, bar_w, bar_h = self.WIDTH - 170, 50, 150, 20
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * (self.energy / 100.0))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, fill_w, bar_h))

        # Next Piece Preview
        next_text = self.FONT_UI.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.WIDTH - 170, 100))
        preview_bg_rect = pygame.Rect(self.WIDTH - 170, 130, 150, 100)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_bg_rect, 0, 5)
        
        if self.next_piece:
            shape = self.next_piece['shape']
            color = self.next_piece['color']
            shape_w = len(shape[0]) * self.CELL_SIZE
            shape_h = len(shape) * self.CELL_SIZE
            start_x = preview_bg_rect.centerx - shape_w // 2
            start_y = preview_bg_rect.centery - shape_h // 2
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        px, py = start_x + c * self.CELL_SIZE, start_y + r * self.CELL_SIZE
                        # Simplified drawing for preview
                        pygame.draw.rect(self.screen, self.COLORS[color]['glow'], (px, py, self.CELL_SIZE, self.CELL_SIZE))
                        pygame.draw.rect(self.screen, self.COLORS[color]['main'], (px + 2, py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.FONT_GAMEOVER.render("GAME OVER", True, (255, 50, 50))
        text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(game_over_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window and render the environment
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc. depending on your OS
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Entangled Tetris")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0

    # Game loop for human player
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0 # Note: shift is not used in the game logic

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Optional: Add a delay and reset to play again
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            terminated = False

        clock.tick(30) # Run at 30 FPS for smooth human playback
        
    env.close()