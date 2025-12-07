
# Generated: 2025-08-28T03:29:16.873097
# Source Brief: brief_04942.md
# Brief Index: 4942

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, SHIFT to rotate counter-clockwise. ↓ for soft drop, SPACE for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Block Blitz: A fast-paced, top-down block puzzle game. Strategically place falling blocks to clear lines and achieve a high score before the stack reaches the top."
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
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X_OFFSET = (self.screen_width - self.GRID_WIDTH * self.CELL_SIZE) // 2 - 100
        self.GRID_Y_OFFSET = (self.screen_height - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        # Visuals
        self._init_colors()
        self._init_fonts()
        self._init_tetrominoes()

        # State variables (initialized in reset)
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = None
        self.lines_cleared = None
        self.level = None
        self.game_over = None
        self.win = None
        
        self.fall_speed = None
        self.fall_counter = None
        self.lock_delay_counter = None
        self.line_clear_animation = None
        self.particles = None
        self.steps = None
        
        self.last_action_time = 0
        self.action_cooldown = 2 # frames

        self.rng = None

        # Initialize state
        self.reset()
        self.validate_implementation()

    def _init_colors(self):
        self.COLOR_BG = (25, 30, 45)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (35, 40, 60)
        self.COLOR_UI_BORDER = (60, 70, 90)

        self.TETROMINO_COLORS = [
            (0, 240, 240),   # I (Cyan)
            (240, 240, 0),   # O (Yellow)
            (160, 0, 240),   # T (Purple)
            (0, 0, 240),     # J (Blue)
            (240, 160, 0),   # L (Orange)
            (0, 240, 0),     # S (Green)
            (240, 0, 0),     # Z (Red)
        ]
        self.GHOST_ALPHA = 80
        self.LANDED_DESATURATION = 0.5

    def _init_fonts(self):
        try:
            self.font_main = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_info = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_gameover = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_main = pygame.font.SysFont("Arial", 24)
            self.font_info = pygame.font.SysFont("Arial", 18)
            self.font_gameover = pygame.font.SysFont("Arial", 48)

    def _init_tetrominoes(self):
        # 0: I, 1: O, 2: T, 3: J, 4: L, 5: S, 6: Z
        self.TETROMINOES = [
            [[[0,1],[1,1],[2,1],[3,1]], [[1,0],[1,1],[1,2],[1,3]]], # I
            [[[1,1],[1,2],[2,1],[2,2]]], # O
            [[[0,1],[1,1],[2,1],[1,2]], [[1,0],[1,1],[1,2],[0,1]], [[0,1],[1,1],[2,1],[1,0]], [[1,0],[1,1],[1,2],[2,1]]], # T
            [[[0,0],[0,1],[1,1],[2,1]], [[1,0],[1,1],[1,2],[2,2]], [[0,1],[1,1],[2,1],[2,2]], [[0,0],[1,0],[1,1],[1,2]]], # J
            [[[2,0],[0,1],[1,1],[2,1]], [[1,0],[1,1],[1,2],[0,2]], [[0,1],[1,1],[2,1],[0,2]], [[2,0],[1,0],[1,1],[1,2]]], # L
            [[[1,1],[2,1],[0,2],[1,2]], [[1,1],[1,2],[2,0],[2,1]]], # S
            [[[0,1],[1,1],[1,2],[2,2]], [[2,1],[2,2],[1,0],[1,1]]], # Z
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)
        else:
            # Use a default generator if no seed is provided
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.steps = 0
        self.game_over = False
        self.win = False

        self.fall_speed = 1.0 # cells per second
        self.fall_counter = 0.0
        self.lock_delay_counter = 0
        
        self.line_clear_animation = []
        self.particles = []
        self.last_action_time = 0
        
        self._spawn_piece()
        self._spawn_piece() # Populates current_piece and next_piece

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed

        # Handle line clear animation pause
        if self.line_clear_animation:
            self._update_line_clear_animation()
        else:
            # Process actions
            movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # Action cooldown to prevent single presses from firing too rapidly
            if self.steps > self.last_action_time + self.action_cooldown:
                if movement == 1: # Rotate CW
                    self._rotate(clockwise=True)
                    self.last_action_time = self.steps
                if shift_held: # Rotate CCW
                    self._rotate(clockwise=False)
                    self.last_action_time = self.steps

            # Continuous actions (movement, soft drop)
            if movement == 3: self._move(-1) # Left
            if movement == 4: self._move(1)  # Right
            
            # Hard drop is a single, terminal action for the piece
            if space_pressed:
                drop_reward = self._hard_drop()
                reward += drop_reward
            else:
                # Update gravity
                soft_drop_multiplier = 4.0 if movement == 2 else 1.0
                self.fall_counter += (self.fall_speed * soft_drop_multiplier) / 30.0 # Assuming 30 FPS

                if self.fall_counter >= 1.0:
                    self.fall_counter = 0
                    self._move_down()
        
        # Update particles
        self._update_particles()
        
        # Update steps and check for max steps termination
        self.steps += 1
        terminated = self.game_over or self.win or self.steps >= 10000

        if self.game_over:
            reward -= 100
        if self.win:
            reward += 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _spawn_piece(self):
        self.current_piece = self.next_piece
        piece_id = self.rng.integers(len(self.TETROMINOES))
        self.next_piece = {
            'id': piece_id,
            'shapes': self.TETROMINOES[piece_id],
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0,
            'color': self.TETROMINO_COLORS[piece_id]
        }
        if self.current_piece:
            if self._check_collision(self.current_piece['shapes'][self.current_piece['rotation']], (self.current_piece['x'], self.current_piece['y'])):
                self.game_over = True

    def _move(self, dx):
        if self.current_piece and not self.game_over:
            new_x = self.current_piece['x'] + dx
            if not self._check_collision(self.current_piece['shapes'][self.current_piece['rotation']], (new_x, self.current_piece['y'])):
                self.current_piece['x'] = new_x
                self.lock_delay_counter = 0 # Reset lock delay on move

    def _move_down(self):
        if self.current_piece and not self.game_over:
            new_y = self.current_piece['y'] + 1
            if not self._check_collision(self.current_piece['shapes'][self.current_piece['rotation']], (self.current_piece['x'], new_y)):
                self.current_piece['y'] = new_y
                self.lock_delay_counter = 0
            else:
                self.lock_delay_counter += 1
                if self.lock_delay_counter > 15: # Lock after 0.5s on ground
                    self._lock_piece()

    def _rotate(self, clockwise):
        if self.current_piece and not self.game_over:
            current_rotation = self.current_piece['rotation']
            num_rotations = len(self.current_piece['shapes'])
            new_rotation = (current_rotation + (1 if clockwise else -1) + num_rotations) % num_rotations
            
            new_shape = self.current_piece['shapes'][new_rotation]
            
            # Wall kick checks
            offsets_to_try = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
            for ox, oy in offsets_to_try:
                new_pos = (self.current_piece['x'] + ox, self.current_piece['y'] + oy)
                if not self._check_collision(new_shape, new_pos):
                    self.current_piece['rotation'] = new_rotation
                    self.current_piece['x'] += ox
                    self.current_piece['y'] += oy
                    self.lock_delay_counter = 0 # Reset lock delay on rotate
                    return # Success
            # sound: rotation_fail.wav

    def _hard_drop(self):
        if self.current_piece and not self.game_over:
            rows_dropped = 0
            while not self._check_collision(self.current_piece['shapes'][self.current_piece['rotation']], (self.current_piece['x'], self.current_piece['y'] + 1)):
                self.current_piece['y'] += 1
                rows_dropped += 1
            self._lock_piece()
            # sound: hard_drop.wav
            return rows_dropped * 0.02 # Small reward for efficient dropping

    def _lock_piece(self):
        if not self.current_piece: return
        shape = self.current_piece['shapes'][self.current_piece['rotation']]
        pos_x, pos_y = self.current_piece['x'], self.current_piece['y']
        color_index = self.current_piece['id'] + 1
        
        for x, y in shape:
            grid_x, grid_y = pos_x + x, pos_y + y
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                self.grid[grid_y, grid_x] = color_index
        # sound: piece_lock.wav
        
        self._clear_lines()
        self._spawn_piece()
        self.lock_delay_counter = 0
        self.fall_counter = 0.0

    def _check_collision(self, shape, offset):
        off_x, off_y = offset
        for x, y in shape:
            grid_x, grid_y = off_x + x, off_y + y
            if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                return True # Out of bounds
            if self.grid[grid_y, grid_x] > 0:
                return True # Collides with another block
        return False

    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y] > 0):
                lines_to_clear.append(y)

        num_cleared = len(lines_to_clear)
        if num_cleared > 0:
            # Rewards: 1 line: +10, 4 lines: +40. Interpolating for 2 and 3.
            reward_map = {1: 10, 2: 20, 3: 30, 4: 40}
            self.score += reward_map.get(num_cleared, 0)

            for y in lines_to_clear:
                self.line_clear_animation.append({'y': y, 'timer': 10}) # 10 frames animation
                # sound: line_clear.wav
                self._spawn_clear_particles(y)

            # Defer actual grid clearing until animation is over
            self.lines_cleared += num_cleared
            new_level = 1 + self.lines_cleared // 10
            if new_level > self.level:
                self.level = new_level
                self.fall_speed = 1.0 + (self.level - 1) * 0.2
                # sound: level_up.wav
        
        if self.lines_cleared >= 100:
            self.win = True

    def _update_line_clear_animation(self):
        finished_lines = []
        for anim in self.line_clear_animation:
            anim['timer'] -= 1
            if anim['timer'] <= 0:
                finished_lines.append(anim['y'])
        
        if finished_lines:
            # Sort lines descending to avoid index issues when removing
            finished_lines.sort(reverse=True)
            for y in finished_lines:
                self.grid = np.delete(self.grid, y, axis=0)
                new_row = np.zeros((1, self.GRID_WIDTH), dtype=int)
                self.grid = np.vstack([new_row, self.grid])
            
            # Remove completed animations
            self.line_clear_animation = [a for a in self.line_clear_animation if a['timer'] > 0]

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
            "level": self.level,
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET,
                                self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw landed blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] > 0:
                    color_index = int(self.grid[y, x] - 1)
                    base_color = self.TETROMINO_COLORS[color_index]
                    landed_color = tuple(int(c * self.LANDED_DESATURATION + self.COLOR_GRID[i] * (1-self.LANDED_DESATURATION)) for i, c in enumerate(base_color))
                    self._draw_block(x, y, landed_color)

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_y = self.current_piece['y']
            while not self._check_collision(self.current_piece['shapes'][self.current_piece['rotation']], (self.current_piece['x'], ghost_y + 1)):
                ghost_y += 1
            
            shape = self.current_piece['shapes'][self.current_piece['rotation']]
            for x, y in shape:
                px, py = self.current_piece['x'] + x, ghost_y + y
                self._draw_block(px, py, self.current_piece['color'], alpha=self.GHOST_ALPHA)
        
        # Draw current piece
        if self.current_piece and not self.game_over:
            shape = self.current_piece['shapes'][self.current_piece['rotation']]
            for x, y in shape:
                px, py = self.current_piece['x'] + x, self.current_piece['y'] + y
                self._draw_block(px, py, self.current_piece['color'])
        
        # Draw grid lines
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, 
                             (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE),
                             (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y_OFFSET + y * self.CELL_SIZE))
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG,
                             (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET),
                             (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw line clear animation
        for anim in self.line_clear_animation:
            y = anim['y']
            alpha = 255 * (math.sin(anim['timer'] * math.pi / 5) ** 2) # Flashing effect
            flash_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE))

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), p['color'] + (int(p['alpha']),))
            pygame.gfxdraw.aacircle(self.screen, int(p['x']), int(p['y']), int(p['size']), p['color'] + (int(p['alpha']),))

    def _draw_block(self, x, y, color, alpha=255):
        rect = pygame.Rect(self.GRID_X_OFFSET + x * self.CELL_SIZE, 
                           self.GRID_Y_OFFSET + y * self.CELL_SIZE,
                           self.CELL_SIZE, self.CELL_SIZE)
        
        if alpha < 255:
            surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            surface.fill(color + (alpha,))
            self.screen.blit(surface, rect.topleft)
        else:
            pygame.draw.rect(self.screen, color, rect)
            # Add a subtle 3D effect
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright)
            pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft)
            pygame.draw.line(self.screen, shadow, rect.bottomright, rect.topright)
            pygame.draw.line(self.screen, shadow, rect.bottomright, rect.bottomleft)

    def _render_ui(self):
        ui_x = self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE + 20
        
        # Next Piece Box
        self._draw_ui_box(ui_x, self.GRID_Y_OFFSET, 120, 100, "NEXT")
        if self.next_piece:
            shape = self.next_piece['shapes'][0]
            color = self.next_piece['color']
            xs = [c[0] for c in shape]
            ys = [c[1] for c in shape]
            shape_w = (max(xs) - min(xs) + 1) * self.CELL_SIZE
            shape_h = (max(ys) - min(ys) + 1) * self.CELL_SIZE
            
            for x, y in shape:
                draw_x = ui_x + (120 - shape_w) // 2 + (x - min(xs)) * self.CELL_SIZE
                draw_y = self.GRID_Y_OFFSET + 30 + (100 - 30 - shape_h) // 2 + (y - min(ys)) * self.CELL_SIZE
                rect = pygame.Rect(draw_x, draw_y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)

        # Score Box
        self._draw_ui_box(ui_x, self.GRID_Y_OFFSET + 120, 120, 70, "SCORE")
        score_text = self.font_main.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (ui_x + (120 - score_text.get_width()) // 2, self.GRID_Y_OFFSET + 150))
        
        # Lines Box
        self._draw_ui_box(ui_x, self.GRID_Y_OFFSET + 210, 120, 70, "LINES")
        lines_text = self.font_main.render(f"{self.lines_cleared}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_text, (ui_x + (120 - lines_text.get_width()) // 2, self.GRID_Y_OFFSET + 240))

        # Level Box
        self._draw_ui_box(ui_x, self.GRID_Y_OFFSET + 300, 120, 70, "LEVEL")
        level_text = self.font_main.render(f"{self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (ui_x + (120 - level_text.get_width()) // 2, self.GRID_Y_OFFSET + 330))

        # Game Over / Win Text
        if self.game_over or self.win:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            text = self.font_gameover.render(msg, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(text, text_rect)

    def _draw_ui_box(self, x, y, w, h, title):
        bg_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, bg_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, bg_rect, width=2, border_radius=5)
        
        title_text = self.font_info.render(title, True, self.COLOR_UI_TEXT)
        self.screen.blit(title_text, (x + (w - title_text.get_width()) // 2, y + 5))

    def _spawn_clear_particles(self, y_line):
        # sound: particle_spawn.wav
        for _ in range(30):
            x = self.GRID_X_OFFSET + self.rng.random() * self.GRID_WIDTH * self.CELL_SIZE
            y = self.GRID_Y_OFFSET + y_line * self.CELL_SIZE + self.CELL_SIZE / 2
            angle = self.rng.random() * 2 * math.pi
            speed = 1 + self.rng.random() * 2
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': self.rng.integers(2, 5),
                'alpha': 255,
                'decay': self.rng.integers(3, 8),
                'color': random.choice(self.TETROMINO_COLORS)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['alpha'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.05 # Gravity
            p['alpha'] -= p['decay']
            p['size'] = max(0, p['size'] - 0.05)

    def validate_implementation(self):
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Use a separate screen for display if running directly
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Block Blitz")
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Reset actions
            movement = 0
            space = 0
            shift = 0

            # Map keys to actions
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = np.array([movement, space, shift])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print info for debugging
            # if reward != -0.01:
            #     print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    pygame.quit()