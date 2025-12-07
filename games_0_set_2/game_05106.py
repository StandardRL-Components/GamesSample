import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Hold shift for soft drop, press space for hard drop."
    )

    game_description = (
        "A fast-paced puzzle game. Manipulate falling blocks to clear lines and reach the target score before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    PLAYFIELD_WIDTH = GRID_WIDTH * CELL_SIZE
    PLAYFIELD_HEIGHT = GRID_HEIGHT * CELL_SIZE
    PLAYFIELD_X = (SCREEN_WIDTH - PLAYFIELD_WIDTH) // 2 - 50
    PLAYFIELD_Y = (SCREEN_HEIGHT - PLAYFIELD_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_BORDER = (100, 100, 120)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TIMER_BAR = (70, 180, 255)
    COLOR_TIMER_BAR_BG = (50, 50, 70)
    COLOR_GHOST = (255, 255, 255, 50)

    TETROMINO_COLORS = [
        (0, 240, 240),  # I (Cyan)
        (240, 240, 0),  # O (Yellow)
        (160, 0, 240),  # T (Purple)
        (0, 0, 240),    # J (Blue)
        (240, 160, 0),  # L (Orange)
        (0, 240, 0),    # S (Green)
        (240, 0, 0),    # Z (Red)
    ]

    # Tetromino shapes data
    TETROMINOES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1], [1, 1]],  # O
        [[0, 1, 0], [1, 1, 1]],  # T
        [[1, 0, 0], [1, 1, 1]],  # J
        [[0, 0, 1], [1, 1, 1]],  # L
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 0], [0, 1, 1]],  # Z
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

        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.time_remaining = 0.0
        self.fall_speed = 0.0
        self.fall_progress = 0.0
        self.rotate_cooldown = 0
        self.move_cooldown = 0
        self.line_clear_animation = []
        self.particles = []
        self.last_action_was_drop = False
        self.last_reward_info = ""
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Use Python's random for piece shuffling, seeded for reproducibility
            random.seed(seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.time_remaining = 60.0
        self.fall_speed = 1.0  # units per second
        self.fall_progress = 0.0
        self.rotate_cooldown = 0
        self.move_cooldown = 0
        self.line_clear_animation = []
        self.particles = []
        self.last_action_was_drop = False
        self.last_reward_info = ""

        self.piece_bag = list(range(len(self.TETROMINOES)))
        random.shuffle(self.piece_bag)

        self.next_piece = None # Ensure next_piece is cleared before spawning
        self._spawn_piece()
        self._spawn_piece()  # Once for next, once for current

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        self.last_reward_info = "Time: -0.01"
        self.steps += 1
        
        # --- Time and Cooldowns ---
        delta_time = self.clock.tick(30) / 1000.0
        self.time_remaining = max(0, self.time_remaining - delta_time)
        self.rotate_cooldown = max(0, self.rotate_cooldown - 1)
        self.move_cooldown = max(0, self.move_cooldown - 1)
        
        # --- Handle Action ---
        if not self.game_over and self.current_piece:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            # Hard drop (space) takes priority
            if space_held and not self.last_action_was_drop:
                # Sfx: Hard drop sound
                while self._is_valid_position(self.current_piece, offset_y=1):
                    self.current_piece['y'] += 1
                    self.score += 0.1 # Small reward for dropping
                self._lock_piece()
                self.last_action_was_drop = True
            else:
                self.last_action_was_drop = False
                # Rotation
                if movement in [1, 2] and self.rotate_cooldown == 0:
                    self._rotate_piece(1 if movement == 1 else -1)
                    self.rotate_cooldown = 5 # 5 frames cooldown
                # Movement
                elif movement in [3, 4] and self.move_cooldown == 0:
                    dx = -1 if movement == 3 else 1
                    if self._is_valid_position(self.current_piece, offset_x=dx):
                        # Sfx: Move sound
                        self.current_piece['x'] += dx
                    self.move_cooldown = 3 # 3 frames cooldown

                # --- Update Game Logic (Falling) ---
                fall_multiplier = 5.0 if shift_held else 1.0
                self.fall_progress += self.fall_speed * delta_time * fall_multiplier

                if self.fall_progress >= 1.0:
                    self.fall_progress = 0.0
                    if self._is_valid_position(self.current_piece, offset_y=1):
                        self.current_piece['y'] += 1
                    else:
                        # Sfx: Block lock sound
                        lock_reward = self._lock_piece()
                        reward += lock_reward

        # --- Update animations and check for game end ---
        self._update_animations(delta_time)
        lines_reward = self._clear_lines()
        if lines_reward > 0:
            reward += lines_reward

        terminated = self._check_termination()
        
        # --- Terminal Rewards ---
        if terminated:
            if self.lines_cleared >= 10:
                reward = 100
                self.last_reward_info = "Win: +100"
            elif self.game_over:
                reward = -100
                self.last_reward_info = "Lose (Topped Out): -100"
            elif self.time_remaining <= 0:
                reward = -50
                self.last_reward_info = "Lose (Time Out): -50"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.TETROMINOES)))
            random.shuffle(self.piece_bag)
        
        piece_type = self.piece_bag.pop(0)
        shape = self.TETROMINOES[piece_type]
        
        self.current_piece = self.next_piece
        self.next_piece = {
            'type': piece_type,
            'shape': shape,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'color': self.TETROMINO_COLORS[piece_type]
        }

        if self.current_piece and not self._is_valid_position(self.current_piece):
            self.game_over = True
            # Sfx: Game over sound

    def _rotate_piece(self, direction):
        if not self.current_piece: return
        
        original_shape = self.current_piece['shape']
        
        if direction == 1: # Clockwise
            new_shape = [list(row) for row in zip(*self.current_piece['shape'][::-1])]
        else: # Counter-clockwise
            # FIX: Convert zip object to list before slicing
            new_shape = [list(row) for row in list(zip(*self.current_piece['shape']))[::-1]]
        
        self.current_piece['shape'] = new_shape
        
        # Wall kick logic
        for offset in [0, 1, -1, 2, -2]:
            if self._is_valid_position(self.current_piece, offset_x=offset):
                self.current_piece['x'] += offset
                # Sfx: Rotate sound
                return
                
        self.current_piece['shape'] = original_shape # Revert if no valid position found

    def _lock_piece(self):
        if not self.current_piece: return -0.2 # Penalty for empty lock
        
        for r, row in enumerate(self.current_piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = self.current_piece['x'] + c, self.current_piece['y'] + r
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_piece['type'] + 1
        
        self._spawn_piece()
        self.last_reward_info = "Lock: -0.2"
        return -0.2

    def _clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r] > 0):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            # Sfx: Line clear sound
            for r in lines_to_clear:
                self.grid[r] = -1 # Mark for animation
                self.line_clear_animation.append({'row': r, 'timer': 0.2})
                # Add particles
                for i in range(20):
                    self.particles.append({
                        'x': self.PLAYFIELD_X + random.uniform(0, self.PLAYFIELD_WIDTH),
                        'y': self.PLAYFIELD_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2,
                        'vx': random.uniform(-50, 50), 'vy': random.uniform(-100, 0),
                        'size': random.uniform(2, 5), 'life': 0.5, 'color': (255,255,255)
                    })

            cleared_count = len(lines_to_clear)
            self.lines_cleared += cleared_count
            
            # Score multiplier for multiple lines
            score_bonuses = {1: 10, 2: 30, 3: 60, 4: 100}
            self.score += score_bonuses.get(cleared_count, 0)

            # Increase speed
            if self.lines_cleared % 2 == 0 and cleared_count > 0:
                self.fall_speed += 0.05
            
            # Reward for clearing lines
            reward_bonuses = {1: 1.0, 2: 3.0, 3: 6.0, 4: 10.0}
            reward = reward_bonuses.get(cleared_count, 0)
            self.last_reward_info = f"{cleared_count} Lines: +{reward:.1f}"
            return reward
        
        return 0

    def _is_valid_position(self, piece, offset_x=0, offset_y=0):
        if not piece: return False
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    x = piece['x'] + c + offset_x
                    y = piece['y'] + r + offset_y
                    if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT and self.grid[y, x] == 0):
                        return False
        return True
    
    def _update_animations(self, delta_time):
        # Line clear animation
        if self.line_clear_animation:
            for anim in self.line_clear_animation[:]:
                anim['timer'] -= delta_time
                if anim['timer'] <= 0:
                    self.line_clear_animation.remove(anim)

            # If all animations are done, shift the grid
            if not self.line_clear_animation and np.any(self.grid == -1):
                new_grid = np.zeros_like(self.grid)
                new_row = self.GRID_HEIGHT - 1
                for r in range(self.GRID_HEIGHT - 1, -1, -1):
                    if self.grid[r, 0] != -1:
                        new_grid[new_row] = self.grid[r]
                        new_row -= 1
                self.grid = new_grid
        
        # Particle animation
        for p in self.particles[:]:
            p['x'] += p['vx'] * delta_time
            p['y'] += p['vy'] * delta_time
            p['vy'] += 200 * delta_time # Gravity
            p['life'] -= delta_time
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        max_steps = 1800 # 60 seconds at 30fps
        return self.game_over or self.lines_cleared >= 10 or self.time_remaining <= 0 or self.steps >= max_steps

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw playfield border and grid
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (self.PLAYFIELD_X - 2, self.PLAYFIELD_Y - 2, self.PLAYFIELD_WIDTH + 4, self.PLAYFIELD_HEIGHT + 4), 2)
        for x in range(1, self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X + x * self.CELL_SIZE, self.PLAYFIELD_Y), (self.PLAYFIELD_X + x * self.CELL_SIZE, self.PLAYFIELD_Y + self.PLAYFIELD_HEIGHT))
        for y in range(1, self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X, self.PLAYFIELD_Y + y * self.CELL_SIZE), (self.PLAYFIELD_X + self.PLAYFIELD_WIDTH, self.PLAYFIELD_Y + y * self.CELL_SIZE))

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    color_index = int(self.grid[r, c] - 1)
                    self._draw_block(c, r, self.TETROMINO_COLORS[color_index])
                elif self.grid[r,c] == -1: # Line clear flash
                    flash_color = (255, 255, 255)
                    self._draw_block(c, r, flash_color, is_flash=True)

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_y = self.current_piece['y']
            while self._is_valid_position(self.current_piece, offset_y=ghost_y - self.current_piece['y'] + 1):
                ghost_y += 1
            for r, row in enumerate(self.current_piece['shape']):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece['x'] + c, ghost_y + r, self.current_piece['color'], is_ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            for r, row in enumerate(self.current_piece['shape']):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece['x'] + c, self.current_piece['y'] + r, self.current_piece['color'])
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 0.5))))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['x'] - p['size']), int(p['y'] - p['size'])))

    def _draw_block(self, grid_x, grid_y, color, is_ghost=False, is_flash=False):
        px, py = self.PLAYFIELD_X + grid_x * self.CELL_SIZE, self.PLAYFIELD_Y + grid_y * self.CELL_SIZE
        rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], 50))
            pygame.draw.rect(s, (255, 255, 255, 80), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 1)
            self.screen.blit(s, (px, py))
        elif is_flash:
            pygame.draw.rect(self.screen, color, rect)
        else:
            light_color = tuple(min(255, c + 50) for c in color)
            dark_color = tuple(max(0, c - 50) for c in color)
            
            pygame.draw.rect(self.screen, dark_color, rect)
            pygame.draw.rect(self.screen, color, (px + 2, py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))
            
            # Highlight effect
            pygame.draw.line(self.screen, light_color, (px + 2, py + 2), (px + self.CELL_SIZE - 3, py + 2))
            pygame.draw.line(self.screen, light_color, (px + 2, py + 2), (px + 2, py + self.CELL_SIZE - 3))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Lines
        lines_text = self.font_main.render(f"LINES: {self.lines_cleared} / 10", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - lines_text.get_width() - 20, 20))

        # Time bar
        time_bar_width = 300
        time_bar_x = (self.SCREEN_WIDTH - time_bar_width) / 2
        time_ratio = self.time_remaining / 60.0
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (time_bar_x, 20, time_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (time_bar_x, 20, time_bar_width * time_ratio, 20))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (time_bar_x-1, 19, time_bar_width+2, 22), 1)

        # Next piece preview
        next_box_x, next_box_y = self.PLAYFIELD_X + self.PLAYFIELD_WIDTH + 20, self.PLAYFIELD_Y
        next_box_w, next_box_h = 100, 100
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (next_box_x, next_box_y, next_box_w, next_box_h), 2)
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (next_box_x + (next_box_w - next_text.get_width()) / 2, next_box_y + 5))
        if self.next_piece:
            shape = self.next_piece['shape']
            shape_w = len(shape[0]) * self.CELL_SIZE
            shape_h = len(shape) * self.CELL_SIZE
            start_x = next_box_x + (next_box_w - shape_w) / 2
            start_y = next_box_y + (next_box_h - shape_h) / 2 + 10
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        px, py = start_x + c * self.CELL_SIZE, start_y + r * self.CELL_SIZE
                        rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
                        light_color = tuple(min(255, c + 50) for c in self.next_piece['color'])
                        pygame.draw.rect(self.screen, self.next_piece['color'], rect)
                        pygame.draw.rect(self.screen, light_color, (px+2, py+2, self.CELL_SIZE-4, self.CELL_SIZE-4))

        # Game Over / Win message
        if self.game_over:
            self._render_end_message("GAME OVER")
        elif self.lines_cleared >= 10:
            self._render_end_message("YOU WIN!")
        elif self.time_remaining <= 0 and not self.lines_cleared >= 10:
            self._render_end_message("TIME'S UP!")

    def _render_end_message(self, message):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        end_text = self.font_main.render(message, True, (255, 50, 50) if "OVER" in message or "UP" in message else (50, 255, 50))
        text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "time_remaining": self.time_remaining,
        }

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
    # This block allows you to play the game manually for testing
    # To play, you need to remove the headless mode setting
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for manual play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Mapping keyboard keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not terminated:
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement action
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Lines: {info['lines_cleared']}, Reward: {reward:.2f}, Terminated: {terminated}")

    print(f"Game Over! Final Score: {info['score']:.1f}, Total Reward: {total_reward:.2f}")
    env.close()