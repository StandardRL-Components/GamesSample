
# Generated: 2025-08-27T13:04:16.015392
# Source Brief: brief_00248.md
# Brief Index: 248

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Hold Space for soft drop, press Shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based puzzle game. Clear lines to score points and advance through stages before time runs out or the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_TEXT = (230, 240, 250)
    COLOR_GHOST = (255, 255, 255, 50)
    
    BLOCK_SHAPES = {
        'T': {'coords': [(0, 0), (-1, 0), (1, 0), (0, -1)], 'color': (160, 0, 255)},
        'I': {'coords': [(0, 0), (-1, 0), (1, 0), (2, 0)], 'color': (0, 255, 255)},
        'O': {'coords': [(0, 0), (1, 0), (0, 1), (1, 1)], 'color': (255, 255, 0)},
        'L': {'coords': [(0, 0), (-1, 0), (1, 0), (1, -1)], 'color': (255, 165, 0)},
        'J': {'coords': [(0, 0), (-1, 0), (1, 0), (-1, -1)], 'color': (0, 0, 255)},
        'S': {'coords': [(0, 0), (-1, 0), (0, -1), (1, -1)], 'color': (0, 255, 0)},
        'Z': {'coords': [(0, 0), (1, 0), (0, -1), (-1, -1)], 'color': (255, 0, 0)}
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Initialize state variables
        self.grid = None
        self.current_block = None
        self.next_block_type = None
        self.score = 0
        self.total_lines_cleared = 0
        self.stage = 1
        self.time_remaining = 0.0
        self.game_over = False
        self.steps = 0
        self.fall_progress = 0.0
        self.fall_speed_base = 0
        self.fall_speed_increase_rate = 0
        self.fall_speed_stage_multiplier = 0
        self.rng = None
        self.particles = []
        self.line_clear_animation = None
        
        # Input handling
        self.last_shift_state = False
        self.move_cooldown = 0
        self.rotate_cooldown = 0
        self.MOVE_COOLDOWN_RESET = 0.1 # seconds
        self.ROTATE_COOLDOWN_RESET = 0.15 # seconds

        # This call will also perform the first reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT + 4), dtype=int) # 4 rows buffer
        self.score = 0
        self.total_lines_cleared = 0
        self.stage = 1
        self.time_remaining = 180.0
        self.game_over = False
        self.steps = 0
        
        self.fall_speed_base = 1.0  # cells per second
        self.fall_speed_increase_rate = 0.05
        self.fall_speed_stage_multiplier = 1.0
        self.fall_progress = 0.0
        
        self.last_shift_state = False
        self.move_cooldown = 0
        self.rotate_cooldown = 0
        
        self.particles = []
        self.line_clear_animation = None
        
        self.next_block_type = self.rng.choice(list(self.BLOCK_SHAPES.keys()))
        self._spawn_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        dt = self.clock.tick(30) / 1000.0
        self.steps += 1
        reward = 0

        if self.game_over:
            return self._get_observation(), -10, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Update cooldowns
        self.move_cooldown = max(0, self.move_cooldown - dt)
        self.rotate_cooldown = max(0, self.rotate_cooldown - dt)

        # Handle line clear animation pause
        if self.line_clear_animation:
            self.line_clear_animation['timer'] -= dt
            if self.line_clear_animation['timer'] <= 0:
                self._finish_line_clear()
                self.line_clear_animation = None
        else:
            # --- Handle Input ---
            self._handle_input(movement, shift_held)

            # --- Game Logic Update ---
            self.time_remaining -= dt
            
            current_fall_speed = self._get_current_fall_speed()
            if space_held: # Soft drop
                current_fall_speed *= 10.0
            
            self.fall_progress += current_fall_speed * dt
            
            if self.fall_progress >= 1.0:
                self.fall_progress -= 1.0
                self.current_block['y'] += 1
                if self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'])):
                    self.current_block['y'] -= 1
                    placement_reward, holes_created = self._place_block()
                    reward += placement_reward
                    
                    lines_cleared, clear_reward = self._check_and_clear_lines()
                    reward += clear_reward
                    if lines_cleared > 0:
                        self._update_difficulty()
                    
                    self._spawn_block()
                    if self.game_over:
                        reward -= 10.0 # Penalty for topping out
        
        # Update particles
        self._update_particles(dt)

        terminated = self.game_over or self.total_lines_cleared >= 100 or self.time_remaining <= 0
        if terminated and self.total_lines_cleared >= 100:
            reward += 100 # Victory reward
        elif terminated and self.time_remaining <= 0:
            reward -= 10.0 # Penalty for time out

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "lines": self.total_lines_cleared,
            "stage": self.stage,
            "time_remaining": self.time_remaining,
            "steps": self.steps,
        }

    # --- Game Logic Helpers ---
    def _handle_input(self, movement, shift_held):
        # Movement
        if movement in [3, 4] and self.move_cooldown <= 0:
            dx = -1 if movement == 3 else 1
            self._move(dx)
            self.move_cooldown = self.MOVE_COOLDOWN_RESET
        # Rotation
        elif movement in [1, 2] and self.rotate_cooldown <= 0:
            direction = 1 if movement == 1 else -1 # 1: CW, -1: CCW
            self._rotate(direction)
            self.rotate_cooldown = self.ROTATE_COOLDOWN_RESET
        # Hard drop (on press, not hold)
        if shift_held and not self.last_shift_state:
            self._hard_drop()
        self.last_shift_state = shift_held

    def _spawn_block(self):
        self.current_block = {
            'type': self.next_block_type,
            'shape': self.BLOCK_SHAPES[self.next_block_type]['coords'],
            'color': self.BLOCK_SHAPES[self.next_block_type]['color'],
            'x': self.GRID_WIDTH // 2,
            'y': 2, # Start in buffer zone
        }
        self.next_block_type = self.rng.choice(list(self.BLOCK_SHAPES.keys()))
        self.fall_progress = 0.0
        if self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'])):
            self.game_over = True

    def _place_block(self):
        # Calculate holes created for reward
        pre_place_holes = self._count_holes()
        
        for x_off, y_off in self.current_block['shape']:
            grid_x, grid_y = self.current_block['x'] + x_off, self.current_block['y'] + y_off
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT + 4:
                self.grid[grid_x, grid_y] = list(self.BLOCK_SHAPES.keys()).index(self.current_block['type']) + 1
        
        # sfx: block_place.wav
        post_place_holes = self._count_holes()
        holes_created = max(0, post_place_holes - pre_place_holes)
        
        # Reward for placing a block, penalized by holes created
        reward = 0.1 - (0.1 * holes_created)
        return reward, holes_created

    def _count_holes(self):
        holes = 0
        for x in range(self.GRID_WIDTH):
            found_block = False
            for y in range(self.GRID_HEIGHT + 4):
                if self.grid[x, y] != 0:
                    found_block = True
                elif found_block and self.grid[x, y] == 0:
                    holes += 1
        return holes

    def _check_and_clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT + 4):
            if all(self.grid[x, y] != 0 for x in range(self.GRID_WIDTH)):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            return 0, 0.0

        self.line_clear_animation = {'lines': lines_to_clear, 'timer': 0.2}
        
        # Reward calculation
        num_cleared = len(lines_to_clear)
        self.score += [0, 100, 300, 500, 800][num_cleared] * self.stage
        
        # Base reward + bonus for multi-clear
        reward = num_cleared * 1.0
        if num_cleared > 1:
            reward += (num_cleared -1) * 2.0
        
        # sfx: line_clear.wav
        for y in lines_to_clear:
            for i in range(15):
                self._spawn_particle(self.GRID_X + self.rng.integers(0, self.GRID_WIDTH * self.CELL_SIZE), 
                                     self.GRID_Y + (y - 4) * self.CELL_SIZE)
        return num_cleared, reward

    def _finish_line_clear(self):
        lines_cleared = self.line_clear_animation['lines']
        for y in sorted(lines_cleared, reverse=True):
            self.grid[:, 1:y+1] = self.grid[:, 0:y]
            self.grid[:, 0] = 0
        self.total_lines_cleared += len(lines_cleared)

    def _update_difficulty(self):
        new_stage = 1 + self.total_lines_cleared // 33
        if new_stage > self.stage:
            self.stage = new_stage
            self.fall_speed_stage_multiplier *= 1.2
            # sfx: stage_up.wav
    
    def _get_current_fall_speed(self):
        speed = self.fall_speed_base + (self.total_lines_cleared // 20) * self.fall_speed_increase_rate * self.fall_speed_stage_multiplier
        return speed

    # --- Collision and Movement Helpers ---
    def _check_collision(self, shape, offset):
        off_x, off_y = offset
        for x_off, y_off in shape:
            x, y = off_x + x_off, off_y + y_off
            if not (0 <= x < self.GRID_WIDTH and y < self.GRID_HEIGHT + 4):
                return True # Out of bounds
            if y >= 0 and self.grid[x, y] != 0:
                return True # Collides with existing block
        return False

    def _rotate(self, direction):
        if self.current_block['type'] == 'O': return # Can't rotate a square
        
        # sfx: rotate.wav
        rotated_shape = []
        for x, y in self.current_block['shape']:
            if direction == 1: # Clockwise
                rotated_shape.append((y, -x))
            else: # Counter-clockwise
                rotated_shape.append((-y, x))
        
        # Wall kick logic
        for kick_x, kick_y in [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]:
            if not self._check_collision(rotated_shape, (self.current_block['x'] + kick_x, self.current_block['y'] + kick_y)):
                self.current_block['shape'] = rotated_shape
                self.current_block['x'] += kick_x
                self.current_block['y'] += kick_y
                return

    def _move(self, dx):
        if not self._check_collision(self.current_block['shape'], (self.current_block['x'] + dx, self.current_block['y'])):
            self.current_block['x'] += dx
            # sfx: move.wav

    def _hard_drop(self):
        # sfx: hard_drop.wav
        y = self.current_block['y']
        while not self._check_collision(self.current_block['shape'], (self.current_block['x'], y + 1)):
            y += 1
        self.current_block['y'] = y
        self.fall_progress = 1.0 # Force placement on next frame

    # --- Rendering Helpers ---
    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))
        
        # Draw landed blocks
        for x in range(self.GRID_WIDTH):
            for y in range(4, self.GRID_HEIGHT + 4):
                if self.grid[x, y] != 0:
                    color_index = int(self.grid[x, y] - 1)
                    block_type = list(self.BLOCK_SHAPES.keys())[color_index]
                    color = self.BLOCK_SHAPES[block_type]['color']
                    self._draw_cell(x, y - 4, color)

        # Draw ghost piece
        if self.current_block and not self.line_clear_animation:
            ghost_y = self.current_block['y']
            while not self._check_collision(self.current_block['shape'], (self.current_block['x'], ghost_y + 1)):
                ghost_y += 1
            for x_off, y_off in self.current_block['shape']:
                if ghost_y + y_off >= 4:
                    self._draw_cell(self.current_block['x'] + x_off, ghost_y + y_off - 4, self.COLOR_GHOST, is_ghost=True)

        # Draw current block
        if self.current_block and not self.line_clear_animation:
            for x_off, y_off in self.current_block['shape']:
                if self.current_block['y'] + y_off >= 4:
                    self._draw_cell(self.current_block['x'] + x_off, self.current_block['y'] + y_off - 4, self.current_block['color'])
        
        # Draw line clear animation
        if self.line_clear_animation:
            flash_color = (255, 255, 255, 255 * (self.line_clear_animation['timer'] / 0.2))
            for y in self.line_clear_animation['lines']:
                rect = pygame.Rect(self.GRID_X, self.GRID_Y + (y - 4) * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                pygame.gfxdraw.box(self.screen, rect, flash_color)

        # Draw particles
        self._render_particles()

        # Draw grid outline
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.GRID_X - 1, self.GRID_Y - 1, self.GRID_WIDTH * self.CELL_SIZE + 2, self.GRID_HEIGHT * self.CELL_SIZE + 2), 1)

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        x, y = self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2)
        else:
            light_color = tuple(min(255, c + 50) for c in color[:3])
            dark_color = tuple(max(0, c - 50) for c in color[:3])
            pygame.gfxdraw.box(self.screen, rect, color)
            pygame.draw.line(self.screen, light_color, (x, y), (x + self.CELL_SIZE - 1, y))
            pygame.draw.line(self.screen, light_color, (x, y), (x, y + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, dark_color, (x + self.CELL_SIZE - 1, y), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, dark_color, (x, y + self.CELL_SIZE - 1), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1))
            pygame.draw.rect(self.screen, (0,0,0), rect, 1)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Lines
        lines_text = self.font_main.render(f"LINES: {self.total_lines_cleared}/100", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - lines_text.get_width() - 20, 20))

        # Time
        time_str = f"{int(self.time_remaining // 60)}:{int(self.time_remaining % 60):02d}"
        time_text = self.font_title.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 - time_text.get_width() // 2, 15))
        
        # Stage
        stage_text = self.font_main.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH // 2 - stage_text.get_width() // 2, 50))

        # Next Piece
        next_text = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 100, 80))
        next_box_rect = pygame.Rect(self.SCREEN_WIDTH - 120, 100, 100, 80)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_box_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, next_box_rect, 1)

        if self.next_block_type:
            shape_info = self.BLOCK_SHAPES[self.next_block_type]
            for x_off, y_off in shape_info['coords']:
                x = next_box_rect.centerx + x_off * self.CELL_SIZE - self.CELL_SIZE/2
                y = next_box_rect.centery + y_off * self.CELL_SIZE
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.gfxdraw.box(self.screen, rect, shape_info['color'])
                pygame.draw.rect(self.screen, (0,0,0), rect, 1)

        # Game Over / Victory
        if self.game_over:
            self._render_end_screen("GAME OVER")
        elif self.total_lines_cleared >= 100:
            self._render_end_screen("VICTORY!")

    def _render_end_screen(self, message):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        end_text = self.font_title.render(message, True, self.COLOR_TEXT)
        text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(end_text, text_rect)

    def _spawn_particle(self, x, y):
        self.particles.append({
            'x': x, 'y': y,
            'vx': self.rng.uniform(-100, 100), 'vy': self.rng.uniform(-150, -50),
            'size': self.rng.uniform(2, 5),
            'lifetime': self.rng.uniform(0.3, 0.8),
            'color': random.choice([(255,255,255), (255,255,0), (200,200,255)])
        })

    def _update_particles(self, dt):
        for p in self.particles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] += 200 * dt # gravity
            p['lifetime'] -= dt
            p['size'] = max(0, p['size'] - 2*dt)
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['size'] > 0]

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (p['x'], p['y'], int(p['size']), int(p['size'])))

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset (which is called in __init__)
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Grid Fall")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated:
            # On game over, wait for a key press to reset
            keys = pygame.key.get_pressed()
            if any(keys):
                obs, info = env.reset()
                terminated = False
            continue

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(30)

    pygame.quit()