
# Generated: 2025-08-27T22:15:55.540399
# Source Brief: brief_03064.md
# Brief Index: 3064

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop, Shift to hold a piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, falling block puzzle. Clear lines to score points, but don't let the stack reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    BLOCK_SIZE = 18
    GRID_LINE_WIDTH = 1

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_GHOST = (255, 255, 255, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (30, 30, 40)
    COLOR_UI_BORDER = (60, 60, 70)

    # Tetromino shapes and their colors
    SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1, 0], [0, 1, 1]],  # Z
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 1], [0, 1, 0]],  # T
        [[1, 1, 1], [1, 0, 0]],  # L
        [[1, 1, 1], [0, 0, 1]],  # J
        [[1, 1], [1, 1]],  # O
    ]
    COLORS = [
        (0, 240, 240),  # I (Cyan)
        (240, 0, 0),    # Z (Red)
        (0, 240, 0),    # S (Green)
        (160, 0, 240),  # T (Purple)
        (240, 160, 0),  # L (Orange)
        (0, 0, 240),    # J (Blue)
        (240, 240, 0),  # O (Yellow)
    ]

    # Game parameters
    MAX_STEPS = 10000
    WIN_SCORE = 1000
    MOVE_DELAY_FRAMES = 3  # Auto-repeat delay for horizontal movement

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
        self.font_small = pygame.font.SysFont("Consolas", 18)

        self.grid_pixel_width = self.GRID_WIDTH * self.BLOCK_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.grid_top_left_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_top_left_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2

        # Initialize state variables that are not reset every episode
        self.render_mode = render_mode

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[0 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0.5  # units per frame
        self.fall_counter = 0.0

        self.bag = list(range(len(self.SHAPES)))
        random.shuffle(self.bag)
        
        self.current_block = self._new_block()
        self.next_block = self._new_block()
        self.held_block = None
        self.can_hold = True

        # Action handling state
        self.prev_space_held = False
        self.prev_shift_held = False
        self.prev_up_held = False
        self.move_timer = 0
        
        # Visual effects
        self.particles = []
        self.line_clear_effects = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Processing ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        up_press = movement == 1 and not self.prev_up_held
        
        if space_press:
            reward += self._hard_drop()
            # Hard drop places the block, so we proceed to the next game tick logic
        elif shift_press:
            if self._hold_block():
                reward -= 0.02
        else: # Only process other movements if not hard dropping
            if up_press:
                self._rotate_block()
                reward -= 0.02
            
            # Horizontal Movement with auto-repeat
            if movement in [3, 4]: # Left or Right
                if self.move_timer <= 0:
                    dx = -1 if movement == 3 else 1
                    if not self._check_collision(self.current_block['shape'], (self.current_block['x'] + dx, self.current_block['y'])):
                        self.current_block['x'] += dx
                        reward -= 0.02
                    self.move_timer = self.MOVE_DELAY_FRAMES
                self.move_timer -= 1
            else:
                self.move_timer = 0
                
            # Soft Drop
            current_fall_speed = self.fall_speed * 5 if movement == 2 else self.fall_speed
            self.fall_counter += current_fall_speed
        
        # --- Game Logic Update ---
        if self.fall_counter >= 1.0:
            fall_steps = int(self.fall_counter)
            self.fall_counter -= fall_steps
            for _ in range(fall_steps):
                if not self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'] + 1)):
                    self.current_block['y'] += 1
                else:
                    self._place_block()
                    reward += 0.1 # Reward for placing a block
                    
                    lines_cleared, clear_reward = self._clear_lines()
                    reward += clear_reward
                    if lines_cleared > 0:
                        self._update_difficulty()

                    self._spawn_new_block()
                    break # Exit fall loop as block has been placed

        # --- Update previous action states ---
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        self.prev_up_held = (movement == 1)

        # --- Termination Check ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.game_over:
            reward -= 100
        elif self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
        
        self._update_particles()

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Helper Methods: Game Logic ---

    def _new_block(self):
        if not self.bag:
            self.bag = list(range(len(self.SHAPES)))
            random.shuffle(self.bag)
        
        shape_index = self.bag.pop()
        shape = self.SHAPES[shape_index]
        return {
            'shape_index': shape_index,
            'shape': shape,
            'color': self.COLORS[shape_index],
            'x': self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0
        }

    def _spawn_new_block(self):
        self.current_block = self.next_block
        self.next_block = self._new_block()
        self.can_hold = True
        self.fall_counter = 0.0

        if self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'])):
            self.game_over = True

    def _check_collision(self, shape, pos):
        px, py = pos
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = px + x, py + y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True  # Wall collision
                    if self.grid[grid_y][grid_x] != 0:
                        return True  # Other block collision
        return False

    def _place_block(self):
        # sfx: block_place.wav
        shape = self.current_block['shape']
        px, py = self.current_block['x'], self.current_block['y']
        color_index = self.current_block['shape_index'] + 1
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[py + y][px + x] = color_index

    def _clear_lines(self):
        lines_to_clear = []
        for r, row in enumerate(self.grid):
            if all(cell != 0 for cell in row):
                lines_to_clear.append(r)

        if lines_to_clear:
            # sfx: line_clear.wav
            for r in lines_to_clear:
                self.line_clear_effects.append({'y': r, 'timer': 10})
                for x in range(self.GRID_WIDTH):
                    self._create_particles(self.grid_top_left_x + x * self.BLOCK_SIZE + self.BLOCK_SIZE//2, 
                                           self.grid_top_left_y + r * self.BLOCK_SIZE + self.BLOCK_SIZE//2, 
                                           self.COLORS[self.grid[r][x]-1])
                self.grid.pop(r)
                self.grid.insert(0, [0 for _ in range(self.GRID_WIDTH)])
            
            num_cleared = len(lines_to_clear)
            self.score += [0, 100, 300, 500, 800][num_cleared]
            reward = [0, 1, 3, 5, 10][num_cleared]
            return num_cleared, reward
        return 0, 0
    
    def _update_difficulty(self):
        level = self.score // 200
        self.fall_speed = 0.5 + level * 0.1

    def _rotate_block(self):
        # sfx: rotate.wav
        shape = self.current_block['shape']
        rotated_shape = [list(row) for row in zip(*shape[::-1])]
        if not self._check_collision(rotated_shape, (self.current_block['x'], self.current_block['y'])):
            self.current_block['shape'] = rotated_shape
        # Basic wall kick
        else:
            for dx in [-1, 1, -2, 2]:
                if not self._check_collision(rotated_shape, (self.current_block['x'] + dx, self.current_block['y'])):
                    self.current_block['x'] += dx
                    self.current_block['shape'] = rotated_shape
                    return

    def _hard_drop(self):
        # sfx: hard_drop.wav
        dy = 0
        while not self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'] + dy + 1)):
            dy += 1
        self.current_block['y'] += dy
        self.fall_counter = 1.0 # Force placement on next logic tick
        return 0.01 * dy # Small reward for dropping

    def _hold_block(self):
        # sfx: hold.wav
        if self.can_hold:
            self.can_hold = False
            if self.held_block:
                self.current_block, self.held_block = self.held_block, self.current_block
                self.current_block['x'] = self.GRID_WIDTH // 2 - len(self.current_block['shape'][0]) // 2
                self.current_block['y'] = 0
            else:
                self.held_block = self.current_block
                self._spawn_new_block()
            
            if self.held_block and self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'])):
                 self.game_over = True
            return True
        return False
        
    # --- Helper Methods: Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _draw_block(self, surface, x, y, color, size, is_ghost=False):
        rect = pygame.Rect(x, y, size, size)
        if is_ghost:
            pygame.gfxdraw.box(surface, rect, color)
        else:
            pygame.draw.rect(surface, color, rect)
            # Add a subtle 3D effect
            l_color = tuple(max(0, c - 40) for c in color)
            d_color = tuple(max(0, c - 80) for c in color)
            pygame.draw.line(surface, l_color, rect.topleft, rect.topright, 2)
            pygame.draw.line(surface, l_color, rect.topleft, rect.bottomleft, 2)
            pygame.draw.line(surface, d_color, rect.bottomright, rect.topright, 2)
            pygame.draw.line(surface, d_color, rect.bottomright, rect.bottomleft, 2)

    def _draw_shape(self, surface, shape_data, offset_x, offset_y, centered=False):
        shape = shape_data['shape']
        color = shape_data['color']
        
        if centered:
            shape_w = len(shape[0]) * self.BLOCK_SIZE
            shape_h = len(shape) * self.BLOCK_SIZE
            offset_x -= shape_w // 2
            offset_y -= shape_h // 2
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(surface, offset_x + c * self.BLOCK_SIZE, offset_y + r * self.BLOCK_SIZE, color, self.BLOCK_SIZE)

    def _render_game(self):
        # Draw grid background and lines
        grid_rect = pygame.Rect(self.grid_top_left_x, self.grid_top_left_y, self.grid_pixel_width, self.grid_pixel_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_top_left_x + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (x, self.grid_top_left_y), (x, self.grid_top_left_y + self.grid_pixel_height), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_top_left_y + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (self.grid_top_left_x, y), (self.grid_top_left_x + self.grid_pixel_width, y), self.GRID_LINE_WIDTH)

        # Draw placed blocks
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell != 0:
                    color = self.COLORS[cell - 1]
                    self._draw_block(self.screen, self.grid_top_left_x + c * self.BLOCK_SIZE, self.grid_top_left_y + r * self.BLOCK_SIZE, color, self.BLOCK_SIZE)

        # Draw line clear effect
        for effect in self.line_clear_effects:
            effect['timer'] -= 1
            if effect['timer'] > 0:
                alpha = 255 * (effect['timer'] / 10)
                flash_surface = pygame.Surface((self.grid_pixel_width, self.BLOCK_SIZE), pygame.SRCALPHA)
                flash_surface.fill((255, 255, 255, alpha))
                self.screen.blit(flash_surface, (self.grid_top_left_x, self.grid_top_left_y + effect['y'] * self.BLOCK_SIZE))
        self.line_clear_effects = [e for e in self.line_clear_effects if e['timer'] > 0]

        if not self.game_over:
            # Draw ghost block
            dy = 0
            while not self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'] + dy + 1)):
                dy += 1
            ghost_x = self.grid_top_left_x + self.current_block['x'] * self.BLOCK_SIZE
            ghost_y = self.grid_top_left_y + (self.current_block['y'] + dy) * self.BLOCK_SIZE
            for r, row in enumerate(self.current_block['shape']):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen, ghost_x + c * self.BLOCK_SIZE, ghost_y + r * self.BLOCK_SIZE, self.COLOR_GHOST, self.BLOCK_SIZE, is_ghost=True)

            # Draw current block
            block_x = self.grid_top_left_x + self.current_block['x'] * self.BLOCK_SIZE
            block_y = self.grid_top_left_y + self.current_block['y'] * self.BLOCK_SIZE
            self._draw_shape(self.screen, self.current_block, block_x, block_y)
        
        # Draw particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            rect = pygame.Rect(int(p['pos'][0]), int(p['pos'][1]), p['size'], p['size'])
            pygame.gfxdraw.box(self.screen, rect, color)


    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Next block display
        next_box_rect = pygame.Rect(self.grid_top_left_x + self.grid_pixel_width + 20, self.grid_top_left_y, 120, 100)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, next_box_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, next_box_rect, 2)
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (next_box_rect.centerx - next_text.get_width() // 2, next_box_rect.top + 5))
        self._draw_shape(self.screen, self.next_block, next_box_rect.centerx, next_box_rect.centery + 10, centered=True)

        # Held block display
        hold_box_rect = pygame.Rect(self.grid_top_left_x - 140, self.grid_top_left_y, 120, 100)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, hold_box_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, hold_box_rect, 2)
        hold_text = self.font_small.render("HOLD", True, self.COLOR_TEXT)
        self.screen.blit(hold_text, (hold_box_rect.centerx - hold_text.get_width() // 2, hold_box_rect.top + 5))
        if self.held_block:
            self._draw_shape(self.screen, self.held_block, hold_box_rect.centerx, hold_box_rect.centery + 10, centered=True)
            if not self.can_hold: # Dim if unavailable
                 dim_surface = pygame.Surface(hold_box_rect.size, pygame.SRCALPHA)
                 dim_surface.fill((0,0,0,128))
                 self.screen.blit(dim_surface, hold_box_rect.topleft)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            win_text = self.font_main.render(status_text, True, (255, 255, 50))
            text_rect = win_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(win_text, text_rect)


    def _create_particles(self, x, y, color):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(20, 40)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.randint(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# --- Example Usage ---
if __name__ == "__main__":
    # To run the game with manual controls
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Falling Block Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action Mapping for Manual Play ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
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
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # The env automatically handles the game over screen, so we just wait for reset
            
        # --- Render to the Display Window ---
        # The observation is (H, W, C), but pygame blit needs a surface
        # We can get the surface directly from the env for display
        surf = env.screen 
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    pygame.quit()