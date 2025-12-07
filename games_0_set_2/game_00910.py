
# Generated: 2025-08-27T15:11:28.256051
# Source Brief: brief_00910.md
# Brief Index: 910

        
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
        "Controls: ←→ to move, ↓ for soft drop, ↑ to rotate. Press space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced pixel-block puzzler. Clear lines against the clock to advance through stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    BLOCK_SIZE = 18
    
    GRID_LINE_WIDTH = 1
    GRID_AREA_WIDTH = GRID_WIDTH * BLOCK_SIZE
    GRID_AREA_HEIGHT = GRID_HEIGHT * BLOCK_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_AREA_HEIGHT)

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_ACCENT = (255, 215, 0)
    COLOR_PANEL = (30, 30, 40)
    COLOR_FLASH = (255, 255, 255)

    # Tetromino shapes and colors
    SHAPES = {
        'I': [[(0, 0), (0, 1), (0, 2), (0, 3)]],
        'O': [[(0, 0), (0, 1), (1, 0), (1, 1)]],
        'T': [[(0, 1), (1, 1), (2, 1), (1, 0)], [(1, 0), (1, 1), (1, 2), (0, 1)], [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (1, 1), (1, 2), (2, 1)]],
        'L': [[(0, 1), (1, 1), (2, 1), (2, 0)], [(1, 0), (1, 1), (1, 2), (2, 2)], [(0, 1), (1, 1), (2, 1), (0, 2)], [(0, 0), (1, 0), (1, 1), (1, 2)]],
        'J': [[(0, 1), (1, 1), (2, 1), (0, 0)], [(1, 0), (1, 1), (1, 2), (2, 0)], [(0, 1), (1, 1), (2, 1), (2, 2)], [(0, 2), (1, 0), (1, 1), (1, 2)]],
        'S': [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
        'Z': [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (1, 1), (0, 1), (0, 2)]]
    }
    
    SHAPE_COLORS = {
        'I': (0, 255, 255), 'O': (255, 255, 0), 'T': (128, 0, 128),
        'L': (255, 165, 0), 'J': (0, 0, 255), 'S': (0, 255, 0), 'Z': (255, 0, 0)
    }

    # Game parameters
    FPS = 30
    TIME_PER_STAGE = 60 # seconds
    MAX_STAGES = 3
    LINES_PER_STAGE = 10
    
    # Input cooldowns (in frames)
    MOVE_COOLDOWN = 3
    ROTATE_COOLDOWN = 6

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
        
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        self.reset()
        
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.stage = 1
        self.total_lines_cleared_in_stage = 0
        
        self.timer = self.TIME_PER_STAGE * self.FPS
        self.game_over = False
        self.win = False
        self.steps = 0
        
        self.fall_counter = 0
        self.fall_speed = 20 # frames per drop

        self.move_cooldown_timer = 0
        self.rotate_cooldown_timer = 0
        
        self.previous_space_state = 0
        
        self.particles = []
        self.line_clear_animation = []

        self._spawn_new_block(first=True)
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        self.steps += 1
        self.timer = max(0, self.timer - 1)
        
        self._update_animations()

        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # --- Handle Input and Movement ---
            is_hard_drop = space_held and not self.previous_space_state
            
            if is_hard_drop:
                # // Sound effect: Hard drop
                reward += self._hard_drop()
                self._lock_block()
            else:
                # Cooldowns
                if self.move_cooldown_timer > 0: self.move_cooldown_timer -= 1
                if self.rotate_cooldown_timer > 0: self.rotate_cooldown_timer -= 1
                
                # Horizontal Movement
                if self.move_cooldown_timer == 0:
                    if movement == 3: # Left
                        if self._is_valid_move(self.block_x - 1, self.block_y, self.block_rotation):
                            self.block_x -= 1
                            self.move_cooldown_timer = self.MOVE_COOLDOWN
                            reward -= 0.05
                    elif movement == 4: # Right
                        if self._is_valid_move(self.block_x + 1, self.block_y, self.block_rotation):
                            self.block_x += 1
                            self.move_cooldown_timer = self.MOVE_COOLDOWN
                            reward -= 0.05
                
                # Rotation
                if movement == 1 and self.rotate_cooldown_timer == 0: # Up
                    next_rotation = (self.block_rotation + 1) % len(self.SHAPES[self.block_shape])
                    if self._is_valid_move(self.block_x, self.block_y, next_rotation):
                        self.block_rotation = next_rotation
                        self.rotate_cooldown_timer = self.ROTATE_COOLDOWN
                        reward += 0.01
                        # // Sound effect: Rotate
                
                # Soft Drop
                if movement == 2: # Down
                    self.fall_counter += self.fall_speed // 2 # Speed up drop
                    reward += 0.1

                # Auto Fall
                self.fall_counter += 1
                if self.fall_counter >= self.fall_speed:
                    self.fall_counter = 0
                    if self._is_valid_move(self.block_x, self.block_y + 1, self.block_rotation):
                        self.block_y += 1
                    else:
                        self._lock_block()

            # --- Check for landed block and process ---
            if self.block_locked:
                lines_cleared, line_reward = self._check_and_clear_lines()
                reward += line_reward
                
                if lines_cleared > 0:
                    self.total_lines_cleared_in_stage += lines_cleared
                    if self.total_lines_cleared_in_stage >= self.LINES_PER_STAGE:
                        # Stage clear
                        reward += 100
                        self.stage += 1
                        if self.stage > self.MAX_STAGES:
                            self.win = True
                            self.game_over = True
                        else:
                            # // Sound effect: Stage Clear
                            self.total_lines_cleared_in_stage = 0
                            self.timer = self.TIME_PER_STAGE * self.FPS
                
                if not self.game_over:
                    self._spawn_new_block()
                    if not self._is_valid_move(self.block_x, self.block_y, self.block_rotation):
                        self.game_over = True # Top out
                        reward -= 100 # Penalty for topping out
            
            self.previous_space_state = space_held

        # --- Check Termination Conditions ---
        if self.timer <= 0 and not self.game_over:
            self.game_over = True
            reward -= 100 # Penalty for time out
        
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_ui_panels()
        self._render_grid()
        self._render_locked_blocks()
        if not self.game_over:
            self._render_ghost_block()
            self._render_current_block()
        self._render_animations()
        self._render_ui_text()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "lines_to_clear": self.LINES_PER_STAGE - self.total_lines_cleared_in_stage,
            "time_left": self.timer / self.FPS
        }

    # --- Helper and Logic Methods ---

    def _spawn_new_block(self, first=False):
        if first:
            self.next_block_shape = self.np_random.choice(list(self.SHAPES.keys()))
        
        self.block_shape = self.next_block_shape
        self.next_block_shape = self.np_random.choice(list(self.SHAPES.keys()))
        
        self.block_color = self.SHAPE_COLORS[self.block_shape]
        self.block_rotation = 0
        self.block_x = self.GRID_WIDTH // 2 - 1
        self.block_y = 0
        self.block_locked = False

    def _is_valid_move(self, x, y, rotation):
        shape_coords = self.SHAPES[self.block_shape][rotation]
        for dx, dy in shape_coords:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
                return False
            if self.grid[ny, nx] != 0:
                return False
        return True

    def _lock_block(self):
        # // Sound effect: Block land
        shape_coords = self.SHAPES[self.block_shape][self.block_rotation]
        for dx, dy in shape_coords:
            nx, ny = self.block_x + dx, self.block_y + dy
            if 0 <= ny < self.GRID_HEIGHT:
                self.grid[ny, nx] = list(self.SHAPE_COLORS.values()).index(self.block_color) + 1
        self.block_locked = True

    def _hard_drop(self):
        drop_y = self.block_y
        while self._is_valid_move(self.block_x, drop_y + 1, self.block_rotation):
            drop_y += 1
        
        reward = (drop_y - self.block_y) * 0.2 # Small reward for distance
        self.block_y = drop_y
        return reward

    def _check_and_clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y, :] != 0):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            return 0, 0
            
        # // Sound effect: Line clear
        for y in lines_to_clear:
            self.grid[y, :] = 0
            self.line_clear_animation.append({'y': y, 'timer': 10})
            self._create_particles_for_line(y)

        # Shift rows down
        new_grid = np.zeros_like(self.grid)
        new_y = self.GRID_HEIGHT - 1
        for y in range(self.GRID_HEIGHT - 1, -1, -1):
            if y not in lines_to_clear:
                new_grid[new_y, :] = self.grid[y, :]
                new_y -= 1
        self.grid = new_grid

        num_cleared = len(lines_to_clear)
        self.score += (100 * num_cleared) * num_cleared # Bonus for multi-clears
        reward = 5 if num_cleared > 1 else 1
        return num_cleared, reward

    # --- Animation and Particle Methods ---

    def _update_animations(self):
        # Line clear flash
        self.line_clear_animation = [anim for anim in self.line_clear_animation if anim['timer'] > 0]
        for anim in self.line_clear_animation:
            anim['timer'] -= 1

        # Particles
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['timer'] -= 1
            if p['timer'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _create_particles_for_line(self, y):
        for x in range(self.GRID_WIDTH):
            for _ in range(5):
                color_val = self.np_random.integers(150, 256)
                color = (color_val, color_val, self.np_random.integers(200, 256))
                self.particles.append({
                    'pos': [self.GRID_X_OFFSET + x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2, self.GRID_Y_OFFSET + y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2],
                    'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 1)],
                    'timer': self.np_random.integers(15, 30),
                    'color': color,
                    'size': self.np_random.integers(2, 5)
                })

    # --- Rendering Methods ---
    
    def _draw_block(self, surface, x, y, color, size):
        outer_rect = pygame.Rect(x, y, size, size)
        inner_rect = pygame.Rect(x + 2, y + 2, size - 4, size - 4)
        
        light_color = tuple(min(255, c + 60) for c in color)
        dark_color = tuple(max(0, c - 60) for c in color)
        
        pygame.draw.rect(surface, dark_color, outer_rect)
        pygame.draw.rect(surface, color, inner_rect)
        pygame.gfxdraw.pixel(surface, int(x + 3), int(y + 3), light_color)

    def _render_ui_panels(self):
        # Main game area panel
        pygame.draw.rect(self.screen, self.COLOR_PANEL, (self.GRID_X_OFFSET - 10, self.GRID_Y_OFFSET - 10, self.GRID_AREA_WIDTH + 20, self.GRID_AREA_HEIGHT + 20), border_radius=5)
        # Left panel
        pygame.draw.rect(self.screen, self.COLOR_PANEL, (20, 20, 180, 360), border_radius=5)
        # Right panel
        pygame.draw.rect(self.screen, self.COLOR_PANEL, (self.SCREEN_WIDTH - 200, 20, 180, 200), border_radius=5)

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(self.GRID_X_OFFSET + x * self.BLOCK_SIZE,
                                   self.GRID_Y_OFFSET + y * self.BLOCK_SIZE,
                                   self.BLOCK_SIZE, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, self.GRID_LINE_WIDTH)

    def _render_locked_blocks(self):
        colors = list(self.SHAPE_COLORS.values())
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    color_index = int(self.grid[y, x] - 1)
                    color = colors[color_index]
                    self._draw_block(self.screen,
                                     self.GRID_X_OFFSET + x * self.BLOCK_SIZE,
                                     self.GRID_Y_OFFSET + y * self.BLOCK_SIZE,
                                     color, self.BLOCK_SIZE)

    def _render_current_block(self):
        shape_coords = self.SHAPES[self.block_shape][self.block_rotation]
        for dx, dy in shape_coords:
            x = self.GRID_X_OFFSET + (self.block_x + dx) * self.BLOCK_SIZE
            y = self.GRID_Y_OFFSET + (self.block_y + dy) * self.BLOCK_SIZE
            self._draw_block(self.screen, x, y, self.block_color, self.BLOCK_SIZE)

    def _render_ghost_block(self):
        ghost_y = self.block_y
        while self._is_valid_move(self.block_x, ghost_y + 1, self.block_rotation):
            ghost_y += 1
        
        shape_coords = self.SHAPES[self.block_shape][self.block_rotation]
        for dx, dy in shape_coords:
            x = self.GRID_X_OFFSET + (self.block_x + dx) * self.BLOCK_SIZE
            y = self.GRID_Y_OFFSET + (ghost_y + dy) * self.BLOCK_SIZE
            rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
            
            # Create a semi-transparent surface
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            s.fill((*self.block_color, 60))
            self.screen.blit(s, rect.topleft)
            pygame.draw.rect(self.screen, self.block_color, rect, 1)

    def _render_animations(self):
        # Line clear flash
        for anim in self.line_clear_animation:
            y = anim['y']
            alpha = int(255 * (anim['timer'] / 10))
            flash_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.BLOCK_SIZE, self.GRID_AREA_WIDTH, self.BLOCK_SIZE)
            s = pygame.Surface(flash_rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_FLASH, alpha))
            self.screen.blit(s, flash_rect.topleft)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['timer'] / 30.0))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

    def _render_ui_text(self):
        # Left Panel: Game Info
        score_title = self.font_medium.render("SCORE", True, self.COLOR_TEXT_ACCENT)
        score_value = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_title, (40, 40))
        self.screen.blit(score_value, (40, 70))
        
        stage_title = self.font_medium.render("STAGE", True, self.COLOR_TEXT_ACCENT)
        stage_value = self.font_large.render(f"{self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_title, (40, 140))
        self.screen.blit(stage_value, (40, 170))

        lines_title = self.font_medium.render("LINES", True, self.COLOR_TEXT_ACCENT)
        lines_left = self.LINES_PER_STAGE - self.total_lines_cleared_in_stage
        lines_value = self.font_large.render(f"{lines_left}", True, self.COLOR_TEXT)
        self.screen.blit(lines_title, (40, 240))
        self.screen.blit(lines_value, (40, 270))

        # Right Panel: Timer and Next Block
        time_title = self.font_medium.render("TIME", True, self.COLOR_TEXT_ACCENT)
        time_seconds = math.ceil(self.timer / self.FPS)
        time_color = (255, 100, 100) if time_seconds <= 10 else self.COLOR_TEXT
        time_value = self.font_large.render(f"{time_seconds}", True, time_color)
        self.screen.blit(time_title, (self.SCREEN_WIDTH - 180, 40))
        self.screen.blit(time_value, (self.SCREEN_WIDTH - 180, 70))
        
        next_title = self.font_medium.render("NEXT", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(next_title, (self.SCREEN_WIDTH - 180, 130))
        
        # Render next block
        next_shape_coords = self.SHAPES[self.next_block_shape][0]
        next_color = self.SHAPE_COLORS[self.next_block_shape]
        for dx, dy in next_shape_coords:
            x = self.SCREEN_WIDTH - 150 + dx * self.BLOCK_SIZE
            y = 170 + dy * self.BLOCK_SIZE
            self._draw_block(self.screen, x, y, next_color, self.BLOCK_SIZE)

        # Game Over / Win Text
        if self.game_over:
            text = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(text, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Mapping from Pygame keys to our action space
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # For human play, we need a separate screen
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Fall")
    clock = pygame.time.Clock()

    while running:
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        if keys[pygame.K_r]: # Reset key
            obs, info = env.reset()
            total_reward = 0

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)

        clock.tick(GameEnv.FPS)

    pygame.quit()