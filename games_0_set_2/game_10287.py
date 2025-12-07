import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:15:26.401710
# Source Brief: brief_00287.md
# Brief Index: 287
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Stack falling blocks onto rising columns to score points before time runs out. "
        "Land blocks carefully and create tall stacks for bonus points."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move the falling blocks. Press space to drop them into place."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 16
        self.BLOCK_SIZE = self.SCREEN_WIDTH // self.GRID_COLS

        # Gameplay Constants
        self.MAX_STEPS = 6000  # 60 seconds at 100 steps/sec
        self.WIN_SCORE = 1000
        self.TIME_LIMIT_SECONDS = 60.0
        self.STEPS_PER_SECOND = self.MAX_STEPS / self.TIME_LIMIT_SECONDS

        # Physics
        self.INITIAL_COLUMN_GEN_SPEED = 1.0  # pixels per second
        self.COLUMN_ACCELERATION = 0.005 # speed increase per second
        self.SLOW_FALL_SPEED = 1.0 # pixels per step

        # Colors (Bright, high-contrast)
        self.COLOR_BG = (15, 23, 42)
        self.COLOR_GRID = (30, 41, 59)
        self.COLOR_COLUMN = (51, 65, 85)
        self.COLOR_UI_TEXT = (226, 232, 240)
        self.COLOR_TIMER_BAR = (34, 197, 94)
        self.COLOR_TIMER_BAR_BG = (71, 85, 105)
        self.COLOR_WIN = (74, 222, 128)
        self.COLOR_LOSE = (248, 113, 113)
        self.BLOCK_COLORS = [
            (251, 146, 60), (249, 115, 22), (239, 68, 68),
            (168, 85, 247), (139, 92, 246), (99, 102, 241),
            (59, 130, 246), (37, 99, 235), (14, 165, 233),
            (20, 184, 166), (16, 185, 129), (34, 197, 94)
        ]

        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_elapsed = 0.0
        self.column_gen_speed = 0.0
        self.column_heights = []
        self.placed_blocks = []
        self.player_col = 0
        self.player_y = 0.0
        self.player_colors = []
        self.particles = []
        self.flash_effects = []
        self.last_space_held = False
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_elapsed = 0.0
        self.column_gen_speed = self.INITIAL_COLUMN_GEN_SPEED
        
        self.column_heights = [self.np_random.uniform(20, 80) for _ in range(self.GRID_COLS)]
        self.placed_blocks = [[] for _ in range(self.GRID_COLS)]
        
        self.particles = []
        self.flash_effects = []
        self.last_space_held = False
        
        self._spawn_player_blocks()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if space_pressed:
            # // Sound effect: Block drop
            reward += self._place_player_blocks()
        else:
            self._handle_movement(movement)
            reward += self._apply_gravity()

        self._update_columns()
        self._update_effects()

        self.steps += 1
        self.time_elapsed = self.steps / self.STEPS_PER_SECOND
        self.column_gen_speed += self.COLUMN_ACCELERATION / self.STEPS_PER_SECOND

        terminated = self._check_termination()
        truncated = False # This environment does not truncate
        if terminated and not self.win: # Timeout
            reward -= 10
        elif terminated and self.win: # Win
            reward += 100

        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _spawn_player_blocks(self):
        self.player_col = self.np_random.integers(0, self.GRID_COLS - 2)
        self.player_y = -self.BLOCK_SIZE * 2
        self.player_colors = [random.choice(self.BLOCK_COLORS) for _ in range(3)]

    def _handle_movement(self, movement):
        # Action space: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 3: # Left
            self.player_col -= 1
        elif movement == 4: # Right
            self.player_col += 1
        self.player_col %= self.GRID_COLS

    def _apply_gravity(self):
        self.player_y += self.SLOW_FALL_SPEED
        
        landed = False
        for i in range(3):
            col = (self.player_col + i) % self.GRID_COLS
            landing_y = self._get_landing_y(col)
            if self.player_y >= landing_y:
                landed = True
                break
        
        if landed:
            # // Sound effect: Block land (soft)
            # To ensure it sits flush, align y with the lowest landing spot
            min_landing_y = min(self._get_landing_y((self.player_col + i) % self.GRID_COLS) for i in range(3))
            self.player_y = min_landing_y
            return self._place_player_blocks()
        return 0

    def _place_player_blocks(self):
        reward = 0.1 * 3 # +0.1 for each successful block placement
        
        for i in range(3):
            col = (self.player_col + i) % self.GRID_COLS
            
            self.placed_blocks[col].append(self.player_colors[i])
            
            # Create landing particles
            landing_y = self._get_landing_y(col) + self.BLOCK_SIZE
            self._create_particles(col * self.BLOCK_SIZE + self.BLOCK_SIZE / 2, landing_y)

            # Check for stack score
            if len(self.placed_blocks[col]) >= 5:
                # // Sound effect: Score!
                reward += 10
                self.score += 10
                self.flash_effects.append({'col': col, 'life': 20})

        self._spawn_player_blocks()
        return reward

    def _get_landing_y(self, col_idx):
        column_base_y = self.SCREEN_HEIGHT - self.column_heights[col_idx]
        num_blocks_in_stack = len(self.placed_blocks[col_idx])
        landing_y = column_base_y - (num_blocks_in_stack + 1) * self.BLOCK_SIZE
        return landing_y

    def _update_columns(self):
        delta_height = self.column_gen_speed / self.STEPS_PER_SECOND
        for i in range(self.GRID_COLS):
            self.column_heights[i] += delta_height

    def _update_effects(self):
        # Update particles
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Update flash effects
        for f in self.flash_effects[:]:
            f['life'] -= 1
            if f['life'] <= 0:
                self.flash_effects.remove(f)

    def _create_particles(self, x, y):
        for _ in range(10):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': self.np_random.uniform(-1.5, 1.5),
                'vy': self.np_random.uniform(-2, 0),
                'life': self.np_random.integers(15, 30),
                'color': (255, 255, 255),
                'size': self.np_random.uniform(2, 5)
            })

    def _check_termination(self):
        self.win = self.score >= self.WIN_SCORE
        timeout = self.steps >= self.MAX_STEPS
        self.game_over = self.win or timeout
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame produces (width, height, 3) arrays, but we need (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background_grid()
        self._render_columns()
        self._render_placed_blocks()
        self._render_flash_effects()
        self._render_player_blocks()
        self._render_particles()

    def _render_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.BLOCK_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.BLOCK_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_columns(self):
        for i in range(self.GRID_COLS):
            height = self.column_heights[i]
            rect = pygame.Rect(i * self.BLOCK_SIZE, self.SCREEN_HEIGHT - height, self.BLOCK_SIZE, height)
            pygame.draw.rect(self.screen, self.COLOR_COLUMN, rect)

    def _render_placed_blocks(self):
        for col_idx, column_of_blocks in enumerate(self.placed_blocks):
            column_base_y = self.SCREEN_HEIGHT - self.column_heights[col_idx]
            for i, block_color in enumerate(column_of_blocks):
                block_y = column_base_y - (i + 1) * self.BLOCK_SIZE
                rect = pygame.Rect(col_idx * self.BLOCK_SIZE, int(block_y), self.BLOCK_SIZE, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, block_color, rect)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1) # Outline

    def _render_player_blocks(self):
        for i in range(3):
            col = (self.player_col + i) % self.GRID_COLS
            x = col * self.BLOCK_SIZE
            y = int(self.player_y)
            
            # Glow effect
            glow_size = self.BLOCK_SIZE + 8
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.player_colors[i], 80), glow_surf.get_rect())
            self.screen.blit(glow_surf, (x - 4, y - 4))

            # Block
            rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, self.player_colors[i], rect)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2) # White outline

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (int(p['x']), int(p['y']), int(p['size']), int(p['size'])))

    def _render_flash_effects(self):
        for f in self.flash_effects:
            alpha = int(255 * (f['life'] / 20))
            flash_surface = pygame.Surface((self.BLOCK_SIZE, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (f['col'] * self.BLOCK_SIZE, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer Bar
        timer_width = 200
        timer_height = 20
        time_left_ratio = max(0, (self.TIME_LIMIT_SECONDS - self.time_elapsed) / self.TIME_LIMIT_SECONDS)
        
        bar_bg_rect = pygame.Rect(self.SCREEN_WIDTH - timer_width - 10, 10, timer_width, timer_height)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, bar_bg_rect, border_radius=5)
        
        bar_fill_rect = pygame.Rect(self.SCREEN_WIDTH - timer_width - 10, 10, timer_width * time_left_ratio, timer_height)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, bar_fill_rect, border_radius=5)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_LOSE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.TIME_LIMIT_SECONDS - self.time_elapsed,
            "win": self.win
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Column Stacker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # 0=none, 3=left, 4=right
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds before resetting
            obs, info = env.reset()
            total_reward = 0

        clock.tick(100) # Match the internal step rate for smooth manual play
        
    env.close()