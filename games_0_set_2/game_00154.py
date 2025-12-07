
# Generated: 2025-08-27T12:45:45.658151
# Source Brief: brief_00154.md
# Brief Index: 154

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade puzzle game where the player clears a grid of colored blocks
    against a timer. The goal is to clear 50% of the blocks by clicking
    on groups of same-colored adjacent blocks. Larger groups yield more points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to select and clear a block group."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear 50% of the block grid within the time limit! "
        "Clicking a block removes it and all adjacent blocks of the same color. "
        "Bigger groups give bonus points."
    )

    # Frames auto-advance for smooth timer and animations.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    BLOCK_SIZE = 40
    FPS = 30
    GAME_DURATION_SECONDS = 30
    MAX_STEPS = GAME_DURATION_SECONDS * FPS + 10  # A little buffer

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (44, 48, 60)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    BLOCK_COLORS = [
        (255, 82, 82),   # Red
        (52, 152, 219),  # Blue
        (46, 204, 113),  # Green
        (241, 196, 15),   # Yellow
        (155, 89, 182),  # Purple
    ]
    TIMER_COLORS = {
        "high": (46, 204, 113),
        "medium": (241, 196, 15),
        "low": (255, 82, 82),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- Game State Variables (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.timer = None
        self.initial_block_count = None
        self.blocks_cleared = None
        self.particles = None
        self.click_cooldown = None
        self.last_space_state = None

        # Call reset to initialize the state for the first time
        self.reset()
        
        # This check is for development and ensures API compliance
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS * self.FPS
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []
        self.click_cooldown = 0
        self.last_space_state = 0

        # --- Generate Grid ---
        self.grid = self.np_random.integers(
            1, len(self.BLOCK_COLORS) + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int8
        )
        self.initial_block_count = self.GRID_WIDTH * self.GRID_HEIGHT
        self.blocks_cleared = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement, space_raw, _ = action
        space_pressed = space_raw == 1
        
        # Detect rising edge of space press to treat it as a single click
        is_click_action = space_pressed and not self.last_space_state
        self.last_space_state = space_pressed

        reward = 0
        cleared_count = 0

        # --- Update Game Logic ---
        self._update_cursor(movement)
        
        if is_click_action:
            # Sound effect placeholder: # sfx_click()
            cleared_count = self._clear_blocks_at_cursor()
        
        self._update_particles()
        
        # --- Calculate Reward ---
        if cleared_count > 0:
            reward += cleared_count  # +1 per block
            if cleared_count > 5:
                reward += 5  # Bonus for large groups
                # Sound effect placeholder: # sfx_combo()
        elif movement == 0 and not is_click_action:
            reward -= 0.1 # Small penalty for doing nothing

        self.score += reward

        # --- Update Timers and Counters ---
        self.steps += 1
        self.timer -= 1

        # --- Check Termination Conditions ---
        cleared_percentage = self.blocks_cleared / self.initial_block_count if self.initial_block_count > 0 else 0
        
        win = cleared_percentage >= 0.5
        lose_time = self.timer <= 0
        lose_steps = self.steps >= self.MAX_STEPS
        
        terminated = win or lose_time or lose_steps
        
        if terminated and not self.game_over:
            self.game_over = True
            if win:
                reward += 100 # Big reward for winning
                # Sound effect placeholder: # sfx_win()
            else:
                reward -= 100 # Big penalty for losing
                # Sound effect placeholder: # sfx_lose()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    def _update_cursor(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        # Clamp cursor to grid boundaries
        self.cursor_pos[0] = max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0]))
        self.cursor_pos[1] = max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1]))

    def _clear_blocks_at_cursor(self):
        x, y = self.cursor_pos
        target_color_idx = self.grid[x, y]

        if target_color_idx == 0: # Empty space
            return 0

        q = deque([(x, y)])
        visited = set([(x, y)])
        blocks_to_clear = []

        while q:
            cx, cy = q.popleft()
            blocks_to_clear.append((cx, cy))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[nx, ny] == target_color_idx:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        
        if len(blocks_to_clear) < 2: # Only clear groups of 2 or more
            # Sound effect placeholder: # sfx_invalid_click()
            return 0

        for bx, by in blocks_to_clear:
            self.grid[bx, by] = 0 # Set to empty
            self._create_particles(bx, by, target_color_idx)
        
        self.blocks_cleared += len(blocks_to_clear)
        return len(blocks_to_clear)
    
    def _create_particles(self, grid_x, grid_y, color_idx):
        px = grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        py = grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        color = self.BLOCK_COLORS[color_idx - 1]

        for _ in range(10): # Number of particles per block
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # lifetime -= 1
            if p[4] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        cleared_percentage = self.blocks_cleared / self.initial_block_count if self.initial_block_count > 0 else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cleared_percentage": cleared_percentage
        }

    def _render_game(self):
        # --- Draw Grid and Blocks ---
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.BLOCK_SIZE, y * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                color_idx = self.grid[x, y]
                if color_idx > 0:
                    color = self.BLOCK_COLORS[color_idx - 1]
                    inner_rect = rect.inflate(-4, -4)
                    pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)
        
        # --- Draw Particles ---
        for p in self.particles:
            x, y, _, _, lifetime, color = p
            size = max(1, int(lifetime / 4))
            pygame.draw.rect(self.screen, color, (int(x - size/2), int(y - size/2), size, size))

        # --- Draw Cursor ---
        if not self.game_over:
            cursor_rect = pygame.Rect(
                self.cursor_pos[0] * self.BLOCK_SIZE,
                self.cursor_pos[1] * self.BLOCK_SIZE,
                self.BLOCK_SIZE, self.BLOCK_SIZE
            )
            # Highlight effect
            highlight_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surface, (255, 255, 255, 60), (0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE), border_radius=6)
            self.screen.blit(highlight_surface, cursor_rect.topleft)
            
            # Cursor outline
            pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 2, border_radius=6)
            
    def _render_text(self, text, font, position, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        text_surf = font.render(str(text), True, color)
        shadow_surf = font.render(str(text), True, shadow_color)
        self.screen.blit(shadow_surf, (position[0] + 2, position[1] + 2))
        self.screen.blit(text_surf, position)

    def _render_ui(self):
        # --- Score Display ---
        self._render_text(f"SCORE: {int(self.score)}", self.font_large, (15, 10))
        
        # --- Cleared Percentage Display ---
        cleared_percentage = self.blocks_cleared / self.initial_block_count if self.initial_block_count > 0 else 0
        self._render_text(f"CLEARED: {cleared_percentage:.0%}", self.font_small, (15, 45))

        # --- Timer Bar ---
        timer_pct = max(0, self.timer / (self.GAME_DURATION_SECONDS * self.FPS))
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 15
        bar_y = 15

        if timer_pct > 0.5:
            timer_color = self.TIMER_COLORS["high"]
        elif timer_pct > 0.2:
            timer_color = self.TIMER_COLORS["medium"]
        else:
            timer_color = self.TIMER_COLORS["low"]

        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, timer_color, (bar_x, bar_y, int(bar_width * timer_pct), bar_height), border_radius=5)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the screen ---
        # The observation is (H, W, C), but pygame surfaces are (W, H)
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Cleared: {info['cleared_percentage']:.0%}")
            obs, info = env.reset()
            # Add a small delay before restarting
            pygame.time.wait(2000)

        # Control the frame rate
        env.clock.tick(GameEnv.FPS)

    env.close()