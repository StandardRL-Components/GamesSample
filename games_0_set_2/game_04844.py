
# Generated: 2025-08-28T03:11:19.909139
# Source Brief: brief_04844.md
# Brief Index: 4844

        
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
        "Controls: Arrow keys to move cursor. Space to paint. Shift to cycle color."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pixel art image before time runs out. Match colors and positions for a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60

    GRID_SIZE = 16  # 16x16 grid
    
    # Colors (Vaporwave/Retro aesthetic)
    COLOR_BG = (21, 25, 68)
    COLOR_GRID_LINES = (40, 48, 102)
    COLOR_TEXT = (230, 230, 255)
    COLOR_TEXT_ACCENT = (0, 255, 255)
    COLOR_TEXT_WARN = (255, 100, 100)
    COLOR_CURSOR = (255, 255, 255)
    
    PALETTE = [
        (255, 119, 119), # Red
        (255, 187, 119), # Orange
        (255, 238, 119), # Yellow
        (119, 255, 119), # Green
        (119, 187, 255), # Blue
        (187, 119, 255), # Purple
        (255, 119, 204), # Pink
        (255, 255, 255), # White
    ]
    BLANK_COLOR_INDEX = -1 # Special index for unpainted cells
    BLANK_COLOR = (28, 34, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 32)
        self.font_big = pygame.font.Font(None, 72)
        
        # Layout calculations
        self.grid_area_width = self.SCREEN_HEIGHT - 40
        self.cell_size = self.grid_area_width // self.GRID_SIZE
        self.grid_render_size = self.cell_size * self.GRID_SIZE
        self.grid_offset_x = 20
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_render_size) // 2

        self.ui_offset_x = self.grid_offset_x + self.grid_render_size + 30
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.time_remaining = 0
        self.cursor_pos = [0, 0]
        self.selected_color_index = 0
        self.target_grid = None
        self.player_grid = None
        self.last_space_held = False
        self.last_shift_held = False
        self.accuracy = 0.0
        self.particles = []

        self.reset()

        # self.validate_implementation() # Optional: for debugging during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_index = 0
        
        # Generate a new symmetrical target image
        half_width = math.ceil(self.GRID_SIZE / 2)
        half_grid = self.np_random.integers(0, len(self.PALETTE), size=(self.GRID_SIZE, half_width))
        full_grid = np.concatenate((half_grid, np.fliplr(half_grid[:, :self.GRID_SIZE//2])), axis=1)
        self.target_grid = full_grid
        
        self.player_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.BLANK_COLOR_INDEX, dtype=int)
        
        self.last_space_held = False
        self.last_shift_held = False
        self.accuracy = 0.0
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        
        if not self.game_over:
            # Update game logic
            self.time_remaining -= 1
            
            # 1. Handle cursor movement
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] %= self.GRID_SIZE
            self.cursor_pos[1] %= self.GRID_SIZE

            # 2. Handle color cycling (Shift key press)
            if shift_held and not self.last_shift_held:
                self.selected_color_index = (self.selected_color_index + 1) % len(self.PALETTE)
                # SFX: color_cycle_sound

            # 3. Handle painting (Space key press)
            if space_held and not self.last_space_held:
                x, y = self.cursor_pos
                
                # Only allow painting if the cell isn't already correct
                if self.player_grid[y, x] != self.target_grid[y, x]:
                    painted_color_idx = self.selected_color_index
                    target_color_idx = self.target_grid[y, x]

                    self.player_grid[y, x] = painted_color_idx
                    
                    if painted_color_idx == target_color_idx:
                        reward += 1.0
                        # SFX: correct_paint_sound
                        self._create_particles(self.cursor_pos, self.PALETTE[painted_color_idx])
                    else:
                        reward -= 0.2
                        # SFX: wrong_paint_sound
                        self._create_particles(self.cursor_pos, self.COLOR_TEXT_WARN, count=5)

        self.score += reward
        self.steps += 1
        
        # Update state variables
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        correct_pixels = np.sum(self.player_grid == self.target_grid)
        self.accuracy = correct_pixels / (self.GRID_SIZE * self.GRID_SIZE)

        # Check termination conditions
        terminated = False
        if self.time_remaining <= 0:
            terminated = True
            self.game_over = True
        if self.accuracy >= 0.95:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100.0
            self.score += 100.0
            # SFX: win_jingle

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
            "accuracy": self.accuracy,
            "time_remaining": self.time_remaining,
        }

    def _render_game(self):
        # Render player grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.player_grid[y, x]
                color = self.PALETTE[color_idx] if color_idx != self.BLANK_COLOR_INDEX else self.BLANK_COLOR
                
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_render_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)
            # Horizontal
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.cell_size)
            end_pos = (self.grid_offset_x + self.grid_render_size, self.grid_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)

        # Update and draw particles
        self._update_particles()

        # Draw pulsing cursor
        if not self.game_over:
            cursor_x, cursor_y = self.cursor_pos
            pulse = (math.sin(self.steps * 0.4) + 1) / 2  # 0 to 1
            alpha = 100 + pulse * 100
            
            cursor_rect = pygame.Rect(
                self.grid_offset_x + cursor_x * self.cell_size,
                self.grid_offset_y + cursor_y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_CURSOR, alpha), s.get_rect(), border_radius=2)
            pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), 2, border_radius=2)
            self.screen.blit(s, cursor_rect.topleft)

    def _render_ui(self):
        ui_x = self.ui_offset_x
        
        # 1. Target Image
        target_title = self.font_main.render("TARGET", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(target_title, (ui_x, 20))

        target_cell_size = 8
        target_render_size = target_cell_size * self.GRID_SIZE
        target_offset_x = ui_x + (self.SCREEN_WIDTH - ui_x - target_render_size) // 2
        target_offset_y = 50

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.target_grid[y, x]
                color = self.PALETTE[color_idx]
                rect = pygame.Rect(
                    target_offset_x + x * target_cell_size,
                    target_offset_y + y * target_cell_size,
                    target_cell_size,
                    target_cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

        # 2. Palette
        palette_y_start = target_offset_y + target_render_size + 25
        palette_title = self.font_main.render("PALETTE", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(palette_title, (ui_x, palette_y_start))

        swatch_size = 20
        for i, color in enumerate(self.PALETTE):
            py = palette_y_start + 25 + (i // 2) * (swatch_size + 5)
            px = ui_x + (i % 2) * (swatch_size + 5)
            pygame.draw.rect(self.screen, color, (px, py, swatch_size, swatch_size), border_radius=3)
            if i == self.selected_color_index:
                pygame.draw.rect(self.screen, self.COLOR_TEXT_ACCENT, (px-2, py-2, swatch_size+4, swatch_size+4), 2, border_radius=5)

        # 3. Stats
        stats_y_start = 300
        
        # Time
        time_sec = max(0, self.time_remaining / self.FPS)
        time_color = self.COLOR_TEXT if time_sec > 10 else self.COLOR_TEXT_WARN
        time_text = self.font_title.render(f"TIME: {time_sec:.1f}", True, time_color)
        self.screen.blit(time_text, (ui_x, stats_y_start))

        # Accuracy
        acc_text = self.font_title.render(f"ACC: {self.accuracy:.0%}", True, self.COLOR_TEXT)
        self.screen.blit(acc_text, (ui_x, stats_y_start + 30))

        # Score
        score_text = self.font_title.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, stats_y_start + 60))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                msg = "COMPLETE!"
                msg_color = self.COLOR_TEXT_ACCENT
            else:
                msg = "TIME UP!"
                msg_color = self.COLOR_TEXT_WARN
            
            end_text = self.font_big.render(msg, True, msg_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, grid_pos, color, count=15):
        px = self.grid_offset_x + (grid_pos[0] + 0.5) * self.cell_size
        py = self.grid_offset_y + (grid_pos[1] + 0.5) * self.cell_size
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(10, 20)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # lifetime -= 1
            
            radius = max(0, int(p[4] * 0.2))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), radius, p[5])
        
        self.particles = [p for p in self.particles if p[4] > 0]
    
    def close(self):
        pygame.quit()

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Pixel Painter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    # --- Manual Control Mapping ---
    # This maps keyboard keys to the MultiDiscrete action space
    # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    # actions[1]: Space button (0=released, 1=held)
    # actions[2]: Shift button (0=released, 1=held)
    
    action = np.array([0, 0, 0]) # No-op
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # Default to no movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        env.clock.tick(GameEnv.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Accuracy: {info['accuracy']:.0%}")
            pygame.time.wait(3000) # Pause for 3 seconds before closing

    env.close()