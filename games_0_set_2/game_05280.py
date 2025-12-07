
# Generated: 2025-08-28T04:31:27.155342
# Source Brief: brief_05280.md
# Brief Index: 5280

        
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


class FadingParticle:
    """A helper class to render a fading, rising number effect."""
    def __init__(self, text, pos, color, font, lifetime=30):
        self.pos = list(pos)
        self.text = text
        self.color = color
        self.font = font
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        """Update particle's position and lifetime. Returns False if dead."""
        self.lifetime -= 1
        self.pos[1] -= 1  # Move upwards
        return self.lifetime > 0

    def draw(self, surface):
        """Draw the particle with an alpha based on its remaining lifetime."""
        if self.lifetime <= 0:
            return
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        text_surf = self.font.render(self.text, True, self.color)
        text_surf.set_alpha(alpha)
        text_rect = text_surf.get_rect(center=self.pos)
        surface.blit(text_surf, text_rect)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to select a number."
    )

    game_description = (
        "A number puzzle game. Select numbers from the grid to make their sum equal to the target value. "
        "Win by reaching the target, lose by running out of numbers."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 4, 3
        self.CELL_WIDTH = 120
        self.CELL_HEIGHT = 80
        self.GRID_X_START = (self.WIDTH - self.GRID_COLS * self.CELL_WIDTH) // 2
        self.GRID_Y_START = 120

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_POSITIVE = (120, 255, 120)
        self.COLOR_NEGATIVE = (255, 120, 120)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_TARGET = (255, 223, 0)
        self.COLOR_CURSOR = (255, 223, 0)
        
        # Game constants
        self.TARGET_SUM = 50
        self.GRID_SIZE = self.GRID_COLS * self.GRID_ROWS
        self.MAX_STEPS = 1000

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.number_font = pygame.font.Font(None, 48)
        self.ui_font = pygame.font.Font(None, 36)
        self.particle_font = pygame.font.Font(None, 32)
        
        # State variables
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_sum = 0
        self.grid_numbers = []
        self.grid_mask = []
        self.cursor_pos = [0, 0] # [col, row]
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def _generate_solvable_grid(self):
        """Generates a grid of numbers with a guaranteed path to the target sum."""
        solution_len = self.rng.integers(4, 9)
        
        # Generate the solution set
        solution_set = []
        current_sum = 0
        for _ in range(solution_len - 1):
            num = self.rng.integers(-15, 26) # Range of numbers
            solution_set.append(num)
            current_sum += num
        solution_set.append(self.TARGET_SUM - current_sum)
        
        # Generate distractors
        num_distractors = self.GRID_SIZE - solution_len
        distractors = self.rng.integers(-25, 26, size=num_distractors).tolist()
        
        # Combine and shuffle
        final_grid = solution_set + distractors
        self.rng.shuffle(final_grid)
        
        return final_grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_sum = 0
        self.grid_numbers = self._generate_solvable_grid()
        self.grid_mask = [False] * self.GRID_SIZE
        self.cursor_pos = [0, 0]
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_pressed, _ = action
        reward = 0

        # 1. Handle Movement
        if movement == 1: # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2: # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_ROWS
        elif movement == 3: # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4: # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_COLS
            
        # 2. Handle Selection
        if space_pressed:
            index = self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]
            if not self.grid_mask[index]:
                # // Sound: select_number.wav
                number_value = self.grid_numbers[index]
                
                dist_before = abs(self.TARGET_SUM - self.current_sum)
                self.current_sum += number_value
                dist_after = abs(self.TARGET_SUM - self.current_sum)
                
                reward = 0.5 if dist_after < dist_before else -0.5
                
                self.grid_mask[index] = True
                
                # Create particle effect
                cell_center_x = self.GRID_X_START + self.cursor_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH // 2
                cell_center_y = self.GRID_Y_START + self.cursor_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT // 2
                particle_text = f"+{number_value}" if number_value >= 0 else str(number_value)
                particle_color = self.COLOR_POSITIVE if number_value >= 0 else self.COLOR_NEGATIVE
                self.particles.append(FadingParticle(particle_text, (cell_center_x, cell_center_y), particle_color, self.particle_font))
        
        # 3. Update game state
        self.steps += 1
        self.score += reward
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # 4. Check for termination
        terminated = False
        if self.current_sum == self.TARGET_SUM:
            # // Sound: win_game.wav
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        elif all(self.grid_mask):
            # // Sound: lose_game.wav
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_grid(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                index = r * self.GRID_COLS + c
                rect = pygame.Rect(
                    self.GRID_X_START + c * self.CELL_WIDTH,
                    self.GRID_Y_START + r * self.CELL_HEIGHT,
                    self.CELL_WIDTH,
                    self.CELL_HEIGHT
                )
                
                # Draw cell background
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2, border_radius=8)

                # Draw number if not selected
                if not self.grid_mask[index]:
                    num = self.grid_numbers[index]
                    color = self.COLOR_POSITIVE if num >= 0 else self.COLOR_NEGATIVE
                    text_surf = self.number_font.render(str(num), True, color)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)
    
    def _render_cursor(self):
        rect = pygame.Rect(
            self.GRID_X_START + self.cursor_pos[0] * self.CELL_WIDTH,
            self.GRID_Y_START + self.cursor_pos[1] * self.CELL_HEIGHT,
            self.CELL_WIDTH,
            self.CELL_HEIGHT
        )
        
        # Draw a glowing effect
        for i in range(5, 0, -1):
            alpha = 150 - i * 25
            glow_rect = rect.inflate(i*2, i*2)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*self.COLOR_CURSOR, alpha), shape_surf.get_rect(), border_radius=10)
            self.screen.blit(shape_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=8)

    def _render_particles(self):
        for particle in self.particles:
            particle.draw(self.screen)

    def _render_ui(self):
        # Current Sum
        sum_text_surf = self.ui_font.render("Current Sum", True, self.COLOR_UI_TEXT)
        self.screen.blit(sum_text_surf, (40, 30))
        sum_val_surf = self.number_font.render(str(self.current_sum), True, self.COLOR_UI_TEXT)
        self.screen.blit(sum_val_surf, (40, 60))

        # Target Sum
        target_text_surf = self.ui_font.render("Target", True, self.COLOR_TARGET)
        target_rect = target_text_surf.get_rect(centerx=self.WIDTH / 2)
        target_rect.top = 30
        self.screen.blit(target_text_surf, target_rect)

        target_val_surf = self.number_font.render(str(self.TARGET_SUM), True, self.COLOR_TARGET)
        target_val_rect = target_val_surf.get_rect(centerx=self.WIDTH / 2)
        target_val_rect.top = 60
        self.screen.blit(target_val_surf, target_val_rect)

        # Remaining
        remaining_count = self.GRID_SIZE - sum(self.grid_mask)
        remaining_text_surf = self.ui_font.render("Remaining", True, self.COLOR_UI_TEXT)
        rem_rect = remaining_text_surf.get_rect(right=self.WIDTH - 40, top=30)
        self.screen.blit(remaining_text_surf, rem_rect)
        
        remaining_val_surf = self.number_font.render(str(remaining_count), True, self.COLOR_UI_TEXT)
        rem_val_rect = remaining_val_surf.get_rect(right=self.WIDTH - 40, top=60)
        self.screen.blit(remaining_val_surf, rem_val_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "current_sum": self.current_sum,
            "remaining_numbers": self.GRID_SIZE - sum(self.grid_mask)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to action space
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
    }
    
    # Setup a window to display the environment
    pygame.display.set_caption("Number Sum Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_q:
                    running = False

        # If an action was triggered, step the environment
        # For this turn-based game, we only step on player input
        if any(action):
            if not done:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
            else:
                print("Game Over. Press 'R' to reset.")

        # Rendering
        # The observation is already the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()