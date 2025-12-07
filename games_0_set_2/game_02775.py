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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling block. Press space to drop it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically drop colored blocks to create horizontal or vertical matches of 3 or more. Plan ahead to create chain reactions and maximize your score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 10
    GRID_ROWS = 10
    BLOCK_SIZE = 36
    GRID_LINE_WIDTH = 2

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (0, 0, 0),  # 0: Empty
        (255, 80, 80),   # 1: Red
        (80, 255, 80),   # 2: Green
        (80, 80, 255),   # 3: Blue
        (255, 255, 80),  # 4: Yellow
        (255, 80, 255),  # 5: Magenta
        (80, 255, 255),  # 6: Cyan
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 18)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)

        # Calculate grid position to center it
        self.grid_width = self.GRID_COLS * self.BLOCK_SIZE
        self.grid_height = self.GRID_ROWS * self.BLOCK_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = self.SCREEN_HEIGHT - self.grid_height - 10 # 10px from bottom

        # State variables are initialized in reset()
        self.np_random = None
        self.grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.drop_column = 0
        self.current_block_color = 0
        self.next_block_color = 0
        self.num_colors = 0
        self.particles = []
        self.last_match_info = []

        # The validation must be performed on a reset environment
        # self.validate_implementation() # This line is commented out as validation is now part of the test script
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.drop_column = self.GRID_COLS // 2
        self.num_colors = 3
        self.current_block_color = self.np_random.integers(1, self.num_colors + 1)
        self.next_block_color = self.np_random.integers(1, self.num_colors + 1)
        self.particles = []
        self.last_match_info = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = self.game_over

        self._update_particles()
        self.last_match_info.clear()

        if terminated:
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Action Handling ---
        # Handle movement. Only allow one move per step for responsiveness.
        if movement == 3:  # Left
            self.drop_column = max(0, self.drop_column - 1)
        elif movement == 4:  # Right
            self.drop_column = min(self.GRID_COLS - 1, self.drop_column + 1)

        # Handle drop
        if space_pressed:
            self.steps += 1

            # Find where the block will land
            landing_row = -1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, self.drop_column] == 0:
                    landing_row = r
                    break

            # If column is full, game over
            if landing_row == -1:
                self.game_over = True
                terminated = True
                reward -= 100.0
            else:
                # Place the block
                self.grid[landing_row, self.drop_column] = self.current_block_color

                # --- Match and Gravity Logic ---
                total_cleared_this_turn = 0
                chain_reaction_count = 0
                while True:
                    matched_blocks = self._find_matches()
                    if not matched_blocks:
                        break

                    chain_reaction_count += 1
                    num_cleared = len(matched_blocks)
                    total_cleared_this_turn += num_cleared

                    # Add reward for clearing, with a bonus for chains
                    reward += (10.0 * num_cleared) * (1.0 + 0.5 * (chain_reaction_count - 1))
                    self.score += (10 * num_cleared) * (1 + 0.5 * (chain_reaction_count - 1))

                    self._clear_blocks(matched_blocks)
                    self._apply_gravity()

                # If no blocks were cleared on drop, check for near-miss or bad placement
                if total_cleared_this_turn == 0:
                    if self._check_near_miss(landing_row, self.drop_column):
                        reward += 1.0  # Continuous feedback for setting up a match
                    else:
                        reward -= 0.2 # Penalty for a "wasted" block

                # Advance to the next block
                self.current_block_color = self.next_block_color
                self._update_difficulty()
                self.next_block_color = self.np_random.integers(1, self.num_colors + 1)

        # --- Termination Checks ---
        if self.score >= 500 and not terminated:
            reward += 100.0
            self.game_over = True
            terminated = True

        if self.steps >= 1000 and not terminated:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_matches(self):
        matched_blocks = set()
        # Check horizontal matches
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                color = self.grid[r, c]
                if color != 0 and color == self.grid[r, c+1] and color == self.grid[r, c+2]:
                    for i in range(self.GRID_COLS - c):
                        if self.grid[r, c+i] == color:
                            matched_blocks.add((r, c+i))
                        else:
                            break
        # Check vertical matches
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                color = self.grid[r, c]
                if color != 0 and color == self.grid[r+1, c] and color == self.grid[r+2, c]:
                    for i in range(self.GRID_ROWS - r):
                        if self.grid[r+i, c] == color:
                            matched_blocks.add((r+i, c))
                        else:
                            break
        return matched_blocks

    def _clear_blocks(self, matched_blocks):
        # Sound effect placeholder: # sfx_match.play()
        for r, c in matched_blocks:
            color_index = self.grid[r, c]
            if color_index > 0:
                self.last_match_info.append(((r, c), color_index))
                self.grid[r, c] = 0

    def _apply_gravity(self):
        # Sound effect placeholder: # sfx_fall.play()
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _check_near_miss(self, r, c):
        color = self.grid[r, c]
        if color == 0: return False
        # Check horizontal
        if c > 0 and self.grid[r, c-1] == color: return True
        if c < self.GRID_COLS - 1 and self.grid[r, c+1] == color: return True
        # Check vertical
        if r < self.GRID_ROWS - 1 and self.grid[r+1, c] == color: return True
        return False

    def _update_difficulty(self):
        if self.score >= 450 and self.num_colors < 6:
            self.num_colors = 6
        elif self.score >= 350 and self.num_colors < 5:
            self.num_colors = 5
        elif self.score >= 250 and self.num_colors < 4:
            self.num_colors = 4

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background and lines
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.grid_width, self.grid_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        for i in range(self.GRID_COLS + 1):
            x = self.grid_offset_x + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_height), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_ROWS + 1):
            y = self.grid_offset_y + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_width, y), self.GRID_LINE_WIDTH)

        # Draw blocks in grid
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid is not None:
                    color_index = self.grid[r, c]
                    if color_index != 0:
                        self._draw_block(c, r, color_index)

        # Create particles from recent matches
        for (r, c), color_index in self.last_match_info:
            px = self.grid_offset_x + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
            py = self.grid_offset_y + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
            color = self.BLOCK_COLORS[color_index]
            for _ in range(15): # Number of particles per block
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(20, 40)
                self.particles.append([ [px, py], [vx, vy], life, color ])

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p[3], p[0], max(0, int(p[2] / 5)))

        # Draw dropper preview block
        if not self.game_over:
            self._draw_block(self.drop_column, -1.2, self.current_block_color, is_preview=True)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[1][1] += 0.1 # Gravity
            p[2] -= 1

    def _draw_block(self, c, r, color_index, is_preview=False):
        x = self.grid_offset_x + c * self.BLOCK_SIZE
        y = self.grid_offset_y + r * self.BLOCK_SIZE
        color = self.BLOCK_COLORS[color_index]

        rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)

        # Draw 3D-ish effect
        highlight = tuple(min(255, val + 50) for val in color)
        shadow = tuple(max(0, val - 50) for val in color)

        pygame.draw.rect(self.screen, shadow, rect)
        inner_rect = rect.inflate(-self.GRID_LINE_WIDTH*2, -self.GRID_LINE_WIDTH*2)
        pygame.draw.rect(self.screen, color, inner_rect)

        # Top-left highlight
        pygame.draw.line(self.screen, highlight, inner_rect.topleft, inner_rect.topright, 2)
        pygame.draw.line(self.screen, highlight, inner_rect.topleft, inner_rect.bottomleft, 2)

        if is_preview:
            # Add a pulsing alpha effect
            alpha = 128 + int(127 * math.sin(pygame.time.get_ticks() / 200.0))
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            s.fill((255, 255, 255, alpha))
            self.screen.blit(s, (x, y))

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Next block preview
        next_text = self.font_title.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 120, 20))
        self._draw_block(
            (self.SCREEN_WIDTH - 100) / self.BLOCK_SIZE,
            (50) / self.BLOCK_SIZE,
            self.next_block_color
        )

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "YOU WON!" if self.score >= 500 else "GAME OVER"
        text_surf = self.font_game_over.render(message, True, (255, 255, 100))
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "num_colors": self.num_colors,
        }

    def close(self):
        pygame.quit()

    # The validation function is not part of the standard environment and is removed.
    # It is good practice for testing but should not be in the final class definition
    # to avoid issues like the one encountered. The test harness will perform validation.


if __name__ == '__main__':
    # --- Manual Play Example ---
    # The following code is for demonstration and manual testing purposes.
    # It is not part of the required GameEnv class.
    # To run this, you'll need to have pygame installed and a display available.
    # You can comment out the os.environ line at the top to run with a display.
    
    # --- Validation Step (similar to the original `validate_implementation`) ---
    def validate_environment(env_instance):
        print("Validating environment...")
        # Test action space
        assert env_instance.action_space.shape == (3,)
        assert env_instance.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and observation space
        obs, info = env_instance.reset()
        assert obs.shape == (env_instance.SCREEN_HEIGHT, env_instance.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = env_instance.action_space.sample()
        obs, reward, term, trunc, info = env_instance.step(test_action)
        assert obs.shape == (env_instance.SCREEN_HEIGHT, env_instance.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    env = GameEnv()
    validate_environment(env) # Perform validation
    
    # --- Manual Play ---
    # To run this, you might need to remove/comment the `os.environ` line at the top
    # if you are not in a headless environment.
    try:
        obs, info = env.reset()
        pygame.display.set_caption("Block Dropper")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        running = True
        while running:
            action = [0, 0, 0] # Default no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action[1] = 1 # Press space to drop
                    if event.key == pygame.K_r: # Reset game
                        obs, info = env.reset()
                    if event.key == pygame.K_ESCAPE:
                        running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4

            obs, reward, terminated, truncated, info = env.step(action)
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.time.wait(3000) # Wait 3 seconds
                obs, info = env.reset()

            env.clock.tick(15) 
    except pygame.error as e:
        print("\nCould not start manual play.")
        print("This is expected in a headless environment (like the one for evaluation).")
        print(f"Pygame error: {e}")
    finally:
        env.close()