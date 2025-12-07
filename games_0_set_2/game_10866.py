import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:11:45.366590
# Source Brief: brief_00866.md
# Brief Index: 866
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a grid-based puzzle game.
    The player manipulates a 5x5 grid of colored blocks to create matches of 3 or more.
    This triggers chain reactions for a high score within a time limit.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement direction (0:none, 1:up, 2:down, 3:left, 4:right)
    - actions[1]: Action button (Space) (0:released, 1:held)
    - actions[2]: Modifier button (Shift) (0:released, 1:held)

    Control Scheme:
    - Movement only: Moves the cursor.
    - Movement + Space: Swaps the block under the cursor with the one in the specified direction.
    - Shift: Cycles the color of the block under the cursor (costs points).

    Observation Space: A (400, 640, 3) RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Match 3 or more blocks by swapping or changing their colors to score points. Create chain reactions for a high score before time runs out!"
    user_guide = "Use arrow keys (↑↓←→) to move the cursor. Hold an arrow key and press space to swap blocks. Hold shift to cycle the color of the selected block."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 180
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    WIN_SCORE = 500

    GRID_SIZE = 5
    NUM_BLOCK_TYPES = 4
    GRID_AREA_WIDTH = 300
    GRID_AREA_HEIGHT = 300
    CELL_SIZE = GRID_AREA_WIDTH // GRID_SIZE
    GRID_LINE_WIDTH = 2
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2 + 20

    # --- Colors ---
    COLOR_BG = (15, 23, 42)  # Dark Slate Blue
    COLOR_GRID_LINES = (51, 65, 85) # Slate Gray
    BLOCK_COLORS = [
        (239, 68, 68),   # Red
        (59, 130, 246),  # Blue
        (34, 197, 94),   # Green
        (234, 179, 8),   # Yellow
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (226, 232, 240)
    COLOR_TEXT_SHADOW = (30, 41, 59)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- Game State ---
        # These are initialized properly in reset()
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.time_left = None
        self.game_over = None
        self.particles = None
        self.animations = None
        
        # Initialize state variables
        # self.reset() # reset is called by the wrapper, no need to call it here
        
        # --- Validation ---
        # self.validate_implementation() # Optional: uncomment to run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_STEPS
        self.game_over = False
        self.cursor_pos = [0, 0] # [row, col]
        self.particles = []
        self.animations = []
        self._populate_initial_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        
        reward = self._handle_action(action)
        reward += self._process_matches()

        terminated = (self.score >= self.WIN_SCORE) or (self.time_left <= 0) or (self.steps >= self.MAX_STEPS)
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Goal-oriented reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_action(self, action):
        """Interprets the action and updates the game state accordingly."""
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # Action priority: Shift (color change) > Space (swap) > Movement (cursor)
        if shift_held:
            # Cycle block color under cursor
            r, c = self.cursor_pos
            current_color_idx = self.grid[r][c]
            self.grid[r][c] = (current_color_idx % self.NUM_BLOCK_TYPES) + 1
            self.score = max(0, self.score - 1)
            reward -= 1.0
            # SFX: color_change_sound
            self._create_particles(r, c, self.BLOCK_COLORS[self.grid[r][c]-1], 5)

        elif space_held and movement != 0:
            # Swap block with adjacent block
            r, c = self.cursor_pos
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            nr, nc = r + dr, c + dc

            if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                self.grid[r][c], self.grid[nr][nc] = self.grid[nr][nc], self.grid[r][c]
                # SFX: swap_sound
        else:
            # Move cursor
            if movement != 0:
                dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
                self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dr, 0, self.GRID_SIZE - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dc, 0, self.GRID_SIZE - 1)
        return reward

    def _process_matches(self):
        """Handles the entire chain reaction process: find, clear, fall, refill."""
        total_reward = 0
        chain_level = 1

        while True:
            matches = self._find_matches()
            if not matches:
                break

            # SFX: match_sound_chain_{chain_level}
            total_reward += 1.0 * chain_level # Chain reaction reward
            num_cleared = len(matches)
            total_reward += 0.1 * num_cleared # Per-block reward
            self.score += num_cleared * chain_level

            for r, c in matches:
                self._create_particles(r, c, self.BLOCK_COLORS[self.grid[r][c]-1])
                self.grid[r][c] = 0 # Mark as empty

            self._apply_gravity_and_refill()
            chain_level += 1
        
        return total_reward

    def _find_matches(self):
        """Finds all horizontal and vertical matches of 3 or more."""
        matches = set()
        # Horizontal
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r][c] != 0 and self.grid[r][c] == self.grid[r][c+1] == self.grid[r][c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r][c] != 0 and self.grid[r][c] == self.grid[r+1][c] == self.grid[r+2][c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _apply_gravity_and_refill(self):
        """Makes blocks fall into empty spaces and refills the top rows."""
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r][c] != 0:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = 0
                    empty_row -= 1
            # Refill from top
            for r in range(empty_row, -1, -1):
                self.grid[r][c] = self.np_random.integers(1, self.NUM_BLOCK_TYPES + 1)

    def _populate_initial_grid(self):
        """Fills the grid with random blocks, ensuring no initial matches."""
        self.grid = self.np_random.integers(1, self.NUM_BLOCK_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        while len(self._find_matches()) > 0:
            # This is a bit brute-force but fine for a 5x5 grid
            self.grid = self.np_random.integers(1, self.NUM_BLOCK_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            
    def _create_particles(self, r, c, color, count=15):
        """Creates an explosion of particles at a grid cell."""
        px, py = self._grid_to_pixel(r, c)
        center_x, center_y = px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifespan, 'color': color})

    def _update_and_draw_particles(self):
        """Updates particle physics and draws them."""
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(0, int((p['life'] / 40) * 5))
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

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
            "time_left": self.time_left // self.FPS,
            "cursor_pos": list(self.cursor_pos)
        }

    def _grid_to_pixel(self, r, c):
        """Converts grid coordinates to pixel coordinates."""
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
        return x, y

    def _render_game(self):
        """Renders the grid, blocks, cursor, and particles."""
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y),
                             (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT),
                             self.GRID_LINE_WIDTH)
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                             (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE),
                             (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE),
                             self.GRID_LINE_WIDTH)
        
        # Draw blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                block_type = self.grid[r][c]
                if block_type > 0:
                    color = self.BLOCK_COLORS[block_type - 1]
                    px, py = self._grid_to_pixel(r, c)
                    block_rect = pygame.Rect(px + self.GRID_LINE_WIDTH, py + self.GRID_LINE_WIDTH,
                                             self.CELL_SIZE - self.GRID_LINE_WIDTH, self.CELL_SIZE - self.GRID_LINE_WIDTH)
                    pygame.draw.rect(self.screen, color, block_rect, border_radius=8)

        # Draw cursor
        r, c = self.cursor_pos
        px, py = self._grid_to_pixel(r, c)
        cursor_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=10)
        
        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        """Renders the score and timer."""
        # Render Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(midleft=(25, 35))
        self.screen.blit(score_text, score_rect)
        
        score_label = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        score_label_rect = score_label.get_rect(midleft=(25, 70))
        self.screen.blit(score_label, score_label_rect)

        # Render Timer
        time_remaining_sec = max(0, self.time_left // self.FPS)
        minutes = time_remaining_sec // 60
        seconds = time_remaining_sec % 60
        time_str = f"{minutes:01}:{seconds:02}"
        time_text = self.font_large.render(time_str, True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(midright=(self.SCREEN_WIDTH - 25, 35))
        self.screen.blit(time_text, time_rect)
        
        time_label = self.font_small.render("TIME", True, self.COLOR_TEXT)
        time_label_rect = time_label.get_rect(midright=(self.SCREEN_WIDTH - 25, 70))
        self.screen.blit(time_label, time_label_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_message = "VICTORY!" if self.score >= self.WIN_SCORE else "TIME UP"
            end_text = self.font_large.render(end_message, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def render(self):
        """The render method for human playback."""
        return self._get_observation()
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Obs dtype is {test_obs.dtype}"
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # We need a real display for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shifting Sands")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping for human play ---
        movement = 0 # none
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render the game ---
        frame = env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()