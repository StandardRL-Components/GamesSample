
# Generated: 2025-08-27T15:05:53.285090
# Source Brief: brief_00886.md
# Brief Index: 886

        
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
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to cycle the selected tile's color."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade puzzle game. Race against time to clear colorful tiles by matching 3 or more of the same color."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 12, 8
    NUM_COLORS = 5
    GAME_TIME_SECONDS = 30
    FPS = 30

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_LINES = (50, 55, 65)
    COLOR_EMPTY = (35, 38, 48)
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Grid and tile dimensions
        self.UI_HEIGHT = 50
        self.GRID_PX_HEIGHT = self.HEIGHT - self.UI_HEIGHT
        self.TILE_SIZE = min(self.WIDTH // self.GRID_COLS, self.GRID_PX_HEIGHT // self.GRID_ROWS)
        self.GRID_PX_WIDTH = self.GRID_COLS * self.TILE_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_PX_WIDTH) // 2
        self.GRID_OFFSET_Y = self.UI_HEIGHT

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.prev_space_held = False
        self.particles = []

        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.GAME_TIME_SECONDS
        self.max_steps = self.GAME_TIME_SECONDS * self.FPS
        
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_COLS, self.GRID_ROWS))
        
        # Ensure no initial matches
        while self._find_all_matches():
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_COLS, self.GRID_ROWS))

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.prev_space_held = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # --- Update game logic ---
        self.steps += 1
        self.time_left -= 1.0 / self.FPS

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS
        
        # 2. Handle color cycle action
        is_space_press = space_held and not self.prev_space_held
        if is_space_press:
            cx, cy = self.cursor_pos
            current_color = self.grid[cx, cy]
            # Cycle color: 1 -> 2 -> ... -> NUM_COLORS -> 1
            self.grid[cx, cy] = (current_color % self.NUM_COLORS) + 1
            # SFX: Color change sound
            
            # 3. Check for matches and process cascades
            total_cleared_this_action = 0
            cascaded_reward = 0
            
            while True:
                matches = self._find_all_matches()
                if not matches:
                    break
                
                num_cleared = len(matches)
                total_cleared_this_action += num_cleared
                
                # Calculate reward for this cascade wave
                cascaded_reward += num_cleared * 0.1
                if num_cleared == 4: cascaded_reward += 1
                elif num_cleared >= 5: cascaded_reward += 2
                
                # Clear tiles and create particles
                for x, y in matches:
                    self._create_particles(x, y, self.grid[x, y])
                    self.grid[x, y] = 0 # 0 represents an empty space
                # SFX: Tile clear sound
                
                self._apply_gravity_and_refill()
                # SFX: Tiles falling sound

            if total_cleared_this_action == 0:
                reward = -0.02 # Penalty for a useless action
            else:
                reward = cascaded_reward
                self.score += total_cleared_this_action

        self.prev_space_held = space_held

        # 4. Update particles
        self._update_particles()
        
        # 5. Check for termination
        terminated = False
        all_cleared = np.all(self.grid == 0)
        
        if self.time_left <= 0 or self.steps >= self.max_steps:
            terminated = True
            reward += -10 # Penalty for running out of time
            # SFX: Game over sound
        elif all_cleared:
            terminated = True
            reward += 100 # Big reward for winning
            # SFX: Victory fanfare
            
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_all_matches(self):
        matches = set()
        # Horizontal matches
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS - 2):
                color = self.grid[x, y]
                if color != 0 and color == self.grid[x+1, y] and color == self.grid[x+2, y]:
                    matches.add((x, y))
                    matches.add((x+1, y))
                    matches.add((x+2, y))
        # Vertical matches
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS - 2):
                color = self.grid[x, y]
                if color != 0 and color == self.grid[x, y+1] and color == self.grid[x, y+2]:
                    matches.add((x, y))
                    matches.add((x, y+1))
                    matches.add((x, y+2))
        return matches

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_COLS):
            empty_slots = 0
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[x, y + empty_slots] = self.grid[x, y]
                    self.grid[x, y] = 0
            # Refill top
            for i in range(empty_slots):
                self.grid[x, i] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _create_particles(self, grid_x, grid_y, color_index):
        px = self.GRID_OFFSET_X + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
        color = self.TILE_COLORS[color_index - 1]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.randint(15, 30) # frames
            self.particles.append([px, py, vx, vy, life, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2] # x
            p[1] += p[3] # y
            p[3] += 0.1 # gravity
            p[4] -= 1 # life

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_PX_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.HEIGHT))
            
        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = self.grid[c, r]
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.TILE_SIZE,
                    self.GRID_OFFSET_Y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                
                color = self.COLOR_EMPTY if color_index == 0 else self.TILE_COLORS[color_index - 1]
                
                # Draw main tile with a slight 3D effect
                pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))
                darker_color = tuple(max(0, val - 40) for val in color)
                pygame.draw.line(self.screen, darker_color, rect.bottomleft, rect.bottomright, 1)
                pygame.draw.line(self.screen, darker_color, rect.topright, rect.bottomright, 1)

        # Draw particles
        for p in self.particles:
            pos = (int(p[0]), int(p[1]))
            life_ratio = p[4] / 30.0
            radius = int(3 * life_ratio)
            if radius > 0:
                color = tuple(int(c * life_ratio) for c in p[5])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cx * self.TILE_SIZE,
            self.GRID_OFFSET_Y + cy * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        line_width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer bar
        timer_bar_width = 200
        timer_bar_height = 20
        timer_x = self.WIDTH - timer_bar_width - 20
        timer_y = 15
        
        time_ratio = max(0, self.time_left / self.GAME_TIME_SECONDS)
        
        # Color changes from green to yellow to red
        if time_ratio > 0.5:
            bar_color = (80, 200, 80)
        elif time_ratio > 0.2:
            bar_color = (255, 200, 80)
        else:
            bar_color = (200, 80, 80)
            
        # Background
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, (timer_x, timer_y, timer_bar_width, timer_bar_height))
        # Foreground
        pygame.draw.rect(self.screen, bar_color, (timer_x, timer_y, int(timer_bar_width * time_ratio), timer_bar_height))
        # Border
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (timer_x, timer_y, timer_bar_width, timer_bar_height), 1)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    # Set this to 'human' to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    env = GameEnv()
    
    if render_mode == "human":
        pygame.display.set_caption("Color Grid Puzzle")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        
    obs, info = env.reset()
    done = False
    
    # --- Manual Play ---
    # For manual play, we map keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    total_reward = 0
    start_time = time.time()
    
    while not done:
        action = [0, 0, 0] # Default no-op action
        
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            for key, move_action in key_to_action.items():
                if keys[key]:
                    action[0] = move_action
                    break # Only one movement per frame
                    
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
        else:
            # Random agent
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        if render_mode == "human":
            # Blit the env's surface to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS)
            
    end_time = time.time()
    
    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")
    print(f"Duration: {end_time - start_time:.2f}s")
    
    env.close()