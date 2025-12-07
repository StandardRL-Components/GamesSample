
# Generated: 2025-08-28T06:54:41.615840
# Source Brief: brief_03073.md
# Brief Index: 3073

        
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
        "Controls: Use arrow keys (↑↓←→) to move the selector. Press Space to swap the selected fruits."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match cascading fruits in a frantic race against time to achieve a high score before the clock runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 8, 8
    CELL_SIZE = 40
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 10 # Offset for UI

    # Game parameters
    FPS = 60
    MAX_TIME = 60  # seconds
    WIN_SCORE = 500
    MAX_STEPS = 3600 # 60s * 60fps
    NUM_FRUIT_TYPES = 6
    ANIMATION_SPEED = 1 / (FPS * 0.15)  # Animation takes 0.15 seconds

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_SELECTOR = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)
    FRUIT_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
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
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 20)
        
        # Etc...        
        
        # Initialize state variables
        self.grid = None
        self.selector_pos = None
        self.score = None
        self.timer = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = None
        
        self.game_state = None # 'IDLE', 'SWAPPING', 'CLEARING', 'FALLING'
        self.animation_progress = None
        self.swap_info = None
        self.revert_swap = None
        self.fruits_to_clear = None
        self.fall_map = None
        self.particles = None
        self.last_reward = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.score = 0
        self.timer = self.MAX_TIME
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        
        self.game_state = 'IDLE'
        self.animation_progress = 0
        self.swap_info = None
        self.revert_swap = False
        self.fruits_to_clear = set()
        self.fall_map = {}
        self.particles = []
        
        self.selector_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2 - 1]
        
        # Initialize grid without any starting matches
        while True:
            self.grid = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS))
            if not self._find_matches():
                break
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        self.steps += 1
        reward = 0

        # --- Update Game State Machine ---
        if self.game_state == 'IDLE':
            self._handle_selector_move(movement)
            is_swap_triggered = space_held and not self.prev_space_held
            if is_swap_triggered:
                self.swap_info = (tuple(self.selector_pos), (self.selector_pos[0], self.selector_pos[1] + 1))
                self.animation_progress = 0
                self.game_state = 'SWAPPING'
                self.revert_swap = False
                # Sound: swap_start.wav

        elif self.game_state == 'SWAPPING':
            self.animation_progress += self.ANIMATION_SPEED
            if self.animation_progress >= 1:
                self._finalize_swap()
                if self.revert_swap:
                    self.game_state = 'IDLE'
                    self.revert_swap = False
                else:
                    matches = self._find_matches()
                    if matches:
                        reward_info = self._calculate_match_reward(matches)
                        reward += reward_info["reward"]
                        self.score += reward_info["cleared_count"]
                        self.fruits_to_clear = matches
                        self.animation_progress = 0
                        self.game_state = 'CLEARING'
                        # Sound: match_success.wav
                    else:
                        self.animation_progress = 0
                        self.revert_swap = True # Trigger swap-back
                        # Sound: invalid_swap.wav

        elif self.game_state == 'CLEARING':
            self.animation_progress += self.ANIMATION_SPEED
            if self.animation_progress >= 1:
                self._create_particles()
                self._clear_fruits()
                self._prepare_fall()
                self.animation_progress = 0
                self.game_state = 'FALLING'

        elif self.game_state == 'FALLING':
            self.animation_progress += self.ANIMATION_SPEED
            if self.animation_progress >= 1:
                self._finalize_fall()
                self._refill_grid()
                matches = self._find_matches()
                if matches:
                    reward_info = self._calculate_match_reward(matches)
                    reward += reward_info["reward"]
                    self.score += reward_info["cleared_count"]
                    self.fruits_to_clear = matches
                    self.animation_progress = 0
                    self.game_state = 'CLEARING'
                    # Sound: match_success_chain.wav
                else:
                    self.game_state = 'IDLE'

        self.prev_space_held = space_held
        
        # --- Update Timer and Particles ---
        if not self.game_over:
            self.timer -= 1 / self.FPS
        self._update_particles()
        
        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
            else:
                reward -= 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _check_termination(self):
        return self.timer <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer
        }

    # --- Helper methods for game logic ---

    def _handle_selector_move(self, movement):
        r, c = self.selector_pos
        if movement == 1: # Up
            self.selector_pos[0] = max(0, r - 1)
        elif movement == 2: # Down
            self.selector_pos[0] = min(self.GRID_ROWS - 1, r + 1)
        elif movement == 3: # Left
            self.selector_pos[1] = max(0, c - 1)
        elif movement == 4: # Right
            self.selector_pos[1] = min(self.GRID_COLS - 2, c + 1)
    
    def _finalize_swap(self):
        (r1, c1), (r2, c2) = self.swap_info
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                fruit = self.grid[r, c]
                if fruit == 0: continue
                
                if c < self.GRID_COLS - 2 and self.grid[r, c+1] == fruit and self.grid[r, c+2] == fruit:
                    for i in range(3): matches.add((r, c+i))
                
                if r < self.GRID_ROWS - 2 and self.grid[r+1, c] == fruit and self.grid[r+2, c] == fruit:
                    for i in range(3): matches.add((r+i, c))
        return matches

    def _calculate_match_reward(self, matches):
        reward = 0
        total_cleared_in_swap = 0
        
        processed_matches = set()
        for r_start, c_start in matches:
            if (r_start, c_start) in processed_matches:
                continue

            component = set()
            q = [(r_start, c_start)]
            processed_matches.add((r_start, c_start))
            component.add((r_start, c_start))
            
            head = 0
            while head < len(q):
                r, c = q[head]
                head += 1
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in matches and (nr, nc) not in processed_matches:
                        processed_matches.add((nr, nc))
                        component.add((nr, nc))
                        q.append((nr, nc))

            size = len(component)
            total_cleared_in_swap += size
            if size == 3: reward += 1
            elif size == 4: reward += 2
            else: reward += 3
        
        if total_cleared_in_swap > 5:
            reward += 5
            
        return {"reward": reward, "cleared_count": total_cleared_in_swap}

    def _create_particles(self):
        for r, c in self.fruits_to_clear:
            fruit_type = self.grid[r, c]
            if fruit_type == 0: continue
            color = self.FRUIT_COLORS[fruit_type - 1]
            px = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE / 2
            py = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2
            for _ in range(15):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifespan = self.np_random.integers(20, 40)
                self.particles.append({'pos': [px, py], 'vel': vel, 'life': lifespan, 'color': color})

    def _clear_fruits(self):
        for r, c in self.fruits_to_clear:
            self.grid[r, c] = 0

    def _prepare_fall(self):
        self.fall_map = {}
        for c in range(self.GRID_COLS):
            empty_count = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.fall_map[(r, c)] = empty_count

    def _finalize_fall(self):
        for (r, c), fall_dist in self.fall_map.items():
            self.grid[r + fall_dist, c] = self.grid[r, c]
            self.grid[r, c] = 0

    def _refill_grid(self):
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    # --- Rendering methods ---

    def _render_game(self):
        self._render_grid_bg()
        self._render_fruits()
        self._render_selector()
        self._render_particles()

    def _render_grid_bg(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), 1)

    def _render_fruits(self):
        prog = min(1.0, self.animation_progress)
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                fruit_type = self.grid[r, c]
                if fruit_type == 0: continue
                color = self.FRUIT_COLORS[fruit_type - 1]
                
                center_x = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                radius = int(self.CELL_SIZE * 0.4)

                if self.game_state == 'SWAPPING' and self.swap_info:
                    (r1, c1), (r2, c2) = self.swap_info
                    if (r, c) == (r1, c1): center_x += self.CELL_SIZE * prog
                    elif (r, c) == (r2, c2): center_x -= self.CELL_SIZE * prog
                
                if self.game_state == 'CLEARING' and (r,c) in self.fruits_to_clear:
                    radius = int(radius * (1 - prog))
                
                if self.game_state == 'FALLING' and (r,c) in self.fall_map:
                    fall_dist = self.fall_map[(r,c)]
                    center_y += fall_dist * self.CELL_SIZE * prog

                if radius > 0:
                    pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), radius, color)

    def _render_selector(self):
        if self.game_state != 'IDLE': return
        r, c = self.selector_pos
        rect1 = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        rect2 = pygame.Rect(self.GRID_X + (c + 1) * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect1, 3, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect2, 3, border_radius=5)
    
    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(5 * (p['life'] / 40.0)))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1]), size, size))

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 5))
        
        timer_ratio = max(0, self.timer / self.MAX_TIME)
        bar_width = self.SCREEN_WIDTH - 40
        bar_height = 20
        
        bar_color = (80, 200, 80)
        if timer_ratio < 0.2: bar_color = (200, 80, 80)
        elif timer_ratio < 0.5: bar_color = (200, 200, 80)
            
        pygame.draw.rect(self.screen, self.COLOR_GRID, (20, self.SCREEN_HEIGHT - 30, bar_width, bar_height))
        pygame.draw.rect(self.screen, bar_color, (20, self.SCREEN_HEIGHT - 30, bar_width * timer_ratio, bar_height))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            msg = "YOU WON!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text, text_rect)

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

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    # To run with manual controls, uncomment this block
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # pygame.display.set_caption("Fruit Matcher")
    # screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # done = False
    # while not done:
    #     movement = 0 # none
    #     space = 0 # released
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
        
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
        
    #     if keys[pygame.K_SPACE]: space = 1
        
    #     action = [movement, space, 0]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Blit the observation to the display screen
    #     frame = np.transpose(obs, (1, 0, 2))
    #     surf = pygame.surfarray.make_surface(frame)
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    #     if reward != 0:
    #         print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

    # print(f"Final Score: {info['score']}")
    # env.close()

    # --- Random Agent Test ---
    env = GameEnv()
    total_reward = 0
    obs, info = env.reset()
    start_time = time.time()
    for i in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 200 == 0:
            print(f"Step {i+1}/{2000} | Score: {info['score']} | Total Reward: {total_reward:.2f}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()
    
    end_time = time.time()
    print(f"Ran 2000 steps in {end_time - start_time:.2f} seconds.")
    env.close()