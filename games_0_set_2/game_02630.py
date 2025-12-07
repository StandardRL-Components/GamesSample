
# Generated: 2025-08-27T20:56:58.034861
# Source Brief: brief_02630.md
# Brief Index: 2630

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a block and clear matching groups."
    )

    game_description = (
        "Clear the grid by selecting groups of 2 or more same-colored blocks. Race against the clock to achieve the highest score. Bigger matches and clearing entire colors give bonus points!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_SIZE = 5
        self.NUM_COLORS = 3
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_TIME = 60  # seconds
        self.MAX_STEPS = 1000
        self.MIN_MATCH_SIZE = 2

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 32)
        except FileNotFoundError:
            # Fallback if default font is not found (e.g., in minimal containers)
            self.font_large = pygame.font.SysFont("sans", 48)
            self.font_small = pygame.font.SysFont("sans", 32)


        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID_BG = (40, 50, 60)
        self.COLOR_GRID_LINE = (60, 75, 90)
        self.COLOR_EMPTY = (50, 60, 70)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
        ]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        
        # Grid layout
        self.CELL_SIZE = 60
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # Game state variables (will be initialized in reset)
        self.grid = None
        self.visual_offsets = None
        self.cursor_pos = None
        self.time_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_space_held = None
        self.last_movement_action = None
        self.particles = None
        self.color_counts = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.visual_offsets = np.zeros_like(self.grid, dtype=float)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.time_remaining = self.MAX_TIME * self.FPS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.last_movement_action = 0
        self.particles = []
        self._update_color_counts()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        self.time_remaining -= 1

        # Handle input and game logic
        reward += self._handle_input(action)
        self._update_animations()

        # Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.time_remaining <= 0:
                reward -= 100 # Penalty for time out
            elif np.all(self.grid == 0):
                reward += 100 # Bonus for clearing grid

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Handle cursor movement (on press, not hold)
        if movement != 0 and movement != self.last_movement_action:
            if movement == 1: # Up
                self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_SIZE
            elif movement == 2: # Down
                self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
            elif movement == 3: # Left
                self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_SIZE
            elif movement == 4: # Right
                self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        self.last_movement_action = movement

        # Handle block selection (on press, not hold)
        if space_held and not self.last_space_held:
            # sfx: click_block
            reward += self._process_selection()
        self.last_space_held = space_held
        
        return reward

    def _process_selection(self):
        r, c = self.cursor_pos
        if self.grid[r, c] == 0:
            # sfx: invalid_selection
            return 0

        color_to_match = self.grid[r, c]
        matches = self._find_matches(r, c, color_to_match)

        if len(matches) < self.MIN_MATCH_SIZE:
            # sfx: no_match
            return 0
        
        # sfx: match_success
        reward = len(matches)
        self.score += len(matches)
        
        # Find center for particle effect
        avg_x = sum(pos[1] for pos in matches) / len(matches)
        avg_y = sum(pos[0] for pos in matches) / len(matches)
        
        # Clear matched blocks
        for match_r, match_c in matches:
            self.grid[match_r, match_c] = 0
        
        # Check for color clear bonus
        self._update_color_counts()
        if self.color_counts[color_to_match - 1] == 0:
            reward += 10
            self.score += 10
            # sfx: color_clear_bonus

        self._apply_gravity_and_refill()
        self._create_particles(avg_x, avg_y, color_to_match, len(matches))
        return reward

    def _find_matches(self, start_r, start_c, color_id):
        q = deque([(start_r, start_c)])
        visited = set([(start_r, start_c)])
        matches = []

        while q:
            r, c = q.popleft()
            matches.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and \
                   (nr, nc) not in visited and self.grid[nr, nc] == color_id:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return matches
    
    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            write_idx = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != write_idx:
                        # Animate drop
                        fall_dist = write_idx - r
                        self.visual_offsets[r,c] += fall_dist * self.CELL_SIZE
                        
                        # Swap data
                        self.grid[write_idx, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                        
                        # Swap visual offsets
                        self.visual_offsets[write_idx, c] = self.visual_offsets[r,c]
                        self.visual_offsets[r,c] = 0

                    write_idx -= 1
            
            # Refill column
            for r in range(write_idx, -1, -1):
                self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
                self.visual_offsets[r, c] = (write_idx - r + 1) * self.CELL_SIZE + self.GRID_HEIGHT


    def _update_animations(self):
        # Update block fall animations
        self.visual_offsets = np.maximum(0, self.visual_offsets - (self.CELL_SIZE / 4))

        # Update particles
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2  # Gravity
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles
        
    def _create_particles(self, grid_x, grid_y, color_id, num_blocks):
        px = self.GRID_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.BLOCK_COLORS[color_id - 1]
        
        num_particles = min(80, 10 + num_blocks * 5)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})

    def _update_color_counts(self):
        counts = [0, 0, 0]
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r,c] > 0:
                    counts[self.grid[r,c]-1] += 1
        self.color_counts = counts

    def _check_termination(self):
        if self.time_remaining <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if np.all(self.grid == 0):
            return True
        return False

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
            "time_remaining": int(self.time_remaining / self.FPS),
        }

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT))

        # Draw blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_id = self.grid[r, c]
                x = self.GRID_X + c * self.CELL_SIZE
                y = self.GRID_Y + r * self.CELL_SIZE - self.visual_offsets[r,c]
                
                block_rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                
                if y < self.GRID_Y: # Don't draw blocks above the grid area
                    continue

                if color_id == 0:
                    color = self.COLOR_EMPTY
                else:
                    color = self.BLOCK_COLORS[color_id - 1]
                
                pygame.draw.rect(self.screen, color, block_rect.inflate(-6, -6), border_radius=8)
                
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, 
                             (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE), 
                             (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + i * self.CELL_SIZE), 2)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, 
                             (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y), 
                             (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT), 2)
        
        # Draw particles
        for p in self.particles:
            size = max(1, p['life'] / 5)
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0], p['pos'][1], size, size))

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_x = self.GRID_X + cursor_c * self.CELL_SIZE
        cursor_y = self.GRID_Y + cursor_r * self.CELL_SIZE
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        thickness = int(2 + pulse * 3)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE), thickness, border_radius=8)

    def _render_ui(self):
        # Render Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Render Timer
        time_left = max(0, self.time_remaining // self.FPS)
        time_color = self.COLOR_TEXT if time_left > 10 else (255, 100, 100)
        timer_text = self.font_large.render(f"Time: {time_left}", True, time_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(timer_text, timer_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            win_condition = np.all(self.grid == 0)
            
            status_text_str = "GRID CLEARED!" if win_condition else "TIME'S UP!"
            status_text = self.font_large.render(status_text_str, True, self.COLOR_TEXT)
            status_rect = status_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20))

            self.screen.blit(overlay, (0, 0))
            self.screen.blit(status_text, status_rect)
            self.screen.blit(final_score_text, final_score_rect)

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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set auto_advance to False for human play
    env.auto_advance = False

    # Pygame setup for human play
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Matcher")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print(env.user_guide)

    while running:
        movement_action = 0
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                if event.key == pygame.K_UP:
                    movement_action = 1
                elif event.key == pygame.K_DOWN:
                    movement_action = 2
                elif event.key == pygame.K_LEFT:
                    movement_action = 3
                elif event.key == pygame.K_RIGHT:
                    movement_action = 4
                if event.key == pygame.K_SPACE:
                    space_action = 1
        
        # For human play, we only step when an action is taken
        if movement_action != 0 or space_action != 0:
            action = [movement_action, space_action, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            
            if terminated:
                print("--- Episode Finished ---")
                print(f"Final Score: {info['score']}")
                # Optional: auto-reset after a delay
                # pygame.time.wait(2000)
                # obs, info = env.reset()
                # total_reward = 0

        # Update and render the display
        # In human mode, we get the latest observation from the step call
        # or just call _get_observation if no action was taken
        if not (movement_action != 0 or space_action != 0):
             obs = env._get_observation()
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()