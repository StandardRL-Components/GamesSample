import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a block and clear any matching adjacent blocks. Press Shift to end the game."
    )

    game_description = (
        "A minimalist puzzle game. Match groups of adjacent colored blocks to clear them from the grid. Score points for each block cleared and earn bonuses for large groups. Clear the entire board for a huge bonus, but the game ends if no more matches are possible."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 10
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.SCREEN_HEIGHT) // 2
        self.GRID_OFFSET_Y = 0
        self.CELL_SIZE = self.SCREEN_HEIGHT // self.GRID_HEIGHT
        
        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_EMPTY = (50, 60, 70)
        self.COLORS = {
            0: self.COLOR_EMPTY,
            1: (220, 50, 50),   # Red
            2: (50, 220, 50),   # Green
            3: (50, 100, 220),  # Blue
        }
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_FAIL = (255, 0, 0, 150)

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game State Variables
        self.grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.hint_blocks = set()
        self.fail_animation_timer = 0
        self.max_steps = 1000

        # Initialize state
        # We need a seed for the first reset to initialize self.np_random
        super().reset(seed=random.randint(0, 1000000000))
        self._generate_board() # Ensure a valid board is created before validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.fail_animation_timer = 0
        self.cursor_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        
        self._generate_board()
        self._find_all_possible_matches()
        
        return self._get_observation(), self._get_info()

    def _generate_board(self):
        """Generates a new board, ensuring at least one match is possible."""
        attempts = 0
        while attempts < 100: # Safety break
            self.grid = self.np_random.integers(1, len(self.COLORS), size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if self._check_for_any_matches():
                return
            attempts += 1
        # Failsafe: if no valid board is found, create one with a guaranteed match
        self.grid = self.np_random.integers(1, len(self.COLORS), size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        self.grid[0, 0] = self.grid[0, 1] = 1

    def step(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        truncated = False

        if self.fail_animation_timer > 0:
            self.fail_animation_timer -= 1

        if shift_press:
            terminated = True
            # No reward modification for manual reset
        else:
            # 1. Handle cursor movement
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] %= self.GRID_WIDTH
            self.cursor_pos[1] %= self.GRID_HEIGHT

            # 2. Handle block matching on space press
            if space_press:
                reward += self._handle_match_action()

        self.steps += 1

        # 3. Check for termination conditions
        if not terminated:
            is_board_clear = np.all(self.grid == 0)
            no_matches_left = not self._check_for_any_matches()
            
            if self.steps >= self.max_steps:
                truncated = True
                terminated = True # Gymnasium standard is to set both to True
            elif is_board_clear:
                reward += 100  # Goal-oriented reward for clearing the board
                terminated = True
            elif no_matches_left:
                reward -= 50  # Penalty for getting stuck
                terminated = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_match_action(self):
        """Processes a match action at the cursor's position."""
        cx, cy = self.cursor_pos
        target_color = self.grid[cy, cx]

        if target_color == 0:  # Cannot match an empty cell
            self.fail_animation_timer = 5
            return -0.2

        matched_blocks = self._find_connected_blocks(cx, cy)

        if len(matched_blocks) > 1:
            # Successful match
            for r, c in matched_blocks:
                self._spawn_particles(c, r, self.COLORS[self.grid[r, c]])
                self.grid[r, c] = 0
            
            self._apply_gravity()
            self._find_all_possible_matches()

            reward = len(matched_blocks)  # +1 for each block
            if len(matched_blocks) >= 4:
                reward += 5  # Bonus for large matches
            self.score += int(reward)
            return reward
        else:
            # Failed match attempt
            self.fail_animation_timer = 5
            return -0.2

    def _find_connected_blocks(self, start_x, start_y):
        """Finds all connected blocks of the same color using BFS."""
        target_color = self.grid[start_y, start_x]
        if target_color == 0:
            return []

        q = deque([(start_y, start_x)])
        visited = set([(start_y, start_x)])
        connected = []

        while q:
            r, c = q.popleft()
            connected.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                    if (nr, nc) not in visited and self.grid[nr, nc] == target_color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return connected

    def _apply_gravity(self):
        """Shifts non-empty blocks down to fill empty spaces."""
        for c in range(self.GRID_WIDTH):
            write_r = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != write_r:
                        self.grid[write_r, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    write_r -= 1
    
    def _check_for_any_matches(self):
        """Checks if any matches are possible on the entire board."""
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    # Check right neighbor
                    if c + 1 < self.GRID_WIDTH and self.grid[r, c] == self.grid[r, c + 1]:
                        return True
                    # Check down neighbor
                    if r + 1 < self.GRID_HEIGHT and self.grid[r, c] == self.grid[r + 1, c]:
                        return True
        return False

    def _find_all_possible_matches(self):
        """Finds all groups of matchable blocks to provide hints."""
        self.hint_blocks.clear()
        visited = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0 and (r, c) not in visited:
                    group = self._find_connected_blocks(c, r)
                    if len(group) > 1:
                        for block_r, block_c in group:
                            self.hint_blocks.add((block_r, block_c))
                            visited.add((block_r, block_c))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))
        grid_surface.fill(self.COLOR_GRID)

        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 0.2 + 0.8 # Varies between 0.8 and 1.0

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_val = self.grid[r, c]
                base_color = self.COLORS[color_val]
                
                rect = pygame.Rect(c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                if color_val == 0:
                    pygame.draw.rect(grid_surface, self.COLOR_EMPTY, rect.inflate(-2, -2))
                else:
                    # Hinting: make potential matches pulse
                    if (r,c) in self.hint_blocks:
                        draw_color = tuple(min(255, int(val * pulse)) for val in base_color)
                    else:
                        draw_color = base_color
                    
                    # Draw block with a border effect
                    pygame.draw.rect(grid_surface, draw_color, rect.inflate(-4, -4), border_radius=4)
                    
        # Draw cursor
        cursor_rect = pygame.Rect(
            self.cursor_pos[0] * self.CELL_SIZE, 
            self.cursor_pos[1] * self.CELL_SIZE, 
            self.CELL_SIZE, 
            self.CELL_SIZE
        )
        
        cursor_alpha = (math.sin(self.steps * 0.3) + 1) / 2 * 100 + 155 # 155-255
        cursor_color = (*self.COLOR_CURSOR, int(cursor_alpha))
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, cursor_color, s.get_rect(), 4, border_radius=6)
        grid_surface.blit(s, cursor_rect.topleft)

        # Draw failure animation
        if self.fail_animation_timer > 0:
            fail_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            alpha = (self.fail_animation_timer / 5) * self.COLOR_FAIL[3]
            fail_surface.fill((*self.COLOR_FAIL[:3], alpha))
            grid_surface.blit(fail_surface, cursor_rect.topleft)

        self.screen.blit(grid_surface, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y))

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))

        steps_text = self.font_small.render(f"Step: {self.steps}/{self.max_steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (15, 50))

    def _spawn_particles(self, x, y, color):
        px = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})
            
    def _update_and_draw_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            
            size = max(0, int(p['life'] / 5))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], size)

        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "possible_matches": len(self.hint_blocks) > 0,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Matcher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        # Event handling for manual play
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                elif event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            print("--- New Game Started ---")
            
        clock.tick(10) # Control the speed of the manual play

    env.close()