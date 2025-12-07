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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to break the selected block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Break blocks to start chain reactions. Clear the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_ROWS, self.GRID_COLS = 8, 10
        self.CELL_SIZE = 38
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20
        self.MAX_MOVES = 30
        self.MAX_STABILITY = 3
        
        # Colors
        self.COLOR_BG = pygame.Color("#1a1c2c")
        self.COLOR_GRID = pygame.Color("#303452")
        self.COLOR_CURSOR = pygame.Color("#ffffff")
        self.COLOR_TEXT = pygame.Color("#f0f0f0")
        self.COLOR_STABILITY = [
            None, # 0 stability is empty
            pygame.Color("#f2cd54"), # 1
            pygame.Color("#f07e48"), # 2
            pygame.Color("#d45a68"), # 3
        ]
        self.COLOR_CHAIN_LINE = pygame.Color("#8fd3ff")

        # Fonts
        self.font_ui = pygame.font.Font(None, 32)
        self.font_msg = pygame.font.Font(None, 64)

        # Game state variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_space_held = False
        self.particles = []
        self.chain_lines = []
        
        # Initialize state
        # self.reset() is called by the gym wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.grid = self.np_random.integers(1, self.MAX_STABILITY + 1, size=(self.GRID_ROWS, self.GRID_COLS))
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_space_held = False
        self.particles = []
        self.chain_lines = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1 # Unused in this game
        
        reward = 0
        terminated = False
        
        if not self.game_over:
            # Handle cursor movement
            if movement == 1: # Up
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: # Down
                self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            elif movement == 3: # Left
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: # Right
                self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

            # Handle block breaking on space PRESS (not hold)
            space_pressed = space_held and not self.last_space_held
            if space_pressed:
                turn_reward = self._execute_break()
                if turn_reward > 0: # A move is only used if an action was successful
                    self.moves_left -= 1
                    reward += turn_reward
        
        self.last_space_held = space_held
        
        # Update animations regardless of game state
        self._update_animations()

        # Check for termination conditions
        all_blocks_cleared = np.sum(self.grid) == 0
        if not self.game_over:
            if all_blocks_cleared:
                terminated = True
                self.game_over = True
                self.win = True
                reward += 100  # Win bonus
                self.score += 100
                # sound: game_win.wav
            elif self.moves_left <= 0:
                terminated = True
                self.game_over = True
                self.win = False
                # sound: game_over.wav

        self.steps += 1
        
        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )
    
    def _execute_break(self):
        r, c = self.cursor_pos[1], self.cursor_pos[0]
        
        if self.grid[r, c] == 0:
            return 0  # Cannot break an empty space

        # sound: block_break_initial.wav
        grid_before = self.grid.copy()
        total_reward = 0
        
        q = deque([(r, c)])
        broken_coords = set([(r, c)])
        
        # Initial break
        self._add_particles(r, c, self.grid[r,c], 30)
        self.grid[r, c] = 0

        while q:
            curr_r, curr_c = q.popleft()
            
            # Propagate to orthogonal neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and self.grid[nr, nc] > 0:
                    self.grid[nr, nc] -= 1
                    self._add_chain_line(curr_r, curr_c, nr, nc)
                    
                    if self.grid[nr, nc] == 0 and (nr, nc) not in broken_coords:
                        # sound: block_break_chain.wav
                        self._add_particles(nr, nc, grid_before[nr,nc], 15)
                        q.append((nr, nc))
                        broken_coords.add((nr, nc))

        # Calculate rewards
        # 1. Per-block reward
        num_broken = len(broken_coords)
        total_reward += num_broken

        # 2. Row/Column clear rewards
        rows_before = np.sum(grid_before, axis=1) > 0
        rows_after = np.sum(self.grid, axis=1) > 0
        cleared_rows = np.sum(rows_before & ~rows_after)
        total_reward += cleared_rows * 20

        cols_before = np.sum(grid_before, axis=0) > 0
        cols_after = np.sum(self.grid, axis=0) > 0
        cleared_cols = np.sum(cols_before & ~cols_after)
        total_reward += cleared_cols * 10
        
        self.score += total_reward
        return total_reward

    def _add_particles(self, r, c, stability, count):
        px, py = self._get_cell_center(r, c)
        color = self.COLOR_STABILITY[stability]
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})

    def _add_chain_line(self, r1, c1, r2, c2):
        start_pos = self._get_cell_center(r1, c1)
        end_pos = self._get_cell_center(r2, c2)
        self.chain_lines.append({'start': start_pos, 'end': end_pos, 'life': 10})

    def _update_animations(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Update chain lines
        for line in self.chain_lines[:]:
            line['life'] -= 1
            if line['life'] <= 0:
                self.chain_lines.remove(line)

    def _get_cell_rect(self, r, c):
        return pygame.Rect(
            self.GRID_OFFSET_X + c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + r * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )

    def _get_cell_center(self, r, c):
        rect = self._get_cell_rect(r, c)
        return rect.centerx, rect.centery

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        # Draw blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                stability = self.grid[r, c]
                if stability > 0:
                    rect = self._get_cell_rect(r, c).inflate(-4, -4)
                    color = self.COLOR_STABILITY[stability]
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
                    # Add a subtle inner glow/highlight
                    highlight_color = color.lerp((255,255,255), 0.3)
                    pygame.draw.rect(self.screen, highlight_color, (rect.x, rect.y, rect.width, 5), border_top_left_radius=4, border_top_right_radius=4)


        # Draw chain lines
        for line in self.chain_lines:
            alpha = int(255 * (line['life'] / 10))
            color = self.COLOR_CHAIN_LINE
            rgba_color = (color.r, color.g, color.b, alpha)
            pygame.gfxdraw.line(self.screen, int(line['start'][0]), int(line['start'][1]), int(line['end'][0]), int(line['end'][1]), rgba_color)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            base_color = p['color']
            rgba_color = (base_color.r, base_color.g, base_color.b, alpha)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, rgba_color)

        # Draw cursor
        if not self.game_over:
            cursor_rect = self._get_cell_rect(self.cursor_pos[1], self.cursor_pos[0])
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
            thickness = 2 + int(pulse * 2)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, thickness, border_radius=5)

    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 15))

        # Render moves left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 15))

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = "#54f28a" if self.win else "#d45a68"
            msg_text = self.font_msg.render(msg, True, pygame.Color(color))
            msg_rect = msg_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # For manual play
    obs, info = env.reset()
    terminated = False
    
    # Pygame window for human interaction
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")

    action = [0, 0, 0] # no-op, released, released
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward}, Moves Left: {info['moves_left']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS for human play

    print("Game Over!")
    print(f"Final Score: {info['score']}")
    
    # Keep the window open for a few seconds to see the result
    if env.game_over:
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < 3000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            pygame.event.pump()

    env.close()