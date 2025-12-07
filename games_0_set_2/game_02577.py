
# Generated: 2025-08-28T05:21:23.806006
# Source Brief: brief_02577.md
# Brief Index: 2577

        
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

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a block and clear matching groups."
    )

    # User-facing game description
    game_description = (
        "Clear the grid by matching groups of 3 or more same-colored blocks. Plan your moves to create chain reactions and maximize your score before you run out of moves."
    )

    # The game state is static until a user submits an action.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 12
    GRID_HEIGHT = 8
    BLOCK_SIZE = 40
    GRID_LINE_WIDTH = 2
    MAX_MOVES = 20
    MAX_STEPS = 1000
    MIN_MATCH_SIZE = 3

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 45, 60)
    COLOR_GRID_LINES = (40, 60, 80)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 0)
    
    # Block colors (index 0 is empty)
    COLORS = [
        (0, 0, 0),  # 0: Empty
        (255, 80, 80),   # 1: Red
        (80, 255, 80),   # 2: Green
        (80, 150, 255),  # 3: Blue
        (255, 255, 80),  # 4: Yellow
        (200, 80, 255),  # 5: Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Calculate grid position to center it
        self.grid_render_width = self.GRID_WIDTH * self.BLOCK_SIZE
        self.grid_render_height = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_render_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_render_height) // 2

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.space_pressed_last_frame = False
        self.particles = []
        
        self.reset()

        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.grid = self.np_random.integers(1, len(self.COLORS), size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.space_pressed_last_frame = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        # --- Action processing ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[0] -= 1  # Up
        elif movement == 2: self.cursor_pos[0] += 1  # Down
        elif movement == 3: self.cursor_pos[1] -= 1  # Left
        elif movement == 4: self.cursor_pos[1] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_HEIGHT - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_WIDTH - 1)

        # 2. Handle block selection (on key press, not hold)
        space_pressed = space_held and not self.space_pressed_last_frame
        if space_pressed:
            reward, blocks_cleared = self._attempt_match()
            self.moves_left -= 1
            if blocks_cleared > 0:
                # Sound effect placeholder
                # play_sound('match_success')
                self._apply_gravity_and_refill()
            else:
                # Sound effect placeholder
                # play_sound('match_fail')
                pass


        self.space_pressed_last_frame = space_held
        
        # 3. Update game effects
        self._update_particles()

        # 4. Check for termination conditions
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _attempt_match(self):
        row, col = self.cursor_pos
        target_color_idx = self.grid[row, col]

        if target_color_idx == 0:  # Empty block
            return -0.1, 0

        connected_blocks = self._find_connected_blocks(row, col)
        
        if len(connected_blocks) < self.MIN_MATCH_SIZE:
            # Invalid move
            return -0.1, 0
        
        # Valid move
        num_cleared = len(connected_blocks)
        for r, c in connected_blocks:
            self._spawn_particles(r, c, self.grid[r, c])
            self.grid[r, c] = 0  # Set to empty

        self.score += num_cleared
        reward = num_cleared
        
        # Bonus for larger clears
        if num_cleared > 5:
            reward += 5
            self.score += 5

        return reward, num_cleared

    def _find_connected_blocks(self, start_row, start_col):
        target_color = self.grid[start_row, start_col]
        if target_color == 0:
            return []

        q = deque([(start_row, start_col)])
        visited = set([(start_row, start_col)])
        
        while q:
            r, c = q.popleft()
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                    if (nr, nc) not in visited and self.grid[nr, nc] == target_color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        
        return list(visited)

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_WIDTH):
            write_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[write_row, c], self.grid[r, c] = self.grid[r, c], self.grid[write_row, c]
                    write_row -= 1
            
            # Refill empty top cells
            for r in range(write_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(1, len(self.COLORS))

    def _check_termination(self):
        # Win condition: all blocks cleared
        if np.all(self.grid == 0):
            return True, 100
        
        # Lose condition: out of moves
        if self.moves_left <= 0:
            return True, -50
            
        # Lose condition: max steps reached
        if self.steps >= self.MAX_STEPS:
            return True, -50

        return False, 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.grid_render_width, self.grid_render_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw blocks and grid lines
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r, c]
                block_rect = pygame.Rect(
                    self.grid_offset_x + c * self.BLOCK_SIZE,
                    self.grid_offset_y + r * self.BLOCK_SIZE,
                    self.BLOCK_SIZE,
                    self.BLOCK_SIZE
                )
                
                if color_idx > 0:
                    main_color = self.COLORS[color_idx]
                    shadow_color = tuple(max(0, val - 40) for val in main_color)
                    highlight_color = tuple(min(255, val + 40) for val in main_color)
                    
                    pygame.draw.rect(self.screen, shadow_color, block_rect)
                    inner_rect = block_rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, main_color, inner_rect)
                    
                    # Top-left highlight for 3D effect
                    pygame.draw.line(self.screen, highlight_color, inner_rect.topleft, inner_rect.topright, 2)
                    pygame.draw.line(self.screen, highlight_color, inner_rect.topleft, inner_rect.bottomleft, 2)

        # Draw grid lines over blocks
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_render_height), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_render_width, y), self.GRID_LINE_WIDTH)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_c * self.BLOCK_SIZE,
            self.grid_offset_y + cursor_r * self.BLOCK_SIZE,
            self.BLOCK_SIZE,
            self.BLOCK_SIZE
        )
        # Pulsing effect for cursor
        pulse = (math.sin(self.steps * 0.3) + 1) / 2  # Varies between 0 and 1
        line_width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
        self.screen.blit(score_text, score_rect)

        # Moves display
        moves_text = self.font_medium.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 25))
        self.screen.blit(moves_text, moves_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = np.all(self.grid == 0)
            msg = "YOU WIN!" if win_condition else "GAME OVER"
            
            end_text = self.font_large.render(msg, True, (255, 255, 100))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
        }

    def _spawn_particles(self, r, c, color_idx):
        px = self.grid_offset_x + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = self.grid_offset_y + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.COLORS[color_idx]
        
        for _ in range(10): # Spawn 10 particles per block
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'color': color,
                'radius': random.uniform(2, 5),
                'lifespan': random.randint(20, 40)
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] *= 0.95 # Shrink
            if p['lifespan'] > 0 and p['radius'] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for interactive play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Buster")
    clock = pygame.time.Clock()
    running = True

    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'r' key

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS for interactive play

    env.close()