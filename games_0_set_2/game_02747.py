
# Generated: 2025-08-28T05:49:35.020325
# Source Brief: brief_02747.md
# Brief Index: 2747

        
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
        "Controls: Arrow keys to move cursor. Space to swap with the tile in the direction of your last movement."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist match-3 puzzle game. Swap adjacent tiles to create lines of 3 or more. Clear the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 8
    TILE_SIZE = 40
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_WIDTH * TILE_SIZE) // 2
    GRID_MARGIN_Y = (SCREEN_HEIGHT - GRID_HEIGHT * TILE_SIZE) // 2 + 20
    
    # Colors
    COLOR_BG = pygame.Color("#1A1A2E")
    COLOR_GRID_BG = pygame.Color("#16213E")
    COLOR_TEXT = pygame.Color("#E94560")
    COLOR_SCORE_TEXT = pygame.Color("#F0F0F0")
    
    TILE_COLORS = [
        pygame.Color("#FF5733"),  # Orange
        pygame.Color("#33FF57"),  # Green
        pygame.Color("#3357FF"),  # Blue
        pygame.Color("#FF33A1"),  # Pink
        pygame.Color("#FFFF33"),  # Yellow
        pygame.Color("#8E44AD"),  # Purple
    ]

    NUM_TILE_TYPES = len(TILE_COLORS)
    MAX_MOVES = 15
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables initialized in reset
        self.grid = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.last_swap_invalid = False
        self.invalid_swap_timer = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.cursor_pos = (self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2)
        self.last_move_dir = None
        self.particles = []
        self.invalid_swap_timer = 0
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement = action[0]
        space_pressed = action[1] == 1
        
        self.last_swap_invalid = False
        if self.invalid_swap_timer > 0:
            self.invalid_swap_timer -= 1

        if movement != 0:
            self._move_cursor(movement)
        
        if space_pressed and self.last_move_dir is not None:
            reward = self._attempt_swap()
        
        self.score += reward

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            while self._find_all_matches(self.grid):
                matches = self._find_all_matches(self.grid)
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
            
            if self._has_possible_moves():
                break

    def _has_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                        temp_grid = self.grid.copy()
                        temp_grid[r, c], temp_grid[nr, nc] = temp_grid[nr, nc], temp_grid[r, c]
                        if self._find_all_matches(temp_grid):
                            return True
        return False

    def _move_cursor(self, direction):
        r, c = self.cursor_pos
        if direction == 1:  # Up
            self.cursor_pos = ((r - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT, c)
            self.last_move_dir = (-1, 0)
        elif direction == 2:  # Down
            self.cursor_pos = ((r + 1) % self.GRID_HEIGHT, c)
            self.last_move_dir = (1, 0)
        elif direction == 3:  # Left
            self.cursor_pos = (r, (c - 1 + self.GRID_WIDTH) % self.GRID_WIDTH)
            self.last_move_dir = (0, -1)
        elif direction == 4:  # Right
            self.cursor_pos = (r, (c + 1) % self.GRID_WIDTH)
            self.last_move_dir = (0, 1)

    def _attempt_swap(self):
        r1, c1 = self.cursor_pos
        dr, dc = self.last_move_dir
        r2, c2 = r1 + dr, c1 + dc

        if not (0 <= r2 < self.GRID_HEIGHT and 0 <= c2 < self.GRID_WIDTH):
            return 0

        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        matches = self._find_all_matches(self.grid)
        
        if not matches:
            # Invalid move, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self.last_swap_invalid = True
            self.invalid_swap_timer = 5 # Frames to show invalid effect
            return -0.1

        # Valid move
        self.moves_left -= 1
        total_reward = 0
        total_cleared_tiles = 0

        # Cascade loop
        while True:
            current_matches = self._find_all_matches(self.grid)
            if not current_matches:
                break
            
            num_cleared = len(current_matches)
            total_cleared_tiles += num_cleared
            
            match_lengths = self._get_match_lengths(current_matches)
            for length in match_lengths:
                if length == 3: total_reward += 1
                elif length == 4: total_reward += 2
                else: total_reward += 3

            self._clear_and_drop(current_matches)
            self._refill_grid()
        
        if total_cleared_tiles >= (self.GRID_WIDTH * self.GRID_HEIGHT) * 0.25:
            total_reward += 5

        if np.all(self.grid == 0):
            self.game_over = True
            self.win = True
            total_reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            self.win = False
            total_reward -= 10
        
        return total_reward

    def _find_all_matches(self, grid):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if grid[r, c] == 0: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches
    
    def _get_match_lengths(self, matches):
        # A simplified way to estimate reward based on number of matched tiles
        # This is an approximation and doesn't perfectly count distinct 3, 4, 5-matches
        if not matches: return []
        
        visited = set()
        lengths = []
        
        rows = [[] for _ in range(self.GRID_HEIGHT)]
        cols = [[] for _ in range(self.GRID_WIDTH)]
        
        for r,c in matches:
            rows[r].append(c)
            cols[c].append(r)
        
        for r in range(self.GRID_HEIGHT):
            if not rows[r]: continue
            rows[r].sort()
            count = 1
            for i in range(1, len(rows[r])):
                if rows[r][i] == rows[r][i-1] + 1:
                    count += 1
                else:
                    if count >= 3: lengths.append(count)
                    count = 1
            if count >= 3: lengths.append(count)
            
        for c in range(self.GRID_WIDTH):
            if not cols[c]: continue
            cols[c].sort()
            count = 1
            for i in range(1, len(cols[c])):
                if cols[c][i] == cols[c][i-1] + 1:
                    count += 1
                else:
                    if count >= 3: lengths.append(count)
                    count = 1
            if count >= 3: lengths.append(count)
        
        return lengths if lengths else [3] # Ensure at least a base reward


    def _clear_and_drop(self, matches):
        for r, c in matches:
            if self.grid[r, c] != 0:
                self._create_particles(r, c)
                self.grid[r, c] = 0
        
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[r, c], self.grid[empty_row, c] = self.grid[empty_row, c], self.grid[r, c]
                    empty_row -= 1

    def _refill_grid(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
    
    def _create_particles(self, r, c):
        tile_type = self.grid[r, c]
        if tile_type == 0: return
        color = self.TILE_COLORS[tile_type - 1]
        center_x = self.GRID_MARGIN_X + c * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.GRID_MARGIN_Y + r * self.TILE_SIZE + self.TILE_SIZE // 2
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append([[center_x, center_y], vel, color, life])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[1][1] += 0.1  # Gravity
            p[3] -= 1
        self.particles = [p for p in self.particles if p[3] > 0]
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_MARGIN_X, self.GRID_MARGIN_Y, self.GRID_WIDTH * self.TILE_SIZE, self.GRID_HEIGHT * self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # Draw tiles
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                tile_type = self.grid[r, c]
                if tile_type > 0:
                    color = self.TILE_COLORS[tile_type - 1]
                    rect = pygame.Rect(
                        self.GRID_MARGIN_X + c * self.TILE_SIZE + 4,
                        self.GRID_MARGIN_Y + r * self.TILE_SIZE + 4,
                        self.TILE_SIZE - 8,
                        self.TILE_SIZE - 8
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=6)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_MARGIN_X + cursor_c * self.TILE_SIZE,
            self.GRID_MARGIN_Y + cursor_r * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, width=3, border_radius=8)

        # Draw invalid swap effect
        if self.last_swap_invalid and self.invalid_swap_timer > 0:
            alpha = 100 * (self.invalid_swap_timer / 5)
            s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            s.fill((255, 0, 0, alpha))
            self.screen.blit(s, cursor_rect.topleft)

        # Draw particles
        for p in self.particles:
            pos, _, color, life = p
            radius = max(0, life / 8)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)

    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_SCORE_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))
        
        # Render moves left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Render game over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "win": self.win,
        }

    def close(self):
        pygame.font.quit()
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Match-3 Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # Game loop for human control
    while not done:
        # Action defaults
        movement = 0 # no-op
        space_pressed = 0
        shift_pressed = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_pressed = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_pressed = 1
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
        
        action = np.array([movement, space_pressed, shift_pressed])
        
        # Only step if an action was taken, as auto_advance is False
        if not np.array_equal(action, [0, 0, 0]):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to reset.")
        
        # Rendering
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()