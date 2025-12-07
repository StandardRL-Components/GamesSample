
# Generated: 2025-08-27T20:25:25.478881
# Source Brief: brief_02452.md
# Brief Index: 2452

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem. "
        "Move to an adjacent gem and press Space again to swap. "
        "Hold Shift to reshuffle the board (costs moves)."
    )

    game_description = (
        "A vibrant match-3 puzzle game. Swap adjacent gems to create lines of three or more. "
        "Create cascading combos to maximize your score and reach the target before you run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 8
        self.GEM_SIZE = 40
        self.NUM_GEM_TYPES = 6
        self.MAX_MOVES = 50
        self.TARGET_SCORE = 250
        self.MAX_STEPS = 1000
        self.RESHUFFLE_COST = 5

        self.X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.GEM_SIZE) // 2
        self.Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.GEM_SIZE) // 2

        # --- Colors and Visuals ---
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_SELECT = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 150, 50),  # Orange
            (200, 80, 255),  # Purple
            (255, 255, 100), # Yellow
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.last_action_was_invalid_swap = False

        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cursor_pos = [0, 0]
        self.selected_gem = None
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.last_action_was_invalid_swap = False

        self.grid = self._create_initial_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.last_action_was_invalid_swap = False

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Action Handling ---
        if movement > 0:
            self._move_cursor(movement)
        
        if shift_pressed:
            if self.moves_left >= self.RESHUFFLE_COST:
                self._reshuffle_board()
                self.moves_left -= self.RESHUFFLE_COST
                self.selected_gem = None
                reward -= 1.0 # Penalty for manual reshuffle
                # sfx: board_shuffle

        elif space_pressed:
            reward += self._handle_selection()

        # --- Termination Check ---
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 50
            terminated = True
            # sfx: win_game
        elif self.moves_left <= 0:
            reward -= 50
            terminated = True
            # sfx: lose_game
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_selection(self):
        reward = 0
        cy, cx = self.cursor_pos
        
        if self.selected_gem is None:
            self.selected_gem = (cy, cx)
            # sfx: select_gem
        else:
            sy, sx = self.selected_gem
            # Deselect if clicking the same gem
            if (sy, sx) == (cy, cx):
                self.selected_gem = None
                return reward
            
            # Check for adjacency
            if abs(sy - cy) + abs(sx - cx) == 1:
                # Perform swap
                self._swap_gems(sy, sx, cy, cx)
                self.moves_left -= 1
                # sfx: swap_gems

                # Resolve board and get rewards
                cleared_info = self._resolve_board()
                
                if cleared_info["gems_cleared"] == 0:
                    # Invalid swap, swap back
                    self._swap_gems(sy, sx, cy, cx)
                    self.moves_left += 1 # Refund move
                    reward = -0.1
                    self.last_action_was_invalid_swap = True
                    # sfx: invalid_swap
                else:
                    self.score += cleared_info["gems_cleared"]
                    reward += cleared_info["reward"]
            
            self.selected_gem = None
        
        return reward

    def _move_cursor(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_HEIGHT - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_WIDTH - 1, self.cursor_pos[1] + 1)
        # sfx: cursor_move

    def _resolve_board(self):
        total_gems_cleared = 0
        total_reward = 0
        combo_multiplier = 1.0

        while True:
            matches = self._find_matches()
            if not matches:
                break

            num_cleared = len(matches)
            total_gems_cleared += num_cleared
            
            # sfx: match_clear
            if combo_multiplier > 1.0:
                # sfx: combo_clear
                pass

            base_reward = num_cleared * 1.0
            if num_cleared > 3:
                base_reward += 5.0 # Bonus for 4+ match
            
            total_reward += base_reward * combo_multiplier

            for r, c in matches:
                self._create_particles(r, c, self.grid[r, c])

            self._clear_gems(matches)
            self._gems_fall()
            self._refill_board()
            
            combo_multiplier += 0.5

        if total_gems_cleared > 0 and not self._find_possible_moves():
            self._reshuffle_board()
            # sfx: board_shuffle_auto
        
        return {"gems_cleared": total_gems_cleared, "reward": total_reward}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.Y_OFFSET + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, y), (self.X_OFFSET + self.GRID_WIDTH * self.GEM_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.X_OFFSET + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.Y_OFFSET), (x, self.Y_OFFSET + self.GRID_HEIGHT * self.GEM_SIZE))

        # Draw gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    self._draw_gem(r, c, gem_type)
        
        # Draw selection and cursor
        self._draw_cursor()
        if self.selected_gem:
            self._draw_selection()
            
        # Draw particles
        self._update_and_draw_particles()

    def _draw_gem(self, r, c, gem_type):
        x = self.X_OFFSET + c * self.GEM_SIZE + self.GEM_SIZE // 2
        y = self.Y_OFFSET + r * self.GEM_SIZE + self.GEM_SIZE // 2
        radius = self.GEM_SIZE // 2 - 4
        color = self.GEM_COLORS[gem_type - 1]
        
        # Main gem body
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        
        # Highlight
        highlight_color = tuple(min(255, val + 60) for val in color)
        pygame.gfxdraw.aacircle(self.screen, x - radius//3, y - radius//3, radius//3, highlight_color)
        pygame.gfxdraw.filled_circle(self.screen, x - radius//3, y - radius//3, radius//3, highlight_color)

    def _draw_cursor(self):
        r, c = self.cursor_pos
        rect = pygame.Rect(
            self.X_OFFSET + c * self.GEM_SIZE,
            self.Y_OFFSET + r * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, self.COLOR_CURSOR, shape_surf.get_rect(), border_radius=5)
        self.screen.blit(shape_surf, rect)

    def _draw_selection(self):
        r, c = self.selected_gem
        x = self.X_OFFSET + c * self.GEM_SIZE + self.GEM_SIZE // 2
        y = self.Y_OFFSET + r * self.GEM_SIZE + self.GEM_SIZE // 2
        radius = self.GEM_SIZE // 2 - 2
        
        # Pulsating effect
        pulse = abs(math.sin(self.steps * 0.3))
        color = (
            int(self.COLOR_SELECT[0] * (0.7 + 0.3 * pulse)),
            int(self.COLOR_SELECT[1] * (0.7 + 0.3 * pulse)),
            int(self.COLOR_SELECT[2] * (0.7 + 0.3 * pulse)),
        )

        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius-1, color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 10))
        
        # Moves
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 10))

        # Target Score
        target_text = self.font_small.render(f"Target: {self.TARGET_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (self.SCREEN_WIDTH - target_text.get_width() - 20, 45))

        # Invalid Swap feedback
        if self.last_action_was_invalid_swap:
            invalid_text = self.font_small.render("Invalid Swap!", True, self.GEM_COLORS[0])
            pos = (self.SCREEN_WIDTH // 2 - invalid_text.get_width() // 2, self.SCREEN_HEIGHT - 30)
            self.screen.blit(invalid_text, pos)

    def _create_particles(self, r, c, gem_type):
        x = self.X_OFFSET + c * self.GEM_SIZE + self.GEM_SIZE // 2
        y = self.Y_OFFSET + r * self.GEM_SIZE + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(10, 20)
            self.particles.append([x, y, vx, vy, color, lifetime])

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[5] -= 1    # lifetime -= 1
            if p[5] > 0:
                active_particles.append(p)
                size = max(1, int(p[5] / 4))
                pygame.draw.circle(self.screen, p[4], (int(p[0]), int(p[1])), size)
        self.particles = active_particles

    # --- Core Game Logic Helpers ---

    def _create_initial_board(self):
        for _ in range(100): # Safety break
            grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            while self._find_matches(grid):
                matches = self._find_matches(grid)
                for r, c in matches:
                    grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
            
            if self._find_possible_moves(grid):
                return grid
        
        # Fallback if no valid board is found
        return self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def _find_matches(self, grid=None):
        if grid is None:
            grid = self.grid
        
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

    def _find_possible_moves(self, grid=None):
        if grid is None:
            grid = self.grid
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                temp_grid = grid.copy()
                # Try swapping right
                if c < self.GRID_WIDTH - 1:
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_matches(temp_grid): return True
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c] # Swap back
                # Try swapping down
                if r < self.GRID_HEIGHT - 1:
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_matches(temp_grid): return True
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c] # Swap back
        return False

    def _swap_gems(self, r1, c1, r2, c2):
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _clear_gems(self, matches):
        for r, c in matches:
            self.grid[r, c] = 0

    def _gems_fall(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
    
    def _refill_board(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
    
    def _reshuffle_board(self):
        flat_gems = self.grid.flatten()
        non_empty_gems = flat_gems[flat_gems > 0]
        self.np_random.shuffle(non_empty_gems)
        
        new_grid = np.zeros_like(self.grid)
        gem_idx = 0
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r,c] > 0:
                    new_grid[r,c] = non_empty_gems[gem_idx]
                    gem_idx += 1
        self.grid = new_grid

        # Ensure the new board is valid
        if self._find_matches() or not self._find_possible_moves():
            self.grid = self._create_initial_board()

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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

if __name__ == "__main__":
    # Example of how to run the environment
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # To play manually, you would need to map keyboard events to actions
    # This loop demonstrates random actions
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Moves Left: {info['moves_left']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print("Game Over!")
            obs, info = env.reset()
    
    env.close()