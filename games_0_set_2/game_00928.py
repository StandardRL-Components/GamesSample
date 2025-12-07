
# Generated: 2025-08-27T15:13:54.479550
# Source Brief: brief_00928.md
# Brief Index: 928

        
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
        "Controls: Use arrow keys to move the selector. Press space to swap with the gem in your last moved direction. Hold shift to reshuffle (costs 1 move)."
    )

    game_description = (
        "A strategic puzzle game. Swap adjacent gems to create matches of three or more. Plan your moves to create combos and reach the target score before you run out of turns."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 6
        self.TARGET_SCORE = 1000
        self.MAX_MOVES = 20
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.GEM_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (200, 80, 255),   # Purple
            (255, 160, 80),   # Orange
        ]

        # Visuals
        self.CELL_SIZE = 42
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20
        self.GEM_RADIUS = self.CELL_SIZE // 2 - 4
        
        # State variables (initialized in reset)
        self.grid = None
        self.selector_pos = None
        self.last_move_dir = None
        self.score = None
        self.moves_remaining = None
        self.game_over = None
        self.steps = None
        self.just_matched_coords = None
        self.just_reshuffled = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.steps = 0
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.last_move_dir = [0, 1]  # Default to down
        self.just_matched_coords = []
        self.just_reshuffled = False

        self._init_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.just_matched_coords.clear()
        self.just_reshuffled = False
        
        reward = 0
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        action_taken = False

        if shift_held and self.moves_remaining > 0:
            # Reshuffle action
            self.moves_remaining -= 1
            action_taken = True
            reward -= 5  # Small penalty for reshuffling
            self._reshuffle_board()
            self.just_reshuffled = True
            # sfx: board_shuffle

        elif space_held and self.moves_remaining > 0:
            # Swap action
            self.moves_remaining -= 1
            action_taken = True
            
            p1 = self.selector_pos
            p2 = [p1[0] + self.last_move_dir[0], p1[1] + self.last_move_dir[1]]

            if 0 <= p2[0] < self.GRID_SIZE and 0 <= p2[1] < self.GRID_SIZE:
                self._swap_gems(p1, p2)
                
                match_reward = self._resolve_cascades()
                if match_reward > 0:
                    reward += match_reward
                    # sfx: match_success
                else:
                    # No match, swap back
                    self._swap_gems(p1, p2)
                    reward += -0.1
                    # sfx: swap_fail
            else:
                # Invalid swap attempt (out of bounds)
                reward += -0.1
        
        elif movement > 0:
            # Selector movement (no move cost)
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            self.selector_pos[0] = (self.selector_pos[0] + dx) % self.GRID_SIZE
            self.selector_pos[1] = (self.selector_pos[1] + dy) % self.GRID_SIZE
            self.last_move_dir = [dx, dy]

        # After any action that changes the board, check for softlock
        if action_taken and not self._find_possible_moves():
            self._reshuffle_board()
            self.just_reshuffled = True # Visual feedback for auto-shuffle
            # sfx: auto_shuffle_warning

        # Check for termination conditions
        if self.score >= self.TARGET_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx: win_game
        elif self.moves_remaining <= 0:
            reward += -50
            terminated = True
            self.game_over = True
            # sfx: lose_game

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_remaining": self.moves_remaining,
            "steps": self.steps,
        }

    def _init_grid(self):
        self.grid = np.random.randint(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        while self._check_for_matches() or not self._find_possible_moves():
            self._reshuffle_board(force_no_matches=True)

    def _reshuffle_board(self, force_no_matches=False):
        flat_gems = self.grid.flatten().tolist()
        random.shuffle(flat_gems)
        self.grid = np.array(flat_gems).reshape((self.GRID_SIZE, self.GRID_SIZE))
        if force_no_matches:
            while self._check_for_matches():
                self._remove_initial_matches()
        if not self._find_possible_moves():
             self._init_grid() # Failsafe, regenerate if shuffle fails

    def _remove_initial_matches(self):
        matches = self._check_for_matches()
        if not matches:
            return
        for r, c in matches:
            allowed_gems = list(range(1, self.NUM_GEM_TYPES + 1))
            if c > 0 and self.grid[r, c] == self.grid[r, c-1]:
                if self.grid[r,c] in allowed_gems: allowed_gems.remove(self.grid[r,c])
            if r > 0 and self.grid[r, c] == self.grid[r-1, c]:
                if self.grid[r,c] in allowed_gems: allowed_gems.remove(self.grid[r,c])
            self.grid[r, c] = random.choice(allowed_gems) if allowed_gems else 1


    def _swap_gems(self, p1, p2):
        y1, x1 = p1
        y2, x2 = p2
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]

    def _check_for_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != 0:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != 0:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Check swap right
                if c < self.GRID_SIZE - 1:
                    self._swap_gems((r, c), (r, c + 1))
                    if self._check_for_matches():
                        self._swap_gems((r, c), (r, c + 1))
                        return True
                    self._swap_gems((r, c), (r, c + 1))
                # Check swap down
                if r < self.GRID_SIZE - 1:
                    self._swap_gems((r, c), (r + 1, c))
                    if self._check_for_matches():
                        self._swap_gems((r, c), (r + 1, c))
                        return True
                    self._swap_gems((r, c), (r + 1, c))
        return False

    def _resolve_cascades(self):
        total_reward = 0
        combo_multiplier = 1.0
        while True:
            matches = self._check_for_matches()
            if not matches:
                break
            
            # Score calculation
            num_matched = len(matches)
            reward = num_matched * combo_multiplier
            if num_matched >= 5:
                reward += 10  # Bonus for large clusters
            total_reward += reward
            self.score += int(reward * 10) # Scale reward to score

            self.just_matched_coords.extend(list(matches))
            # sfx: combo_ding

            # Remove matched gems
            for r, c in matches:
                self.grid[r, c] = 0

            # Apply gravity and refill
            self._apply_gravity_and_refill()
            combo_multiplier += 0.5
        
        return total_reward

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            empty_slots = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[r + empty_slots, c] = self.grid[r, c]
                    self.grid[r, c] = 0
            # Refill top
            for r in range(empty_slots):
                self.grid[r, c] = np.random.randint(1, self.NUM_GEM_TYPES + 1)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=8)

        # Draw gems
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    cx = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                    cy = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    color = self.GEM_COLORS[gem_type - 1]
                    
                    # Main gem body
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.GEM_RADIUS, color)
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, self.GEM_RADIUS, tuple(min(255, x+30) for x in color))

                    # Sheen effect
                    sheen_cx = cx - self.GEM_RADIUS // 3
                    sheen_cy = cy - self.GEM_RADIUS // 3
                    pygame.gfxdraw.filled_circle(self.screen, sheen_cx, sheen_cy, self.GEM_RADIUS // 3, (255, 255, 255, 80))

        # Draw selector
        sel_r, sel_c = self.selector_pos
        sel_x = self.GRID_X + sel_c * self.CELL_SIZE
        sel_y = self.GRID_Y + sel_r * self.CELL_SIZE
        selector_rect = pygame.Rect(sel_x, sel_y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsating effect for selector
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        line_width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, line_width, border_radius=5)

        # Draw match/reshuffle effects
        if self.just_matched_coords:
            for r, c in self.just_matched_coords:
                cx = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                cy = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                # Starburst effect
                for i in range(5):
                    angle = i * (2 * math.pi / 5) + (pygame.time.get_ticks() * 0.01)
                    x1 = cx + math.cos(angle) * (self.GEM_RADIUS)
                    y1 = cy + math.sin(angle) * (self.GEM_RADIUS)
                    x2 = cx + math.cos(angle) * (self.GEM_RADIUS + 10)
                    y2 = cy + math.sin(angle) * (self.GEM_RADIUS + 10)
                    pygame.draw.aaline(self.screen, (255, 255, 200), (x1, y1), (x2, y2))
        
        if self.just_reshuffled:
            reshuffle_text = self.font_small.render("Reshuffled!", True, (255, 200, 0))
            text_rect = reshuffle_text.get_rect(center=(self.WIDTH // 2, self.GRID_Y - 20))
            self.screen.blit(reshuffle_text, text_rect)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves display
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.TARGET_SCORE:
                end_text = self.font_large.render("YOU WIN!", True, (100, 255, 100))
            else:
                end_text = self.font_large.render("GAME OVER", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Matcher")
    
    running = True
    terminated = False
    
    action = np.array([0, 0, 0]) # No-op, release, release

    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not terminated:
                current_action = np.array([0, 0, 0])
                if event.key == pygame.K_UP:
                    current_action[0] = 1
                elif event.key == pygame.K_DOWN:
                    current_action[0] = 2
                elif event.key == pygame.K_LEFT:
                    current_action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    current_action[0] = 4
                elif event.key == pygame.K_SPACE:
                    current_action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    current_action[2] = 1
                
                # Step the environment with the single key press action
                obs, reward, terminated, truncated, info = env.step(current_action)
                print(f"Action: {current_action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                 print("Resetting environment.")
                 obs, info = env.reset()
                 terminated = False

        # Rendering
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS for human play

    env.close()