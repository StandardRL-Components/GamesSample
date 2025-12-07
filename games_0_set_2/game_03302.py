
# Generated: 2025-08-27T22:57:12.203345
# Source Brief: brief_03302.md
# Brief Index: 3302

        
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

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "then move to an adjacent gem and press Space again to swap. "
        "Press Shift to reset the cursor to the top-left."
    )

    # User-facing description of the game
    game_description = (
        "A strategic puzzle game. Swap adjacent gems to create matches of three or more. "
        "Clear 100 gems before you run out of moves to win! "
        "Longer matches and chain reactions give more points."
    )

    # Frames advance on action, not automatically
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 6
        self.GEM_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.GEM_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.GEM_SIZE) // 2 + 20
        self.MAX_MOVES = 50
        self.WIN_GEMS_COLLECTED = 100
        
        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CURSOR = (255, 255, 0, 150)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SCORE = (255, 200, 0)
        self.COLOR_MOVES = (200, 255, 200)
        self.COLOR_GEMS = (255, 100, 100)
        self.GEM_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (255, 80, 255),   # Magenta
            (80, 255, 255),   # Cyan
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.score = None
        self.gems_collected = None
        self.moves_remaining = None
        self.game_over = None
        self.terminated = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.animation_frame = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.gems_collected = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.terminated = False
        self.cursor_pos = [0, 0]
        self.selected_gem_pos = None
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._create_initial_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        reward = 0
        
        # --- Handle Actions ---
        if shift_press:
            self.cursor_pos = [0, 0]
        
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1) # Down
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1) # Right

        if space_press:
            # sound: 'select_gem.wav'
            if self.selected_gem_pos is None:
                self.selected_gem_pos = list(self.cursor_pos)
            else:
                # Check for adjacency
                x1, y1 = self.selected_gem_pos
                x2, y2 = self.cursor_pos
                if abs(x1 - x2) + abs(y1 - y2) == 1:
                    reward += self._handle_swap()
                else: # Not adjacent, select new gem
                    self.selected_gem_pos = list(self.cursor_pos)

        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # --- Check Termination ---
        if not self.game_over and (self.gems_collected >= self.WIN_GEMS_COLLECTED or self.moves_remaining <= 0):
            self.game_over = True
            self.terminated = True
            if self.gems_collected >= self.WIN_GEMS_COLLECTED:
                reward += 100 # Win bonus
                # sound: 'win_game.wav'
            else:
                reward -= 50 # Loss penalty
                # sound: 'lose_game.wav'

        return self._get_observation(), reward, self.terminated, False, self._get_info()

    def _handle_swap(self):
        pos1 = tuple(self.selected_gem_pos)
        pos2 = tuple(self.cursor_pos)
        self.moves_remaining -= 1
        
        self._swap_gems(pos1, pos2)
        
        total_chain_reward = 0
        chain_multiplier = 1.0

        while True:
            matches = self._find_matches()
            if not matches:
                # If the initial swap created no match, swap back and penalize
                if chain_multiplier == 1.0:
                    self._swap_gems(pos1, pos2)
                    # sound: 'invalid_swap.wav'
                    self.selected_gem_pos = None
                    return -0.1 # Invalid move penalty
                else: # End of a successful chain reaction
                    break
            
            # sound: 'match_found.wav'
            num_gems_cleared = len(matches)
            
            # Calculate reward for this step of the chain
            base_reward = num_gems_cleared
            for match_set in self._group_matches(matches):
                if len(match_set) == 4: base_reward += 5
                elif len(match_set) >= 5: base_reward += 10
            
            total_chain_reward += base_reward * chain_multiplier
            
            self.gems_collected += num_gems_cleared
            self.score += int(base_reward * chain_multiplier * 10)

            # Clear matched gems
            for x, y in matches:
                self.grid[y][x] = -1 # Mark as empty
            
            self._apply_gravity()
            self._refill_board()
            
            chain_multiplier += 0.5 # Increase multiplier for next chain
        
        # After a successful swap, check for softlock
        if not self._find_possible_matches():
            # sound: 'reshuffle.wav'
            self._create_initial_grid() # Reshuffle
            
        self.selected_gem_pos = None
        return total_chain_reward

    def _get_observation(self):
        self.animation_frame += 1
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_gems()
        self._render_cursor()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "gems_collected": self.gems_collected,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": tuple(self.cursor_pos),
        }

    # --- Rendering Helpers ---
    def _render_grid(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.GEM_SIZE,
                    self.GRID_OFFSET_Y + y * self.GEM_SIZE,
                    self.GEM_SIZE, self.GEM_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_gems(self):
        radius = self.GEM_SIZE // 2 - 4
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                gem_type = self.grid[y][x]
                if gem_type != -1:
                    center_x = self.GRID_OFFSET_X + x * self.GEM_SIZE + self.GEM_SIZE // 2
                    center_y = self.GRID_OFFSET_Y + y * self.GEM_SIZE + self.GEM_SIZE // 2
                    color = self.GEM_COLORS[gem_type]
                    
                    # Pulsing effect for selected gem
                    if self.selected_gem_pos and [x, y] == self.selected_gem_pos:
                        pulse = (math.sin(self.animation_frame * 0.2) + 1) / 2
                        color = tuple(min(255, int(c + pulse * 40)) for c in color)
                        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius + 2, color)
                    
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + x * self.GEM_SIZE,
            self.GRID_OFFSET_Y + y * self.GEM_SIZE,
            self.GEM_SIZE, self.GEM_SIZE
        )
        # Use a surface for transparency
        s = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), 4, border_radius=4)
        self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))
        
        # Gems Collected
        gems_text = self.font_medium.render(f"Gems: {self.gems_collected}/{self.WIN_GEMS_COLLECTED}", True, self.COLOR_GEMS)
        gems_rect = gems_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(gems_text, gems_rect)
        
        # Moves Remaining
        moves_text = self.font_medium.render(f"Moves: {self.moves_remaining}", True, self.COLOR_MOVES)
        moves_rect = moves_text.get_rect(midbottom=(self.WIDTH // 2, self.HEIGHT - 10))
        self.screen.blit(moves_text, moves_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "YOU WIN!" if self.gems_collected >= self.WIN_GEMS_COLLECTED else "GAME OVER"
        text = self.font_large.render(msg, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text, text_rect)

    # --- Game Logic Helpers ---
    def _create_initial_grid(self):
        while True:
            self.grid = [
                [self.np_random.integers(0, self.NUM_GEM_TYPES) for _ in range(self.GRID_SIZE)]
                for _ in range(self.GRID_SIZE)
            ]
            if not self._find_matches() and self._find_possible_matches():
                break

    def _swap_gems(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.grid[y1][x1], self.grid[y2][x2] = self.grid[y2][x2], self.grid[y1][x1]

    def _find_matches(self):
        matches = set()
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if x < self.GRID_SIZE - 2 and self.grid[y][x] == self.grid[y][x+1] == self.grid[y][x+2] != -1:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                if y < self.GRID_SIZE - 2 and self.grid[y][x] == self.grid[y+1][x] == self.grid[y+2][x] != -1:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches

    def _group_matches(self, matches):
        # Helper to find distinct match lines for scoring (e.g., a 5-gem T-shape is a 3-match and another 3-match)
        # This is a simplified grouping for reward calculation.
        match_sets = []
        matches_copy = set(matches)
        while matches_copy:
            start_node = matches_copy.pop()
            q = [start_node]
            visited = {start_node}
            component = {start_node}
            while q:
                (cx, cy) = q.pop(0)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in matches_copy and neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        q.append(neighbor)
            matches_copy -= component
            match_sets.append(component)
        return match_sets

    def _apply_gravity(self):
        for x in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y][x] != -1:
                    self.grid[empty_row][x], self.grid[y][x] = self.grid[y][x], self.grid[empty_row][x]
                    empty_row -= 1

    def _refill_board(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == -1:
                    self.grid[y][x] = self.np_random.integers(0, self.NUM_GEM_TYPES)
    
    def _find_possible_matches(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Try swapping right
                if x < self.GRID_SIZE - 1:
                    self._swap_gems((x, y), (x + 1, y))
                    if self._find_matches():
                        self._swap_gems((x, y), (x + 1, y))
                        return True
                    self._swap_gems((x, y), (x + 1, y))
                # Try swapping down
                if y < self.GRID_SIZE - 1:
                    self._swap_gems((x, y), (x, y + 1))
                    if self._find_matches():
                        self._swap_gems((x, y), (x, y + 1))
                        return True
                    self._swap_gems((x, y), (x, y + 1))
        return False

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

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # Arrow keys: Move cursor
    # Space: Select/Swap
    # Left Shift: Reset cursor
    # Q: Quit
    
    pygame.display.set_caption("Gem Matcher Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = [0, 0, 0] # No-op, space released, shift released
    
    while not done:
        # --- Pygame event loop for manual control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
        keys = pygame.key.get_pressed()
        
        # Reset action at the start of the loop
        action = [0, 0, 0]
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Buttons
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        if keys[pygame.K_q]:
            done = True
            
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Gems: {info['gems_collected']}, Moves: {info['moves_remaining']}")
        
        if terminated:
            print("Game Over!")

        # --- Render the environment to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit manual play speed
        
    pygame.quit()