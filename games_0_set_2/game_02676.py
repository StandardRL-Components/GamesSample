
# Generated: 2025-08-27T21:06:14.413945
# Source Brief: brief_02676.md
# Brief Index: 2676

        
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
        "Controls: Arrows to move cursor. Press Space to select a gem, "
        "then move to an adjacent gem and press Space again to swap. Press Shift to cancel."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more. "
        "Clear the entire board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 6
    MOVES_LIMIT = 25
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_SELECTED_BG = (60, 70, 100)
    
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width = 640
        self.height = 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        self.grid_rect = pygame.Rect(
            (self.width - self.height) // 2, 0, self.height, self.height
        )
        self.gem_size = self.grid_rect.width // self.GRID_WIDTH
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.MOVES_LIMIT
        
        self.selector_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_gem_pos = None
        
        self.grid = self._generate_board()
        self.total_gems = self.GRID_WIDTH * self.GRID_HEIGHT
        
        self.last_turn_effects = [] # To draw effects from the last move

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        terminated = False
        self.last_turn_effects.clear()

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle player input ---
        # This part of the logic does not consume a move and has no reward
        
        # 1. Cancel selection
        if shift_held and self.selected_gem_pos:
            # sfx: deselect_sound
            self.selected_gem_pos = None

        # 2. Move selector
        if movement != 0:
            if movement == 1: self.selector_pos[0] -= 1
            elif movement == 2: self.selector_pos[0] += 1
            elif movement == 3: self.selector_pos[1] -= 1
            elif movement == 4: self.selector_pos[1] += 1
            self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.GRID_HEIGHT - 1)
            self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.GRID_WIDTH - 1)

        # 3. Primary action (select/swap)
        if space_held:
            if not self.selected_gem_pos:
                # Select a gem
                # sfx: select_sound
                self.selected_gem_pos = list(self.selector_pos)
            else:
                # Attempt a swap
                dist = abs(self.selected_gem_pos[0] - self.selector_pos[0]) + \
                       abs(self.selected_gem_pos[1] - self.selector_pos[1])
                
                if dist == 1: # Is adjacent
                    # --- This is a turn-consuming action ---
                    self.moves_left -= 1
                    # sfx: swap_sound
                    reward = self._process_swap()
                    self.selected_gem_pos = None # Deselect after swap
                else:
                    # sfx: invalid_select_sound
                    self.selected_gem_pos = list(self.selector_pos) # Select new gem instead
        
        terminated = self._check_termination()
        if terminated:
            reward += 50 if self.win else -50

        if self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_swap(self):
        p1 = self.selected_gem_pos
        p2 = self.selector_pos
        
        # Perform swap
        self.grid[p1[0], p1[1]], self.grid[p2[0], p2[1]] = self.grid[p2[0], p2[1]], self.grid[p1[0], p1[1]]

        total_reward = 0
        combo_multiplier = 1
        
        while True:
            matches = self._find_matches()
            if not matches:
                # If the very first check finds no matches, it was an invalid swap
                if combo_multiplier == 1:
                    # Swap back
                    self.grid[p1[0], p1[1]], self.grid[p2[0], p2[1]] = self.grid[p2[0], p2[1]], self.grid[p1[0], p1[1]]
                    # sfx: invalid_swap_sound
                    return -0.1
                break

            # sfx: match_sound
            gems_cleared_this_cascade = 0
            for match in matches:
                gems_cleared_this_cascade += len(match)
                for r, c in match:
                    if self.grid[r, c] != 0: # Avoid double counting cleared gems
                        self.last_turn_effects.append(
                            ("explosion", (r, c), self.grid[r, c])
                        )
                        self.grid[r, c] = 0 # Mark as cleared
                        self.total_gems -= 1
            
            reward_this_cascade = gems_cleared_this_cascade
            if gems_cleared_this_cascade > 3:
                reward_this_cascade += 5 # Combo bonus
                # sfx: combo_sound
            
            total_reward += reward_this_cascade * combo_multiplier
            
            self._apply_gravity()
            self._fill_top_rows()
            
            combo_multiplier += 1

        # After all cascades, check if any moves are left
        if not self._find_possible_moves() and self.total_gems > 0:
            # sfx: shuffle_sound
            self._shuffle_board()

        return total_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _check_termination(self):
        if self.total_gems <= 0:
            self.game_over = True
            self.win = True
            return True
        if self.moves_left <= 0:
            self.game_over = True
            self.win = False
            return True
        return False

    def _render_game(self):
        # Draw grid background
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.grid_rect.left + c * self.gem_size,
                    self.grid_rect.top + r * self.gem_size,
                    self.gem_size, self.gem_size
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw selected gem background
        if self.selected_gem_pos:
            r, c = self.selected_gem_pos
            rect = pygame.Rect(
                self.grid_rect.left + c * self.gem_size,
                self.grid_rect.top + r * self.gem_size,
                self.gem_size, self.gem_size
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED_BG, rect)

        # Draw gems and effects
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    is_selected = self.selected_gem_pos and self.selected_gem_pos == [r, c]
                    self._draw_gem(self.screen, gem_type, r, c, is_selected)
        
        # Draw effects from last turn
        for effect in self.last_turn_effects:
            if effect[0] == "explosion":
                r, c = effect[1]
                gem_type = effect[2]
                self._draw_explosion(self.screen, r, c, gem_type)

        # Draw selector
        r, c = self.selector_pos
        rect = pygame.Rect(
            self.grid_rect.left + c * self.gem_size,
            self.grid_rect.top + r * self.gem_size,
            self.gem_size, self.gem_size
        )
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, 3)

    def _draw_gem(self, surface, gem_type, r, c, is_selected):
        rect = pygame.Rect(
            self.grid_rect.left + c * self.gem_size,
            self.grid_rect.top + r * self.gem_size,
            self.gem_size, self.gem_size
        )
        center = rect.center
        
        pulse = 0
        if is_selected:
            pulse = 3 * math.sin(self.steps * 0.3)

        size = int(self.gem_size * 0.35 + pulse)
        color = self.GEM_COLORS[gem_type - 1]
        
        # Draw a slightly darker version for depth
        dark_color = tuple(max(0, val - 50) for val in color)

        if gem_type == 1: # Circle
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], size, dark_color)
            pygame.gfxdraw.filled_circle(surface, center[0]-2, center[1]-2, size, color)
        elif gem_type == 2: # Square
            points = [(center[0]-size, center[1]-size), (center[0]+size, center[1]-size),
                      (center[0]+size, center[1]+size), (center[0]-size, center[1]+size)]
            pygame.gfxdraw.filled_polygon(surface, points, dark_color)
            points_bright = [(p[0]-2, p[1]-2) for p in points]
            pygame.gfxdraw.filled_polygon(surface, points_bright, color)
        elif gem_type == 3: # Diamond
            points = [(center[0], center[1]-size), (center[0]+size, center[1]),
                      (center[0], center[1]+size), (center[0]-size, center[1])]
            pygame.gfxdraw.filled_polygon(surface, points, dark_color)
            points_bright = [(p[0]-2, p[1]-2) for p in points]
            pygame.gfxdraw.filled_polygon(surface, points_bright, color)
        elif gem_type == 4: # Triangle
            points = [(center[0], center[1]-size), (center[0]+size, center[1]+size//2),
                      (center[0]-size, center[1]+size//2)]
            pygame.gfxdraw.filled_polygon(surface, points, dark_color)
            points_bright = [(p[0]-2, p[1]-2) for p in points]
            pygame.gfxdraw.filled_polygon(surface, points_bright, color)
        elif gem_type == 5: # Hexagon
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                points.append((center[0] + size * math.cos(angle), center[1] + size * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(surface, points, dark_color)
            points_bright = [(p[0]-2, p[1]-2) for p in points]
            pygame.gfxdraw.filled_polygon(surface, points_bright, color)
        elif gem_type == 6: # Star
            points = []
            for i in range(10):
                r_val = size if i % 2 == 0 else size / 2
                angle = math.pi / 5 * i - math.pi / 2
                points.append((center[0] + r_val * math.cos(angle), center[1] + r_val * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(surface, points, dark_color)
            points_bright = [(p[0]-2, p[1]-2) for p in points]
            pygame.gfxdraw.filled_polygon(surface, points_bright, color)

    def _draw_explosion(self, surface, r, c, gem_type):
        rect = pygame.Rect(
            self.grid_rect.left + c * self.gem_size,
            self.grid_rect.top + r * self.gem_size,
            self.gem_size, self.gem_size
        )
        center = rect.center
        color = self.GEM_COLORS[gem_type - 1]
        
        for i in range(8):
            angle = math.pi / 4 * i
            start_pos = (
                center[0] + 5 * math.cos(angle),
                center[1] + 5 * math.sin(angle)
            )
            end_pos = (
                center[0] + self.gem_size * 0.6 * math.cos(angle),
                center[1] + self.gem_size * 0.6 * math.sin(angle)
            )
            pygame.draw.line(surface, color, start_pos, end_pos, 3)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.width - moves_text.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.width/2, self.height/2))
            pygame.draw.rect(self.screen, self.COLOR_BG, msg_rect.inflate(20, 20))
            self.screen.blit(msg_surf, msg_rect)

    def _generate_board(self):
        while True:
            grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_matches(grid) and self._find_possible_moves(grid):
                return grid

    def _find_matches(self, grid=None):
        if grid is None:
            grid = self.grid
        
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem = grid[r, c]
                if gem == 0: continue
                
                # Horizontal
                if c < self.GRID_WIDTH - 2 and grid[r, c+1] == gem and grid[r, c+2] == gem:
                    match = {(r, c), (r, c+1), (r, c+2)}
                    i = c + 3
                    while i < self.GRID_WIDTH and grid[r, i] == gem:
                        match.add((r, i))
                        i += 1
                    matches.add(frozenset(match))

                # Vertical
                if r < self.GRID_HEIGHT - 2 and grid[r+1, c] == gem and grid[r+2, c] == gem:
                    match = {(r, c), (r+1, c), (r+2, c)}
                    i = r + 3
                    while i < self.GRID_HEIGHT and grid[i, c] == gem:
                        match.add((i, c))
                        i += 1
                    matches.add(frozenset(match))
        return list(matches)

    def _find_possible_moves(self, grid=None):
        if grid is None:
            grid = self.grid
        
        temp_grid = grid.copy()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Swap right
                if c < self.GRID_WIDTH - 1:
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_matches(temp_grid):
                        return True
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c] # Swap back
                # Swap down
                if r < self.GRID_HEIGHT - 1:
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_matches(temp_grid):
                        return True
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c] # Swap back
        return False

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
    
    def _fill_top_rows(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
                    self.total_gems += 1
    
    def _shuffle_board(self):
        flat_gems = self.grid[self.grid > 0].flatten()
        self.np_random.shuffle(flat_gems)
        
        new_grid = np.zeros_like(self.grid)
        k = 0
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r,c] > 0:
                    new_grid[r,c] = flat_gems[k]
                    k += 1

        self.grid = new_grid
        
        if not self._find_possible_moves() or self._find_matches():
            # Extremely rare case: reshuffle resulted in instant match or still no moves
            # The simplest solution is a full reset of the board state
            self.grid = self._generate_board()

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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((env.width, env.height))
    
    terminated = False
    
    print(env.game_description)
    print(env.user_guide)

    action = [0, 0, 0] # No-op, no space, no shift

    while not terminated:
        # --- Human Controls ---
        # Reset action at the start of each frame
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    print("--- Game Reset ---")

        # --- Step the environment ---
        # We only step if a key was pressed, because auto_advance is False
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Reward: {reward:.1f}, Score: {info['score']}, Moves Left: {info['moves_left']}")
            if terminated:
                print("--- GAME OVER ---")
                print(f"Final Score: {info['score']}")

        # --- Render the game ---
        # Transpose observation back to pygame's (width, height, channels) format
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit frame rate

    env.close()