
# Generated: 2025-08-28T03:58:48.239309
# Source Brief: brief_02177.md
# Brief Index: 2177

        
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

    user_guide = (
        "Controls: Arrow keys to move selector. Space to select a gem. Select an adjacent gem to swap. Shift to cancel selection."
    )

    game_description = (
        "A colorful match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. Create cascades for big points! Collect 100 gems in 50 moves to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.np_random = None

        # --- Game Constants ---
        self.GRID_ROWS = 8
        self.GRID_COLS = 8
        self.NUM_GEM_TYPES = 6
        self.WIN_SCORE = 100
        self.MAX_MOVES = 50

        # --- Visuals ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 70)
        self.COLOR_UI_TEXT = (220, 230, 255)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_SELECTED_GEM = (255, 255, 100)
        
        self.GEM_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (255, 80, 255),   # Magenta
            (80, 255, 255),   # Cyan
        ]
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid_area_width = 360
        self.grid_area_height = 360
        self.grid_offset_x = (self.screen_width - self.grid_area_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_area_height) // 2
        self.cell_size = self.grid_area_width // self.GRID_COLS
        self.gem_radius = int(self.cell_size * 0.4)

        # --- State Variables ---
        self.grid = None
        self.selector_pos = None
        self.selected_gem = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.win_status = ""
        self.particles = []
        self.prev_space_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win_status = ""
        self.selector_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_gem = None
        self.particles = []
        self.prev_space_held = False
        
        self._create_new_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        if shift_action:
            self.selected_gem = None
        
        if movement == 1: self.selector_pos[1] = max(0, self.selector_pos[1] - 1) # Up
        elif movement == 2: self.selector_pos[1] = min(self.GRID_ROWS - 1, self.selector_pos[1] + 1) # Down
        elif movement == 3: self.selector_pos[0] = max(0, self.selector_pos[0] - 1) # Left
        elif movement == 4: self.selector_pos[0] = min(self.GRID_COLS - 1, self.selector_pos[0] + 1) # Right

        space_pressed = space_action and not self.prev_space_held
        if space_pressed:
            if self.selected_gem is None:
                self.selected_gem = tuple(self.selector_pos)
                # SFX: select_gem.wav
            else:
                current_selection = tuple(self.selector_pos)
                if current_selection == self.selected_gem:
                    self.selected_gem = None # Deselect
                    # SFX: deselect.wav
                elif self._is_adjacent(self.selected_gem, current_selection):
                    # This is a swap attempt, which counts as a move.
                    self.moves_left -= 1
                    
                    self._swap_gems(self.selected_gem, current_selection)
                    
                    matches1 = self._find_matches_at(self.selected_gem[0], self.selected_gem[1])
                    matches2 = self._find_matches_at(current_selection[0], current_selection[1])
                    all_matches = matches1.union(matches2)

                    if not all_matches:
                        # Invalid swap, swap back
                        self._swap_gems(self.selected_gem, current_selection)
                        reward = -0.1
                        # SFX: invalid_swap.wav
                    else:
                        # Successful swap, start cascade
                        cascade_reward = self._handle_cascades(all_matches)
                        reward += cascade_reward
                        # SFX: match_success.wav
                    
                    self.selected_gem = None
                else:
                    # Selected a non-adjacent gem, make it the new selection
                    self.selected_gem = current_selection
                    # SFX: select_gem.wav
        
        self.prev_space_held = space_action
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            terminated = True
            self.win_status = "YOU WIN!"
            reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
            self.win_status = "GAME OVER"
            # No penalty for losing, just lack of win reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    # --- Game Logic Helpers ---
    def _create_new_board(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_COLS, self.GRID_ROWS))
        while self._find_all_matches():
             self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_COLS, self.GRID_ROWS))

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _swap_gems(self, pos1, pos2):
        self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]

    def _find_matches_at(self, x, y):
        matches = set()
        gem_type = self.grid[x, y]
        
        # Horizontal
        h_matches = {(x, y)}
        for i in range(x - 1, -1, -1):
            if self.grid[i, y] == gem_type: h_matches.add((i, y))
            else: break
        for i in range(x + 1, self.GRID_COLS):
            if self.grid[i, y] == gem_type: h_matches.add((i, y))
            else: break
        if len(h_matches) >= 3: matches.update(h_matches)

        # Vertical
        v_matches = {(x, y)}
        for j in range(y - 1, -1, -1):
            if self.grid[x, j] == gem_type: v_matches.add((x, j))
            else: break
        for j in range(y + 1, self.GRID_ROWS):
            if self.grid[x, j] == gem_type: v_matches.add((x, j))
            else: break
        if len(v_matches) >= 3: matches.update(v_matches)
        
        return matches

    def _find_all_matches(self):
        all_matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[c, r] == -1: continue
                
                # Horizontal check
                if c < self.GRID_COLS - 2 and self.grid[c, r] == self.grid[c+1, r] == self.grid[c+2, r]:
                    all_matches.update([(c, r), (c+1, r), (c+2, r)])
                
                # Vertical check
                if r < self.GRID_ROWS - 2 and self.grid[c, r] == self.grid[c, r+1] == self.grid[c, r+2]:
                    all_matches.update([(c, r), (c, r+1), (c, r+2)])
        return all_matches

    def _handle_cascades(self, initial_matches):
        total_reward = 0
        matches = initial_matches
        
        while matches:
            num_matched = len(matches)
            total_reward += num_matched # +1 per gem
            if num_matched == 4: total_reward += 5 # Bonus for 4
            if num_matched >= 5: total_reward += 10 # Bonus for 5+
            self.score += num_matched

            for x, y in matches:
                if self.grid[x, y] != -1:
                    self._create_particles(x, y, self.GEM_COLORS[self.grid[x, y]])
                    self.grid[x, y] = -1 # Mark as empty
            
            self._apply_gravity_and_refill()
            matches = self._find_all_matches()
            if matches:
                # SFX: cascade.wav
                pass
        return total_reward

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[c, r] != -1:
                    if r != empty_row:
                        self.grid[c, empty_row] = self.grid[c, r]
                        self.grid[c, r] = -1
                    empty_row -= 1
            
            for r in range(empty_row, -1, -1):
                self.grid[c, r] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    # --- Rendering Helpers ---
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.grid_area_width, self.grid_area_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=10)

        # Draw gems
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[c, r]
                if gem_type != -1:
                    center_x = self.grid_offset_x + c * self.cell_size + self.cell_size // 2
                    center_y = self.grid_offset_y + r * self.cell_size + self.cell_size // 2
                    color = self.GEM_COLORS[gem_type]
                    
                    # Draw filled circle with anti-aliasing
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.gem_radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.gem_radius, color)
                    
                    # Highlight for selected gem
                    if self.selected_gem == (c, r):
                        pygame.draw.circle(self.screen, self.COLOR_SELECTED_GEM, (center_x, center_y), self.gem_radius + 3, 3)

        # Draw selector
        sel_x = self.grid_offset_x + self.selector_pos[0] * self.cell_size
        sel_y = self.grid_offset_y + self.selector_pos[1] * self.cell_size
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, (sel_x, sel_y, self.cell_size, self.cell_size), 3, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Gems: {self.score} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Moves
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(moves_text, moves_rect)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_status, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(end_text, end_rect)

    def _create_particles(self, grid_x, grid_y, color):
        center_x = self.grid_offset_x + grid_x * self.cell_size + self.cell_size // 2
        center_y = self.grid_offset_y + grid_y * self.cell_size + self.cell_size // 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed
            lifespan = random.randint(20, 40)
            self.particles.append([center_x, center_y, vel_x, vel_y, color, lifespan])
    
    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2] # pos_x += vel_x
            p[1] += p[3] # pos_y += vel_y
            p[5] -= 1    # lifespan -= 1
            if p[5] > 0:
                active_particles.append(p)
                radius = max(0, int((p[5] / 40) * 5))
                pygame.draw.circle(self.screen, p[4], (int(p[0]), int(p[1])), radius)
        self.particles = active_particles

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override screen to be a display screen
    env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Gem Swap")
    
    action = [0, 0, 0] # No-op, no space, no shift
    
    print(env.user_guide)

    while not done:
        # --- Human Input Handling ---
        movement = 0
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Step the environment ---
        # Since auto_advance is False, we only step when there's an action.
        # For human play, we step every frame to register key presses.
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        # --- Render to the display ---
        # The observation is already the rendered frame, so we just need to put it on the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit frame rate for human play

    print("Game Over!")
    pygame.time.wait(2000)
    env.close()