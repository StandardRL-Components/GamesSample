
# Generated: 2025-08-28T00:01:58.766813
# Source Brief: brief_03662.md
# Brief Index: 3662

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrows to move the cursor. Press space to select a gem, "
        "then move to an adjacent gem and press space again to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more. "
        "Collect 50 gems within 20 moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_GEM_TYPES = 6
    GEM_GOAL = 50
    MAX_MOVES = 20
    MAX_STEPS = 1000

    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 40, 55)
    COLOR_GRID_LINE = (50, 60, 75)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 160, 80),  # Orange
    ]
    COLOR_WHITE = (240, 240, 240)
    COLOR_GOLD = (255, 215, 0)
    COLOR_SILVER = (192, 192, 192)

    # Animation Timings (in steps/frames)
    SWAP_FRAMES = 8
    MATCH_FRAMES = 10
    FALL_FRAMES = 6

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.grid = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.last_space_held = False
        self.animation_state = None
        self.animation_timer = 0
        self.animation_data = {}
        self.win_bonus_given = False
        
        # Calculate layout
        self.GEM_SIZE = 38
        self.GRID_PIXEL_WIDTH = self.GRID_WIDTH * self.GEM_SIZE
        self.GRID_PIXEL_HEIGHT = self.GRID_HEIGHT * self.GEM_SIZE
        self.GRID_TOP = (self.SCREEN_HEIGHT - self.GRID_PIXEL_HEIGHT) // 2
        self.GRID_LEFT = (self.SCREEN_WIDTH - self.GRID_PIXEL_WIDTH) // 2

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
        self.win_bonus_given = False
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_gem_pos = None
        self.last_space_held = False
        self.animation_state = None
        self.animation_timer = 0
        self.animation_data = {}

        self._initialize_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        
        movement, space_held, _ = action
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if self.animation_state is not None:
            reward = self._update_animations()
        elif not self.game_over:
            self._handle_input(movement, space_pressed)

        terminated = (self.moves_left <= 0 or self.score >= self.GEM_GOAL or self.steps >= self.MAX_STEPS)
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.GEM_GOAL and not self.win_bonus_given:
                reward += 50
                self.win_bonus_given = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Game Logic ---

    def _initialize_board(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            self._remove_and_refill_instant(matches)
        
        if not self._check_for_valid_moves():
            self._reshuffle_board()

    def _handle_input(self, movement, space_pressed):
        # Move cursor
        if movement == 1 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 2 and self.cursor_pos[0] < self.GRID_HEIGHT - 1: self.cursor_pos[0] += 1
        elif movement == 3 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 4 and self.cursor_pos[1] < self.GRID_WIDTH - 1: self.cursor_pos[1] += 1
        
        if space_pressed:
            r, c = self.cursor_pos
            if self.selected_gem_pos is None:
                self.selected_gem_pos = [r, c]
                # sfx: select_gem
            else:
                if self.selected_gem_pos == [r, c]:
                    self.selected_gem_pos = None
                    # sfx: deselect_gem
                elif self._are_adjacent(self.selected_gem_pos, [r, c]):
                    self._start_swap_animation(self.selected_gem_pos, [r, c])
                else:
                    self.selected_gem_pos = [r, c]
                    # sfx: select_gem

    def _are_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def _start_swap_animation(self, pos1, pos2):
        self.animation_state = 'swapping'
        self.animation_timer = self.SWAP_FRAMES
        self.animation_data = {'pos1': pos1, 'pos2': pos2}
        self._swap_gems(pos1, pos2)
        self.selected_gem_pos = None
        # sfx: swap_start

    def _swap_gems(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _update_animations(self):
        self.animation_timer -= 1
        reward = 0

        if self.animation_timer > 0:
            return 0

        if self.animation_state == 'swapping':
            matches = self._find_all_matches()
            if matches:
                self.moves_left -= 1
                reward = self._start_match_animation(matches)
                # sfx: match_success
            else:
                self._swap_gems(self.animation_data['pos1'], self.animation_data['pos2']) # Swap back
                self.animation_state = 'reverting'
                self.animation_timer = self.SWAP_FRAMES
                # sfx: swap_fail
        
        elif self.animation_state == 'reverting':
            self.animation_state = None
            self.animation_data = {}

        elif self.animation_state == 'matching':
            self._apply_gravity_and_refill()
            self.animation_state = 'falling'
            self.animation_timer = self.FALL_FRAMES
            # sfx: gems_fall

        elif self.animation_state == 'falling':
            matches = self._find_all_matches()
            if matches:
                reward = self._start_match_animation(matches)
                # sfx: chain_reaction
            else:
                self.animation_state = None
                self.animation_data = {}
                if not self._check_for_valid_moves():
                    self._reshuffle_board()
                    # sfx: board_shuffle
        
        return reward

    def _start_match_animation(self, matches):
        reward = 0
        num_matched = len(matches)
        
        reward += num_matched # Base reward
        if num_matched == 4: reward += 5
        elif num_matched >= 5: reward += 10
        
        self.score += num_matched
        
        self.animation_state = 'matching'
        self.animation_timer = self.MATCH_FRAMES
        self.animation_data = {'matches': matches}
        
        for r, c in matches:
            self.grid[r, c] = -1 # Mark for removal
        
        return reward

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem = self.grid[r, c]
                if gem == -1: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[r, c+1] == gem and self.grid[r, c+2] == gem:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[r+1, c] == gem and self.grid[r+2, c] == gem:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _check_for_valid_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    self._swap_gems((r, c), (r, c + 1))
                    if self._find_all_matches():
                        self._swap_gems((r, c), (r, c + 1)) # Swap back
                        return True
                    self._swap_gems((r, c), (r, c + 1)) # Swap back
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    self._swap_gems((r, c), (r + 1, c))
                    if self._find_all_matches():
                        self. _swap_gems((r, c), (r + 1, c)) # Swap back
                        return True
                    self._swap_gems((r, c), (r + 1, c)) # Swap back
        return False

    def _reshuffle_board(self):
        self._initialize_board() # Re-run the full initialization logic

    def _remove_and_refill_instant(self, matches):
        for r, c in matches:
            self.grid[r, c] = -1
        self._apply_gravity_and_refill()

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = (self.GRID_LEFT, self.GRID_TOP, self.GRID_PIXEL_WIDTH, self.GRID_PIXEL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # Draw gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type == -1:
                    continue
                
                x = self.GRID_LEFT + c * self.GEM_SIZE
                y = self.GRID_TOP + r * self.GEM_SIZE
                
                # Handle animations
                if self.animation_state in ['swapping', 'reverting']:
                    pos1, pos2 = self.animation_data['pos1'], self.animation_data['pos2']
                    p = self.animation_timer / self.SWAP_FRAMES
                    if self.animation_state == 'swapping': p = 1 - p
                    
                    if [r,c] == pos1:
                        x = self._lerp(self.GRID_LEFT + pos2[1] * self.GEM_SIZE, self.GRID_LEFT + pos1[1] * self.GEM_SIZE, p)
                        y = self._lerp(self.GRID_TOP + pos2[0] * self.GEM_SIZE, self.GRID_TOP + pos1[0] * self.GEM_SIZE, p)
                    elif [r,c] == pos2:
                        x = self._lerp(self.GRID_LEFT + pos1[1] * self.GEM_SIZE, self.GRID_LEFT + pos2[1] * self.GEM_SIZE, p)
                        y = self._lerp(self.GRID_TOP + pos1[0] * self.GEM_SIZE, self.GRID_TOP + pos2[0] * self.GEM_SIZE, p)

                self._draw_gem(x, y, gem_type, (r,c))

        # Draw grid lines over gems
        for i in range(1, self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_LEFT + i * self.GEM_SIZE, self.GRID_TOP), (self.GRID_LEFT + i * self.GEM_SIZE, self.GRID_TOP + self.GRID_PIXEL_HEIGHT))
        for i in range(1, self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_LEFT, self.GRID_TOP + i * self.GEM_SIZE), (self.GRID_LEFT + self.GRID_PIXEL_WIDTH, self.GRID_TOP + i * self.GEM_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, grid_rect, 2, 10)

        # Draw cursor and selection
        self._draw_cursor()

    def _draw_gem(self, x, y, gem_type, pos):
        size = self.GEM_SIZE
        padding = 4
        color = self.GEM_COLORS[gem_type]
        
        # Matching animation
        if self.animation_state == 'matching' and tuple(pos) in self.animation_data['matches']:
            p = 1 - (self.animation_timer / self.MATCH_FRAMES)
            size_mod = math.sin(p * math.pi) * (self.GEM_SIZE * 0.5)
            size += size_mod
            padding -= size_mod / 2
            
            # Flash effect
            flash_alpha = int(255 * math.sin(p * math.pi))
            flash_color = (255, 255, 255, flash_alpha)
            flash_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(flash_surf, flash_color, (0,0,size,size), border_radius=int(size*0.3))
            self.screen.blit(flash_surf, (x + self.GEM_SIZE/2 - size/2, y + self.GEM_SIZE/2 - size/2))


        rect = pygame.Rect(x + padding, y + padding, size - 2 * padding, size - 2 * padding)
        
        # Draw gem body with anti-aliasing
        pygame.gfxdraw.box(self.screen, rect, (*color, 200)) # slightly transparent body
        pygame.gfxdraw.rectangle(self.screen, rect, color) # solid border
        
        # Highlight
        highlight_rect = rect.copy()
        highlight_rect.width = int(highlight_rect.width * 0.7)
        highlight_rect.height = int(highlight_rect.height * 0.4)
        highlight_rect.x += int(rect.width * 0.15)
        highlight_rect.y += int(rect.height * 0.1)
        pygame.gfxdraw.box(self.screen, highlight_rect, (255, 255, 255, 60))

    def _draw_cursor(self):
        if self.animation_state is not None: return

        # Pulse effect for selected gem
        if self.selected_gem_pos:
            r, c = self.selected_gem_pos
            x = self.GRID_LEFT + c * self.GEM_SIZE
            y = self.GRID_TOP + r * self.GEM_SIZE
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            width = int(self._lerp(2, 4, pulse))
            alpha = int(self._lerp(150, 255, pulse))
            pygame.draw.rect(self.screen, (*self.COLOR_WHITE, alpha), (x+1, y+1, self.GEM_SIZE-2, self.GEM_SIZE-2), width, border_radius=8)

        # Draw main cursor
        r, c = self.cursor_pos
        x = self.GRID_LEFT + c * self.GEM_SIZE
        y = self.GRID_TOP + r * self.GEM_SIZE
        pygame.draw.rect(self.screen, self.COLOR_GOLD, (x, y, self.GEM_SIZE, self.GEM_SIZE), 3, border_radius=8)

    def _render_ui(self):
        # Score
        score_text = self.font_m.render(f"GEMS: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (20, 20))

        # Moves
        moves_text = self.font_m.render(f"MOVES: {self.moves_left}", True, self.COLOR_WHITE)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 20))
        
        # Progress Bar
        bar_y = self.SCREEN_HEIGHT - 40
        bar_w = self.SCREEN_WIDTH - 40
        bar_h = 20
        progress = min(1.0, self.score / self.GEM_GOAL)
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (20, bar_y, bar_w, bar_h), border_radius=5)
        if progress > 0:
            pygame.draw.rect(self.screen, self.COLOR_GOLD, (20, bar_y, bar_w * progress, bar_h), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_SILVER, (20, bar_y, bar_w, bar_h), 2, border_radius=5)
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.GEM_GOAL:
                msg = "YOU WIN!"
                color = self.COLOR_GOLD
            else:
                msg = "GAME OVER"
                color = self.COLOR_WHITE
            
            end_text = self.font_l.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def validate_implementation(self):
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()