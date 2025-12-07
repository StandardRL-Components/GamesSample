
# Generated: 2025-08-28T05:50:15.741254
# Source Brief: brief_02743.md
# Brief Index: 2743

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "then move to an adjacent gem and press Space again to swap. Hold Shift to deselect."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of three or more "
        "of the same color. Reach the target score before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 5
        self.WIN_SCORE = 1000
        self.MAX_MOVES = 20
        self.MAX_STEPS = self.MAX_MOVES * 5 # Generous step limit

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 0)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        
        # --- Game State (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.score = 0
        self.moves_remaining = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.particles = []
        
        # --- Visuals ---
        self.board_rect = None
        self.gem_size = 0
        self.board_offset_x = 0
        self.board_offset_y = 0
        self._calculate_layout()
        
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def _calculate_layout(self):
        board_height = int(self.HEIGHT * 0.9)
        self.gem_size = board_height // self.GRID_SIZE
        board_width = self.gem_size * self.GRID_SIZE
        self.board_offset_x = (self.WIDTH - board_width) // 2
        self.board_offset_y = (self.HEIGHT - board_height) + (board_height - self.gem_size * self.GRID_SIZE) // 2
        self.board_rect = pygame.Rect(self.board_offset_x, self.board_offset_y, board_width, self.gem_size * self.GRID_SIZE)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            # Fallback to a default generator if no seed is provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gem = None
        self.prev_space_held = False
        self.particles = []
        
        self._generate_initial_grid()

        return self._get_observation(), self._get_info()

    def _generate_initial_grid(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._find_possible_moves():
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_SIZE
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_SIZE
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE

        # 2. Handle deselection
        if shift_held:
            if self.selected_gem:
                self.selected_gem = None
                # Small penalty for canceling a selection? Or neutral? Let's go neutral.

        # 3. Handle selection/swap (on key press, not hold)
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            if self.selected_gem is None:
                self.selected_gem = tuple(self.cursor_pos)
                # Sound: select_gem.wav
            else:
                # Attempting a swap
                x1, y1 = self.selected_gem
                x2, y2 = self.cursor_pos
                
                # Check for adjacency
                if abs(x1 - x2) + abs(y1 - y2) == 1:
                    self.moves_remaining -= 1
                    
                    # Perform swap
                    self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
                    
                    # Check for matches
                    total_cleared_gems = self._process_matches()
                    
                    if total_cleared_gems > 0:
                        # Sound: match_success.wav
                        reward += total_cleared_gems
                        if not self._find_possible_moves():
                           self._reshuffle_grid()
                    else:
                        # Invalid swap (no match), swap back
                        # Sound: swap_fail.wav
                        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
                        # No reward, but move is consumed
                else:
                    # Not adjacent
                    reward = -0.1 # Penalty for invalid selection
                    # Sound: error.wav

                self.selected_gem = None

        self.prev_space_held = space_held
        self.steps += 1
        
        # 4. Check for termination
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.moves_remaining <= 0:
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_matches(self):
        total_gems_cleared = 0
        chain_multiplier = 1.0

        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            gems_in_this_pass = len(matches)
            total_gems_cleared += gems_in_this_pass
            
            # Award points based on match size and chain
            base_score = gems_in_this_pass
            if gems_in_this_pass == 4: base_score += 5
            if gems_in_this_pass >= 5: base_score += 10
            
            self.score += int(base_score * chain_multiplier)
            
            for x, y in matches:
                self._create_particles(x, y, self.grid[y, x])
                self.grid[y, x] = -1 # Mark as empty
            
            self._apply_gravity()
            self._refill_grid()
            
            chain_multiplier += 0.5 # Increase multiplier for next chain
            # Sound: chain_reaction.wav
        
        return total_gems_cleared

    def _find_matches(self):
        matched_coords = set()
        # Horizontal matches
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE - 2):
                if self.grid[y, x] == self.grid[y, x+1] == self.grid[y, x+2] and self.grid[y, x] != -1:
                    matched_coords.add((x, y))
                    matched_coords.add((x+1, y))
                    matched_coords.add((x+2, y))
        # Vertical matches
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE - 2):
                if self.grid[y, x] == self.grid[y+1, x] == self.grid[y+2, x] and self.grid[y, x] != -1:
                    matched_coords.add((x, y))
                    matched_coords.add((x, y+1))
                    matched_coords.add((x, y+2))
        return matched_coords

    def _apply_gravity(self):
        for x in range(self.GRID_SIZE):
            empty_slots = []
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y, x] == -1:
                    empty_slots.append(y)
                elif empty_slots:
                    new_y = empty_slots.pop(0)
                    self.grid[new_y, x] = self.grid[y, x]
                    self.grid[y, x] = -1
                    empty_slots.append(y)

    def _refill_grid(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y, x] == -1:
                    self.grid[y, x] = self.np_random.integers(0, self.NUM_GEM_TYPES)
    
    def _find_possible_moves(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Try swapping right
                if x < self.GRID_SIZE - 1:
                    self.grid[y, x], self.grid[y, x+1] = self.grid[y, x+1], self.grid[y, x]
                    if self._find_matches():
                        self.grid[y, x], self.grid[y, x+1] = self.grid[y, x+1], self.grid[y, x]
                        return True
                    self.grid[y, x], self.grid[y, x+1] = self.grid[y, x+1], self.grid[y, x]
                # Try swapping down
                if y < self.GRID_SIZE - 1:
                    self.grid[y, x], self.grid[y+1, x] = self.grid[y+1, x], self.grid[y, x]
                    if self._find_matches():
                        self.grid[y, x], self.grid[y+1, x] = self.grid[y+1, x], self.grid[y, x]
                        return True
                    self.grid[y, x], self.grid[y+1, x] = self.grid[y+1, x], self.grid[y, x]
        return False

    def _reshuffle_grid(self):
        # Sound: reshuffle.wav
        while True:
            flat_grid = self.grid.flatten()
            self.np_random.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_SIZE, self.GRID_SIZE))
            if self._find_possible_moves():
                break

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._update_and_render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": self.cursor_pos,
            "selected_gem": self.selected_gem,
        }

    def _render_grid(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, self.board_rect, border_radius=5)
        for i in range(1, self.GRID_SIZE):
            # Vertical lines
            x = self.board_offset_x + i * self.gem_size
            pygame.draw.line(self.screen, self.COLOR_BG, (x, self.board_offset_y), (x, self.board_rect.bottom -1), 2)
            # Horizontal lines
            y = self.board_offset_y + i * self.gem_size
            pygame.draw.line(self.screen, self.COLOR_BG, (self.board_offset_x, y), (self.board_rect.right - 1, y), 2)

    def _render_gems(self):
        padding = int(self.gem_size * 0.1)
        gem_radius = (self.gem_size - 2 * padding) // 2
        
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                gem_type = self.grid[y, x]
                if gem_type == -1: continue

                center_x = self.board_offset_x + x * self.gem_size + self.gem_size // 2
                center_y = self.board_offset_y + y * self.gem_size + self.gem_size // 2
                
                color = self.GEM_COLORS[gem_type]
                
                # Gem highlight/shadow for 3D effect
                highlight = tuple(min(255, c + 50) for c in color)
                shadow = tuple(max(0, c - 50) for c in color)
                
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, gem_radius, shadow)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(gem_radius * 0.9), color)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, gem_radius, shadow)
                
                # Add a small glint
                glint_x = center_x + gem_radius // 3
                glint_y = center_y - gem_radius // 3
                pygame.gfxdraw.filled_circle(self.screen, glint_x, glint_y, gem_radius // 5, (255, 255, 255, 150))

    def _render_cursor(self):
        cursor_x, cursor_y = self.cursor_pos
        rect = pygame.Rect(
            self.board_offset_x + cursor_x * self.gem_size,
            self.board_offset_y + cursor_y * self.gem_size,
            self.gem_size,
            self.gem_size
        )
        
        # Pulsing effect for the cursor
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        cursor_color = (*self.COLOR_CURSOR, alpha)
        
        # Create a temporary surface for transparency
        s = pygame.Surface((self.gem_size, self.gem_size), pygame.SRCALPHA)
        pygame.draw.rect(s, cursor_color, s.get_rect(), 5, border_radius=8)
        self.screen.blit(s, rect.topleft)

        if self.selected_gem:
            sel_x, sel_y = self.selected_gem
            sel_rect = pygame.Rect(
                self.board_offset_x + sel_x * self.gem_size,
                self.board_offset_y + sel_y * self.gem_size,
                self.gem_size,
                self.gem_size
            )
            pygame.draw.rect(self.screen, (255, 255, 255), sel_rect, 3, border_radius=8)


    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 20))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "You Win!" if self.score >= self.WIN_SCORE else "Game Over"
            end_text = self.font_main.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, grid_x, grid_y, gem_type):
        center_x = self.board_offset_x + grid_x * self.gem_size + self.gem_size // 2
        center_y = self.board_offset_y + grid_y * self.gem_size + self.gem_size // 2
        color = self.GEM_COLORS[gem_type]

        for _ in range(15): # Number of particles per gem
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30) # Frames to live
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'color': color, 'radius': random.uniform(2, 5)})
    
    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(s, (p['pos'][0]-p['radius'], p['pos'][1]-p['radius']))

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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override screen for direct rendering
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Matcher")
    
    done = False
    
    # Game loop
    while not done:
        # Action mapping from keyboard
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'r' to reset
                    obs, info = env.reset()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we control the step rate
        env.clock.tick(30) # Run at 30 FPS for smooth controls

    env.close()