
# Generated: 2025-08-27T17:05:50.126891
# Source Brief: brief_01426.md
# Brief Index: 1426

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to select a gem, "
        "then move to an adjacent gem and press Space to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of three or more. "
        "Score points and trigger cascading combos to reach the target score before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 6
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    TARGET_SCORE = 1000
    MAX_MOVES = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTION = (255, 255, 255, 100)
    
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_score_popup = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 60, bold=True)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.score = None
        self.moves_left = None
        self.steps = None
        self.game_over = None
        self.game_won = None
        self.particles = None
        self.score_popups = None
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem_pos = None
        
        self.particles = []
        self.score_popups = []

        self._create_grid()
        while not self._find_possible_moves():
            self._create_grid() # Reshuffle until a valid move exists

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        # 1. Handle cursor movement
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
            
        # 2. Handle gem selection/deselection
        if shift_pressed:
            if self.selected_gem_pos == self.cursor_pos:
                self.selected_gem_pos = None # Deselect
            else:
                self.selected_gem_pos = list(self.cursor_pos) # Select
                
        # 3. Handle swap attempt
        if space_pressed and self.selected_gem_pos is not None:
            reward = self._attempt_swap()
            self.selected_gem_pos = None # Clear selection after attempt

        # 4. Check for termination conditions
        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100
            else:
                reward -= 50 # Ran out of moves
        
        # Anti-softlock: reshuffle if no moves are possible
        if not self.game_over and not self._find_possible_moves():
            self._create_grid() # This is a free reshuffle
            while not self._find_possible_moves():
                self._create_grid()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _attempt_swap(self):
        p1 = self.selected_gem_pos
        p2 = self.cursor_pos
        
        # Check if swap is adjacent
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
            return 0 # Not an adjacent swap, no penalty, just no action

        # Perform swap
        self._swap_gems(p1, p2)
        
        # Check for matches
        matches = self._find_matches()
        if not matches:
            # No match, swap back
            self._swap_gems(p1, p2) # Revert
            return -0.2 # Penalty for invalid move

        # Match found, process it
        self.moves_left -= 1
        
        total_reward = 0
        is_cascade = False
        
        while matches:
            gems_cleared = len(matches)
            total_reward += gems_cleared
            if is_cascade:
                total_reward += 5 # Cascade bonus
            
            # Create particles and popups
            score_for_match = gems_cleared * 10
            self.score += score_for_match
            
            # Find center of match for popup
            avg_x = sum(pos[0] for pos in matches) / len(matches)
            avg_y = sum(pos[1] for pos in matches) / len(matches)
            popup_pos = self._get_gem_screen_pos(avg_x, avg_y)
            self.score_popups.append([f"+{score_for_match}", popup_pos, 60]) # 60 frames lifetime

            for x, y in matches:
                self._create_particles(x, y, self.grid[y, x])
                self.grid[y, x] = -1 # Mark as empty
            
            # sfx: gem_match.wav
            self._handle_gravity()
            self._refill_grid()
            
            matches = self._find_matches()
            is_cascade = True
            # sfx: cascade.wav
            
        return total_reward

    def _create_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        # Ensure no initial matches
        while self._find_matches():
            matches = self._find_matches()
            for x, y in matches:
                self.grid[y, x] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
    
    def _swap_gems(self, p1, p2):
        self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = self.grid[p2[1], p2[0]], self.grid[p1[1], p1[0]]
        
    def _find_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[y, x]
                if gem_type == -1: continue
                
                # Horizontal match
                if x < self.GRID_WIDTH - 2 and self.grid[y, x+1] == gem_type and self.grid[y, x+2] == gem_type:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                
                # Vertical match
                if y < self.GRID_HEIGHT - 2 and self.grid[y+1, x] == gem_type and self.grid[y+2, x] == gem_type:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches

    def _find_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Try swapping right
                if x < self.GRID_WIDTH - 1:
                    self._swap_gems((x, y), (x + 1, y))
                    if self._find_matches():
                        self._swap_gems((x, y), (x + 1, y)) # Swap back
                        return True
                    self._swap_gems((x, y), (x + 1, y)) # Swap back
                
                # Try swapping down
                if y < self.GRID_HEIGHT - 1:
                    self._swap_gems((x, y), (x, y + 1))
                    if self._find_matches():
                        self._swap_gems((x, y), (x, y + 1)) # Swap back
                        return True
                    self._swap_gems((x, y), (x, y + 1)) # Swap back
        return False

    def _handle_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] != -1:
                    self._swap_gems((x, y), (x, empty_row))
                    empty_row -= 1
                    
    def _refill_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] == -1:
                    self.grid[y, x] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
                    # sfx: gem_fall.wav

    def _check_termination(self):
        if self.score >= self.TARGET_SCORE:
            self.game_over = True
            self.game_won = True
        elif self.moves_left <= 0:
            self.game_over = True
            self.game_won = False
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_won = False
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "selected_gem": self.selected_gem_pos
        }

    # --- Rendering Methods ---
    
    def _get_gem_screen_pos(self, x, y):
        grid_render_width = self.SCREEN_HEIGHT - 40
        self.gem_size = grid_render_width // self.GRID_HEIGHT
        grid_offset_x = (self.SCREEN_WIDTH - grid_render_width) / 2
        grid_offset_y = (self.SCREEN_HEIGHT - grid_render_width) / 2
        
        screen_x = grid_offset_x + x * self.gem_size + self.gem_size / 2
        screen_y = grid_offset_y + y * self.gem_size + self.gem_size / 2
        return int(screen_x), int(screen_y)

    def _render_game(self):
        self._update_and_draw_particles()
        
        # Draw grid background
        grid_render_width = self.SCREEN_HEIGHT - 40
        grid_offset_x = (self.SCREEN_WIDTH - grid_render_width) / 2
        grid_offset_y = (self.SCREEN_HEIGHT - grid_render_width) / 2
        pygame.draw.rect(self.screen, self.COLOR_GRID, (grid_offset_x, grid_offset_y, grid_render_width, grid_render_width), border_radius=10)

        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[y, x]
                if gem_type != -1:
                    self._draw_gem(x, y, gem_type)
        
        # Draw selection highlight
        if self.selected_gem_pos is not None:
            sx, sy = self._get_gem_screen_pos(self.selected_gem_pos[0], self.selected_gem_pos[1])
            s = pygame.Surface((self.gem_size, self.gem_size), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECTION)
            self.screen.blit(s, (sx - self.gem_size/2, sy - self.gem_size/2))
            
        # Draw cursor
        cx_grid, cy_grid = self.cursor_pos
        cx, cy = self._get_gem_screen_pos(cx_grid, cy_grid)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cx - self.gem_size/2, cy - self.gem_size/2, self.gem_size, self.gem_size), 3, border_radius=5)
        
        self._draw_score_popups()

    def _draw_gem(self, x, y, gem_type):
        screen_x, screen_y = self._get_gem_screen_pos(x, y)
        radius = int(self.gem_size * 0.4)
        color = self.GEM_COLORS[gem_type - 1]
        
        # Draw with anti-aliasing
        pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, screen_x, screen_y, radius, tuple(min(255, c + 50) for c in color))

        # Add a little shine
        shine_x = screen_x + radius // 3
        shine_y = screen_y - radius // 3
        pygame.gfxdraw.filled_circle(self.screen, shine_x, shine_y, radius // 4, (255, 255, 255, 128))


    def _create_particles(self, x, y, gem_type):
        screen_x, screen_y = self._get_gem_screen_pos(x, y)
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(15): # Create 15 particles per gem
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(20, 40)
            self.particles.append([screen_x, screen_y, vx, vy, color, lifetime])
            # sfx: particle_burst.wav
            
    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[5] -= 1    # lifetime -= 1
            
            if p[5] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p[5] / 40))
                color = p[4] + (alpha,)
                pygame.draw.circle(self.screen, color, (int(p[0]), int(p[1])), 2)

    def _draw_score_popups(self):
        for popup in self.score_popups[:]:
            text, pos, lifetime = popup
            popup[2] -= 1
            if lifetime <= 0:
                self.score_popups.remove(popup)
                continue
            
            alpha = int(255 * (lifetime / 60))
            text_surf = self.font_score_popup.render(text, True, (255, 255, 100))
            text_surf.set_alpha(alpha)
            
            pos[1] -= 0.5 # Move up
            text_rect = text_surf.get_rect(center=pos)
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 10))
        
        # Moves display
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)
        
        # Game over display
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert 'score' in info and 'moves_left' in info
        
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Matcher")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    terminated = False
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human play
        
    env.close()