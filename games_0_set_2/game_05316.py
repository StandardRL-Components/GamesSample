
# Generated: 2025-08-28T04:38:29.542660
# Source Brief: brief_05316.md
# Brief Index: 5316

        
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
        "Controls: Use Up/Down to select crystal type. "
        "Hold Space to place at the first empty spot, or Shift to place at the last empty spot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Place crystals to create chains of 3 or more. Reach 1000 points in 20 moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.MAX_MOVES = 20
        self.WIN_SCORE = 1000
        self.CRYSTAL_TYPES = 3

        # --- Visuals ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 220, 240)
        self.CRYSTAL_PALETTE = {
            1: {'base': (255, 60, 60), 'light': (255, 150, 150), 'dark': (180, 20, 20)}, # Red
            2: {'base': (60, 255, 60), 'light': (150, 255, 150), 'dark': (20, 180, 20)}, # Green
            3: {'base': (80, 120, 255), 'light': (160, 180, 255), 'dark': (40, 60, 180)}, # Blue
        }
        
        self.TILE_WIDTH = 60
        self.TILE_HEIGHT = self.TILE_WIDTH / 2
        self.TILE_DEPTH = 20 # Visual height of the crystal block
        
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

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
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
            self.font_huge = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_large = pygame.font.SysFont("sans", 24)
            self.font_small = pygame.font.SysFont("sans", 16)
            self.font_huge = pygame.font.SysFont("sans", 48)
        
        # --- State Variables ---
        self.grid = None
        self.score = 0
        self.moves_left = 0
        self.selected_crystal_type = 1
        self.particles = []
        self.game_over = False
        self.win_state = ""
        
        # Initialize state
        self.reset()

        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.selected_crystal_type = self.np_random.integers(1, self.CRYSTAL_TYPES + 1)
        self.particles = []
        self.game_over = False
        self.win_state = ""
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        placed_crystal = False

        # --- Handle Actions ---
        # 1. Cycle crystal type
        if movement == 1: # Up
            self.selected_crystal_type = (self.selected_crystal_type % self.CRYSTAL_TYPES) + 1
        elif movement == 2: # Down
            self.selected_crystal_type = ((self.selected_crystal_type - 2 + self.CRYSTAL_TYPES) % self.CRYSTAL_TYPES) + 1

        # 2. Place crystal
        if space_held or shift_held:
            placed_crystal, pos = self._handle_placement(place_top_left=space_held)
            if placed_crystal:
                # sfx: crystal_place.wav
                self.moves_left -= 1
                match_reward = self._process_matches()
                reward += match_reward

        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            self.win_state = "YOU WIN!"
            reward += 100 # Goal-oriented reward
            # sfx: game_win.wav
        elif self.moves_left <= 0:
            terminated = True
            self.game_over = True
            self.win_state = "GAME OVER"
            reward -= 100 # Goal-oriented penalty
            # sfx: game_lose.wav

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_placement(self, place_top_left):
        """Finds an empty spot and places the selected crystal."""
        if place_top_left:
            # Iterate from top-left to bottom-right
            for y in range(self.GRID_SIZE):
                for x in range(self.GRID_SIZE):
                    if self.grid[y, x] == 0:
                        self.grid[y, x] = self.selected_crystal_type
                        self.selected_crystal_type = self.np_random.integers(1, self.CRYSTAL_TYPES + 1)
                        return True, (x, y)
        else: # Place bottom-right
            # Iterate from bottom-right to top-left
            for y in range(self.GRID_SIZE - 1, -1, -1):
                for x in range(self.GRID_SIZE - 1, -1, -1):
                    if self.grid[y, x] == 0:
                        self.grid[y, x] = self.selected_crystal_type
                        self.selected_crystal_type = self.np_random.integers(1, self.CRYSTAL_TYPES + 1)
                        return True, (x, y)
        return False, None # Grid is full

    def _process_matches(self):
        """Handles the entire match-finding, clearing, and gravity cascade."""
        total_reward = 0
        chain_multiplier = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            # sfx: match_clear.wav
            num_cleared = len(matches)
            total_reward += num_cleared # +1 per crystal
            self.score += num_cleared * 10 # Base score per crystal
            
            if chain_multiplier > 0:
                total_reward += 5 # Chain reaction bonus
                self.score += num_cleared * 10 * chain_multiplier # Chain score bonus

            for x, y in matches:
                crystal_type = self.grid[y, x]
                self._create_particles(x, y, self.CRYSTAL_PALETTE[crystal_type]['base'])
                self.grid[y, x] = 0 # Clear the crystal
            
            self._apply_gravity()
            chain_multiplier += 1
        
        return total_reward

    def _find_matches(self):
        """Finds all horizontal and vertical matches of 3 or more."""
        matches = set()
        # Horizontal
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE - 2):
                if self.grid[y, x] != 0 and self.grid[y, x] == self.grid[y, x+1] == self.grid[y, x+2]:
                    ctype = self.grid[y, x]
                    for i in range(3): matches.add((x+i, y))
                    # Check for longer matches
                    for i in range(3, self.GRID_SIZE - x):
                        if self.grid[y, x+i] == ctype: matches.add((x+i, y))
                        else: break
        # Vertical
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE - 2):
                if self.grid[y, x] != 0 and self.grid[y, x] == self.grid[y+1, x] == self.grid[y+2, x]:
                    ctype = self.grid[y, x]
                    for i in range(3): matches.add((x, y+i))
                    # Check for longer matches
                    for i in range(3, self.GRID_SIZE - y):
                        if self.grid[y+i, x] == ctype: matches.add((x, y+i))
                        else: break
        return matches

    def _apply_gravity(self):
        """Shifts crystals down to fill empty spaces."""
        for x in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y, x] != 0:
                    self.grid[empty_row, x], self.grid[y, x] = self.grid[y, x], self.grid[empty_row, x]
                    empty_row -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "steps": 0, # Not used in this turn-based game
        }

    def _render_game(self):
        self._draw_grid()
        self._update_and_draw_particles()
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                crystal_type = self.grid[y, x]
                if crystal_type != 0:
                    self._draw_iso_cube(x, y, self.CRYSTAL_PALETTE[crystal_type])

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _draw_grid(self):
        for y in range(self.GRID_SIZE + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_SIZE, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_SIZE + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_SIZE)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _draw_iso_cube(self, x, y, colors):
        """Draws a 3D-effect isometric cube."""
        screen_x, screen_y = self._iso_to_screen(x, y)
        
        # Points for the top face
        p_top = (screen_x, screen_y - self.TILE_DEPTH)
        p_left = (screen_x - self.TILE_WIDTH / 2, screen_y - self.TILE_DEPTH + self.TILE_HEIGHT / 2)
        p_right = (screen_x + self.TILE_WIDTH / 2, screen_y - self.TILE_DEPTH + self.TILE_HEIGHT / 2)
        p_bottom = (screen_x, screen_y - self.TILE_DEPTH + self.TILE_HEIGHT)
        
        # Top face (light)
        pygame.gfxdraw.filled_polygon(self.screen, [p_top, p_left, p_bottom, p_right], colors['light'])
        pygame.gfxdraw.aapolygon(self.screen, [p_top, p_left, p_bottom, p_right], colors['light'])
        
        # Left face (base)
        base_left = (p_left[0], p_left[1] + self.TILE_DEPTH)
        base_bottom = (p_bottom[0], p_bottom[1] + self.TILE_DEPTH)
        pygame.gfxdraw.filled_polygon(self.screen, [p_left, p_bottom, base_bottom, base_left], colors['base'])
        pygame.gfxdraw.aapolygon(self.screen, [p_left, p_bottom, base_bottom, base_left], colors['base'])
        
        # Right face (dark)
        base_right = (p_right[0], p_right[1] + self.TILE_DEPTH)
        pygame.gfxdraw.filled_polygon(self.screen, [p_right, p_bottom, base_bottom, base_right], colors['dark'])
        pygame.gfxdraw.aapolygon(self.screen, [p_right, p_bottom, base_bottom, base_right], colors['dark'])

    def _create_particles(self, grid_x, grid_y, color):
        screen_x, screen_y = self._iso_to_screen(grid_x, grid_y)
        for _ in range(15): # Create 15 particles per cleared crystal
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.uniform(15, 30)
            self.particles.append({'pos': [screen_x, screen_y], 'vel': vel, 'life': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                size = int(5 * (p['life'] / 30))
                if size > 0:
                    temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, (*p['color'], alpha), (size, size), size)
                    self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Moves
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))
        
        # Next crystal
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.WIDTH // 2 - next_text.get_width() // 2, self.HEIGHT - 70))
        self._draw_iso_cube(-0.5, self.GRID_SIZE + 3.5, self.CRYSTAL_PALETTE[self.selected_crystal_type])
        
        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_huge.render(self.win_state, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Grid")
    
    done = False
    clock = pygame.time.Clock()
    
    # Initial render
    frame = env._get_observation()
    frame = np.transpose(frame, (1, 0, 2))
    surf = pygame.surfarray.make_surface(frame)
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    while not done:
        movement, space, shift = 0, 0, 0
        
        # Human-friendly keyboard mapping
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # Only step if an action is taken in this turn-based game
        action_taken = any(action)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'R'
                    obs, info = env.reset()
                    action_taken = True 
                if event.key == pygame.K_ESCAPE:
                    done = True
        
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            print(f"Action: {action}, Reward: {reward}, Score: {info['score']}, Moves: {info['moves_left']}")
        
        # Rendering
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    pygame.quit()