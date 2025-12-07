
# Generated: 2025-08-27T15:04:42.258173
# Source Brief: brief_00880.md
# Brief Index: 880

        
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
        "Controls: Use arrow keys to move the selector. Press Space to select a crystal. "
        "With a crystal selected, use an arrow key to swap it. Press Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent crystals in an isometric grid to match 3 or more. "
        "Create chain reactions to maximize your score before you run out of moves. Clear the board to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.NUM_CRYSTAL_TYPES = 6
        self.INITIAL_MOVES = 20
        self.MAX_STEPS = 1000

        # Visual constants
        self.ISO_TILE_W, self.ISO_TILE_H = 48, 24
        self.ISO_TILE_Z = 18 # Height of the crystal block
        self.GRID_ORIGIN_X = self.WIDTH // 2
        self.GRID_ORIGIN_Y = self.HEIGHT // 2 - self.GRID_ROWS * self.ISO_TILE_H // 2 + 20

        # Colors
        self.COLOR_BG = (25, 30, 50)
        self.COLOR_GRID = (60, 70, 100)
        self.CRYSTAL_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 200, 0)
        self.COLOR_UI_BG = (10, 15, 30, 200)
        self.COLOR_UI_TEXT = (220, 220, 240)
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_crystal = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.INITIAL_MOVES
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_crystal = None
        self.particles = []
        self._generate_initial_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1

        # 1. Update particles
        self._update_particles()
        
        # 2. Handle cursor movement
        if movement != 0:
            prev_cursor_pos = list(self.cursor_pos)
            if movement == 1: self.cursor_pos[0] -= 1  # Up
            elif movement == 2: self.cursor_pos[0] += 1  # Down
            elif movement == 3: self.cursor_pos[1] -= 1  # Left
            elif movement == 4: self.cursor_pos[1] += 1  # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_ROWS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_COLS - 1)

            # If a crystal is selected, this movement triggers a swap
            if self.selected_crystal is not None:
                swap_dir = (self.cursor_pos[0] - prev_cursor_pos[0], self.cursor_pos[1] - prev_cursor_pos[1])
                reward += self._attempt_swap(self.selected_crystal, swap_dir)
                self.selected_crystal = None # Deselect after swap attempt
                self.cursor_pos = list(prev_cursor_pos) # Reset cursor position after swap

        # 3. Handle selection/deselection
        if space_pressed and self.selected_crystal is None:
            r, c = self.cursor_pos
            if self.grid[r][c] > 0:
                self.selected_crystal = [r, c]
        elif shift_pressed and self.selected_crystal is not None:
            self.selected_crystal = None

        # 4. Check for termination
        terminated = self._check_termination()
        if self.win:
            reward += 100 # Victory bonus

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_swap(self, pos1, direction):
        r1, c1 = pos1
        r2, c2 = r1 + direction[0], c1 + direction[1]

        if not (0 <= r2 < self.GRID_ROWS and 0 <= c2 < self.GRID_COLS):
            return 0 # Invalid swap, out of bounds

        # Perform the swap
        self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]
        self.moves_left -= 1
        # sfx: swap_sound

        # Check for matches and process chain reactions
        chain_reward = self._handle_matches_and_gravity()
        
        # In many match-3 games, invalid swaps (that create no match) are reversed.
        # The brief implies they are not, and just cost a move. We follow the brief.
        
        return chain_reward
        
    def _handle_matches_and_gravity(self):
        total_reward = 0
        chain_level = 1
        while True:
            matches = self._find_matches()
            if not matches:
                break

            # Calculate reward for this wave of matches
            cleared_count = len(matches)
            total_reward += cleared_count # +1 per crystal
            
            # Event-based reward for match size
            if cleared_count == 3: total_reward += 5
            elif cleared_count == 4: total_reward += 10
            elif cleared_count >= 5: total_reward += 20
            
            total_reward += (chain_level - 1) * 5 # Chain reaction bonus

            # Clear matched crystals and create particles
            for r, c in matches:
                self._create_particles(r, c, self.grid[r][c])
                self.grid[r][c] = 0 # 0 represents an empty cell
            # sfx: match_clear_sound
            
            self.score += total_reward

            # Apply gravity and refill
            self._apply_gravity()
            self._refill_board()
            chain_level += 1
        
        return total_reward

    def _find_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r][c] > 0 and self.grid[r][c] == self.grid[r][c+1] == self.grid[r][c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical matches
        for r in range(self.GRID_ROWS - 2):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] > 0 and self.grid[r][c] == self.grid[r+1][c] == self.grid[r+2][c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] > 0:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = 0
                    empty_row -= 1

    def _refill_board(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == 0:
                    self.grid[r][c] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)

    def _generate_initial_grid(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS))
            # Clear any initial matches
            while True:
                initial_matches = self._find_matches()
                if not initial_matches:
                    break
                for r, c in initial_matches:
                    self.grid[r][c] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)
            
            # Ensure at least one move is possible
            if self._check_possible_moves():
                break

    def _check_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                original_crystal = self.grid[r][c]
                # Check swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                    if self._find_matches():
                        self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                        return True
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                    if self._find_matches():
                        self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                        return True
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
        return False

    def _check_termination(self):
        if self.game_over:
            return True
        
        crystals_left = np.count_nonzero(self.grid)
        if crystals_left == 0:
            self.win = True
            self.game_over = True
            return True
        
        if self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False
    
    def _grid_to_iso(self, r, c):
        x = self.GRID_ORIGIN_X + (c - r) * (self.ISO_TILE_W / 2)
        y = self.GRID_ORIGIN_Y + (c + r) * (self.ISO_TILE_H / 2)
        return int(x), int(y)

    def _draw_iso_cube(self, surface, x, y, z, color):
        top_color = color
        side_color1 = tuple(max(0, val - 40) for val in color)
        side_color2 = tuple(max(0, val - 60) for val in color)
        
        points_top = [
            (x, y - z),
            (x + self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2 - z),
            (x, y + self.ISO_TILE_H - z),
            (x - self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2 - z),
        ]
        points_side1 = [
            (x, y + self.ISO_TILE_H - z),
            (x - self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2 - z),
            (x - self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2),
            (x, y + self.ISO_TILE_H),
        ]
        points_side2 = [
            (x, y + self.ISO_TILE_H - z),
            (x + self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2 - z),
            (x + self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2),
            (x, y + self.ISO_TILE_H),
        ]
        
        pygame.gfxdraw.filled_polygon(surface, points_side1, side_color1)
        pygame.gfxdraw.aapolygon(surface, points_side1, side_color1)
        pygame.gfxdraw.filled_polygon(surface, points_side2, side_color2)
        pygame.gfxdraw.aapolygon(surface, points_side2, side_color2)
        pygame.gfxdraw.filled_polygon(surface, points_top, top_color)
        pygame.gfxdraw.aapolygon(surface, points_top, top_color)

    def _draw_iso_selector(self, surface, r, c, color, thickness):
        x, y = self._grid_to_iso(r, c)
        points = [
            (x, y),
            (x + self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2),
            (x, y + self.ISO_TILE_H),
            (x - self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2),
        ]
        pygame.draw.lines(surface, color, True, points, thickness)

    def _create_particles(self, r, c, crystal_type):
        x, y = self._grid_to_iso(r, c)
        color = self.CRYSTAL_COLORS[crystal_type - 1]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append([[x, y], vel, life, color])
    
    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[1][1] += 0.1 # gravity
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid floor
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                x, y = self._grid_to_iso(r, c)
                points = [
                    (x, y + self.ISO_TILE_H),
                    (x + self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2),
                    (x, y),
                    (x - self.ISO_TILE_W // 2, y + self.ISO_TILE_H // 2),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw crystals
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                crystal_type = self.grid[r][c]
                if crystal_type > 0:
                    x, y = self._grid_to_iso(r, c)
                    color = self.CRYSTAL_COLORS[crystal_type - 1]
                    self._draw_iso_cube(self.screen, x, y, self.ISO_TILE_Z, color)

        # Draw selectors
        if self.selected_crystal:
            r, c = self.selected_crystal
            self._draw_iso_selector(self.screen, r, c, self.COLOR_SELECTED, 3)
        
        r, c = self.cursor_pos
        self._draw_iso_selector(self.screen, r, c, self.COLOR_CURSOR, 2)

        # Draw particles
        for p in self.particles:
            pos, _, life, color = p
            size = max(1, int(life / 8))
            pygame.draw.rect(self.screen, color, (int(pos[0]), int(pos[1]), size, size))

    def _render_ui(self):
        # Moves left
        ui_panel = pygame.Rect(self.WIDTH - 180, 10, 170, 50)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_panel, border_radius=8)
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (ui_panel.x + 15, ui_panel.y + 13))

        # Score
        score_panel = pygame.Rect(10, 10, 200, 50)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, score_panel, border_radius=8)
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (score_panel.x + 15, score_panel.y + 13))

        # Crystals left
        crystals_left = np.count_nonzero(self.grid)
        crystals_panel = pygame.Rect(10, self.HEIGHT - 60, 200, 50)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, crystals_panel, border_radius=8)
        crystals_text = self.font_main.render(f"Crystals Left: {crystals_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystals_text, (crystals_panel.x + 15, crystals_panel.y + 13))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            message = "YOU WIN!" if self.win else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_SELECTED if self.win else self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "crystals_left": np.count_nonzero(self.grid),
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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