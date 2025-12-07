
# Generated: 2025-08-28T04:49:03.633699
# Source Brief: brief_02423.md
# Brief Index: 2423

        
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
    """
    A turn-based puzzle game where the player rotates tiles on a grid to match target patterns.
    
    The player controls a selector to choose a tile and can rotate it clockwise or counter-clockwise.
    Each rotation costs one move. The goal is to complete all five target patterns before running out of moves.
    
    The action space is MultiDiscrete([5, 2, 2]):
    - actions[0]: Movement of the selector (0=none, 1=up, 2=down, 3=left, 4=right).
    - actions[1]: Rotate clockwise (1=pressed).
    - actions[2]: Rotate counter-clockwise (1=pressed).
    
    Rotation actions take precedence over movement and consume a move.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move selector. Space to rotate tile clockwise, Shift to rotate counter-clockwise."
    )

    # Short, user-facing description of the game
    game_description = (
        "A strategic puzzle game. Rotate tiles to match all five target patterns before you run out of moves."
    )

    # The game is turn-based, so it only advances state upon receiving an action.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 5
        self.TILE_SIZE = 60
        self.GRID_MARGIN = 10
        self.GRID_WIDTH = self.GRID_HEIGHT = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_ORIGIN = (
            self.SCREEN_WIDTH - self.GRID_WIDTH - self.GRID_MARGIN * 4,
            (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2
        )
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000
        self.NUM_PATTERNS = 5

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Fonts ---
        try:
            self.font_main = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
            self.font_tiny = pygame.font.Font(None, 16)
        except IOError:
            self.font_main = pygame.font.SysFont("sans", 32)
            self.font_small = pygame.font.SysFont("sans", 24)
            self.font_tiny = pygame.font.SysFont("sans", 16)

        # --- Colors (Nord Palette for aesthetics) ---
        self.COLOR_BG = (46, 52, 64)          # nord0
        self.COLOR_GRID_BG = (59, 66, 82)     # nord1
        self.COLOR_GRID_LINE = (76, 86, 106)  # nord3
        self.COLOR_TILE_SHAPE = (136, 192, 208) # nord8
        self.COLOR_CURSOR = (235, 203, 139)   # nord13
        self.COLOR_TEXT = (236, 239, 244)     # nord6
        self.COLOR_CORRECT = (163, 190, 140)  # nord14
        self.COLOR_INCORRECT = (191, 97, 106) # nord11
        self.COLOR_EFFECT = (216, 222, 233)   # nord5

        # --- State Variables ---
        self.grid = None
        self.target_patterns = None
        self.completed_patterns = None
        self.cursor_pos = None
        self.moves_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.visual_effects = [] # List to manage temporary visual effects

        # Initialize state variables
        self.reset()

        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.completed_patterns = set()
        self.visual_effects = []
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        """Generates the tile grid and target patterns."""
        # Define 5 patterns as lists of (row, col) coordinates
        self.target_patterns = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],  # 2x2 top-left square
            [(3, 3), (3, 4), (4, 3), (4, 4)],  # 2x2 bottom-right square
            [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)], # Center horizontal line
            [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)], # Center vertical line
            [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]  # Main anti-diagonal
        ]
        
        # Scramble the grid. A solved state has all tiles with orientation 0 (up).
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        num_scrambles = self.np_random.integers(25, 40)
        for _ in range(num_scrambles):
            r, c = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()
            # Ensure we don't just undo the last rotation
            if self.grid[r, c] == 0:
                self.grid[r, c] = self.np_random.choice([1, 2, 3])
            else:
                self.grid[r, c] = (self.grid[r, c] + self.np_random.choice([-1, 1])) % 4
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        # --- Action Handling ---
        rotation = 0
        if space_held: rotation = 1  # Clockwise
        elif shift_held: rotation = -1 # Counter-clockwise

        if rotation != 0:
            # --- Rotation Logic ---
            old_correct_tiles = self._count_correct_tiles_in_patterns()
            
            r, c = self.cursor_pos
            self.grid[r, c] = (self.grid[r, c] + rotation) % 4
            self.moves_remaining -= 1
            
            # Add a visual effect for the rotation
            # format: [type, pos, radius, max_radius, lifetime]
            self.visual_effects.append(["circle_out", (r, c), 0, self.TILE_SIZE // 2, 10])

            new_correct_tiles = self._count_correct_tiles_in_patterns()
            current_completed = self._get_currently_completed_patterns()
            
            # Reward for getting closer/further from solution
            reward += (new_correct_tiles - old_correct_tiles) * 0.1
            
            # Reward for newly completed patterns
            newly_completed = current_completed - self.completed_patterns
            for pattern_idx in newly_completed:
                reward += 10.0
                # Add a flash effect for the completed pattern UI
                # format: [type, id, lifetime]
                self.visual_effects.append(["flash", pattern_idx, 20])
            
            self.completed_patterns = current_completed

        elif movement > 0:
            # --- Movement Logic ---
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dr) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dc) % self.GRID_SIZE

        self.steps += 1
        self.score += reward

        # --- Termination Check ---
        if len(self.completed_patterns) == self.NUM_PATTERNS:
            reward += 100.0  # Win bonus
            self.score += 100.0
            terminated = True
            self.game_over = True
        elif self.moves_remaining <= 0:
            reward -= 50.0  # Loss penalty
            self.score -= 50.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

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
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "completed_patterns": len(self.completed_patterns),
        }

    def _render_game(self):
        # Draw main grid background
        grid_rect = pygame.Rect(self.GRID_ORIGIN[0], self.GRID_ORIGIN[1], self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw tiles and grid lines
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_rect = pygame.Rect(
                    self.GRID_ORIGIN[0] + c * self.TILE_SIZE,
                    self.GRID_ORIGIN[1] + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                self._draw_tile(self.screen, tile_rect.center, self.grid[r, c])
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, tile_rect, 1)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_ORIGIN[0] + cursor_c * self.TILE_SIZE,
            self.GRID_ORIGIN[1] + cursor_r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Process and draw visual effects
        self._update_and_draw_effects()

    def _draw_tile(self, surface, center, orientation):
        """Draws a single triangular tile indicating orientation."""
        cx, cy = center
        size = self.TILE_SIZE * 0.35
        
        # Points for a triangle pointing up (orientation 0)
        p1 = (cx, cy - size)
        p2 = (cx - size, cy + size * 0.5)
        p3 = (cx + size, cy + size * 0.5)
        
        angle = orientation * -90  # Pygame rotates CCW, so use negative for CW
        
        def rotate(p, angle_deg, origin):
            ox, oy = origin
            px, py = p
            angle_rad = math.radians(angle_deg)
            qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
            qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
            return int(qx), int(qy)

        points = [rotate(p1, angle, center), rotate(p2, angle, center), rotate(p3, angle, center)]
        pygame.gfxdraw.aapolygon(surface, points, self.COLOR_TILE_SHAPE)
        pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_TILE_SHAPE)

    def _render_ui(self):
        # --- Main Info Panel ---
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 60))

        # --- Pattern Previews ---
        preview_title = self.font_small.render("Target Patterns", True, self.COLOR_TEXT)
        self.screen.blit(preview_title, (20, 120))
        
        preview_size = 60
        preview_padding = 10
        start_y = 150
        
        for i, pattern in enumerate(self.target_patterns):
            is_complete = i in self.completed_patterns
            preview_y = start_y + i * (preview_size + preview_padding)
            
            # Check for flash effect
            is_flashing = any(fx[0] == 'flash' and fx[1] == i for fx in self.visual_effects)
            border_color = self.COLOR_CORRECT if is_complete or is_flashing else self.COLOR_GRID_LINE
            
            # Draw preview box
            preview_rect = pygame.Rect(20, preview_y, preview_size, preview_size)
            pygame.draw.rect(self.screen, self.COLOR_GRID_BG, preview_rect)
            pygame.draw.rect(self.screen, border_color, preview_rect, 2 if is_complete or is_flashing else 1)
            
            # Draw mini-grid inside
            mini_tile_size = preview_size / self.GRID_SIZE
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    is_in_pattern = (r, c) in pattern
                    if is_in_pattern:
                        is_correct = self.grid[r, c] == 0
                        color = self.COLOR_CORRECT if is_correct else self.COLOR_INCORRECT
                        mini_rect = pygame.Rect(
                            20 + c * mini_tile_size,
                            preview_y + r * mini_tile_size,
                            math.ceil(mini_tile_size), math.ceil(mini_tile_size)
                        )
                        pygame.draw.rect(self.screen, color, mini_rect)
    
    def _update_and_draw_effects(self):
        new_effects = []
        for effect in self.visual_effects:
            effect[-1] -= 1  # Decrement lifetime
            if effect[-1] > 0:
                # Draw effect
                if effect[0] == "circle_out":
                    _, (r, c), radius, max_radius, lifetime = effect
                    center_x = self.GRID_ORIGIN[0] + c * self.TILE_SIZE + self.TILE_SIZE // 2
                    center_y = self.GRID_ORIGIN[1] + r * self.TILE_SIZE + self.TILE_SIZE // 2
                    
                    current_radius = int(max_radius * (1 - lifetime / 10))
                    if current_radius > 0:
                        alpha = int(255 * (lifetime / 10))
                        color = self.COLOR_EFFECT + (alpha,)
                        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, current_radius, color)
                
                # Flash effect is handled in UI render, just keep it in the list
                new_effects.append(effect)
        self.visual_effects = new_effects

    def _count_correct_tiles_in_patterns(self):
        """Counts total number of correctly oriented tiles across all defined patterns."""
        count = 0
        for pattern in self.target_patterns:
            for r, c in pattern:
                if self.grid[r, c] == 0:  # Correct orientation is 0 (up)
                    count += 1
        return count

    def _get_currently_completed_patterns(self):
        """Returns a set of indices for currently completed patterns."""
        completed = set()
        for i, pattern in enumerate(self.target_patterns):
            if all(self.grid[r, c] == 0 for r, c in pattern):
                completed.add(i)
        return completed

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Tile Rotator")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    print(env.game_description)
    print(env.user_guide)
    
    running = True
    while running:
        # --- Action Mapping for Human Player ---
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    terminated = False
                    print("--- Game Reset ---")

        if not terminated:
            action = [movement, space, shift]
            # Only step if an action is taken
            if any(a > 0 for a in action):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_remaining']}")
                if terminated:
                    print(f"--- Game Over --- Final Score: {info['score']:.2f}")

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    env.close()