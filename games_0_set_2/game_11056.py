import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:22:27.234603
# Source Brief: brief_01056.md
# Brief Index: 1056
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    GameEnv: Manipulate collapsing tiles through portals to create matching sets.

    The player controls a selector on a grid of tiles. Each tile has a state (color).
    Activating a tile triggers a "portal", changing the state of its orthogonal neighbors.
    The goal is to make all tiles "stable". A tile is stable if it has at least one
    neighbor of the same color. The game is won when the entire grid is stable.
    Difficulty increases with each win by growing the grid and adding more tile states.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate a grid of tiles using portals to create matching sets. A tile becomes stable "
        "when it has a neighbor of the same color; stabilize the entire grid to win."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selector. Press space to activate a portal, "
        "changing the color of adjacent tiles. Press shift to reset the puzzle."
    )
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = pygame.Color("#1a1c2c")
    COLOR_GRID = pygame.Color("#3a3c4c")
    COLOR_TEXT = pygame.Color("#e0e0e0")
    COLOR_SELECTOR = pygame.Color("#ffff00")
    TILE_PALETTE = [
        pygame.Color("#26c6da"),  # Cyan
        pygame.Color("#ec407a"),  # Pink
        pygame.Color("#ffee58"),  # Yellow
        pygame.Color("#ff7043"),  # Orange
        pygame.Color("#66bb6a"),  # Green
        pygame.Color("#7e57c2"),  # Deep Purple
    ]

    # Game Parameters
    MAX_EPISODE_STEPS = 1000
    MAX_GRID_SIZE = 10
    MAX_TILE_STATES = 6
    
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
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 28, bold=True)
        
        # --- Game State Attributes ---
        self.current_level = -1
        self.won_last_game = False
        self.grid_rows = 0
        self.grid_cols = 0
        self.num_tile_states = 0
        self.grid = None
        self.selector_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.stability_cache = (0, 0)
        self.active_animations = deque()

        # --- Initial Reset ---
        # The first call to reset will set current_level to 0
        self.reset()
        
        # --- Validation ---
        # self.validate_implementation() # Commented out for submission
    
    def _update_level_parameters(self):
        """Calculates grid size and tile states based on the current level."""
        self.grid_rows = min(self.MAX_GRID_SIZE, 4 + 2 * (self.current_level // 2))
        self.grid_cols = self.grid_rows
        self.num_tile_states = min(self.MAX_TILE_STATES, 3 + self.current_level // 3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.won_last_game:
            self.current_level += 1
        elif self.current_level == -1: # First run
             self.current_level = 0
        
        self._update_level_parameters()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won_last_game = False
        self.last_space_held = False
        self.selector_pos = [self.grid_rows // 2, self.grid_cols // 2]
        self.active_animations.clear()

        self._generate_puzzle()
        self.stability_cache = self._check_stability()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        """Creates a solvable puzzle by starting from a solved state and working backwards."""
        # A "solved" state is one where all tiles are stable. The simplest is a uniform grid.
        self.grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        
        # Apply a number of inverse operations to shuffle the grid
        num_shuffles = int((self.grid_rows * self.grid_cols) * 1.5)
        for _ in range(num_shuffles):
            r = self.np_random.integers(0, self.grid_rows)
            c = self.np_random.integers(0, self.grid_cols)
            self._activate_portal([r, c], inverse=True)

    def _activate_portal(self, pos, inverse=False):
        """Changes the state of neighbors of the given tile position."""
        r, c = pos
        delta = -1 if inverse else 1
        
        # Orthogonal neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                self.grid[nr, nc] = (self.grid[nr, nc] + delta) % self.num_tile_states
                # Add a visual effect for the tile change
                if not inverse:
                    self.active_animations.append({'type': 'flash', 'pos': (nr, nc), 'duration': 0.2, 'timer': 0.2})

    def _check_stability(self):
        """
        Calculates grid stability.
        Returns: (total_stable_tiles, num_2x2_sets)
        """
        is_stable = np.zeros_like(self.grid, dtype=bool)
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                current_state = self.grid[r, c]
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                        if self.grid[nr, nc] == current_state:
                            is_stable[r, c] = True
                            break # Found a match, tile is stable
        
        num_2x2_sets = 0
        for r in range(self.grid_rows - 1):
            for c in range(self.grid_cols - 1):
                state = self.grid[r, c]
                if (state == self.grid[r+1, c] and
                    state == self.grid[r, c+1] and
                    state == self.grid[r+1, c+1]):
                    num_2x2_sets += 1
                    
        return np.sum(is_stable), num_2x2_sets

    def step(self, action):
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        
        # --- Handle Actions ---
        if shift_action:
            # SFX: Play puzzle_restart sound
            reward = -100  # Penalty for giving up
            self.score += reward
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()
            
        # Movement
        if movement == 1: self.selector_pos[0] -= 1  # Up
        elif movement == 2: self.selector_pos[0] += 1  # Down
        elif movement == 3: self.selector_pos[1] -= 1  # Left
        elif movement == 4: self.selector_pos[1] += 1  # Right
        # SFX: Play cursor_move sound if movement != 0

        # Wrap selector position
        self.selector_pos[0] %= self.grid_rows
        self.selector_pos[1] %= self.grid_cols

        # Activation (on rising edge of space press)
        space_pressed = space_action and not self.last_space_held
        old_stable_count, old_2x2_sets = self.stability_cache
        
        if space_pressed:
            # SFX: Play portal_activate sound
            self._activate_portal(self.selector_pos)
            self.active_animations.append({
                'type': 'burst', 'pos': self.selector_pos, 'duration': 0.5, 'timer': 0.5
            })

        self.last_space_held = space_action

        # --- Calculate Rewards & State ---
        new_stable_count, new_2x2_sets = self._check_stability()
        self.stability_cache = (new_stable_count, new_2x2_sets)
        
        if space_pressed:
            stability_change = new_stable_count - old_stable_count
            reward += max(0, stability_change) * 1.0 + min(0, stability_change) * 0.1
            
            set_bonus = (new_2x2_sets - old_2x2_sets) * 5.0
            reward += set_bonus
        
        self.score += reward
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.won_last_game:
                # SFX: Play puzzle_win sound
                terminal_reward = 100
            else: # Max steps reached
                # SFX: Play puzzle_fail sound
                terminal_reward = -10
            reward += terminal_reward
            self.score += terminal_reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        stable_count, _ = self.stability_cache
        total_tiles = self.grid_rows * self.grid_cols
        
        win = stable_count == total_tiles
        loss_timeout = self.steps >= self.MAX_EPISODE_STEPS
        
        if win:
            self.won_last_game = True
        
        self.game_over = win or loss_timeout
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Calculate Grid Layout ---
        screen_w, screen_h = self.screen.get_size()
        grid_area_size = min(screen_w - 40, screen_h - 80)
        self.cell_size = grid_area_size / self.grid_rows
        grid_w = self.cell_size * self.grid_cols
        grid_h = self.cell_size * self.grid_rows
        self.grid_topleft = ((screen_w - grid_w) / 2, (screen_h - grid_h + 40) / 2)

        # --- Draw Tiles and Grid ---
        is_stable_map = self._get_stability_map()
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x = self.grid_topleft[0] + c * self.cell_size
                y = self.grid_topleft[1] + r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                state = self.grid[r, c]
                base_color = self.TILE_PALETTE[state % len(self.TILE_PALETTE)]
                
                if is_stable_map[r, c]:
                    # Stable: bright, pulsating
                    pulse = (math.sin(pygame.time.get_ticks() * 0.002) + 1) / 2
                    tile_color = base_color.lerp(pygame.Color("white"), 0.2 + pulse * 0.2)
                    pygame.draw.rect(self.screen, tile_color, rect)
                else:
                    # Unstable: desaturated
                    tile_color = base_color.lerp(self.COLOR_BG, 0.6)
                    pygame.draw.rect(self.screen, tile_color, rect)

        # Draw grid lines over tiles
        for i in range(self.grid_rows + 1):
            start_pos = (self.grid_topleft[0], self.grid_topleft[1] + i * self.cell_size)
            end_pos = (self.grid_topleft[0] + grid_w, self.grid_topleft[1] + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)
        for i in range(self.grid_cols + 1):
            start_pos = (self.grid_topleft[0] + i * self.cell_size, self.grid_topleft[1])
            end_pos = (self.grid_topleft[0] + i * self.cell_size, self.grid_topleft[1] + grid_h)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)

        # --- Draw Selector ---
        sel_r, sel_c = self.selector_pos
        sel_x = self.grid_topleft[0] + sel_c * self.cell_size
        sel_y = self.grid_topleft[1] + sel_r * self.cell_size
        sel_rect = pygame.Rect(sel_x, sel_y, self.cell_size, self.cell_size)
        
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        width = int(2 + pulse * 3)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, sel_rect, width, border_radius=4)
        
        # --- Draw Animations ---
        self._update_and_draw_animations()

    def _get_stability_map(self):
        """Generates a boolean map of stable tiles for rendering."""
        is_stable = np.zeros_like(self.grid, dtype=bool)
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                current_state = self.grid[r, c]
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and self.grid[nr, nc] == current_state:
                        is_stable[r, c] = True
                        break
        return is_stable

    def _update_and_draw_animations(self):
        """Processes and renders all active animations like bursts and flashes."""
        # Use a fixed delta time for consistency if needed, but for simple timers it's ok
        # dt = self.clock.get_time() / 1000.0
        
        new_animations = deque()
        while self.active_animations:
            anim = self.active_animations.popleft()
            anim['timer'] -= 1 / 30.0 # Assume 30 FPS for visual updates
            
            if anim['timer'] > 0:
                progress = 1 - (anim['timer'] / anim['duration'])
                
                if anim['type'] == 'burst':
                    r, c = anim['pos']
                    center_x = int(self.grid_topleft[0] + (c + 0.5) * self.cell_size)
                    center_y = int(self.grid_topleft[1] + (r + 0.5) * self.cell_size)
                    radius = int(progress * self.cell_size * 1.5)
                    alpha = int(255 * (1 - progress))
                    if alpha > 0 and radius > 0:
                        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (*self.COLOR_SELECTOR[:3], alpha))
                        
                elif anim['type'] == 'flash':
                    r, c = anim['pos']
                    x = self.grid_topleft[0] + c * self.cell_size
                    y = self.grid_topleft[1] + r * self.cell_size
                    flash_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    alpha = int(255 * (1 - progress**2)) # Fast fade out
                    flash_surface.fill((255, 255, 255, alpha))
                    self.screen.blit(flash_surface, (x, y))

                new_animations.append(anim)
        self.active_animations = new_animations

    def _render_ui(self):
        stable_count, _ = self.stability_cache
        total_tiles = self.grid_rows * self.grid_cols
        
        # Title
        title_surf = self.font_title.render("Quantum Stabilizer", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (self.screen.get_width() // 2 - title_surf.get_width() // 2, 10))

        # Left Info
        level_text = f"Level: {self.current_level + 1}"
        score_text = f"Score: {self.score:.1f}"
        level_surf = self.font_main.render(level_text, True, self.COLOR_TEXT)
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (20, self.screen.get_height() - 30))
        self.screen.blit(score_surf, (150, self.screen.get_height() - 30))
        
        # Right Info
        steps_text = f"Step: {self.steps}/{self.MAX_EPISODE_STEPS}"
        stable_text = f"Stable: {stable_count}/{total_tiles}"
        steps_surf = self.font_main.render(steps_text, True, self.COLOR_TEXT)
        stable_surf = self.font_main.render(stable_text, True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (self.screen.get_width() - stable_surf.get_width() - 200, self.screen.get_height() - 30))
        self.screen.blit(stable_surf, (self.screen.get_width() - stable_surf.get_width() - 20, self.screen.get_height() - 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "stability": self.stability_cache[0] / (self.grid_rows * self.grid_cols) if (self.grid_rows * self.grid_cols) > 0 else 0
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Set a real video driver to see the game window
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Quantum Stabilizer")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0
        
        # Get keyboard state for continuous presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Level: {info['level']+1}")
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()