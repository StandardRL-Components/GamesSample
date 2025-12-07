
# Generated: 2025-08-27T17:19:04.289083
# Source Brief: brief_01488.md
# Brief Index: 1488

        
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
        "Controls: Arrow keys to move the cursor. Spacebar to place a light-redirecting crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Place crystals to redirect a light beam and illuminate all target crystals before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    CELL_SIZE = 40
    
    NUM_TARGET_CRYSTALS = 10
    INITIAL_PLACEMENT_CRYSTALS = 15
    MAX_STEPS = 600

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_LIGHT_SOURCE = (255, 255, 200)
    COLOR_PLACED_CRYSTAL = (255, 200, 0)
    COLOR_TARGET_UNLIT = (100, 150, 255)
    COLOR_TARGET_LIT = (0, 255, 255)
    COLOR_LIGHT_BEAM = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Internal state variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.light_source_pos = None
        self.target_crystals = None
        self.placed_crystals = None
        self.lit_crystals = None
        self.light_path_segments = None
        self.remaining_placement_crystals = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None
        
        # Initialize state variables
        self.reset()

    def _grid_to_pixel(self, gx, gy):
        """Converts grid coordinates to pixel coordinates for the center of the cell."""
        return (
            int(gx * self.CELL_SIZE + self.CELL_SIZE / 2),
            int(gy * self.CELL_SIZE + self.CELL_SIZE / 2)
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.remaining_placement_crystals = self.INITIAL_PLACEMENT_CRYSTALS
        
        # Initialize grid
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int) # 0: empty, 1: target, 2: placed, 3: source
        
        # Place light source (fixed on the left edge)
        source_gy = self.np_random.integers(1, self.GRID_ROWS - 1)
        self.light_source_pos = (0, source_gy)
        self.grid[self.light_source_pos] = 3

        # Place target crystals
        self.target_crystals = set()
        # Ensure target crystals are not on the very edge columns to make it solvable
        while len(self.target_crystals) < self.NUM_TARGET_CRYSTALS:
            gx = self.np_random.integers(1, self.GRID_COLS - 1)
            gy = self.np_random.integers(0, self.GRID_ROWS)
            if self.grid[gx, gy] == 0:
                self.target_crystals.add((gx, gy))
                self.grid[gx, gy] = 1

        self.placed_crystals = set()
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self._recalculate_light_paths()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        # shift_held = action[2] == 1 is unused per brief

        self.steps += 1
        reward = -0.01  # Small penalty for each step

        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
        
        # 2. Handle crystal placement
        placed_new_crystal = False
        if space_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            if self.grid[cursor_tuple] == 0 and self.remaining_placement_crystals > 0:
                # Place a new crystal
                self.placed_crystals.add(cursor_tuple)
                self.grid[cursor_tuple] = 2
                self.remaining_placement_crystals -= 1
                placed_new_crystal = True
                # Sound placeholder: place_crystal.wav

        # 3. Recalculate light paths and score if state changed
        if placed_new_crystal:
            old_lit_count = len(self.lit_crystals)
            self._recalculate_light_paths()
            new_lit_count = len(self.lit_crystals)
            
            newly_lit_this_step = new_lit_count - old_lit_count
            if newly_lit_this_step > 0:
                reward += newly_lit_this_step * 5.0
                # Sound placeholder: crystal_lit.wav

        self.score += reward

        # 4. Check for termination
        terminated = False
        win = len(self.lit_crystals) == self.NUM_TARGET_CRYSTALS
        timeout = self.steps >= self.MAX_STEPS
        no_more_moves = self.remaining_placement_crystals == 0 and not win

        if win:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
            # Sound placeholder: win.wav
        elif timeout or no_more_moves:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
            # Sound placeholder: lose.wav
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _recalculate_light_paths(self):
        self.lit_crystals = set()
        self.light_path_segments = []
        
        # Queue of beams to trace: (start_pos, direction_vector)
        beams_to_trace = [ (self.light_source_pos, (1, 0)) ]
        
        # To prevent infinite reflection loops
        processed_reflections = set()

        while beams_to_trace:
            start_pos, direction = beams_to_trace.pop(0)
            
            current_pos = start_pos
            for _ in range(max(self.GRID_COLS, self.GRID_ROWS)): # Safety break
                next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                
                # Check grid boundaries
                if not (0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS):
                    self.light_path_segments.append((start_pos, current_pos))
                    break

                cell_content = self.grid[next_pos]
                
                if cell_content == 1: # Target crystal
                    self.lit_crystals.add(next_pos)
                    current_pos = next_pos # Light passes through
                    continue

                if cell_content == 2: # Placed (reflector) crystal
                    self.light_path_segments.append((start_pos, next_pos))
                    
                    # Reflection logic: 90 degree turn
                    # Horizontal (dx, 0) -> Vertical (0, dx)
                    # Vertical (0, dy) -> Horizontal (-dy, 0)
                    new_direction = (-direction[1], direction[0])
                    
                    reflection_key = (next_pos, new_direction)
                    if reflection_key not in processed_reflections:
                        processed_reflections.add(reflection_key)
                        beams_to_trace.append((next_pos, new_direction))
                    break
                
                current_pos = next_pos
            else: # Loop finished without break (hit nothing)
                self.light_path_segments.append((start_pos, current_pos))


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw light source
        sx, sy = self._grid_to_pixel(*self.light_source_pos)
        self._draw_glowing_shape(sx, sy, self.CELL_SIZE // 3, self.COLOR_LIGHT_SOURCE, 'rect')

        # Draw target crystals
        for gx, gy in self.target_crystals:
            px, py = self._grid_to_pixel(gx, gy)
            is_lit = (gx, gy) in self.lit_crystals
            color = self.COLOR_TARGET_LIT if is_lit else self.COLOR_TARGET_UNLIT
            self._draw_glowing_shape(px, py, self.CELL_SIZE // 4, color, 'diamond', glow=is_lit)

        # Draw placed crystals
        for gx, gy in self.placed_crystals:
            px, py = self._grid_to_pixel(gx, gy)
            self._draw_glowing_shape(px, py, self.CELL_SIZE // 4, self.COLOR_PLACED_CRYSTAL, 'circle', glow=True)

        # Draw light beams
        for start_g, end_g in self.light_path_segments:
            start_p = self._grid_to_pixel(*start_g)
            end_p = self._grid_to_pixel(*end_g)
            pygame.draw.line(self.screen, self.COLOR_LIGHT_BEAM, start_p, end_p, 3)
            # Add a subtle glow to the beam
            pygame.draw.line(self.screen, (*self.COLOR_LIGHT_BEAM, 50), start_p, end_p, 7)

        # Draw cursor
        cursor_px, cursor_py = self._grid_to_pixel(*self.cursor_pos)
        cursor_rect = pygame.Rect(cursor_px - self.CELL_SIZE/2, cursor_py - self.CELL_SIZE/2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)


    def _draw_glowing_shape(self, x, y, size, color, shape, glow=True):
        if glow:
            for i in range(4, 0, -1):
                glow_color = (*color, 60 // i)
                pygame.gfxdraw.filled_circle(self.screen, x, y, int(size * (1 + i * 0.2)), glow_color)
        
        if shape == 'circle':
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color)
        elif shape == 'rect':
            rect = pygame.Rect(x - size, y - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, rect)
        elif shape == 'diamond':
            points = [
                (x, y - size), (x + size, y),
                (x, y + size), (x - size, y)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)


    def _render_ui(self):
        # Lit crystals count
        lit_text = f"Lit: {len(self.lit_crystals)} / {self.NUM_TARGET_CRYSTALS}"
        text_surf = self.font_small.render(lit_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (10, 10))

        # Remaining placement crystals
        rem_text = f"Crystals: {self.remaining_placement_crystals}"
        text_surf = self.font_small.render(rem_text, True, self.COLOR_PLACED_CRYSTAL)
        self.screen.blit(text_surf, (10, 30))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = f"Time: {time_left}"
        text_surf = self.font_small.render(time_text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)
        
        # Game Over Text
        if self.game_over:
            win = len(self.lit_crystals) == self.NUM_TARGET_CRYSTALS
            msg = "VICTORY!" if win else "FAILURE"
            color = self.COLOR_TARGET_LIT if win else (255, 50, 50)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lit_crystals": len(self.lit_crystals),
            "remaining_crystals": self.remaining_placement_crystals,
            "cursor_pos": list(self.cursor_pos)
        }

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
        
        # Test reset
        obs, info = self.reset(seed=123)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space (after a reset)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment for human play
if __name__ == '__main__':
    import time
    import sys

    # Check for a command-line argument to run validation
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        env_val = GameEnv()
        env_val.validate_implementation()
        env_val.close()
        sys.exit()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    # Track pressed state to only act on key down for spacebar
    space_was_pressed = False

    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    terminated = False
                    space_was_pressed = False

        if terminated:
            # On game over, wait for 'R' to reset
            pass
        else:
            # Key polling for continuous actions (movement)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # For discrete actions like placing a crystal, we want to act on press, not hold.
            space_is_pressed = keys[pygame.K_SPACE]
            if space_is_pressed and not space_was_pressed:
                action[1] = 1
            space_was_pressed = space_is_pressed

            # Step the environment only if an action is taken
            if np.any(action):
                obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate
        
    env.close()