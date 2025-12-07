
# Generated: 2025-08-27T22:24:49.464565
# Source Brief: brief_03114.md
# Brief Index: 3114

        
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
        "Controls: Arrows to move selector. Shift to cycle drop direction. Space to drop crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Isometric puzzle game. Push crystals to create lines of 5 of the same color. You have 3 moves per level."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 10
    MOVES_PER_EPISODE = 3

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (60, 70, 80)
    COLOR_UI_TEXT = (220, 220, 220)
    CRYSTAL_COLORS = {
        1: {"base": (255, 50, 50), "light": (255, 150, 150), "dark": (180, 20, 20)}, # Red
        2: {"base": (50, 255, 50), "light": (150, 255, 150), "dark": (20, 180, 20)}, # Green
        3: {"base": (50, 100, 255), "light": (150, 180, 255), "dark": (20, 50, 180)}, # Blue
    }
    SELECTOR_COLOR = (255, 255, 0)
    DROP_ARROW_COLOR = (255, 165, 0)

    # Isometric rendering
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80

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
        self.font_big = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables
        self.grid = None
        self.selector_pos = None
        self.drop_direction_idx = None
        self.drop_directions = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_matches = None
        self.animation_timer = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.moves_left = self.MOVES_PER_EPISODE
        self.game_over = False
        self.selector_pos = (self.GRID_WIDTH // 2, 0)
        self.drop_direction_idx = 0
        self.drop_directions = [(-1, 0), (1, 0), (0, 1)] # Left, Right, Down
        self.last_matches = []
        self.animation_timer = 0
        
        self._initialize_grid()
        
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        while True:
            self.grid = self.np_random.integers(0, len(self.CRYSTAL_COLORS) + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            
            # Fill top rows with empty space
            self.grid[:self.GRID_HEIGHT // 2, :] = 0
            
            # Apply gravity to settle initial state
            self._apply_gravity()

            # Ensure no pre-existing long alignments
            initial_alignments = self._find_all_alignments()
            if not any(len(line) >= 4 for line in initial_alignments):
                break

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        terminated = self.game_over

        if terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle player input ---
        # 1. Cycle drop direction (if shift is pressed)
        if shift_held:
            self.drop_direction_idx = (self.drop_direction_idx + 1) % len(self.drop_directions)

        # 2. Move selector
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            new_x = (self.selector_pos[0] + dx) % self.GRID_WIDTH
            new_y = (self.selector_pos[1] + dy) % self.GRID_HEIGHT
            self.selector_pos = (new_x, new_y)

        self.last_matches = []
        self.animation_timer = 0

        # 3. Execute drop (if space is pressed) - this is a "move"
        if space_held:
            was_valid_move, moved_crystals = self._execute_drop()
            if was_valid_move:
                self.moves_left -= 1
                
                # Check for new alignments and calculate reward
                new_alignments = self._find_new_alignments(moved_crystals)
                
                if not new_alignments:
                    reward = -0.2 # Penalty for move with no alignment
                else:
                    alignment_counts = {5: 0, 4: 0, 3: 0, 2: 0}
                    all_matched_coords = set()
                    for line in new_alignments:
                        alignment_counts[len(line)] += 1
                        all_matched_coords.update(line)
                    
                    self.last_matches = list(all_matched_coords)
                    self.animation_timer = 15 # frames for flash

                    # Calculate reward based on spec
                    if alignment_counts[5] > 0:
                        reward += 100
                        self.game_over = True
                    if alignment_counts[4] > 0:
                        reward += 10 * alignment_counts[4]
                    if alignment_counts[3] > 0:
                        reward += 1 * alignment_counts[3]
                    if alignment_counts[2] > 0:
                        reward += 1 * alignment_counts[2]

                self.score += reward
            else:
                reward = -1 # Penalty for attempting an invalid move
        
        if self.moves_left <= 0:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _execute_drop(self):
        sel_x, sel_y = self.selector_pos
        crystal_color = self.grid[sel_y, sel_x]

        if crystal_color == 0:
            return False, set() # Cannot move an empty space

        drop_dir = self.drop_directions[self.drop_direction_idx]
        target_x, target_y = sel_x + drop_dir[0], sel_y + drop_dir[1]

        # Check boundaries and if target is occupied
        if not (0 <= target_x < self.GRID_WIDTH and 0 <= target_y < self.GRID_HEIGHT):
            return False, set()
        if self.grid[target_y, target_x] != 0:
            return False, set()

        # Perform the move
        self.grid[sel_y, sel_x] = 0
        self.grid[target_y, target_x] = crystal_color
        
        # Apply gravity
        moved_crystals = self._apply_gravity()

        return True, moved_crystals

    def _apply_gravity(self):
        moved_crystals = set()
        for x in range(self.GRID_WIDTH):
            empty_y = -1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] == 0 and empty_y == -1:
                    empty_y = y
                elif self.grid[y, x] != 0 and empty_y != -1:
                    # Drop crystal to the empty spot
                    self.grid[empty_y, x] = self.grid[y, x]
                    self.grid[y, x] = 0
                    moved_crystals.add((x, empty_y))
                    empty_y -= 1
        return moved_crystals

    def _find_new_alignments(self, moved_crystals):
        all_new_alignments = set()
        
        # Also check crystals that were above the moved ones, as they fell
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y,x] != 0:
                    moved_crystals.add((x,y))

        for x, y in moved_crystals:
            color = self.grid[y, x]
            if color == 0: continue

            # Check 4 axes (Horizontal, Vertical, Diagonal /, Diagonal \)
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                line = [(x, y)]
                # Forward
                for i in range(1, self.GRID_WIDTH):
                    nx, ny = x + i * dx, y + i * dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == color:
                        line.append((nx, ny))
                    else:
                        break
                # Backward
                for i in range(1, self.GRID_WIDTH):
                    nx, ny = x - i * dx, y - i * dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == color:
                        line.append((nx, ny))
                    else:
                        break
                
                if len(line) >= 2:
                    # Canonical representation: sorted tuple of coordinates
                    canonical_line = tuple(sorted(line))
                    all_new_alignments.add(canonical_line)
        
        # Filter out sub-lines
        final_alignments = []
        sorted_alignments = sorted(list(all_new_alignments), key=len, reverse=True)
        added_coords = set()

        for line in sorted_alignments:
            is_sub_line = False
            for coord in line:
                if coord in added_coords:
                    is_sub_line = True
                    break
            if not is_sub_line:
                final_alignments.append(line)
                added_coords.update(line)

        return final_alignments

    def _find_all_alignments(self):
        # Used for initialization check
        all_alignments = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.grid[y, x]
                if color == 0: continue
                for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    line = [(x, y)]
                    for i in range(1, 5):
                        nx, ny = x + i * dx, y + i * dy
                        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == color:
                            line.append((nx, ny))
                        else:
                            break
                    if len(line) >= 2:
                        all_alignments.add(tuple(sorted(line)))
        return [list(line) for line in all_alignments]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for y in range(self.GRID_HEIGHT + 1):
            start = self._world_to_iso(0, y)
            end = self._world_to_iso(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_WIDTH + 1):
            start = self._world_to_iso(x, 0)
            end = self._world_to_iso(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Draw crystals
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_id = self.grid[y, x]
                if color_id != 0:
                    is_flashing = self.animation_timer > 0 and (x, y) in self.last_matches
                    self._draw_iso_cube(self.screen, x, y, color_id, is_flashing)
        
        # Draw selector and drop indicator
        self._draw_selector()

        if self.animation_timer > 0:
            self.animation_timer -= 1

    def _draw_selector(self):
        sel_x, sel_y = self.selector_pos
        
        # Draw selector highlight
        points = [
            self._world_to_iso(sel_x, sel_y),
            self._world_to_iso(sel_x + 1, sel_y),
            self._world_to_iso(sel_x + 1, sel_y + 1),
            self._world_to_iso(sel_x, sel_y + 1)
        ]
        
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        color = (255, 255, int(100 + 155 * pulse))
        pygame.draw.lines(self.screen, color, True, points, 2)

        # Draw drop direction arrow if a crystal is selected
        if self.grid[sel_y, sel_x] != 0:
            drop_dir = self.drop_directions[self.drop_direction_idx]
            start_center_x, start_center_y = self._world_to_iso(sel_x + 0.5, sel_y + 0.5)
            
            end_x, end_y = sel_x + drop_dir[0], sel_y + drop_dir[1]
            end_center_x, end_center_y = self._world_to_iso(end_x + 0.5, end_y + 0.5)

            pygame.draw.line(self.screen, self.DROP_ARROW_COLOR, (start_center_x, start_center_y), (end_center_x, end_center_y), 2)
            # Arrowhead
            angle = math.atan2(end_center_y - start_center_y, end_center_x - start_center_x)
            p1 = (end_center_x - 10 * math.cos(angle - math.pi / 6), end_center_y - 10 * math.sin(angle - math.pi / 6))
            p2 = (end_center_x - 10 * math.cos(angle + math.pi / 6), end_center_y - 10 * math.sin(angle + math.pi / 6))
            pygame.draw.polygon(self.screen, self.DROP_ARROW_COLOR, [(end_center_x, end_center_y), p1, p2])

    def _world_to_iso(self, x, y):
        iso_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, x, y, color_id, is_flashing):
        colors = self.CRYSTAL_COLORS[color_id]
        
        top_color = colors["light"]
        side_color1 = colors["base"]
        side_color2 = colors["dark"]

        if is_flashing and (self.animation_timer // 3) % 2 == 0:
            top_color = (255, 255, 255)
            side_color1 = (230, 230, 230)
            side_color2 = (200, 200, 200)

        top_points = [
            self._world_to_iso(x, y),
            self._world_to_iso(x + 1, y),
            self._world_to_iso(x + 1, y + 1),
            self._world_to_iso(x, y + 1)
        ]
        
        # We need to draw from the bottom up to get correct layering
        # The base of the cube is at y+1
        base_y = self._world_to_iso(x, y + 1)[1]
        
        side1_points = [
            self._world_to_iso(x, y + 1),
            self._world_to_iso(x + 1, y + 1),
            self._world_to_iso(x + 1, y + 1)[0], base_y + self.TILE_HEIGHT_HALF * 2,
            self._world_to_iso(x, y + 1)[0], base_y + self.TILE_HEIGHT_HALF * 2,
        ]

        # This logic is flawed for isometric. Let's draw the 3 visible faces.
        p_top_left = self._world_to_iso(x, y)
        p_top_right = self._world_to_iso(x + 1, y)
        p_bottom_right = self._world_to_iso(x + 1, y + 1)
        p_bottom_left = self._world_to_iso(x, y + 1)
        p_center = self._world_to_iso(x+0.5, y+0.5)
        p_center = (p_center[0], p_center[1] + self.TILE_HEIGHT_HALF) # approx base center

        # Draw left face
        pygame.gfxdraw.filled_polygon(surface, [p_top_left, p_bottom_left, (p_bottom_left[0], p_bottom_left[1]+self.TILE_HEIGHT_HALF*2), (p_top_left[0], p_top_left[1]+self.TILE_HEIGHT_HALF*2)], side_color2)
        pygame.gfxdraw.aapolygon(surface, [p_top_left, p_bottom_left, (p_bottom_left[0], p_bottom_left[1]+self.TILE_HEIGHT_HALF*2), (p_top_left[0], p_top_left[1]+self.TILE_HEIGHT_HALF*2)], side_color2)

        # Draw right face
        pygame.gfxdraw.filled_polygon(surface, [p_top_right, p_bottom_right, (p_bottom_right[0], p_bottom_right[1]+self.TILE_HEIGHT_HALF*2), (p_top_right[0], p_top_right[1]+self.TILE_HEIGHT_HALF*2)], side_color1)
        pygame.gfxdraw.aapolygon(surface, [p_top_right, p_bottom_right, (p_bottom_right[0], p_bottom_right[1]+self.TILE_HEIGHT_HALF*2), (p_top_right[0], p_top_right[1]+self.TILE_HEIGHT_HALF*2)], side_color1)

        # Draw top face
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)


    def _render_ui(self):
        # Score
        score_text = self.font_big.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Moves left
        moves_text = self.font_big.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            won = any(len(line) >= 5 for line in self._find_all_alignments())
            msg = "YOU WIN!" if won else "GAME OVER"
            color = (100, 255, 100) if won else (255, 100, 100)
            
            over_text = self.font_big.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    
    running = True
    while running:
        # --- Action mapping for human play ---
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
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
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
        
        action = [movement, space, shift]
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}")
            if done:
                print(f"Episode finished. Final Score: {info['score']:.2f}")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS
        
    env.close()