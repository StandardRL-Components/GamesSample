
# Generated: 2025-08-28T04:22:18.404274
# Source Brief: brief_05232.md
# Brief Index: 5232

        
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
        "Use arrow keys to move the cursor. Press space to select a tile, "
        "then move to an adjacent tile and press space again to swap. "
        "Press shift to cancel a selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A vibrant tile-matching puzzle. Swap adjacent tiles to create matches "
        "of 3 or more and clear the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_COLORS = 6
        self.INITIAL_MOVES = 30
        self.MAX_STEPS = 1000

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (50, 60, 70)
        self.TILE_COLORS = [
            (220, 50, 50),   # Red
            (50, 220, 50),   # Green
            (50, 120, 220),  # Blue
            (220, 220, 50),  # Yellow
            (180, 50, 220),  # Purple
            (240, 140, 40),  # Orange
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECT = (50, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        
        # --- Game State (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.selected_tile_pos = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile_pos = None
        self.moves_left = self.INITIAL_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []

        self._generate_solvable_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0.0

        # --- Unpack Action ---
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        if shift_press and self.selected_tile_pos:
            self.selected_tile_pos = None
            # sfx: cancel_selection

        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        if space_press:
            # sfx: select_tile
            if not self.selected_tile_pos:
                self.selected_tile_pos = list(self.cursor_pos)
            else:
                # Attempt a swap - this is the main game action
                reward = self._handle_swap()
                self.selected_tile_pos = None # Reset selection after swap attempt
        
        # --- Check Termination Conditions ---
        if self.moves_left <= 0:
            if not self.game_over: # Apply penalty only once
                reward += -10.0
                self.game_over = True
        
        if np.sum(self.grid) == 0: # All tiles cleared
            if not self.game_over: # Apply reward only once
                reward += 100.0
                self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated
            self._get_info(),
        )

    def _handle_swap(self):
        pos1 = self.selected_tile_pos
        pos2 = self.cursor_pos
        
        # Check for adjacency
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
            # sfx: invalid_swap
            return 0.0 # No penalty for invalid swap, just no action

        self.moves_left -= 1
        # sfx: swap
        
        # Perform swap
        self._swap_tiles(pos1, pos2)
        
        total_reward = 0
        cascade_multiplier = 1.0

        while True:
            matches = self._find_all_matches()
            if not matches:
                # If the initial swap resulted in no match, swap back
                if cascade_multiplier == 1.0:
                    # sfx: no_match
                    self._swap_tiles(pos1, pos2) # Revert swap
                    return -0.2
                else:
                    break # End of cascade

            # sfx: match_found
            match_reward = 0
            all_matched_coords = set()
            for match in matches:
                match_reward += (len(match) - 2) * cascade_multiplier
                for pos in match:
                    all_matched_coords.add(tuple(pos))

            self.score += int(match_reward * 10) # Score is based on reward
            total_reward += match_reward

            self._clear_tiles_and_create_particles(all_matched_coords)
            self._apply_gravity_and_refill()
            
            cascade_multiplier += 0.5 # Increase reward for cascades
        
        return total_reward

    def _swap_tiles(self, pos1, pos2):
        self.grid[pos1[1], pos1[0]], self.grid[pos2[1], pos2[0]] = \
            self.grid[pos2[1], pos2[0]], self.grid[pos1[1], pos1[0]]

    def _find_all_matches(self):
        matches = []
        # Horizontal matches
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                color = self.grid[r, c]
                if color != 0 and color == self.grid[r, c+1] == self.grid[r, c+2]:
                    match = [(c, r), (c+1, r), (c+2, r)]
                    i = c + 3
                    while i < self.GRID_SIZE and self.grid[r, i] == color:
                        match.append((i, r))
                        i += 1
                    matches.append(match)
                    c = i - 1
        # Vertical matches
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                color = self.grid[r, c]
                if color != 0 and color == self.grid[r+1, c] == self.grid[r+2, c]:
                    match = [(c, r), (c, r+1), (c, r+2)]
                    i = r + 3
                    while i < self.GRID_SIZE and self.grid[i, c] == color:
                        match.append((c, i))
                        i += 1
                    matches.append(match)
                    r = i - 1
        return matches

    def _clear_tiles_and_create_particles(self, coords):
        for (c, r) in coords:
            color_index = self.grid[r, c] - 1
            if color_index >= 0:
                self._create_particles((c, r), self.TILE_COLORS[color_index])
            self.grid[r, c] = 0

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            empty_slots = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[r + empty_slots, c] = self.grid[r, c]
                    self.grid[r, c] = 0
            # Refill top
            for r in range(empty_slots):
                self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _generate_solvable_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._find_all_matches(): continue # No initial matches
            if self._has_possible_moves(): break

    def _has_possible_moves(self):
        # Check horizontal swaps
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 1):
                self._swap_tiles((c, r), (c + 1, r))
                if self._find_all_matches():
                    self._swap_tiles((c, r), (c + 1, r)) # Swap back
                    return True
                self._swap_tiles((c, r), (c + 1, r)) # Swap back
        # Check vertical swaps
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 1):
                self._swap_tiles((c, r), (c, r + 1))
                if self._find_all_matches():
                    self._swap_tiles((c, r), (c, r + 1)) # Swap back
                    return True
                self._swap_tiles((c, r), (c, r + 1)) # Swap back
        return False
        
    def _create_particles(self, grid_pos, color):
        cx = self.grid_rect.left + grid_pos[0] * self.tile_size + self.tile_size / 2
        cy = self.grid_rect.top + grid_pos[1] * self.tile_size + self.tile_size / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append([ [cx, cy], vel, lifespan, color ])
            
    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._calculate_layout()
        self._update_particles()
        
        self._draw_grid()
        self._draw_tiles()
        self._draw_particles()
        self._draw_cursor()
        self._draw_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _calculate_layout(self):
        self.grid_render_size = min(self.WIDTH, self.HEIGHT) * 0.9
        self.tile_size = self.grid_render_size / self.GRID_SIZE
        self.grid_rect = pygame.Rect(
            (self.WIDTH - self.grid_render_size) / 2,
            (self.HEIGHT - self.grid_render_size) / 2,
            self.grid_render_size,
            self.grid_render_size,
        )

    def _draw_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.grid_rect.left + i * self.tile_size, self.grid_rect.top)
            end_pos = (self.grid_rect.left + i * self.tile_size, self.grid_rect.bottom)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.grid_rect.left, self.grid_rect.top + i * self.tile_size)
            end_pos = (self.grid_rect.right, self.grid_rect.top + i * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _draw_tiles(self):
        padding = self.tile_size * 0.08
        radius = int(self.tile_size * 0.2)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.grid[r, c]
                if color_index > 0:
                    color = self.TILE_COLORS[color_index - 1]
                    rect = pygame.Rect(
                        self.grid_rect.left + c * self.tile_size + padding,
                        self.grid_rect.top + r * self.tile_size + padding,
                        self.tile_size - 2 * padding,
                        self.tile_size - 2 * padding
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=radius)

    def _draw_cursor(self):
        # Draw primary selection highlight
        if self.selected_tile_pos:
            c, r = self.selected_tile_pos
            rect = pygame.Rect(
                self.grid_rect.left + c * self.tile_size,
                self.grid_rect.top + r * self.tile_size,
                self.tile_size,
                self.tile_size,
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, width=4, border_radius=int(self.tile_size * 0.2))

        # Draw cursor
        c, r = self.cursor_pos
        rect = pygame.Rect(
            self.grid_rect.left + c * self.tile_size,
            self.grid_rect.top + r * self.tile_size,
            self.tile_size,
            self.tile_size,
        )
        
        # Pulsating effect for cursor
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        cursor_surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), cursor_surface.get_rect(), width=3, border_radius=int(self.tile_size * 0.2))
        self.screen.blit(cursor_surface, rect.topleft)

    def _draw_particles(self):
        for p in self.particles:
            pos, _, lifespan, color = p
            size = max(1, (lifespan / 40) * 5)
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), int(size))
            
    def _draw_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Moves
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(moves_text, moves_rect)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if np.sum(self.grid) == 0:
                end_text = "BOARD CLEARED!"
            else:
                end_text = "GAME OVER"
                
            text_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0, 0))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "selected_tile": self.selected_tile_pos
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame window for human play ---
    pygame.display.set_caption("Tile Matcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not done:
        # Action defaults to no-op
        action = [0, 0, 0] # move, space, shift

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    continue

        # Only step if an action was taken
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print("--- GAME OVER ---")
            print(f"Final Score: {info['score']}")
            pygame.time.wait(2000) # Wait 2 seconds before allowing reset
            # Allow restarting
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        done = True # to exit outer loop
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        done = False
                        waiting_for_reset = False
                        print("--- Game Reset ---")


    env.close()