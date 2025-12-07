
# Generated: 2025-08-28T04:05:05.084400
# Source Brief: brief_05136.md
# Brief Index: 5136

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem. "
        "Move the cursor to an adjacent gem and press Space again to swap. "
        "Press Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match 3 or more gems to clear them. Achieve the target number of sets "
        "before you run out of moves. Swaps that don't create a match still "
        "cost a move."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.GEM_TYPES = 5
        self.MAX_MOVES = 10
        self.TARGET_SETS = 7
        self.MAX_STEPS = 1000

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Visuals ---
        self.BOARD_OFFSET_X = (self.WIDTH - self.HEIGHT) // 2
        self.BOARD_OFFSET_Y = 0
        self.CELL_SIZE = self.HEIGHT // self.GRID_SIZE
        self.GEM_RADIUS = int(self.CELL_SIZE * 0.4)
        
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.GEM_HIGHLIGHTS = [pygame.Color(c).lerp((255,255,255), 0.6) for c in self.GEM_COLORS]
        self._precompute_gem_shapes()

        # --- Initialize state variables ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.last_move_dir = None
        self.score = None
        self.moves_remaining = None
        self.sets_matched = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.particles = None
        
        self.reset()
        self.validate_implementation()

    def _precompute_gem_shapes(self):
        """Precomputes vertices for hexagon gems for faster rendering."""
        self.gem_shape_points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6
            x = self.GEM_RADIUS * math.cos(angle)
            y = self.GEM_RADIUS * math.sin(angle)
            self.gem_shape_points.append((x, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.sets_matched = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gem_pos = None
        self.last_move_dir = (0, 0)
        
        self._create_initial_board()

        return self._get_observation(), self._get_info()

    def _create_initial_board(self):
        """Fills the board with gems, ensuring no initial matches."""
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        # Fill bottom 2 rows
        for r in range(self.GRID_SIZE - 2, self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self.grid[r, c] = self.np_random.integers(1, self.GEM_TYPES + 1)
        
        # Resolve any starting matches
        while self._find_matches()[0]:
            matches, _, gems_to_remove = self._find_matches()
            for r, c in gems_to_remove:
                self.grid[r, c] = self.np_random.integers(1, self.GEM_TYPES + 1)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1

        # 1. Handle Input
        self._handle_movement(movement)
        if shift_pressed:
            self.selected_gem_pos = None
            # sfx: deselect_sound
        if space_pressed:
            reward += self._handle_selection_and_swap()
        
        # 2. Check for Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win:
                reward += 100
            else:
                reward -= 100
        
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
            reward -= 100 # Penalize for running out of time

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        """Updates cursor position based on movement action."""
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.last_move_dir = (dx, dy)
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_SIZE

    def _handle_selection_and_swap(self):
        """Manages gem selection and swapping logic."""
        cx, cy = self.cursor_pos
        
        if self.selected_gem_pos is None:
            # Select a gem
            if self.grid[cy, cx] > 0:
                self.selected_gem_pos = [cx, cy]
                # sfx: select_gem_sound
            return 0
        else:
            # Attempt a swap
            sx, sy = self.selected_gem_pos
            tx, ty = sx + self.last_move_dir[0], sy + self.last_move_dir[1]

            # Check if target is the current cursor and is adjacent
            is_adjacent = abs(sx - cx) + abs(sy - cy) == 1
            if not is_adjacent or not (cx == tx and cy == ty):
                # Invalid swap target, deselect
                self.selected_gem_pos = None
                return 0

            # Perform the swap
            self.moves_remaining -= 1
            self.grid[sy, sx], self.grid[ty, tx] = self.grid[ty, tx], self.grid[sy, sx]
            # sfx: swap_sound
            
            # Process matches and cascades
            cascade_reward = self._process_cascades()

            if cascade_reward == 0:
                # No match found from swap
                self.selected_gem_pos = None
                return -0.1
            else:
                self.selected_gem_pos = None
                return cascade_reward

    def _process_cascades(self):
        """Repeatedly find matches, remove gems, and apply gravity until stable."""
        total_reward = 0
        is_first_match_in_swap = True
        while True:
            has_matches, sets_found, gems_to_remove = self._find_matches()
            if not has_matches:
                break
            
            # sfx: match_sound
            
            # Grant rewards
            reward_per_gem = 1
            reward_per_set = 5
            match_reward = len(gems_to_remove) * reward_per_gem + sets_found * reward_per_set
            total_reward += match_reward
            self.score += match_reward
            self.sets_matched += sets_found

            # Create particle effects
            for r, c in gems_to_remove:
                self._create_particles(c, r, self.grid[r, c])
            
            # Remove gems
            for r, c in gems_to_remove:
                self.grid[r, c] = 0
            
            # Apply gravity and fill new gems
            self._apply_gravity()
            self._fill_new_gems()
        
        return total_reward

    def _find_matches(self):
        """Scans the grid for horizontal and vertical matches of 3 or more."""
        gems_to_remove = set()
        sets_found = 0
        
        # Horizontal matches
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                gem_type = self.grid[r, c]
                if gem_type > 0 and gem_type == self.grid[r, c+1] == self.grid[r, c+2]:
                    sets_found += 1
                    match_len = 2
                    while c + match_len < self.GRID_SIZE and self.grid[r, c+match_len] == gem_type:
                        match_len += 1
                    for i in range(match_len):
                        gems_to_remove.add((r, c + i))

        # Vertical matches
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                gem_type = self.grid[r, c]
                if gem_type > 0 and gem_type == self.grid[r+1, c] == self.grid[r+2, c]:
                    sets_found += 1
                    match_len = 2
                    while r + match_len < self.GRID_SIZE and self.grid[r+match_len, c] == gem_type:
                        match_len += 1
                    for i in range(match_len):
                        gems_to_remove.add((r + i, c))
                        
        return len(gems_to_remove) > 0, sets_found, gems_to_remove

    def _apply_gravity(self):
        """Moves gems down to fill empty spaces."""
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] > 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _fill_new_gems(self):
        """Fills empty spaces at the top of the grid with new gems."""
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.GEM_TYPES + 1)
                else:
                    break # Move to next column once a gem is found

    def _create_particles(self, c, r, gem_type):
        """Generates explosion particles for a matched gem."""
        px = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append([ [px, py], vel, color, lifetime, radius])

    def _check_termination(self):
        if self.sets_matched >= self.TARGET_SETS:
            self.win = True
            return True
        if self.moves_remaining <= 0:
            self.win = False
            return True
        return False

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
            "sets_matched": self.sets_matched,
        }

    def _render_game(self):
        self._update_and_draw_particles()
        self._draw_grid()
        self._draw_gems()
        self._draw_cursor_and_selection()

    def _draw_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_x = self.BOARD_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.BOARD_OFFSET_Y), (start_x, self.HEIGHT), 1)
            # Horizontal lines
            start_y = self.BOARD_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_OFFSET_X, start_y), (self.BOARD_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, start_y), 1)

    def _draw_gems(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    center_x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    points = [(p[0] + center_x, p[1] + center_y) for p in self.gem_shape_points]
                    
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.GEM_COLORS[gem_type - 1])
                    pygame.gfxdraw.aapolygon(self.screen, points, self.GEM_HIGHLIGHTS[gem_type - 1])

    def _draw_cursor_and_selection(self):
        # Draw selected gem highlight
        if self.selected_gem_pos is not None:
            sx, sy = self.selected_gem_pos
            rect = pygame.Rect(
                self.BOARD_OFFSET_X + sx * self.CELL_SIZE,
                self.BOARD_OFFSET_Y + sy * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 3)
            
        # Draw cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(
            self.BOARD_OFFSET_X + cx * self.CELL_SIZE,
            self.BOARD_OFFSET_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Pulsing effect for cursor
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 3, border_radius=4)
        self.screen.blit(cursor_surface, rect.topleft)

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0]  # pos.x += vel.x
            p[0][1] += p[1][1]  # pos.y += vel.y
            p[3] -= 1           # lifetime
            p[4] -= 0.1         # radius
            if p[3] <= 0 or p[4] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p[3] / 20))))
                color = (*p[2], alpha)
                temp_surf = pygame.Surface((p[4]*2, p[4]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p[4], p[4]), p[4])
                self.screen.blit(temp_surf, (p[0][0] - p[4], p[0][1] - p[4]))

    def _render_ui(self):
        # --- Left Panel ---
        score_text = self.font_ui.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # --- Right Panel ---
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}/{self.MAX_MOVES}", True, (255, 255, 255))
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

        sets_text = self.font_ui.render(f"Sets: {self.sets_matched}/{self.TARGET_SETS}", True, (255, 255, 255))
        self.screen.blit(sets_text, (self.WIDTH - sets_text.get_width() - 10, 40))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            msg_text = self.font_msg.render(msg, True, color)
            self.screen.blit(msg_text, (self.WIDTH // 2 - msg_text.get_width() // 2, self.HEIGHT // 2 - msg_text.get_height() // 2))

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
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Event handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
                # Since auto_advance is False, we only step on a key press
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action_taken = True
        
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to restart.")

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for player interaction
        
    env.close()