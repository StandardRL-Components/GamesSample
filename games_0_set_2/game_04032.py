
# Generated: 2025-08-28T01:12:20.797741
# Source Brief: brief_04032.md
# Brief Index: 4032

        
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
        "Use arrow keys to move the cursor. Press space to select a gem, then move to an adjacent "
        "gem and press space again to swap. Press shift to cancel a selection."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of three or more. "
        "Score points by creating combos and chain reactions to reach the target score before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_ROWS * self.CELL_SIZE) // 2
        self.NUM_GEM_TYPES = 6
        self.TARGET_SCORE = 500
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECTED = (255, 255, 255)
        self.GEM_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (255, 80, 255),   # Magenta
            (80, 255, 255),   # Cyan
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game state variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.selection_state = None
        self.selected_pos = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_space_held = None
        self.particles = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selection_state = 0  # 0: idle, 1: one gem selected
        self.selected_pos = None
        self.last_space_held = False
        self.particles = []
        
        self._generate_initial_grid()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # Handle cursor movement
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)
        
        # Handle shift to cancel selection
        if shift_held:
            self.selection_state = 0
            self.selected_pos = None

        # Handle space press (on rising edge)
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            if self.selection_state == 0:
                self.selection_state = 1
                self.selected_pos = tuple(self.cursor_pos)
            elif self.selection_state == 1:
                pos1 = self.selected_pos
                pos2 = tuple(self.cursor_pos)
                
                is_adjacent = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
                if pos1 != pos2 and is_adjacent:
                    # Sound placeholder: player attempts a swap
                    reward = self._attempt_swap(pos1, pos2)
                
                self.selection_state = 0
                self.selected_pos = None
        
        self.last_space_held = space_held
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.score >= self.TARGET_SCORE:
            self.game_over = True
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _attempt_swap(self, pos1, pos2):
        self._swap_gems(pos1, pos2)
        matches = self._find_all_matches()

        if not matches:
            # Invalid move, swap back
            # Sound placeholder: invalid move buzzer
            self._swap_gems(pos1, pos2) # Swap back
            return -0.1

        # Valid move, handle matches and cascades
        # Sound placeholder: successful match sound
        total_reward = 0
        chain_level = 0
        while matches:
            # Calculate reward for this set of matches
            for match in self._group_matches(matches):
                match_len = len(match)
                if match_len == 3: total_reward += 1
                elif match_len == 4: total_reward += 2
                else: total_reward += 4
            
            if chain_level > 0:
                # Sound placeholder: chain reaction chime
                total_reward += 0.5 * len(matches) # Bonus for chain reactions
            
            self.score += len(matches) * 10
            
            # Visuals and grid update
            self._create_match_particles(matches)
            self._clear_gems(matches)
            self._apply_gravity()
            self._fill_new_gems()
            
            chain_level += 1
            matches = self._find_all_matches()

        # Check for game over after a successful move
        if not self._has_possible_moves():
            self.game_over = True
            # Sound placeholder: game over sad trombone
            return total_reward - 10 # Return move reward, but add terminal penalty
            
        return total_reward
        
    def _generate_initial_grid(self):
        while True:
            grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS))
            if not self._find_all_matches(grid) and self._has_possible_moves(grid):
                self.grid = grid
                return

    def _find_all_matches(self, grid=None):
        if grid is None:
            grid = self.grid
        
        matched_coords = set()
        # Horizontal matches
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if grid[r, c] == grid[r, c+1] == grid[r, c+2] and grid[r, c] != 0:
                    matched_coords.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical matches
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if grid[r, c] == grid[r+1, c] == grid[r+2, c] and grid[r, c] != 0:
                    matched_coords.update([(r, c), (r+1, c), (r+2, c)])
        return matched_coords

    def _group_matches(self, matched_coords):
        # Helper to group coordinates into individual lines for scoring
        groups = []
        coords = set(matched_coords)
        while coords:
            start_coord = coords.pop()
            group = {start_coord}
            queue = [start_coord]
            while queue:
                r, c = queue.pop(0)
                # Check neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (r + dr, c + dc)
                    if neighbor in coords:
                        coords.remove(neighbor)
                        group.add(neighbor)
                        queue.append(neighbor)
            groups.append(group)
        return groups

    def _has_possible_moves(self, grid=None):
        if grid is None:
            grid = self.grid
        
        temp_grid = grid.copy()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Test swap right
                if c < self.GRID_COLS - 1:
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_all_matches(temp_grid):
                        return True
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c] # Swap back
                # Test swap down
                if r < self.GRID_ROWS - 1:
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_all_matches(temp_grid):
                        return True
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c] # Swap back
        return False
        
    def _swap_gems(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
    def _clear_gems(self, matched_coords):
        for r, c in matched_coords:
            self.grid[r, c] = 0 # 0 represents an empty space
            
    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
                    
    def _fill_new_gems(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_gems()
        self._draw_cursor()
        self._update_and_draw_particles()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_COLS * self.CELL_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE))

    def _draw_gems(self):
        radius = self.CELL_SIZE // 2 - 5
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    color = self.GEM_COLORS[gem_type - 1]
                    center_x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    
                    # Highlight for selected gem
                    if self.selection_state == 1 and (r, c) == self.selected_pos:
                        pulse = (math.sin(self.steps * 0.3) + 1) / 2
                        highlight_radius = int(radius + 2 + pulse * 3)
                        highlight_color = (255, 255, 255, int(100 + pulse * 100))
                        
                        temp_surf = pygame.Surface((highlight_radius*2, highlight_radius*2), pygame.SRCALPHA)
                        pygame.draw.circle(temp_surf, highlight_color, (highlight_radius, highlight_radius), highlight_radius)
                        self.screen.blit(temp_surf, (center_x - highlight_radius, center_y - highlight_radius), special_flags=pygame.BLEND_RGBA_ADD)


    def _draw_cursor(self):
        r, c = self.cursor_pos
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
        
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=4)
        
    def _create_match_particles(self, matched_coords):
        for r, c in matched_coords:
            # This check is needed because a gem could be part of two matches
            # and already cleared by the time we process the second match group.
            gem_type_at_pos = self.grid[r,c]
            if gem_type_at_pos > 0:
                color = self.GEM_COLORS[gem_type_at_pos - 1]
                center_x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                for _ in range(10): # Spawn 10 particles per gem
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 3)
                    vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                    lifetime = self.np_random.integers(15, 30)
                    self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                size = int(p['life'] / 10) + 1
                
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)), special_flags=pygame.BLEND_RGBA_ADD)

                active_particles.append(p)
        self.particles = active_particles

    def _draw_ui(self):
        score_text = self.font_small.render(f"Score: {self.score}", True, (255, 255, 255))
        target_text = self.font_small.render(f"Target: {self.TARGET_SCORE}", True, (200, 200, 200))
        self.screen.blit(score_text, (10, 5))
        self.screen.blit(target_text, (self.WIDTH - target_text.get_width() - 10, 5))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.TARGET_SCORE:
                # Sound placeholder: victory fanfare
                end_text = self.font_large.render("You Win!", True, (100, 255, 100))
            else:
                end_text = self.font_large.render("Game Over", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "possible_moves": not self.game_over and self._has_possible_moves()
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
        pygame.K_LSHIFT: [0, 0, 1],
        pygame.K_RSHIFT: [0, 0, 1],
    }

    pygame.display.set_caption("Gemstone Grid")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
                elif event.key == pygame.K_r: 
                    obs, info = env.reset()
                    done = False
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q:
                    running = False

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Done: {done}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()