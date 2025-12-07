
# Generated: 2025-08-28T01:34:38.475254
# Source Brief: brief_04152.md
# Brief Index: 4152

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    """
    A puzzle game where the player sorts colored balls into matching groups on a grid.
    The player has a limited number of moves to swap adjacent balls and achieve a fully sorted board.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a ball, "
        "then move to an adjacent ball and press Space again to swap. Press Shift to deselect."
    )

    game_description = (
        "Sort colored balls into matching groups by swapping adjacent balls. "
        "You have a limited number of moves. Plan your swaps carefully to solve the puzzle!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 4, 6
        self.NUM_COLORS = 6
        self.BALLS_PER_COLOR = 4
        self.MAX_MOVES = 20

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 32)
        self.font_big = pygame.font.Font(None, 72)
        
        # --- Visuals ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_CURSOR_SELECT = (0, 255, 128)
        self.COLOR_TEXT = (220, 220, 220)
        self.BALL_COLORS = [
            (231, 76, 60),   # Red
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (241, 196, 15),  # Yellow
            (155, 89, 182),  # Purple
            (230, 126, 34)   # Orange
        ]
        
        self.CELL_SIZE = 64
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2 + 20
        self.BALL_RADIUS = self.CELL_SIZE // 2 - 10

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.selected_ball_pos = None
        self.moves_left = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_state = None
        self.particles = []
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [0, 0]  # [col, row]
        self.selected_ball_pos = None
        self.particles = []

        ball_pool = []
        for color_idx in range(self.NUM_COLORS):
            ball_pool.extend([color_idx] * self.BALLS_PER_COLOR)
        self.np_random.shuffle(ball_pool)

        self.grid = np.array(ball_pool).reshape((self.GRID_ROWS, self.GRID_COLS))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        action_taken = False

        # --- Action Handling ---
        if movement != 0:
            action_taken = True
            col, row = self.cursor_pos
            if movement == 1: self.cursor_pos[1] = (row - 1 + self.GRID_ROWS) % self.GRID_ROWS
            elif movement == 2: self.cursor_pos[1] = (row + 1) % self.GRID_ROWS
            elif movement == 3: self.cursor_pos[0] = (col - 1 + self.GRID_COLS) % self.GRID_COLS
            elif movement == 4: self.cursor_pos[0] = (col + 1) % self.GRID_COLS

        if shift_pressed and self.selected_ball_pos is not None:
            action_taken = True
            self.selected_ball_pos = None
            # sfx: deselect_sound

        elif space_pressed:
            action_taken = True
            cursor_col, cursor_row = self.cursor_pos
            if self.selected_ball_pos is None:
                self.selected_ball_pos = (cursor_col, cursor_row)
                # sfx: select_sound
            else:
                sel_col, sel_row = self.selected_ball_pos
                is_adjacent = abs(sel_col - cursor_col) + abs(sel_row - cursor_row) == 1
                is_same_cell = (sel_col == cursor_col and sel_row == cursor_row)

                if is_same_cell:
                    self.selected_ball_pos = None # Deselect if clicking same ball
                elif is_adjacent:
                    # sfx: swap_sound
                    pre_swap_adj_score = self._calculate_adjacency_score()
                    pre_swap_sorted_groups = self._get_sorted_group_count()

                    val1 = self.grid[sel_row, sel_col]
                    val2 = self.grid[cursor_row, cursor_col]
                    self.grid[sel_row, sel_col] = val2
                    self.grid[cursor_row, cursor_col] = val1

                    self._create_swap_particles(sel_col, sel_row, cursor_col, cursor_row)
                    self.moves_left -= 1

                    post_swap_adj_score = self._calculate_adjacency_score()
                    post_swap_sorted_groups = self._get_sorted_group_count()
                    
                    reward += (post_swap_adj_score - pre_swap_adj_score) * 0.1
                    reward += (post_swap_sorted_groups - pre_swap_sorted_groups) * 5.0

                    self.selected_ball_pos = None
                else:
                    # sfx: error_sound
                    self.selected_ball_pos = (cursor_col, cursor_row) # Select the new ball instead

        self.steps += 1
        self._update_particles()
        self.score += reward

        # --- Termination Check ---
        terminated = False
        is_sorted = self._is_board_sorted()
        if is_sorted:
            terminated = True
            self.win_state = True
            final_reward = 100.0
            reward += final_reward
            self.score += final_reward
            # sfx: win_sound
        elif self.moves_left <= 0:
            terminated = True
            final_reward = -10.0
            reward += final_reward
            self.score += final_reward
            # sfx: lose_sound
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_group_glows()
        self._render_balls()
        self._render_cursor()
        self._render_particles()

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y), 2)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT), 2)

    def _render_group_glows(self):
        for color_idx in range(self.NUM_COLORS):
            is_sorted, positions = self._is_group_sorted(color_idx)
            if is_sorted:
                color = self.BALL_COLORS[color_idx]
                glow_color = (*color, 60) # RGBA with low alpha
                for r, c in positions:
                    cx, cy = self._grid_to_pixel(c, r)
                    glow_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surface, glow_color, (self.CELL_SIZE // 2, self.CELL_SIZE // 2), self.BALL_RADIUS + 8)
                    self.screen.blit(glow_surface, (cx - self.CELL_SIZE // 2, cy - self.CELL_SIZE // 2))

    def _render_balls(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r, c]
                if color_idx != -1:
                    cx, cy = self._grid_to_pixel(c, r)
                    color = self.BALL_COLORS[color_idx]
                    
                    # Shadow
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy + 3, self.BALL_RADIUS, (0, 0, 0, 50))
                    # Main ball
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.BALL_RADIUS, color)
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, self.BALL_RADIUS, color)
                    # Highlight
                    highlight_pos = (cx - self.BALL_RADIUS // 2, cy - self.BALL_RADIUS // 2)
                    pygame.gfxdraw.filled_circle(self.screen, int(highlight_pos[0]), int(highlight_pos[1]), self.BALL_RADIUS // 3, (255, 255, 255, 80))


    def _render_cursor(self):
        # Selection highlight
        if self.selected_ball_pos is not None:
            sel_c, sel_r = self.selected_ball_pos
            rect = pygame.Rect(
                self.GRID_OFFSET_X + sel_c * self.CELL_SIZE,
                self.GRID_OFFSET_Y + sel_r * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_CURSOR_SELECT, rect, 4, border_radius=8)

        # Cursor
        cur_c, cur_r = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + cur_c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + cur_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=8)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p['size'] * (p['life'] / p['max_life'])))
            color = (*p['color'], int(255 * (p['life'] / p['max_life'])))
            surface = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(surface, color, (size,size), size)
            self.screen.blit(surface, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_state else "OUT OF MOVES"
            color = (100, 255, 100) if self.win_state else (255, 100, 100)
            end_text = self.font_big.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _grid_to_pixel(self, col, row):
        x = self.GRID_OFFSET_X + col * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + row * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _calculate_adjacency_score(self):
        score = 0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color = self.grid[r, c]
                # Check right
                if c + 1 < self.GRID_COLS and self.grid[r, c + 1] == color:
                    score += 1
                # Check down
                if r + 1 < self.GRID_ROWS and self.grid[r + 1, c] == color:
                    score += 1
        return score

    def _is_group_sorted(self, color_idx):
        positions = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == color_idx:
                    positions.append((r, c))
        
        if not positions: return True, [] # Vacuously true
        
        q = deque([positions[0]])
        visited = {positions[0]}
        
        while q:
            r, c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and (nr, nc) in positions and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))

        return len(visited) == self.BALLS_PER_COLOR, positions

    def _get_sorted_group_count(self):
        count = 0
        for color_idx in range(self.NUM_COLORS):
            if self._is_group_sorted(color_idx)[0]:
                count += 1
        return count

    def _is_board_sorted(self):
        return self._get_sorted_group_count() == self.NUM_COLORS

    def _create_swap_particles(self, c1, r1, c2, r2):
        pos1 = self._grid_to_pixel(c1, r1)
        pos2 = self._grid_to_pixel(c2, r2)
        color1 = self.BALL_COLORS[self.grid[r1,c1]]
        color2 = self.BALL_COLORS[self.grid[r2,c2]]

        for pos, color in [(pos1, color2), (pos2, color1)]:
            for _ in range(15):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                life = self.np_random.integers(15, 30)
                self.particles.append({
                    'pos': list(pos),
                    'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'life': life,
                    'max_life': life,
                    'color': color,
                    'size': self.np_random.uniform(2, 5)
                })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Example usage and interactive play
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    # Create a display for interactive playing
    pygame.display.set_caption("Ball Sort Gym Environment")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        action = np.array([0, 0, 0])  # no-op
        
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
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    done = False
                    continue
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if not np.all(action == [0,0,0]): # only print on action
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}, Terminated: {terminated}")
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for interactive play
        
    env.close()