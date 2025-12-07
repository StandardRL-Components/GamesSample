
# Generated: 2025-08-27T16:38:08.485015
# Source Brief: brief_01278.md
# Brief Index: 1278

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a start dot, then move to an end dot of the same color and press Space again to connect them. Press Shift to cancel a selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect all pairs of same-colored dots by drawing lines. Lines cannot cross. You have a limited number of moves to solve the puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    MARGIN_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    MARGIN_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2
    
    NUM_COLORS = 4
    MAX_MOVES = 15
    MAX_STEPS = 1000

    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    
    DOT_COLORS = [
        (255, 70, 70),   # Red
        (70, 255, 70),   # Green
        (70, 130, 255),  # Blue
        (255, 150, 50),  # Orange
        (180, 70, 255),  # Purple
        (70, 255, 220),  # Cyan
    ]

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
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.cursor_pos = [0, 0]
        self.dots = []
        self.lines = []
        self.selected_dot_index = None
        self.color_groups = {}
        self.uf_parent = []
        self.uf_size = []
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.lines = []
        self.selected_dot_index = None
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        self.dots = []
        self.color_groups = {}
        
        occupied_positions = set()
        dot_id_counter = 0
        
        colors_to_use = self.DOT_COLORS[:self.NUM_COLORS]

        for color_idx, color in enumerate(colors_to_use):
            self.color_groups[color] = {'dots': [], 'completed': False}
            for _ in range(2): # Create pairs
                while True:
                    pos = (
                        self.np_random.integers(0, self.GRID_WIDTH),
                        self.np_random.integers(0, self.GRID_HEIGHT)
                    )
                    if pos not in occupied_positions:
                        occupied_positions.add(pos)
                        break
                
                dot_info = {
                    'id': dot_id_counter,
                    'pos': pos,
                    'color': color,
                    'color_idx': color_idx
                }
                self.dots.append(dot_info)
                self.color_groups[color]['dots'].append(dot_id_counter)
                dot_id_counter += 1

        # Initialize Union-Find structure
        self.uf_parent = list(range(len(self.dots)))
        self.uf_size = [1] * len(self.dots)

    def step(self, action):
        reward = -0.01  # Small penalty for each action/step
        self.game_over = False
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # 1. Handle deselect action (highest priority)
        if shift_pressed and self.selected_dot_index is not None:
            self.selected_dot_index = None
            # sfx: cancel.wav
        
        # 2. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # 3. Handle select/connect action
        if space_pressed:
            hovered_dot_idx = self._get_dot_at_cursor()
            
            if self.selected_dot_index is None:
                if hovered_dot_idx is not None and not self._is_color_group_complete(self.dots[hovered_dot_idx]['color']):
                    self.selected_dot_index = hovered_dot_idx
                    # sfx: select.wav
            else:
                start_dot_idx = self.selected_dot_index
                end_dot_idx = hovered_dot_idx
                
                self.selected_dot_index = None # Deselect after any connection attempt
                
                if self._validate_connection(start_dot_idx, end_dot_idx):
                    self.moves_left -= 1
                    reward -= 0.1  # Penalty for using a move

                    start_dot = self.dots[start_dot_idx]
                    end_dot = self.dots[end_dot_idx]
                    
                    new_line = (start_dot['pos'], end_dot['pos'])
                    self.lines.append({'line': new_line, 'color': start_dot['color']})
                    # sfx: connect.wav

                    was_merged = self._uf_union(start_dot_idx, end_dot_idx)
                    if was_merged:
                        reward += 1.0

                    if self._is_color_group_complete(start_dot['color']):
                        reward += 5.0
                        # sfx: color_complete.wav
                else:
                    # sfx: invalid_move.wav
                    pass # Invalid move, only base step penalty applies

        # 4. Update game state and check for termination
        self.steps += 1
        is_win = self._check_win_condition()
        is_loss = (self.moves_left <= 0 or self.steps >= self.MAX_STEPS) and not is_win
        self.game_over = is_win or is_loss

        if is_win:
            reward += 50.0
            # sfx: win.wav
        elif is_loss:
            reward -= 50.0
            # sfx: lose.wav
        
        self.score += reward
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    # --- Helper Methods for Game Logic ---

    def _get_dot_at_cursor(self):
        for i, dot in enumerate(self.dots):
            if dot['pos'][0] == self.cursor_pos[0] and dot['pos'][1] == self.cursor_pos[1]:
                return i
        return None

    def _validate_connection(self, start_idx, end_idx):
        if end_idx is None or start_idx == end_idx:
            return False
        
        start_dot = self.dots[start_idx]
        end_dot = self.dots[end_idx]

        # Must be same color
        if start_dot['color'] != end_dot['color']:
            return False
        
        # Must not be already connected
        if self._uf_find(start_idx) == self._uf_find(end_idx):
            return False
            
        # Proposed line must not cross other lines
        proposed_line = (start_dot['pos'], end_dot['pos'])
        for existing_line in self.lines:
            if self._lines_intersect(proposed_line, existing_line['line']):
                return False
        
        return True

    def _check_win_condition(self):
        for color, data in self.color_groups.items():
            if not self._is_color_group_complete(color):
                return False
        return True

    def _is_color_group_complete(self, color):
        dot_indices = self.color_groups[color]['dots']
        if not dot_indices:
            return True
        first_root = self._uf_find(dot_indices[0])
        for dot_idx in dot_indices[1:]:
            if self._uf_find(dot_idx) != first_root:
                return False
        return True

    # --- Union-Find Data Structure ---
    def _uf_find(self, i):
        if self.uf_parent[i] == i:
            return i
        self.uf_parent[i] = self._uf_find(self.uf_parent[i])
        return self.uf_parent[i]

    def _uf_union(self, i, j):
        root_i = self._uf_find(i)
        root_j = self._uf_find(j)
        if root_i != root_j:
            if self.uf_size[root_i] < self.uf_size[root_j]:
                root_i, root_j = root_j, root_i
            self.uf_parent[root_j] = root_i
            self.uf_size[root_i] += self.uf_size[root_j]
            return True
        return False

    # --- Line Intersection Geometry ---
    def _on_segment(self, p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    def _orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counter-clockwise

    def _lines_intersect(self, line1, line2):
        p1, q1 = line1
        p2, q2 = line2
        
        # Two lines sharing an endpoint are not considered intersecting
        if p1 == p2 or p1 == q2 or q1 == p2 or q1 == q2:
            return False

        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        # Collinear cases
        if o1 == 0 and self._on_segment(p1, p2, q1): return True
        if o2 == 0 and self._on_segment(p1, q2, q1): return True
        if o3 == 0 and self._on_segment(p2, p1, q2): return True
        if o4 == 0 and self._on_segment(p2, q1, q2): return True

        return False

    # --- Rendering Methods ---

    def _grid_to_screen(self, gx, gy):
        x = self.MARGIN_X + gx * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.MARGIN_Y + gy * self.CELL_SIZE + self.CELL_SIZE / 2
        return int(x), int(y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start = (self.MARGIN_X + x * self.CELL_SIZE, self.MARGIN_Y)
            end = (self.MARGIN_X + x * self.CELL_SIZE, self.MARGIN_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.MARGIN_X, self.MARGIN_Y + y * self.CELL_SIZE)
            end = (self.MARGIN_X + self.GRID_WIDTH * self.CELL_SIZE, self.MARGIN_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw lines
        for line_info in self.lines:
            start_pos = self._grid_to_screen(line_info['line'][0][0], line_info['line'][0][1])
            end_pos = self._grid_to_screen(line_info['line'][1][0], line_info['line'][1][1])
            pygame.draw.line(self.screen, line_info['color'], start_pos, end_pos, 8)
        
        # Draw dots
        dot_radius = self.CELL_SIZE // 3
        for i, dot in enumerate(self.dots):
            pos = self._grid_to_screen(dot['pos'][0], dot['pos'][1])
            color = dot['color']
            
            is_complete = self._is_color_group_complete(color)
            
            # Draw glow for selected dot
            if i == self.selected_dot_index:
                pulse_alpha = (math.sin(self.steps * 0.3) + 1) / 2
                glow_radius = int(dot_radius * (1.5 + pulse_alpha * 0.5))
                glow_color = color[:3] + (int(pulse_alpha * 100),)
                
                temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, glow_color)
                self.screen.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

            # Draw dot body
            if is_complete:
                # Filled circle for completed dots
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], dot_radius, color)
                # Checkmark
                p1 = (pos[0] - 8, pos[1])
                p2 = (pos[0] - 2, pos[1] + 6)
                p3 = (pos[0] + 8, pos[1] - 8)
                pygame.draw.lines(self.screen, self.COLOR_BG, False, [p1,p2,p3], 3)
            else:
                # Unfilled for active dots
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], dot_radius, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], dot_radius, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], dot_radius-4, self.COLOR_BG)

        # Draw cursor
        cursor_screen_pos = self._grid_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        size = self.CELL_SIZE // 4
        rect = pygame.Rect(cursor_screen_pos[0] - size, cursor_screen_pos[1] - size, size*2, size*2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _render_ui(self):
        # Moves left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 10))

        # Score
        score_text = self.font_main.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(score_text, score_rect)
        
        # Game Over text
        if self.game_over:
            is_win = self._check_win_condition()
            message = "PUZZLE COMPLETE!" if is_win else "OUT OF MOVES"
            color = self.DOT_COLORS[1] if is_win else self.DOT_COLORS[0]
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            end_text = self.font_main.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            overlay.blit(end_text, end_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "win": self._check_win_condition() if not self.game_over else False
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Connect The Dots")
    clock = pygame.time.Clock()

    print(env.user_guide)

    while not done:
        # Action mapping for human play
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
        
        # Only step if an action was taken
        if any([movement, space, shift]):
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    # Keep the window open for a bit after the game ends
    print("Game Over!")
    pygame.time.wait(2000)
    env.close()