
# Generated: 2025-08-28T06:02:42.757163
# Source Brief: brief_02795.md
# Brief Index: 2795

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select/connect. Shift to deselect."
    )

    game_description = (
        "Connect all squares of the same color into a single group by selecting adjacent, matching squares. You have a limited number of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.width, self.height = 640, 400

        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        # --- Visuals & Style ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (50, 65, 80)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_CURSOR = (255, 255, 0, 150)
        self.COLOR_SELECT = (0, 255, 150, 200)
        self.PALETTE = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 150, 50),  # Orange
            (200, 80, 255),  # Purple
            (80, 255, 255),  # Cyan
        ]
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 48, bold=True)

        # --- Game State ---
        self.episodes_won = 0
        self.grid_dim = 4
        self.num_colors = 2
        self.board = []
        self.connections = set()
        self.cursor_pos = (0, 0)
        self.selected_square = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.message = None
        self.message_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Difficulty progression
        self.grid_dim = 4 + (self.episodes_won // 5)
        self.num_colors = min(len(self.PALETTE), 2 + (self.episodes_won // 10))
        
        # Initialize state
        self.board = self._generate_board()
        self.moves_left = self.grid_dim * self.grid_dim // 2 + 5 + (self.grid_dim - 4)
        self.connections = set()
        self.cursor_pos = (self.grid_dim // 2, self.grid_dim // 2)
        self.selected_square = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.message = None
        self.message_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        # Cursor movement
        r, c = self.cursor_pos
        if movement == 1: r -= 1  # Up
        elif movement == 2: r += 1  # Down
        elif movement == 3: c -= 1  # Left
        elif movement == 4: c += 1  # Right
        self.cursor_pos = (max(0, min(self.grid_dim - 1, r)), max(0, min(self.grid_dim - 1, c)))

        # Deselect with Shift
        if shift_pressed and self.selected_square:
            self.selected_square = None
            # sfx: deselect sound

        # Select/Connect with Space
        if space_pressed:
            r, c = self.cursor_pos
            if self.selected_square is None:
                self.selected_square = (r, c)
                # sfx: select sound
            else:
                r1, c1 = self.selected_square
                r2, c2 = r, c

                if (r1, c1) == (r2, c2): # Deselect if same square
                    self.selected_square = None
                else:
                    self.moves_left -= 1
                    
                    color1 = self.board[r1][c1]
                    color2 = self.board[r2][c2]
                    is_adjacent = abs(r1 - r2) + abs(c1 - c2) == 1
                    connection = tuple(sorted(((r1, c1), (r2, c2))))
                    is_new_connection = connection not in self.connections

                    if color1 == color2 and is_adjacent and is_new_connection:
                        # --- Valid Connection ---
                        groups_before = self._count_color_groups(color1)
                        self.connections.add(connection)
                        groups_after = self._count_color_groups(color1)
                        
                        reward += 1.0  # Base reward for connection
                        if groups_after < groups_before:
                            reward += 5.0 # Bonus for merging groups
                        
                        self._create_particles((r1, c1), (r2, c2), self.PALETTE[color1])
                        # sfx: connection success sound
                    else:
                        # --- Invalid Connection ---
                        reward -= 0.1
                        self.message = "Invalid!"
                        self.message_timer = 30
                        # sfx: connection fail sound

                    self.selected_square = None

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Check Termination Conditions ---
        if self._check_win_condition():
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
            self.episodes_won += 1
            self.message = "Complete!"
            self.message_timer = 180
            # sfx: win fanfare
        elif self.moves_left <= 0:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
            self.message = "Out of Moves"
            self.message_timer = 180
            # sfx: lose sound
        elif self.steps >= 1000:
            terminated = True
            self.game_over = True

        self.score += reward
        self._update_particles()
        if self.message_timer > 0:
            self.message_timer -= 1
        else:
            self.message = None

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_board(self):
        return [
            [self.np_random.integers(0, self.num_colors) for _ in range(self.grid_dim)]
            for _ in range(self.grid_dim)
        ]

    def _check_win_condition(self):
        colors_on_board = set(c for row in self.board for c in row)
        if not colors_on_board: return True
        
        for color_idx in colors_on_board:
            if self._count_color_groups(color_idx) > 1:
                return False
        return True

    def _count_color_groups(self, color_idx):
        squares_of_color = set()
        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                if self.board[r][c] == color_idx:
                    squares_of_color.add((r, c))
        
        if not squares_of_color: return 0

        visited = set()
        num_groups = 0
        
        for r_start, c_start in squares_of_color:
            if (r_start, c_start) not in visited:
                num_groups += 1
                q = deque([(r_start, c_start)])
                visited.add((r_start, c_start))
                while q:
                    r, c = q.popleft()
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.grid_dim and 0 <= nc < self.grid_dim:
                            neighbor = (nr, nc)
                            connection = tuple(sorted(((r, c), neighbor)))
                            if neighbor in squares_of_color and neighbor not in visited and connection in self.connections:
                                visited.add(neighbor)
                                q.append(neighbor)
        return num_groups

    def _get_observation(self):
        # --- Calculate grid rendering properties ---
        grid_area_size = min(self.width, self.height) * 0.8
        self.cell_size = grid_area_size / self.grid_dim
        self.grid_pixel_size = self.cell_size * self.grid_dim
        self.offset_x = (self.width - self.grid_pixel_size) / 2
        self.offset_y = (self.height - self.grid_pixel_size) / 2

        # --- Draw Background ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Game Elements ---
        self._render_game()
        self._render_particles()

        # --- Render UI Overlay ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw squares
        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                color_idx = self.board[r][c]
                color = self.PALETTE[color_idx]
                
                center_x = self.offset_x + (c + 0.5) * self.cell_size
                center_y = self.offset_y + (r + 0.5) * self.cell_size
                radius = self.cell_size * 0.35
                
                pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), int(radius), color)
                pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), int(radius), color)

        # Draw grid lines (or lack thereof for connections)
        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                # Check right connection
                if c < self.grid_dim - 1:
                    connection = tuple(sorted(((r, c), (r, c + 1))))
                    if connection not in self.connections:
                        x = self.offset_x + (c + 1) * self.cell_size
                        y1 = self.offset_y + r * self.cell_size
                        y2 = y1 + self.cell_size
                        pygame.draw.line(self.screen, self.COLOR_GRID, (x, y1), (x, y2), 2)
                # Check down connection
                if r < self.grid_dim - 1:
                    connection = tuple(sorted(((r, c), (r + 1, c))))
                    if connection not in self.connections:
                        x1 = self.offset_x + c * self.cell_size
                        x2 = x1 + self.cell_size
                        y = self.offset_y + (r + 1) * self.cell_size
                        pygame.draw.line(self.screen, self.COLOR_GRID, (x1, y), (x2, y), 2)
        
        # Draw selected square highlight
        if self.selected_square is not None:
            r, c = self.selected_square
            rect = pygame.Rect(
                self.offset_x + c * self.cell_size,
                self.offset_y + r * self.cell_size,
                self.cell_size, self.cell_size
            )
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECT)
            self.screen.blit(s, rect.topleft)

        # Draw cursor
        r, c = self.cursor_pos
        rect = pygame.Rect(
            self.offset_x + c * self.cell_size,
            self.offset_y + r * self.cell_size,
            self.cell_size, self.cell_size
        )
        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, rect.topleft)

    def _create_particles(self, pos1, pos2, color):
        r1, c1 = pos1
        r2, c2 = pos2
        mid_x = self.offset_x + ((c1 + c2) / 2 + 0.5) * self.cell_size
        mid_y = self.offset_y + ((r1 + r2) / 2 + 0.5) * self.cell_size
        
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': [mid_x, mid_y], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30))))
            color = p['color'] + (alpha,)
            size = int(p['life'] / 8)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 10))
        
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.width - 15, 10))
        self.screen.blit(score_text, score_rect)

        # Game Message
        if self.message:
            alpha = 255
            if self.message_timer < 30:
                alpha = int(255 * (self.message_timer / 30))
            
            msg_surf = self.font_msg.render(self.message, True, self.COLOR_TEXT)
            msg_surf.set_alpha(alpha)
            msg_rect = msg_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "episodes_won": self.episodes_won,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Color Connect")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, term, trunc, info = env.step(action)
        
        if term or trunc:
            print(f"Game Over. Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Transpose for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()