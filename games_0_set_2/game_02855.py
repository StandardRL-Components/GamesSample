
# Generated: 2025-08-28T06:09:59.725668
# Source Brief: brief_02855.md
# Brief Index: 2855

        
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
    """
    A puzzle game where the player connects adjacent, same-colored dots to clear them from a grid.
    The goal is to connect all dots of each color into single, contiguous groups before running out of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrows to select dots. Press space to connect a highlighted pair. Press shift to deselect."
    )

    # Short, user-facing description of the game
    game_description = (
        "Connect adjacent same-colored dots to clear them. Try to form single large groups for each color to win!"
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.NUM_COLORS = 5
        self.INITIAL_MOVES = 20
        self.MAX_STEPS = 1000  # Failsafe episode termination

        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (50, 55, 65)
        self.DOT_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 240, 80),  # Yellow
            (220, 100, 255), # Magenta
        ]
        self.COLOR_EMPTY = (40, 44, 52)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SELECT_1 = (255, 255, 255)
        self.COLOR_SELECT_2 = (200, 255, 200)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # --- Grid Layout ---
        self.grid_area_width = self.HEIGHT - 40
        self.cell_size = self.grid_area_width // self.GRID_ROWS
        self.dot_radius = int(self.cell_size * 0.35)
        self.grid_offset_x = (self.WIDTH - self.grid_area_width) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_area_width) // 2

        # --- State Variables ---
        self.grid = None
        self.selected_dot_1 = None
        self.selected_dot_2 = None
        self.score = 0
        self.moves_remaining = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []

        # Initialize state variables
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate a valid initial grid
        while True:
            self._generate_grid()
            if self._count_valid_moves() >= 2:
                break

        # Reset game state
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.INITIAL_MOVES
        self.game_over = False
        self.win = False
        self.selected_dot_1 = None
        self.selected_dot_2 = None
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        action_taken = False
        if shift_held:
            self._handle_deselect()
            action_taken = True
        elif space_held:
            connected, dots_cleared, colors_finished = self._handle_connect()
            if connected:
                # SFX: Connect sound
                self.moves_remaining -= 1
                
                # Reward for clearing dots
                reward += dots_cleared
                self.score += dots_cleared
                
                # Bonus reward for finishing a color
                reward += colors_finished * 10
                self.score += colors_finished * 10
                
                action_taken = True
        elif movement != 0:
            self._handle_movement(movement)
            action_taken = True

        self._check_termination()

        if self.game_over:
            if self.win:
                reward += 50
                self.score += 50
            else:
                reward -= 50
                # No score penalty for losing, just reward penalty
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_dots()
        self._render_selections()
        self._render_particles()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    # --- Helper Methods for Game Logic ---

    def _generate_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_ROWS, self.GRID_COLS))

    def _count_valid_moves(self):
        count = 0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color = self.grid[r, c]
                # Check right
                if c + 1 < self.GRID_COLS and self.grid[r, c+1] == color:
                    count += 1
                # Check down
                if r + 1 < self.GRID_ROWS and self.grid[r+1, c] == color:
                    count += 1
        return count

    def _handle_deselect(self):
        if self.selected_dot_2:
            self.selected_dot_2 = None
        elif self.selected_dot_1:
            self.selected_dot_1 = None
        # SFX: Deselect UI sound

    def _handle_connect(self):
        if not self.selected_dot_1 or not self.selected_dot_2:
            return False, 0, 0

        r1, c1 = self.selected_dot_1
        r2, c2 = self.selected_dot_2
        
        # Check for same color and adjacency
        is_adjacent = abs(r1 - r2) + abs(c1 - c2) == 1
        is_same_color = self.grid[r1, c1] == self.grid[r2, c2]

        if not is_adjacent or not is_same_color or self.grid[r1, c1] == 0:
            # SFX: Error/invalid action sound
            self.selected_dot_2 = None # Invalid pair, deselect second dot
            return False, 0, 0

        color_to_clear = self.grid[r1, c1]
        
        # Find all connected dots of this color (flood fill)
        dots_to_clear = set()
        q = deque([self.selected_dot_1])
        visited = {self.selected_dot_1}
        
        while q:
            r, c = q.popleft()
            dots_to_clear.add((r, c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                   (nr, nc) not in visited and self.grid[nr, nc] == color_to_clear:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        
        # Clear dots and create particles
        for r, c in dots_to_clear:
            self.grid[r, c] = 0
            self._create_particles(r, c, color_to_clear)

        dots_cleared_count = len(dots_to_clear)
        self._collapse_grid()

        # Check if any color was fully cleared
        colors_finished = 0
        if np.count_nonzero(self.grid == color_to_clear) == 0:
            colors_finished = 1
            # SFX: Special sound for clearing a color

        # Reset selection
        self.selected_dot_1 = None
        self.selected_dot_2 = None
        
        return True, dots_cleared_count, colors_finished

    def _handle_movement(self, movement):
        # Directions: 1=up, 2=down, 3=left, 4=right
        direction_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = direction_map[movement]

        if not self.selected_dot_1:
            # Select first dot from center
            center_r, center_c = self.GRID_ROWS // 2, self.GRID_COLS // 2
            for i in range(max(self.GRID_ROWS, self.GRID_COLS)):
                r, c = center_r + dr * i, center_c + dc * i
                if 0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS and self.grid[r, c] != 0:
                    self.selected_dot_1 = (r, c)
                    # SFX: Select UI sound
                    break
        else:
            r1, c1 = self.selected_dot_1
            nr, nc = r1 + dr, c1 + dc
            if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                target_color = self.grid[nr, nc]
                if target_color != 0:
                    if target_color == self.grid[r1, c1]:
                        self.selected_dot_2 = (nr, nc)
                        # SFX: Highlight pair sound
                    else:
                        # Change primary selection if moving to a different color
                        self.selected_dot_1 = (nr, nc)
                        self.selected_dot_2 = None
                        # SFX: Select UI sound

    def _collapse_grid(self):
        for c in range(self.GRID_COLS):
            col = self.grid[:, c]
            non_empty = col[col != 0]
            new_col = np.zeros(self.GRID_ROWS, dtype=int)
            if len(non_empty) > 0:
                new_col[-len(non_empty):] = non_empty
            self.grid[:, c] = new_col

    def _check_termination(self):
        if self.moves_remaining <= 0:
            self.game_over = True
            self.win = self._check_win_condition()
            return
        
        if self._count_valid_moves() == 0:
            self.game_over = True
            self.win = self._check_win_condition()
            return

    def _check_win_condition(self):
        unique_colors = {c for c in self.grid.flatten() if c != 0}
        if not unique_colors: return True # Empty board is a win

        visited = np.zeros_like(self.grid, dtype=bool)
        found_groups = {color: 0 for color in unique_colors}

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] != 0 and not visited[r, c]:
                    color = self.grid[r, c]
                    found_groups[color] += 1
                    if found_groups[color] > 1:
                        return False # Found a second group of the same color
                    
                    # Flood fill to mark all dots in this group as visited
                    q = deque([(r, c)])
                    visited[r, c] = True
                    while q:
                        cr, cc = q.popleft()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                               not visited[nr, nc] and self.grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))
        return True

    # --- Helper Methods for Rendering ---

    def _grid_to_pixel(self, r, c):
        x = self.grid_offset_x + c * self.cell_size + self.cell_size // 2
        y = self.grid_offset_y + r * self.cell_size + self.cell_size // 2
        return int(x), int(y)

    def _render_grid(self):
        for i in range(self.GRID_ROWS + 1):
            # Horizontal lines
            y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_area_width, y))
            # Vertical lines
            x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_area_width))

    def _render_dots(self):
        pulse = (math.sin(self.steps * 0.2) + 1) / 4 + 0.75 # Pulse between 0.75 and 1.25
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r, c]
                if color_idx != 0:
                    color = self.DOT_COLORS[color_idx - 1]
                    x, y = self._grid_to_pixel(r, c)
                    
                    is_selected = (self.selected_dot_1 == (r, c)) or (self.selected_dot_2 == (r, c))
                    radius = int(self.dot_radius * pulse) if is_selected else self.dot_radius
                    
                    # Draw filled circle with antialiasing
                    pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _render_selections(self):
        # Draw line between selected dots
        if self.selected_dot_1 and self.selected_dot_2:
            p1 = self._grid_to_pixel(*self.selected_dot_1)
            p2 = self._grid_to_pixel(*self.selected_dot_2)
            pygame.draw.aaline(self.screen, self.COLOR_SELECT_2, p1, p2, 2)
        
        # Draw highlight circles
        if self.selected_dot_1:
            x, y = self._grid_to_pixel(*self.selected_dot_1)
            radius = int(self.dot_radius * 1.4)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_SELECT_1)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius-1, self.COLOR_SELECT_1)

        if self.selected_dot_2:
            x, y = self._grid_to_pixel(*self.selected_dot_2)
            radius = int(self.dot_radius * 1.4)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_SELECT_2)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        # Moves
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 15))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if self.win else "GAME OVER"
        color = (150, 255, 150) if self.win else (255, 150, 150)
        
        text = self.font_game_over.render(message, True, color)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _create_particles(self, r, c, color_idx):
        x, y = self._grid_to_pixel(r, c)
        color = self.DOT_COLORS[color_idx - 1]
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([ [x, y], vel, lifetime, color ])

    def _render_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[2] -= 1 # lifetime
            
            if p[2] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p[2] / 30))
                color = (*p[3], alpha)
                radius = int(self.dot_radius * 0.2 * (p[2]/30))
                if radius > 0:
                    pygame.draw.circle(self.screen, color, p[0], radius)

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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up the display window
    pygame.display.set_caption("Dot Connector")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop for human play
    while not done:
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
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # If an action was taken that changes state, print info
        if any(action):
            print(f"Action: {action}, Reward: {reward}, Info: {info}")
        
        # Small delay to prevent high CPU usage
        pygame.time.wait(30)

    env.close()