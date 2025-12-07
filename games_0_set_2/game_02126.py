
# Generated: 2025-08-27T19:21:44.154506
# Source Brief: brief_02126.md
# Brief Index: 2126

        
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
    """
    A puzzle game where the player clears lines of matching colors on a grid.

    The player controls a cursor on a 10x10 grid. The goal is to draw paths
    connecting cells of the same color and then clear them.

    State:
    - 10x10 grid with colored cells (or empty).
    - Cursor position.
    - Currently drawn path.

    Actions:
    - Move cursor (Up, Down, Left, Right).
    - Start/continue drawing a path (Space).
    - Confirm and clear a path (Shift).

    Rewards:
    - Small positive reward for extending a path.
    - Large positive reward for clearing a path, proportional to its length.
    - Large positive reward for clearing the entire board (win).
    - Large negative reward for getting stuck with no possible moves (loss).
    - Small negative penalty for moving the cursor.

    Termination:
    - Board is cleared (win).
    - No more possible moves (loss).
    - Maximum steps (1000) are reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Hold Space to draw a path over "
        "matching colors. Press Shift to clear the completed path."
    )

    game_description = (
        "A minimalist puzzle game. Find and trace paths of matching colors on "
        "the grid. Clear all colors from the board to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Game Constants ---
        self.GRID_SIZE = 10
        self.NUM_COLORS = 5
        self.MAX_STEPS = 1000
        self.ANIMATION_DURATION = 15  # frames

        # --- Visuals ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_AREA_SIZE = 360
        self.CELL_SIZE = self.GRID_AREA_SIZE // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_AREA_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_AREA_SIZE) // 2
        self.CELL_PADDING = 3
        self.CELL_RADIUS = 4

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (30, 40, 55)
        self.COLOR_GRID_LINES = (40, 55, 70)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_CURSOR = (255, 255, 255)
        self.LINE_COLORS = [
            (52, 152, 219),   # Blue
            (231, 76, 60),    # Red
            (46, 204, 113),   # Green
            (241, 196, 15),   # Yellow
            (155, 89, 182),   # Purple
        ]

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.drawing_path = None
        self.drawing_color = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.possible_moves = None
        self.clear_animations = None

        self.reset()
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.drawing_path = []
        self.drawing_color = -1
        self.clear_animations = []

        self._generate_board()
        self.possible_moves = self._count_possible_moves()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        self._update_animations()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Handle Cursor Movement
        prev_cursor_pos = self.cursor_pos
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_x = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
            new_y = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)
            self.cursor_pos = (new_x, new_y)
            if self.cursor_pos != prev_cursor_pos:
                reward -= 0.01  # Small cost for moving

        # 2. Handle Drawing (Space)
        if space_held:
            cell_color = self.grid[self.cursor_pos[1], self.cursor_pos[0]]
            if not self.drawing_path:
                if cell_color != -1:
                    # Start new path
                    self.drawing_path = [self.cursor_pos]
                    self.drawing_color = cell_color
                    reward += 0.1
            elif self.cursor_pos not in self.drawing_path:
                last_pos = self.drawing_path[-1]
                is_adjacent = abs(last_pos[0] - self.cursor_pos[0]) + abs(last_pos[1] - self.cursor_pos[1]) == 1
                if is_adjacent and cell_color == self.drawing_color:
                    # Extend path
                    self.drawing_path.append(self.cursor_pos)
                    reward += 1.0
                else:
                    # Invalid extension, break path
                    self.drawing_path = []
                    self.drawing_color = -1
        
        # 3. Handle Commit (Shift)
        if shift_held and len(self.drawing_path) > 1:
            path_len = len(self.drawing_path)
            reward += 5 * path_len + path_len**2 * 0.5 # Exponential reward for longer lines

            # Add to animations and clear grid
            self.clear_animations.append({
                "path": list(self.drawing_path),
                "color": self.drawing_color,
                "timer": self.ANIMATION_DURATION,
            })
            for x, y in self.drawing_path:
                self.grid[y, x] = -1
            
            # Reset drawing state
            self.drawing_path = []
            self.drawing_color = -1
            
            self.score += path_len
            self.possible_moves = self._count_possible_moves()
            # SFX: Clear line
        elif shift_held: # Penalty for trying to commit an invalid path
            self.drawing_path = []
            self.drawing_color = -1
            reward -= 1.0

        # 4. Check Termination
        terminated = False
        if np.all(self.grid == -1):
            terminated = True
            reward += 100  # Win bonus
            self.game_over = True
        elif self.possible_moves == 0 and not np.all(self.grid == -1):
            terminated = True
            reward -= 50  # Loss penalty
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "possible_moves": self.possible_moves}

    def _update_animations(self):
        if not self.clear_animations:
            return
        for anim in self.clear_animations:
            anim["timer"] -= 1
        self.clear_animations = [anim for anim in self.clear_animations if anim["timer"] > 0]

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._count_possible_moves() > 5: # Ensure a reasonable number of starting moves
                break

    def _count_possible_moves(self):
        visited = np.zeros_like(self.grid, dtype=bool)
        count = 0
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if not visited[r, c] and self.grid[r, c] != -1:
                    color = self.grid[r, c]
                    component_size = 0
                    q = [(r, c)]
                    visited[r, c] = True
                    head = 0
                    while head < len(q):
                        curr_r, curr_c = q[head]
                        head += 1
                        component_size += 1
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and \
                               not visited[nr, nc] and self.grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    if component_size > 1:
                        count += 1
        return count

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
                         (self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_AREA_SIZE, self.GRID_AREA_SIZE),
                         border_radius=self.CELL_RADIUS * 2)

        # Draw cells
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid[r, c]
                if color_idx != -1:
                    px = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_PADDING
                    py = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_PADDING
                    rect_size = self.CELL_SIZE - 2 * self.CELL_PADDING
                    pygame.draw.rect(self.screen, self.LINE_COLORS[color_idx], 
                                     (px, py, rect_size, rect_size), 
                                     border_radius=self.CELL_RADIUS)
        
        self._render_drawing_path()
        self._render_clear_animation()
        self._render_cursor()

    def _render_drawing_path(self):
        if len(self.drawing_path) < 1:
            return

        # Highlight cells in the path
        for x, y in self.drawing_path:
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            highlight_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surface, (*self.COLOR_CURSOR, 60), 
                             (0, 0, self.CELL_SIZE, self.CELL_SIZE), 
                             border_radius=self.CELL_RADIUS + self.CELL_PADDING)
            self.screen.blit(highlight_surface, (px, py))

        # Draw connecting lines
        if len(self.drawing_path) > 1:
            points = []
            for x, y in self.drawing_path:
                px = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
                py = self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                points.append((px, py))
            
            pygame.draw.lines(self.screen, self.LINE_COLORS[self.drawing_color], False, points, width=8)

    def _render_cursor(self):
        pulse = 1 + 0.1 * math.sin(self.steps * 0.3)
        cx, cy = self.cursor_pos
        px = self.GRID_OFFSET_X + cx * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + cy * self.CELL_SIZE
        
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, 
                         (px, py, self.CELL_SIZE, self.CELL_SIZE), 
                         width=int(3 * pulse), border_radius=self.CELL_RADIUS + 2)

    def _render_clear_animation(self):
        for anim in self.clear_animations:
            progress = anim["timer"] / self.ANIMATION_DURATION
            color = self.LINE_COLORS[anim["color"]]
            
            for x, y in anim["path"]:
                px = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
                py = self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                
                radius = int((self.CELL_SIZE // 2 - self.CELL_PADDING) * progress)
                alpha = int(255 * progress)
                
                if radius > 0:
                    # Use gfxdraw for anti-aliased shapes
                    pygame.gfxdraw.aacircle(self.screen, px, py, radius, (*color, alpha))
                    pygame.gfxdraw.filled_circle(self.screen, px, py, radius, (*color, alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Possible Moves
        moves_text = self.font_large.render(f"MOVES: {self.possible_moves}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if np.all(self.grid == -1):
                end_text_str = "BOARD CLEARED!"
            elif self.possible_moves == 0:
                end_text_str = "NO MOVES LEFT"
            else:
                end_text_str = "TIME UP!"

            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            overlay.blit(end_text, end_rect)
            self.screen.blit(overlay, (0, 0))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Obs shape is {test_obs.shape}"
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Color Lines")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # No-op
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep window open for a bit to see the final state
    pygame.time.wait(2000)
    
    env.close()