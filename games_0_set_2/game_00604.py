
# Generated: 2025-08-27T14:10:43.198573
# Source Brief: brief_00604.md
# Brief Index: 604

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select/connect dots."
    )

    game_description = (
        "A minimalist puzzle game. Connect all dots of the same color without crossing lines."
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
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game Constants ---
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.screen_width - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.screen_height - self.GRID_ROWS * self.CELL_SIZE) // 2
        self.DOT_RADIUS = 14
        self.LINE_WIDTH = 4
        self.MAX_STEPS = 1000

        self.COLORS = [
            (255, 87, 87),    # Red
            (87, 255, 87),    # Green
            (87, 87, 255),    # Blue
            (255, 255, 87),   # Yellow
        ]
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_LINE = (200, 200, 200)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_INVALID = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        
        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = (0, 0)
        self.dots = {}  # (x, y) -> color_index
        self.dots_by_color = []
        self.connections = []
        self.selected_dot = None
        self.last_invalid_attempt = None # For flashing invalid line
        self.particles = []

        self.reset()
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.dots = {}
        self.connections = []
        self.selected_dot = None
        self.last_invalid_attempt = None
        self.particles = []

        self._generate_puzzle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.last_invalid_attempt = None

        movement, space_held, _ = action
        space_click = space_held == 1

        # 1. Handle cursor movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos = (
                (self.cursor_pos[0] + dx) % self.GRID_COLS,
                (self.cursor_pos[1] + dy) % self.GRID_ROWS,
            )

        # 2. Handle click action
        if space_click:
            click_reward = self._handle_click()
            reward += click_reward

        # 3. Update game state
        self.steps += 1
        self._update_particles()

        # 4. Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            is_win = len(self.dots) == 0
            if is_win:
                reward += 50
                # SFX: Win Jingle
            else:
                reward -= 50
                # SFX: Loss Sound

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_click(self):
        clicked_dot = self.dots.get(self.cursor_pos)

        if self.selected_dot is None:
            if clicked_dot is not None:
                # Select a dot
                self.selected_dot = (self.cursor_pos, clicked_dot)
                # SFX: Select
            return 0
        else:
            # A dot is already selected
            selected_pos, selected_color_idx = self.selected_dot
            
            if self.cursor_pos == selected_pos:
                 # Clicked the same dot, deselect
                self.selected_dot = None
                return 0

            if clicked_dot is not None and clicked_dot == selected_color_idx:
                # Attempt to connect to a dot of the same color
                if self._is_path_valid(selected_pos, self.cursor_pos):
                    # Valid connection
                    self.connections.append((selected_pos, self.cursor_pos, selected_color_idx))
                    self.selected_dot = None
                    # SFX: Connect
                    
                    if self._check_set_completion(selected_color_idx):
                        # SFX: Set Clear
                        return 1 + 5 # +1 for connection, +5 for set clear
                    return 1
                else:
                    # Invalid path
                    self.last_invalid_attempt = (selected_pos, self.cursor_pos)
                    self.selected_dot = None
                    # SFX: Invalid Move
                    return -0.1
            else:
                # Clicked empty space or different color, deselect
                self.selected_dot = None
                return 0
    
    def _generate_puzzle(self):
        while True:
            self.dots.clear()
            self.dots_by_color = [[] for _ in self.COLORS]
            
            num_colors = self.np_random.integers(2, len(self.COLORS) + 1)
            dot_counts = self.np_random.integers(2, 5, size=num_colors) * 2 # Pairs of dots
            
            available_pos = list(np.ndindex(self.GRID_COLS, self.GRID_ROWS))
            self.np_random.shuffle(available_pos)

            if sum(dot_counts) > len(available_pos):
                continue # Not enough space, retry

            dot_idx = 0
            for color_idx, count in enumerate(dot_counts):
                for _ in range(count):
                    pos = available_pos[dot_idx]
                    self.dots[pos] = color_idx
                    self.dots_by_color[color_idx].append(pos)
                    dot_idx += 1
            
            if self._are_valid_moves_left():
                break # Found a board with at least one valid move

    def _is_path_valid(self, start_pos, end_pos):
        # Check if path intersects with other dots
        for pos, _ in self.dots.items():
            if pos != start_pos and pos != end_pos:
                if self._is_point_on_segment(start_pos, end_pos, pos):
                    return False
        
        # Check if path intersects with other connections
        for p1, p2, _ in self.connections:
            if self._do_segments_intersect(start_pos, end_pos, p1, p2):
                return False
        return True

    def _are_valid_moves_left(self):
        for color_idx, dot_list in enumerate(self.dots_by_color):
            if len(dot_list) < 2:
                continue
            
            # Check for existing connections for this color
            connected_dots = set()
            for p1, p2, c_idx in self.connections:
                if c_idx == color_idx:
                    connected_dots.add(p1)
                    connected_dots.add(p2)
            
            unconnected_dots = [d for d in dot_list if d not in connected_dots]

            for i in range(len(unconnected_dots)):
                for j in range(i + 1, len(unconnected_dots)):
                    if self._is_path_valid(unconnected_dots[i], unconnected_dots[j]):
                        return True
            
            # Check connecting an unconnected dot to a connected component
            for uc_dot in unconnected_dots:
                for c_dot in connected_dots:
                    if self._is_path_valid(uc_dot, c_dot):
                        return True

        return False

    def _check_set_completion(self, color_idx):
        dots_in_set = self.dots_by_color[color_idx]
        if not dots_in_set:
            return False

        adj = {dot: [] for dot in dots_in_set}
        for p1, p2, c_idx in self.connections:
            if c_idx == color_idx:
                adj[p1].append(p2)
                adj[p2].append(p1)

        q = deque([dots_in_set[0]])
        visited = {dots_in_set[0]}
        while q:
            curr = q.popleft()
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        
        if len(visited) == len(dots_in_set):
            # Set is complete, clear it
            for dot_pos in dots_in_set:
                del self.dots[dot_pos]
                self._create_particles(dot_pos, color_idx)
            
            self.connections = [c for c in self.connections if c[2] != color_idx]
            self.dots_by_color[color_idx] = []
            return True
        
        return False

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if not self.dots: # Win condition
            return True
        if not self._are_valid_moves_left(): # Loss condition
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_COLS * self.CELL_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE))

        # Draw connections
        for p1, p2, _ in self.connections:
            start_pix = self._grid_to_pixel(p1)
            end_pix = self._grid_to_pixel(p2)
            pygame.draw.line(self.screen, self.COLOR_LINE, start_pix, end_pix, self.LINE_WIDTH)

        # Draw invalid attempt flash
        if self.last_invalid_attempt:
            p1, p2 = self.last_invalid_attempt
            start_pix = self._grid_to_pixel(p1)
            end_pix = self._grid_to_pixel(p2)
            pygame.draw.line(self.screen, self.COLOR_INVALID, start_pix, end_pix, self.LINE_WIDTH)

        # Draw particles
        for p in self.particles:
            pos, vel, life, color = p
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(life / 5), color)

        # Draw dots
        for pos, color_idx in self.dots.items():
            pix_pos = self._grid_to_pixel(pos)
            color = self.COLORS[color_idx]
            pygame.gfxdraw.filled_circle(self.screen, pix_pos[0], pix_pos[1], self.DOT_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pix_pos[0], pix_pos[1], self.DOT_RADIUS, color)

        # Draw cursor and selection highlight
        cursor_pix = self._grid_to_pixel(self.cursor_pos)
        
        if self.selected_dot and self.selected_dot[0] == self.cursor_pos:
            # Pulsing effect for selected dot under cursor
            pulse = abs(math.sin(self.steps * 0.3))
            radius = int(self.DOT_RADIUS * (1.5 + pulse * 0.2))
            alpha = int(80 + pulse * 40)
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, radius, radius, radius, (*self.COLOR_CURSOR, alpha))
            self.screen.blit(s, (cursor_pix[0] - radius, cursor_pix[1] - radius))
        else:
            # Normal cursor
            r = self.CELL_SIZE // 2
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_pix[0] - r, cursor_pix[1] - r, self.CELL_SIZE, self.CELL_SIZE), 2, 4)
        
        if self.selected_dot and self.selected_dot[0] != self.cursor_pos:
            # Highlight for the selected dot elsewhere
            selected_pix = self._grid_to_pixel(self.selected_dot[0])
            pulse = abs(math.sin(self.steps * 0.3))
            radius = int(self.DOT_RADIUS * (1.5 + pulse * 0.2))
            alpha = int(80 + pulse * 40)
            color = self.COLORS[self.selected_dot[1]]
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, radius, radius, radius, (*color, alpha))
            self.screen.blit(s, (selected_pix[0] - radius, selected_pix[1] - radius))


    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            is_win = len(self.dots) == 0
            end_text = "COMPLETE!" if is_win else "NO MOVES LEFT"
            end_surf = self.font_main.render(end_text, True, self.COLORS[0] if is_win else self.COLOR_INVALID)
            text_rect = end_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_surf, text_rect)

    def _create_particles(self, grid_pos, color_idx):
        pix_pos = self._grid_to_pixel(grid_pos)
        color = self.COLORS[color_idx]
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particles.append([list(pix_pos), vel, life, color])
    
    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p[0][0] += p[1][0] # pos x
            p[0][1] += p[1][1] # pos y
            p[2] -= 1 # life
            if p[2] > 0:
                new_particles.append(p)
        self.particles = new_particles

    # --- Geometric Helpers ---
    def _is_point_on_segment(self, p1, p2, p3):
        # Check if p3 is on the line segment p1-p2
        return (p3[0] <= max(p1[0], p2[0]) and p3[0] >= min(p1[0], p2[0]) and
                p3[1] <= max(p1[1], p2[1]) and p3[1] >= min(p1[1], p2[1]) and
                (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1]))

    def _orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    def _do_segments_intersect(self, p1, q1, p2, q2):
        # Check if line segment p1q1 and p2q2 intersect.
        # This handles the general case but not endpoints touching,
        # which is fine for this game's logic.
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            # Ensure they are not just touching at an endpoint of an existing line
            if p1 in (p2, q2) or q1 in (p2, q2):
                return False
            return True
        return False

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")
    
    # Create a window to display the game
    pygame.display.set_caption("Connect The Dots")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    action = [0, 0, 0] # No-op, no space, no shift
    
    while not done:
        # --- Human Input Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    print("--- Game Reset ---")

        # Get key presses for movement (this frame only)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.1f}, Score: {info['score']}")

        # --- Render to Screen ---
        # The observation is already the rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Reset action for next frame ---
        action = [0, 0, 0]
        
        # Since auto_advance is False, we need a small delay for human playability
        pygame.time.wait(50) 

    print(f"Game Over! Final Score: {info['score']}")
    env.close()