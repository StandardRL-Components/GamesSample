import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a square. "
        "Move to an adjacent, same-colored square and press Space again to connect. "
        "Shift to deselect."
    )

    game_description = (
        "A minimalist puzzle game. Connect all squares of the same color into a single group. "
        "Making a connection removes the squares, and new ones fall from above. "
        "Plan your moves to unify all colors on the board."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        self.GRID_COLS = 12
        self.GRID_ROWS = 8
        self.CELL_SIZE = 40
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_ROWS * self.CELL_SIZE) // 2
        self.NUM_COLORS = 5

        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID_LINES = (50, 50, 60)
        self.COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Magenta
        ]
        self.BRIGHT_COLORS = [tuple(min(255, int(c * 1.3)) for c in color) for color in self.COLORS]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTION = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.first_selection = None
        self.particles = None
        self.connection_animation = None
        self.unified_colors = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None

        # The reset call is deferred to the user of the environment.
        # The validation part is removed from __init__ to avoid issues
        # with external verifiers that instantiate the class without calling reset.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.first_selection = None
        self.particles = []
        self.connection_animation = None
        self.unified_colors = set()

        while True:
            self._generate_grid()
            if len(self._find_all_valid_moves()) >= 5:
                break
        
        self._check_all_color_unification()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            obs, info = self.reset()
            return obs, 0, True, False, info

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small penalty for taking a step
        self.connection_animation = None # Clear previous frame's animation

        # 1. Handle Deselection
        if shift_pressed and self.first_selection is not None:
            self.first_selection = None
            # sound: deselect.wav

        # 2. Handle Movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_ROWS - 1)

        # 3. Handle Selection/Connection
        if space_pressed:
            cx, cy = self.cursor_pos
            if self.first_selection is None:
                if self.grid[cy][cx] != -1:
                    self.first_selection = [cx, cy]
                    # sound: select.wav
            else:
                sx, sy = self.first_selection
                if self._is_valid_connection((sx, sy), (cx, cy)):
                    reward += 1.0
                    # sound: connect_success.wav
                    
                    color_index = self.grid[sy][sx]
                    self.connection_animation = ((sx, sy), (cx, cy), self.COLORS[color_index])
                    
                    self._create_particles((sx, sy), color_index)
                    self._create_particles((cx, cy), color_index)

                    self.grid[sy][sx] = -1
                    self.grid[cy][cx] = -1
                    
                    self._apply_gravity_and_refill()
                    
                    prev_unified_count = len(self.unified_colors)
                    self._check_all_color_unification()
                    if len(self.unified_colors) > prev_unified_count:
                        reward += 10.0
                        # sound: color_complete.wav
                    
                    self.first_selection = None
                else:
                    reward -= 0.1
                    # sound: connect_fail.wav
                    self.first_selection = None

        self._update_particles()
        self.steps += 1
        
        is_win, is_loss = self._check_termination()
        terminated = is_win or is_loss or self.steps >= self.MAX_STEPS
        truncated = False # Truncation is handled by terminated flag for max steps

        if is_win:
            reward += 100.0
        elif is_loss:
            reward -= 50.0
        
        self.score += reward
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

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
            "unified_colors": len(self.unified_colors),
        }

    # --- Helper Methods ---

    def _generate_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_ROWS, self.GRID_COLS))

    def _is_valid_connection(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        
        if not (0 <= x1 < self.GRID_COLS and 0 <= y1 < self.GRID_ROWS and
                0 <= x2 < self.GRID_COLS and 0 <= y2 < self.GRID_ROWS):
            return False
            
        is_adjacent = abs(x1 - x2) + abs(y1 - y2) == 1
        
        color1 = self.grid[y1][x1]
        color2 = self.grid[y2][x2]
        
        return is_adjacent and color1 == color2 and color1 != -1

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            write_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != -1:
                    if r != write_row:
                        self.grid[write_row][c] = self.grid[r][c]
                    write_row -= 1
            
            for r in range(write_row, -1, -1):
                self.grid[r][c] = self.np_random.integers(0, self.NUM_COLORS)

    def _check_all_color_unification(self):
        self.unified_colors.clear()
        for color_idx in range(self.NUM_COLORS):
            if self._is_color_unified(color_idx):
                self.unified_colors.add(color_idx)

    def _is_color_unified(self, color_idx):
        locations = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == color_idx:
                    locations.append((c, r))

        if not locations:
            return True

        q = deque([locations[0]])
        visited = {locations[0]}
        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in locations and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        
        return len(visited) == len(locations)

    def _find_all_valid_moves(self):
        moves = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check right neighbor
                if c + 1 < self.GRID_COLS and self.grid[r][c] == self.grid[r][c+1] and self.grid[r][c] != -1:
                    moves.append(((c, r), (c + 1, r)))
                # Check bottom neighbor
                if r + 1 < self.GRID_ROWS and self.grid[r][c] == self.grid[r+1][c] and self.grid[r][c] != -1:
                    moves.append(((c, r), (c, r + 1)))
        return moves

    def _check_termination(self):
        is_win = len(self.unified_colors) == self.NUM_COLORS
        is_loss = not is_win and len(self._find_all_valid_moves()) == 0
        return is_win, is_loss

    def _create_particles(self, grid_pos, color_index):
        cx = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        cy = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.COLORS[color_index]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": [cx, cy],
                "vel": vel,
                "radius": self.np_random.uniform(3, 6),
                "life": self.np_random.uniform(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['radius'] *= 0.97
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0.5]

    # --- Rendering Methods ---

    def _render_game(self):
        # Draw grid background and lines
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_COLS * self.CELL_SIZE, self.GRID_ROWS * self.CELL_SIZE)
        pygame.draw.rect(self.screen, (10, 10, 15), grid_rect)
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_ROWS * self.CELL_SIZE))
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_COLS * self.CELL_SIZE, y))

        # Draw squares
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r][c]
                if color_idx != -1:
                    is_unified = color_idx in self.unified_colors
                    base_color = self.BRIGHT_COLORS[color_idx] if is_unified else self.COLORS[color_idx]
                    
                    rect = pygame.Rect(self.GRID_X_OFFSET + c * self.CELL_SIZE, self.GRID_Y_OFFSET + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    
                    if is_unified:
                        pulse = (math.sin(self.steps * 0.2 + c + r) + 1) / 2
                        color = tuple(int(ch * (0.9 + pulse * 0.1)) for ch in base_color)
                        pygame.draw.rect(self.screen, color, rect.inflate(2, 2))
                    
                    pygame.gfxdraw.box(self.screen, rect.inflate(-4, -4), base_color)

        # Draw connection animation
        if self.connection_animation:
            pos1, pos2, color = self.connection_animation
            p1_center = (self.GRID_X_OFFSET + pos1[0] * self.CELL_SIZE + self.CELL_SIZE // 2, self.GRID_Y_OFFSET + pos1[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            p2_center = (self.GRID_X_OFFSET + pos2[0] * self.CELL_SIZE + self.CELL_SIZE // 2, self.GRID_Y_OFFSET + pos2[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            pygame.draw.line(self.screen, (255, 255, 255), p1_center, p2_center, 5)
            pygame.draw.line(self.screen, color, p1_center, p2_center, 3)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color_with_alpha = p['color'] + (alpha,)
            try:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color_with_alpha)
            except TypeError: # Sometimes alpha is not accepted, fallback
                pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))


        # Draw first selection highlight
        if self.first_selection:
            sx, sy = self.first_selection
            rect = pygame.Rect(self.GRID_X_OFFSET + sx * self.CELL_SIZE, self.GRID_Y_OFFSET + sy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 3)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X_OFFSET + cx * self.CELL_SIZE, self.GRID_Y_OFFSET + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        steps_text = self.font_main.render(f"Moves: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 20, 15))

        unified_text = self.font_small.render(f"Unified: {len(self.unified_colors)}/{self.NUM_COLORS}", True, self.COLOR_TEXT)
        self.screen.blit(unified_text, (20, 45))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Switch SDL video driver to play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Connect")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        # Pygame event handling for human input
        # Note: This is a simplified input handler. 
        # For smoother play, you might want to handle key-down events once.
        # This implementation re-triggers actions every frame a key is held.
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # One-shot actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action[2] = 1
        
        # Only step if an action was taken
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
        
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.1f} in {info['steps']} steps.")
                # Render final state
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                pygame.time.wait(2000) # Pause before restarting
                obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(env._get_observation(), (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
            
        # Control the frame rate
        clock.tick(15) 
        
    env.close()