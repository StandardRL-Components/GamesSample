import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select/connect squares. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect all same-colored squares into single groups. Plan your moves to clear the board before you run out of moves or valid connections."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 8
        self.CELL_SIZE = 40
        self.GRID_MARGIN_X = (640 - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_MARGIN_Y = (400 - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.MAX_MOVES = 50
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.COLORS = [
            (255, 70, 70),   # Red
            (70, 200, 255),  # Blue
            (70, 255, 120),  # Green
            (255, 180, 70),  # Orange
            (200, 100, 255), # Purple
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)

        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_square = None
        self.dsu_parent = {}
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.win_message = ""

        # This is called here to ensure np_random is initialized before use.
        # However, reset() will be called again by the environment runner.
        self.reset()
        
        # self.validate_implementation() # Optional: Call for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_square = None
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.win_message = ""

        self._generate_valid_grid()
        self._rebuild_dsu()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        reward = -0.1 # Small penalty for taking a step (no-op)

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH

        # 2. Handle deselection
        if shift_pressed and self.selected_square is not None:
            self.selected_square = None
            reward = 0 # Neutral action

        # 3. Handle selection / connection
        elif space_pressed:
            cx, cy = self.cursor_pos
            if self.grid[cy][cx] is not None:
                if self.selected_square is None:
                    self.selected_square = (cx, cy)
                    reward = 0 # Neutral action
                else:
                    reward = self._handle_connection()

        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.win_message == "VICTORY!":
                self.score += 50
            else:
                self.score -= 10

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _generate_valid_grid(self):
        while True:
            # np_random.choice on a list of tuples/lists returns a numpy array
            self.grid = [[self.np_random.choice(self.COLORS) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
            if self._has_valid_moves():
                break

    def _rebuild_dsu(self):
        self.dsu_parent = {}
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] is not None:
                    self.dsu_parent[(c, r)] = (c, r)

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] is None:
                    continue
                # Check right neighbor
                if c + 1 < self.GRID_WIDTH and self.grid[r][c+1] is not None and np.array_equal(self.grid[r][c], self.grid[r][c+1]):
                    self._dsu_union((c, r), (c + 1, r))
                # Check down neighbor
                if r + 1 < self.GRID_HEIGHT and self.grid[r+1][c] is not None and np.array_equal(self.grid[r][c], self.grid[r+1][c]):
                    self._dsu_union((c, r), (c, r + 1))

    def _dsu_find(self, i):
        if self.dsu_parent.get(i) == i:
            return i
        if i not in self.dsu_parent:
            return None
        self.dsu_parent[i] = self._dsu_find(self.dsu_parent[i])
        return self.dsu_parent[i]

    def _dsu_union(self, i, j):
        root_i = self._dsu_find(i)
        root_j = self._dsu_find(j)
        if root_i is not None and root_j is not None and root_i != root_j:
            self.dsu_parent[root_j] = root_i

    def _handle_connection(self):
        c1, r1 = self.selected_square
        c2, r2 = self.cursor_pos
        
        # Check for valid connection
        is_adjacent = abs(c1 - c2) + abs(r1 - r2) == 1
        is_same_color = np.array_equal(self.grid[r1][c1], self.grid[r2][c2])
        
        if not (is_adjacent and is_same_color):
            self.selected_square = None
            return -0.5 # Penalty for invalid attempt

        # Perform connection
        self.moves_left -= 1
        reward = 1.0
        
        # Remove second square and create particles
        removed_color = self.grid[r2][c2]
        self.grid[r2][c2] = None
        self._spawn_particles((c2, r2), removed_color)

        # Gravity: squares fall down
        for r in range(r2, 0, -1):
            self.grid[r][c2] = self.grid[r-1][c2]
        self.grid[0][c2] = None

        # Column shift: if column is empty, shift columns left
        col_is_empty = all(self.grid[r][c2] is None for r in range(self.GRID_HEIGHT))
        if col_is_empty:
            reward += 5.0 # Column clear bonus
            for c in range(c2, self.GRID_WIDTH - 1):
                for r in range(self.GRID_HEIGHT):
                    self.grid[r][c] = self.grid[r][c+1]
            for r in range(self.GRID_HEIGHT):
                self.grid[r][self.GRID_WIDTH-1] = None

        self.selected_square = None
        self._rebuild_dsu()
        return reward

    def _check_termination(self):
        if self.game_over:
            return True

        # Win condition
        color_sets = {}
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] is not None:
                    color = tuple(self.grid[r][c])
                    if color not in color_sets:
                        color_sets[color] = set()
                    root = self._dsu_find((c, r))
                    if root:
                        color_sets[color].add(root)
        
        is_win = all(len(s) <= 1 for s in color_sets.values()) and any(color_sets)
        if is_win:
            self.game_over = True
            self.win_message = "VICTORY!"
            return True

        # Loss conditions
        if self.moves_left <= 0:
            self.game_over = True
            self.win_message = "OUT OF MOVES"
            return True
        
        if not self._has_valid_moves():
            self.game_over = True
            self.win_message = "NO MOVES LEFT"
            return True

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_message = "TIME LIMIT"
            return True
        
        return False

    def _has_valid_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] is None:
                    continue
                # Check right
                if c + 1 < self.GRID_WIDTH and self.grid[r][c+1] is not None and np.array_equal(self.grid[r][c], self.grid[r][c+1]):
                    return True
                # Check down
                if r + 1 < self.GRID_HEIGHT and self.grid[r+1][c] is not None and np.array_equal(self.grid[r][c], self.grid[r+1][c]):
                    return True
        return False

    def _render_game(self):
        # Draw grid lines and connection lines
        self._draw_connection_lines()
        
        # Draw squares
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color = self.grid[r][c]
                if color is not None:
                    rect = pygame.Rect(
                        self.GRID_MARGIN_X + c * self.CELL_SIZE,
                        self.GRID_MARGIN_Y + r * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=5)

        # Draw selected square highlight
        if self.selected_square:
            c, r = self.selected_square
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 5
            rect = pygame.Rect(
                self.GRID_MARGIN_X + c * self.CELL_SIZE,
                self.GRID_MARGIN_Y + r * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect.inflate(-2, -2), width=int(2 + pulse/3), border_radius=6)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_MARGIN_X + cx * self.CELL_SIZE,
            self.GRID_MARGIN_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=7)
        
        # Update and draw particles
        self._update_and_draw_particles()

    def _draw_connection_lines(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                pos1_center = (
                    self.GRID_MARGIN_X + c * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.GRID_MARGIN_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                )
                
                # Check right neighbor
                if c + 1 < self.GRID_WIDTH and self._dsu_find((c,r)) == self._dsu_find((c+1, r)):
                    pos2_center = (pos1_center[0] + self.CELL_SIZE, pos1_center[1])
                    if self.grid[r][c] is not None:
                        pygame.draw.line(self.screen, self.grid[r][c], pos1_center, pos2_center, 8)
                
                # Check down neighbor
                if r + 1 < self.GRID_HEIGHT and self._dsu_find((c,r)) == self._dsu_find((c, r+1)):
                    pos2_center = (pos1_center[0], pos1_center[1] + self.CELL_SIZE)
                    if self.grid[r][c] is not None:
                        pygame.draw.line(self.screen, self.grid[r][c], pos1_center, pos2_center, 8)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (640 - moves_text.get_width() - 15, 15))

        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_text = self.font_msg.render(self.win_message, True, self.COLOR_CURSOR)
            msg_rect = msg_text.get_rect(center=(320, 200))
            self.screen.blit(msg_text, msg_rect)

    def _spawn_particles(self, grid_pos, color):
        c, r = grid_pos
        px = self.GRID_MARGIN_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_MARGIN_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(3, 8),
                'color': color,
                'life': self.np_random.integers(20, 41)
            })

    def _update_and_draw_particles(self):
        remaining_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] *= 0.95
            p['life'] -= 1
            if p['life'] > 0 and p['radius'] > 0.5:
                # Use gfxdraw for anti-aliased circles
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])
                remaining_particles.append(p)
        self.particles = remaining_particles
        
    def validate_implementation(self):
        ''' Call this at the end of __init__ to verify implementation. '''
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will create a window and render the game
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Connect-the-Squares")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    last_action_time = pygame.time.get_ticks()
    action_delay = 100 # ms between actions for smoother human play

    while running:
        now = pygame.time.get_ticks()
        action = [0, 0, 0] # Default no-op action
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()

        # In auto_advance=False mode, we only step when there's an action
        # For human play, we can poll keys and step periodically
        if now - last_action_time > action_delay:
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            # Only step if there's a meaningful action to avoid rapid state changes
            if movement != 0 or space_held != env.last_space_held or shift_held != env.last_shift_held:
                action = [movement, space_held, shift_held]
                obs, reward, terminated, truncated, info = env.step(action)
                last_action_time = now
                if terminated:
                    print(f"Game Over! Final Score: {info['score']:.1f}")
            else: # If no action, just get the latest observation for rendering animations
                obs = env._get_observation()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if env.game_over and (now - last_action_time > 3000): # Wait 3s after game over
             obs, info = env.reset()

        clock.tick(60) # Limit frame rate

    pygame.quit()