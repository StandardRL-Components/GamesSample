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

    user_guide = (
        "Use arrow keys to move the cursor. Press Space to select a gem, then move to an adjacent "
        "gem and press Space again to swap. Press Shift to cancel a selection."
    )

    game_description = (
        "Swap colorful gems to create lines of 3 or more. Plan your moves to create cascading "
        "combos and reach the target score before you run out of turns!"
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    GEM_TYPES = 6
    CELL_SIZE = 48
    GRID_X_OFFSET = (640 - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y_OFFSET = (400 - GRID_HEIGHT * CELL_SIZE) // 2 + 20

    SCORE_TARGET = 1000
    MAX_MOVES = 25

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID_BG = (25, 35, 55)
    COLOR_CURSOR = (255, 255, 255, 128)
    COLOR_SELECT = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)

    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]
    GEM_COLORS_DARK = [
        (120, 40, 40),
        (40, 120, 40),
        (40, 60, 120),
        (120, 120, 40),
        (120, 40, 120),
        (40, 120, 120),
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.steps = 0
        self.particles = []
        self.last_action_was_press = {'space': False, 'shift': False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.steps = 0
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self.particles = []
        self.last_action_was_press = {'space': False, 'shift': False}

        while True:
            self._init_grid()
            if self._find_all_possible_moves():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.steps += 1

        # --- Handle one-shot button presses ---
        space_pressed = space_action and not self.last_action_was_press['space']
        shift_pressed = shift_action and not self.last_action_was_press['shift']
        self.last_action_was_press['space'] = space_action
        self.last_action_was_press['shift'] = shift_action

        # --- Handle Input ---
        self._handle_movement(movement)

        if shift_pressed:
            self.selected_pos = None
            # Small penalty for canceling a selection? Maybe not.

        if space_pressed:
            reward += self._handle_selection()

        # --- Check for termination ---
        if self.moves_left <= 0 or self.score >= self.SCORE_TARGET:
            self.game_over = True
            if self.score >= self.SCORE_TARGET:
                reward += 100  # Win bonus
            else:
                reward -= 10 # Lose penalty

        return (
            self._get_observation(),
            float(np.clip(reward, -10.0, 100.0)),
            self.game_over,
            False,
            self._get_info()
        )

    def _init_grid(self):
        self.grid = self.np_random.integers(0, self.GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(0, self.GEM_TYPES)

    def _handle_movement(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _handle_selection(self):
        cx, cy = self.cursor_pos
        if self.selected_pos is None:
            self.selected_pos = [cx, cy]
            return 0 # No reward for just selecting
        else:
            sx, sy = self.selected_pos
            # Check for adjacency
            if abs(sx - cx) + abs(sy - cy) == 1:
                return self._attempt_swap(sx, sy, cx, cy)
            else: # Invalid selection (not adjacent)
                self.selected_pos = [cx, cy] # Select the new gem instead
                return 0

    def _attempt_swap(self, r1, c1, r2, c2):
        potential_matches_before = len(self._find_all_possible_moves())

        # Tentative swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        matches1 = self._find_matches_at(r1, c1)
        matches2 = self._find_matches_at(r2, c2)
        all_matches = matches1.union(matches2)

        if not all_matches:
            # Invalid move, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self.selected_pos = None
            return -1 # Penalty for invalid move

        # Valid move
        self.moves_left -= 1
        self.selected_pos = None
        combo_multiplier = 1.0
        total_match_reward = 0.0

        # --- Chain reaction loop ---
        while True:
            matches = self._find_matches()
            if not matches:
                break

            # Score and reward
            num_matched = len(matches)
            match_reward = num_matched * 10 * combo_multiplier
            total_match_reward += match_reward
            self.score += int(match_reward)

            # Create particles
            for r, c in matches:
                self._create_particles((c, r), self.GEM_COLORS[self.grid[r, c]])

            # Remove, drop, and refill
            self._remove_gems(matches)
            self._drop_gems()
            self._refill_gems()

            combo_multiplier += 0.5 # Increase multiplier for cascades

        potential_matches_after = len(self._find_all_possible_moves())
        potential_match_reward = (potential_matches_after - potential_matches_before)

        # Ensure new board state is not deadlocked
        if not self._find_all_possible_moves():
            while True:
                self._init_grid()
                if self._find_all_possible_moves():
                    break

        return total_match_reward + potential_match_reward

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return matches

    def _find_matches_at(self, r, c):
        """Finds matches involving a specific cell."""
        gem_type = self.grid[r, c]
        if gem_type == -1: return set()

        matches = set()
        # Horizontal
        h_line = [ (r, i) for i in range(self.GRID_WIDTH) if self.grid[r, i] == gem_type ]
        h_cont = self._find_contiguous_line(h_line, 1)
        for line in h_cont:
            if len(line) >= 3 and (r,c) in line:
                matches.update(line)
        # Vertical
        v_line = [ (i, c) for i in range(self.GRID_HEIGHT) if self.grid[i, c] == gem_type ]
        v_cont = self._find_contiguous_line(v_line, 0)
        for line in v_cont:
            if len(line) >= 3 and (r,c) in line:
                matches.update(line)
        return matches

    def _find_contiguous_line(self, points, axis):
        if not points: return []
        points.sort(key=lambda p: p[axis])
        groups = []
        current_group = [points[0]]
        for i in range(1, len(points)):
            if points[i][axis] == points[i-1][axis] + 1:
                current_group.append(points[i])
            else:
                groups.append(current_group)
                current_group = [points[i]]
        groups.append(current_group)
        return groups

    def _find_all_possible_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches_at(r, c) or self._find_matches_at(r, c+1):
                        moves.append(((r, c), (r, c+1)))
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c] # Swap back
                # Swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches_at(r, c) or self._find_matches_at(r+1, c):
                        moves.append(((r, c), (r+1, c)))
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c] # Swap back
        return moves

    def _remove_gems(self, matches):
        for r, c in matches:
            self.grid[r, c] = -1 # -1 represents an empty cell

    def _drop_gems(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1

    def _refill_gems(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.GEM_TYPES)

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
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
            "selected_pos": list(self.selected_pos) if self.selected_pos else None,
        }

    def _render_game(self):
        # Grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET,
                                self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=8)

        # Gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    self._draw_gem(c, r, gem_type)

        # Cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X_OFFSET + cx * self.CELL_SIZE,
                                  self.GRID_Y_OFFSET + cy * self.CELL_SIZE,
                                  self.CELL_SIZE, self.CELL_SIZE)

        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), border_radius=6)
        self.screen.blit(s, cursor_rect.topleft)

        # Selection highlight
        if self.selected_pos:
            sx, sy = self.selected_pos
            select_rect = pygame.Rect(self.GRID_X_OFFSET + sx * self.CELL_SIZE,
                                      self.GRID_Y_OFFSET + sy * self.CELL_SIZE,
                                      self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECT, select_rect, 4, border_radius=8)

        # Particles
        self._update_and_draw_particles()

    def _draw_gem(self, c, r, gem_type):
        rect = pygame.Rect(self.GRID_X_OFFSET + c * self.CELL_SIZE,
                           self.GRID_Y_OFFSET + r * self.CELL_SIZE,
                           self.CELL_SIZE, self.CELL_SIZE)
        center = rect.center
        radius = self.CELL_SIZE // 2 - 6

        color = self.GEM_COLORS[gem_type]
        dark_color = self.GEM_COLORS_DARK[gem_type]

        if gem_type == 0: # Circle (Red)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, dark_color)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
        elif gem_type == 1: # Square (Green)
            poly_rect = pygame.Rect(0, 0, radius * 1.8, radius * 1.8)
            poly_rect.center = center
            pygame.draw.rect(self.screen, color, poly_rect, 0, 4)
            pygame.draw.rect(self.screen, dark_color, poly_rect, 3, 4)
        elif gem_type == 2: # Diamond (Blue)
            points = [(center[0], center[1] - radius), (center[0] + radius, center[1]),
                      (center[0], center[1] + radius), (center[0] - radius, center[1])]
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 3: # Triangle (Yellow)
            points = [(center[0], center[1] - radius),
                      (center[0] - radius, center[1] + radius * 0.7),
                      (center[0] + radius, center[1] + radius * 0.7)]
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 4: # Hexagon (Magenta)
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 5: # Star (Cyan)
            points = []
            for i in range(10):
                r_val = radius if i % 2 == 0 else radius * 0.5
                angle = math.pi / 5 * i - math.pi / 2
                points.append((center[0] + r_val * math.cos(angle), center[1] + r_val * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(620, 15))
        self.screen.blit(moves_text, moves_rect)

        if self.game_over:
            s = pygame.Surface((640, 400), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))

            win_text = "YOU WIN!" if self.score >= self.SCORE_TARGET else "GAME OVER"
            end_text_surf = self.font_large.render(win_text, True, self.COLOR_SELECT)
            end_text_rect = end_text_surf.get_rect(center=(320, 200))
            self.screen.blit(end_text_surf, end_text_rect)

    def _create_particles(self, pos, color):
        px, py = self.GRID_X_OFFSET + pos[0] * self.CELL_SIZE + self.CELL_SIZE/2, \
                 self.GRID_Y_OFFSET + pos[1] * self.CELL_SIZE + self.CELL_SIZE/2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_and_draw_particles(self):
        if not self.particles:
            return

        active_particles = []
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # lifetime--
            if p[4] > 0:
                active_particles.append(p)
                radius = max(0, p[4] / 10)
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), int(radius), p[5])
        self.particles = active_particles

if __name__ == "__main__":
    # The validation part from the original code is removed as it's for testing, not running.
    # The main loop is for human play.
    env = GameEnv()
    obs, info = env.reset()

    # Create a display for human viewing
    pygame.display.set_caption("Gem Swap")
    display_surf = pygame.display.set_mode((640, 400))

    running = True
    while running:
        # --- Create an action from keyboard input ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        # --- Handle Pygame events ---
        should_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                should_step = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        # --- Step the environment ---
        # We step on any key press for a responsive human play experience.
        if should_step or any(action):
            obs, reward, terminated, truncated, info = env.step(action)

            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

            if terminated:
                print("Game Over!")
                # The game will show the final screen, press 'r' to restart

        # --- Render for human viewing ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_surf.blit(surf, (0, 0))
        pygame.display.flip()
        env.clock.tick(30)

    pygame.quit()