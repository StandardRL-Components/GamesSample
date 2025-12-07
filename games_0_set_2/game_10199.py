import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:05:16.244211
# Source Brief: brief_00199.md
# Brief Index: 199
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A real-time puzzle game where the player swaps oscillating colored squares
    on a grid to create matches of 3 or more. The goal is to clear a target
    number of squares before time runs out.

    The environment is designed with a focus on visual polish and responsive
    game feel, featuring smooth animations, particle effects, and a clean UI.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Swap colored squares to create matches of three or more. Clear the target number of squares before time runs out to win."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to swap the selected square with its neighbor in the direction of your last move."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 8
    CELL_SIZE = 40
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20

    # Colors
    COLOR_BG = (26, 28, 44)
    COLOR_GRID_LINES = (40, 42, 60)
    SQUARE_COLORS = [
        (243, 139, 168),  # Red/Pink
        (166, 227, 161),  # Green
        (137, 180, 250),  # Blue
        (250, 227, 176),  # Yellow
        (180, 190, 254),  # Lavender
        (148, 226, 213),  # Teal
    ]
    COLOR_TEXT = (205, 214, 244)
    COLOR_CURSOR = (249, 226, 175)

    # Game Parameters
    MAX_STEPS = 90 * metadata['render_fps']  # 90 seconds
    WIN_CONDITION_CLEARS = 30
    INTERP_SPEED = 0.25  # For smooth falling/swapping

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_info = pygame.font.SysFont("Consolas", 18)

        self.grid = []
        self.cursor_pos = [0, 0]
        self.last_move_dir = [0, 0]
        self.last_space_held = False
        self.steps = 0
        self.score = 0.0
        self.cleared_squares_count = 0
        self.game_over = False
        self.particles = []
        self.match_highlights = deque()

        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is a helper, not part of the API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.cleared_squares_count = 0
        self.game_over = False
        self.last_space_held = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.last_move_dir = [1, 0] # Default to right
        self.particles.clear()
        self.match_highlights.clear()

        self._generate_initial_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = bool(space_held)

        self.steps += 1
        reward = 0.0

        # 1. Handle player input
        self._handle_input(movement, space_pressed)

        # 2. Process game logic (cascading matches)
        cascade_reward = self._process_cascades()
        reward += cascade_reward

        # 3. Update animations and timers
        self._update_animations()

        # 4. Continuous adjacency reward
        reward += self._calculate_adjacency_reward()

        # 5. Check for termination
        terminated = self._check_termination()
        truncated = False # No truncation condition other than termination
        if terminated and not self.game_over:
            if self.cleared_squares_count >= self.WIN_CONDITION_CLEARS:
                reward += 100.0 # Win bonus
            else:
                reward -= 100.0 # Time out penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_initial_grid(self):
        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        while True:
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if self.grid[r][c] is None:
                        color_idx = self.np_random.integers(0, len(self.SQUARE_COLORS))
                        self.grid[r][c] = self._create_square(r, c, color_idx)
            if not self._find_matches():
                break
            # If matches exist on spawn, clear and refill
            matches = self._find_matches()
            for r, c in matches:
                self.grid[r][c] = None

    def _create_square(self, r, c, color_idx, start_pos_y_offset=0):
        pixel_x, pixel_y = self._grid_to_pixel(r, c)
        return {
            "color_idx": color_idx,
            "pos": [r, c],
            "pixel_pos": [pixel_x, pixel_y - start_pos_y_offset],
            "target_pixel_pos": [pixel_x, pixel_y],
            "osc_timer": self.np_random.uniform(0, 2 * math.pi),
            "matched": False,
        }

    def _handle_input(self, movement, space_pressed):
        # Move cursor (wraps around)
        if movement == 1: # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS
            self.last_move_dir = [0, -1]
        elif movement == 2: # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_ROWS
            self.last_move_dir = [0, 1]
        elif movement == 3: # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_COLS) % self.GRID_COLS
            self.last_move_dir = [-1, 0]
        elif movement == 4: # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_COLS
            self.last_move_dir = [1, 0]

        # Swap squares
        if space_pressed:
            c1, r1 = self.cursor_pos
            c2, r2 = c1 + self.last_move_dir[0], r1 + self.last_move_dir[1]

            if 0 <= c2 < self.GRID_COLS and 0 <= r2 < self.GRID_ROWS:
                # # Sound placeholder: pygame.mixer.Sound('swap.wav').play()
                sq1, sq2 = self.grid[r1][c1], self.grid[r2][c2]
                if sq1 and sq2: # Can't swap with an empty space
                    self.grid[r1][c1], self.grid[r2][c2] = sq2, sq1
                    sq1["pos"] = [r2, c2]
                    sq2["pos"] = [r1, c1]

    def _process_cascades(self):
        total_reward = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break

            # # Sound placeholder: pygame.mixer.Sound('match.wav').play()
            num_cleared = len(matches)
            if num_cleared == 3: total_reward += 1
            elif num_cleared == 4: total_reward += 2
            elif num_cleared >= 5: total_reward += 3

            self.cleared_squares_count += num_cleared
            self.score += num_cleared

            for r, c in matches:
                square = self.grid[r][c]
                if square:
                    self._spawn_particles(r, c, square["color_idx"])
                    self.match_highlights.append((self._grid_to_pixel(r, c), 6)) # 6 frames duration
                    self.grid[r][c] = None

            self._apply_gravity()
            self._fill_top_rows()
        return total_reward

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] is None: continue
                color = self.grid[r][c]["color_idx"]

                # Horizontal match
                h_match = [ (r, c) ]
                for i in range(1, self.GRID_COLS - c):
                    if self.grid[r][c+i] and self.grid[r][c+i]["color_idx"] == color:
                        h_match.append((r, c+i))
                    else: break
                if len(h_match) >= 3:
                    matches.update(h_match)

                # Vertical match
                v_match = [ (r, c) ]
                for i in range(1, self.GRID_ROWS - r):
                    if self.grid[r+i][c] and self.grid[r+i][c]["color_idx"] == color:
                        v_match.append((r+i, c))
                    else: break
                if len(v_match) >= 3:
                    matches.update(v_match)
        return list(matches)

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] is not None:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = None
                        self.grid[empty_row][c]["pos"] = [empty_row, c]
                    empty_row -= 1

    def _fill_top_rows(self):
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.grid[r][c] is None:
                    color_idx = self.np_random.integers(0, len(self.SQUARE_COLORS))
                    self.grid[r][c] = self._create_square(r, c, color_idx, start_pos_y_offset=self.CELL_SIZE)

    def _update_animations(self):
        # Update squares
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                square = self.grid[r][c]
                if square:
                    # Update target pixel position based on logical grid position
                    target_x, target_y = self._grid_to_pixel(square["pos"][0], square["pos"][1])
                    square["target_pixel_pos"] = [target_x, target_y]

                    # Interpolate pixel position
                    px, py = square["pixel_pos"]
                    tx, ty = square["target_pixel_pos"]
                    square["pixel_pos"][0] += (tx - px) * self.INTERP_SPEED
                    square["pixel_pos"][1] += (ty - py) * self.INTERP_SPEED

                    # Update oscillation
                    square["osc_timer"] += 0.1

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["vel"][1] += 0.1 # Gravity

        # Update match highlights
        new_highlights = deque()
        while self.match_highlights:
            pos, timer = self.match_highlights.popleft()
            if timer > 1:
                new_highlights.append((pos, timer - 1))
        self.match_highlights = new_highlights

    def _calculate_adjacency_reward(self):
        adj_reward = 0.0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                square = self.grid[r][c]
                if not square: continue
                # Check right neighbor
                if c + 1 < self.GRID_COLS and self.grid[r][c+1] and self.grid[r][c+1]["color_idx"] == square["color_idx"]:
                    adj_reward += 0.01
                # Check down neighbor
                if r + 1 < self.GRID_ROWS and self.grid[r+1][c] and self.grid[r+1][c]["color_idx"] == square["color_idx"]:
                    adj_reward += 0.01
        return adj_reward

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self.cleared_squares_count >= self.WIN_CONDITION_CLEARS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame array3d is (width, height, 3), we need (height, width, 3) for gym
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cleared_squares": self.cleared_squares_count,
            "time_left": (self.MAX_STEPS - self.steps) / self.metadata['render_fps']
        }

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT))

        # Draw match highlights
        for pos, timer in self.match_highlights:
            alpha = int(255 * (timer / 6))
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((255, 255, 255, alpha))
            self.screen.blit(s, pos)

        # Draw squares
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                square = self.grid[r][c]
                if square:
                    self._draw_square(square)

        # Draw cursor
        self._draw_cursor()

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 30.0))))
            color = self.SQUARE_COLORS[p["color_idx"]]
            size = int(p["size"] * (p["life"] / 30.0))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, (*color, alpha))

    def _draw_square(self, square):
        osc_x = math.sin(square["osc_timer"]) * 2
        osc_y = math.cos(square["osc_timer"] * 0.7) * 2
        
        px, py = square["pixel_pos"]
        x, y = int(px + osc_x), int(py + osc_y)
        
        color = self.SQUARE_COLORS[square["color_idx"]]
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Draw a slightly smaller, filled rectangle for a beveled look
        inner_rect = rect.inflate(-6, -6)
        pygame.draw.rect(self.screen, color, inner_rect, border_radius=5)
        
        # Draw a border
        pygame.draw.rect(self.screen, tuple(min(255, c+30) for c in color), inner_rect, width=2, border_radius=5)

    def _draw_cursor(self):
        c, r = self.cursor_pos
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        thickness = 2 + int(pulse * 2)
        
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, width=thickness, border_radius=7)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.metadata['render_fps'])
        time_text = self.font_main.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Cleared squares
        cleared_text = self.font_info.render(f"CLEARED: {self.cleared_squares_count} / {self.WIN_CONDITION_CLEARS}", True, self.COLOR_TEXT)
        cleared_rect = cleared_text.get_rect(midbottom=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10))
        self.screen.blit(cleared_text, cleared_rect)

    def _grid_to_pixel(self, r, c):
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
        return x, y

    def _spawn_particles(self, r, c, color_idx):
        # # Sound placeholder: pygame.mixer.Sound('particle_burst.wav').play()
        px, py = self._grid_to_pixel(r, c)
        center_x, center_y = px + self.CELL_SIZE / 2, py + self.CELL_SIZE / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(20, 40),
                "size": self.np_random.integers(3, 6),
                "color_idx": color_idx
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block will not run in the test environment, but is useful for local testing.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Match Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    last_keys = pygame.key.get_pressed()
    
    while running:
        # Default action is "do nothing"
        action = [0, 0, 0] # [movement, space, shift]
        
        # Detect key presses for one-shot actions (like swapping)
        keys = pygame.key.get_pressed()
        
        # Movement is continuous, but we only want to register one press per frame
        # to avoid overly sensitive controls. Here, we'll just take the first one found.
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Space is a hold action
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    terminated = False
        
        if not terminated:
            # The environment advances one frame per step
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Cleared: {info['cleared_squares']}")

            if terminated:
                print("--- GAME OVER ---")
                if info['cleared_squares'] >= GameEnv.WIN_CONDITION_CLEARS:
                    print("YOU WIN!")
                else:
                    print("TIME'S UP!")
                print(f"Final Score: {info['score']}")
        
        # Display the observation from the environment
        # The observation is (H, W, C), but pygame needs (W, H, C)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.metadata['render_fps'])
        
        last_keys = keys
        
    env.close()