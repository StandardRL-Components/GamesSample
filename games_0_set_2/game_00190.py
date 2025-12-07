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
    user_guide = "Controls: Use arrow keys to move the cursor. Press space to swap the selected tile with the one in the direction you last moved. Press shift to reshuffle the board (costs a move and a point)."

    # Must be a short, user-facing description of the game:
    game_description = "A puzzle game where you swap tiles to create matches of 3 or more. Clear the entire board before you run out of moves to win!"

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.TILE_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.TILE_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.TILE_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.NUM_TILE_TYPES = 5
        self.MAX_MOVES = 60

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINES = (40, 50, 70)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 255)
        self.TILE_COLORS = {
            1: (220, 50, 50),  # Red
            2: (50, 220, 50),  # Green
            3: (50, 100, 220),  # Blue
            4: (220, 220, 50),  # Yellow
            5: (180, 50, 220),  # Purple
        }
        self.TILE_HIGHLIGHT_COLORS = {
            k: tuple(min(255, c + 50) for c in v) for k, v in self.TILE_COLORS.items()
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)

        # --- Game State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.last_move_direction = 0  # 0:none, 1:up, 2:down, 3:left, 4:right
        self.moves_remaining = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # This will be properly seeded in reset()
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.last_move_direction = 0
        self.particles = []

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.moves_remaining -= 1
        reward = 0
        self.particles.clear()  # Clear particles from the previous step

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        previous_last_move_direction = self.last_move_direction

        # 1. Handle cursor movement
        if movement > 0:
            self.last_move_direction = movement
            if movement == 1:  # Up
                self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS
            elif movement == 2:  # Down
                self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_ROWS
            elif movement == 3:  # Left
                self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_COLS) % self.GRID_COLS
            elif movement == 4:  # Right
                self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_COLS

        # 2. Handle game actions (Shift overrides Space)
        if shift_held:
            self._reshuffle()
            reward = -1.0
        elif space_held:
            # Use the direction from before this step's movement if no new movement occurred
            swap_dir = self.last_move_direction if movement > 0 else previous_last_move_direction
            if swap_dir != 0:
                swap_reward, cleared_coords = self._perform_swap_and_clear(swap_dir)
                reward += swap_reward
                if swap_reward > 0:
                    for r, c in cleared_coords:
                        self._spawn_particles(c, r)

        # 3. Check for termination
        board_cleared = self._is_board_clear()
        # FIX: Cast the result to a standard Python bool to satisfy Gymnasium's type check.
        # The expression `... or board_cleared` can result in a `numpy.bool_` if `board_cleared` is one.
        terminated = bool(self.moves_remaining <= 0 or board_cleared)
        if terminated:
            self.game_over = True
            if board_cleared:
                reward += 100.0
            else:
                reward -= 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_remaining": self.moves_remaining,
        }

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X_OFFSET, y),
                             (self.GRID_X_OFFSET + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y_OFFSET),
                             (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT))

        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_type = self.grid[r, c]
                if tile_type > 0:
                    self._draw_tile(c, r, tile_type)

        # Draw cursor
        cursor_x = self.GRID_X_OFFSET + self.cursor_pos[0] * self.TILE_SIZE
        cursor_y = self.GRID_Y_OFFSET + self.cursor_pos[1] * self.TILE_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.TILE_SIZE, self.TILE_SIZE), 4,
                         border_radius=5)

        # Draw particles
        self._update_and_draw_particles()

    def _draw_tile(self, c, r, tile_type):
        x = self.GRID_X_OFFSET + c * self.TILE_SIZE
        y = self.GRID_Y_OFFSET + r * self.TILE_SIZE
        color = self.TILE_COLORS[tile_type]
        highlight_color = self.TILE_HIGHLIGHT_COLORS[tile_type]

        # Beveled look
        pygame.draw.rect(self.screen, color, (x + 2, y + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4), border_radius=6)
        pygame.draw.rect(self.screen, highlight_color, (x + 4, y + 4, self.TILE_SIZE - 12, self.TILE_SIZE - 12),
                         border_radius=4)

    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            status_text = "BOARD CLEARED!" if self._is_board_clear() else "OUT OF MOVES"
            win_text_render = self.font_main.render(status_text, True, self.COLOR_CURSOR)
            win_text_rect = win_text_render.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(win_text_render, win_text_rect)

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS))
        while True:
            matches = self._find_matches()
            if not matches:
                if self._check_for_possible_moves():
                    break
                else:  # No matches and no possible moves, reshuffle
                    self._reshuffle(clear_zeros=False)
            else:  # Has matches, fix them
                for r, c in matches:
                    # Pick a new color that is different from its neighbors to break the match
                    original_color = self.grid[r, c]
                    possible_colors = list(range(1, self.NUM_TILE_TYPES + 1))
                    if original_color in possible_colors:
                        possible_colors.remove(original_color)
                    if not possible_colors: # Should not happen with >1 tile type
                         possible_colors = [((original_color % self.NUM_TILE_TYPES) + 1)]
                    self.grid[r, c] = self.np_random.choice(possible_colors)

    def _find_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r, c + 1] == self.grid[r, c + 2]:
                    val = self.grid[r, c]
                    i = c
                    while i < self.GRID_COLS and self.grid[r, i] == val:
                        matches.add((r, i))
                        i += 1
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r + 1, c] == self.grid[r + 2, c]:
                    val = self.grid[r, c]
                    i = r
                    while i < self.GRID_ROWS and self.grid[i, c] == val:
                        matches.add((i, c))
                        i += 1
        return matches

    def _check_for_possible_moves(self):
        temp_grid = np.copy(self.grid)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    temp_grid[r, c], temp_grid[r, c + 1] = temp_grid[r, c + 1], temp_grid[r, c]
                    if self._find_matches_on_grid(temp_grid):
                        return True
                    temp_grid[r, c], temp_grid[r, c + 1] = temp_grid[r, c + 1], temp_grid[r, c]  # Swap back
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    temp_grid[r, c], temp_grid[r + 1, c] = temp_grid[r + 1, c], temp_grid[r, c]
                    if self._find_matches_on_grid(temp_grid):
                        return True
                    temp_grid[r, c], temp_grid[r + 1, c] = temp_grid[r + 1, c], temp_grid[r, c]  # Swap back
        return False

    def _find_matches_on_grid(self, grid):
        # A version of _find_matches that works on a provided grid, for checking possibilities
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r, c + 1] == grid[r, c + 2]:
                    return True
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r + 1, c] == grid[r + 2, c]:
                    return True
        return False

    def _reshuffle(self, clear_zeros=True):
        if clear_zeros and np.any(self.grid > 0):
            non_zero_tiles = self.grid[self.grid > 0].flatten().tolist()
            self.np_random.shuffle(non_zero_tiles)

            new_grid = np.zeros_like(self.grid)
            fill_idx = 0
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if self.grid[r, c] > 0:
                        if fill_idx < len(non_zero_tiles):
                            new_grid[r, c] = non_zero_tiles[fill_idx]
                            fill_idx += 1
            self.grid = new_grid
        else:
            flat_grid = self.grid.flatten()
            self.np_random.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_ROWS, self.GRID_COLS))

        # Ensure no matches and at least one move after reshuffle
        while True:
            if not self._find_matches():
                if self._check_for_possible_moves():
                    break
                else:  # Reshuffled into another dead end, try again
                    self._reshuffle(clear_zeros)
            else:  # Reshuffled into a match, fix it
                self._generate_board()  # Easiest way to guarantee a valid board
                break

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            write_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != write_row:
                        self.grid[write_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    write_row -= 1

    def _perform_swap_and_clear(self, direction):
        c1, r1 = self.cursor_pos
        c2, r2 = c1, r1

        if direction == 1:
            r2 -= 1  # Up
        elif direction == 2:
            r2 += 1  # Down
        elif direction == 3:
            c2 -= 1  # Left
        elif direction == 4:
            c2 += 1  # Right

        if not (0 <= c2 < self.GRID_COLS and 0 <= r2 < self.GRID_ROWS):
            return 0, set()  # Invalid swap

        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        total_cleared_coords = set()
        chain_reward = 0

        while True:
            matches = self._find_matches()
            if not matches:
                break

            num_cleared = len(matches)
            chain_reward += num_cleared
            self.score += num_cleared
            total_cleared_coords.update(matches)

            for r_match, c_match in matches:
                self.grid[r_match, c_match] = 0

            self._apply_gravity()

        if chain_reward == 0:  # No match, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        return chain_reward, total_cleared_coords

    def _is_board_clear(self):
        return np.all(self.grid == 0)

    def _spawn_particles(self, c, r):
        px = self.GRID_X_OFFSET + (c + 0.5) * self.TILE_SIZE
        py = self.GRID_Y_OFFSET + (r + 0.5) * self.TILE_SIZE

        for _ in range(10):  # Spawn 10 particles
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(2, 5)
            color_val_tuple = self.TILE_HIGHLIGHT_COLORS[self.np_random.integers(1, len(self.TILE_HIGHLIGHT_COLORS)+1)]
            self.particles.append([[px, py], vel, color_val_tuple, size])

    def _update_and_draw_particles(self):
        # Since auto_advance=False, particles only live for one frame.
        # We just draw them where they would be after one "tick".
        for p in self.particles:
            p[0][0] += p[1][0]  # pos.x += vel.x
            p[0][1] += p[1][1]  # pos.y += vel.y

            pos = (int(p[0][0]), int(p[0][1]))
            color = p[2]
            size = int(p[3])

            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()

    running = True

    pygame.display.set_caption("Match-3 Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    last_action_time = pygame.time.get_ticks()
    ACTION_COOLDOWN = 150  # ms

    while running:
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        # Only process actions if cooldown has passed
        if current_time - last_action_time > ACTION_COOLDOWN:
            keys = pygame.key.get_pressed()

            movement = 0
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            space_held = keys[pygame.K_SPACE]
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

            # An action is anything other than doing nothing
            if movement != 0 or space_held or shift_held:
                action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
                obs, reward, terminated, truncated, info = env.step(action)
                last_action_time = current_time  # Reset cooldown

                print(
                    f"Action: {action}, Reward: {reward:.2f}, Moves Left: {info['moves_remaining']}, Score: {info['score']}")

                if terminated:
                    print("--- GAME OVER ---")
                    print(f"Final Score: {info['score']}")

        # Render the environment's observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(60)

    env.close()