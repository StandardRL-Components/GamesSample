import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to select a crystal. "
        "Arrows again to select an adjacent empty tile to move to. "
        "Space to cancel selection."
    )

    game_description = (
        "An isometric puzzle game. Move crystals to adjacent empty spaces "
        "to form matches of 3 or more. Plan your moves carefully to "
        "reach the target score before you run out of moves."
    )

    auto_advance = False

    # --- Game Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    NUM_COLORS = 4
    MOVES_LIMIT = 20
    TARGET_CRYSTALS = 50

    # --- Visuals ---
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (60, 60, 70)
    CRYSTAL_COLORS = [
        (255, 80, 80),  # Red
        (80, 255, 80),  # Green
        (80, 150, 255),  # Blue
        (255, 240, 80),  # Yellow
    ]
    TILE_WIDTH = 50
    TILE_HEIGHT = TILE_WIDTH * 0.5

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

        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_info = pygame.font.SysFont("monospace", 18)
        self.font_gameover = pygame.font.SysFont("monospace", 48, bold=True)

        self.origin_x = self.screen.get_width() // 2
        self.origin_y = 100

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)

        # A seed is required for the first reset to create the np_random generator
        self.reset(seed=random.randint(0, 1_000_000_000))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.moves_left = self.MOVES_LIMIT
        self.game_over = False

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_crystal = None
        self.animations = []
        self.particles = []

        # State machine for game flow
        # INPUT: waiting for player action
        # ANIMATING: waiting for animations (swap, fall) to finish
        # RESOLVE: board has changed, check for matches and cascades
        self.game_phase = "INPUT"

        # Loop until a valid starting board is generated.
        # A valid board has no initial matches (handled by _generate_board)
        # and at least one possible move.
        while True:
            self._generate_board()
            if self._check_for_possible_moves():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Since auto_advance is False, this function processes one logical step
        # which might involve a chain of internal state changes (match -> fall -> match)
        movement, space_held, shift_held = action
        reward = 0
        terminated = False

        self.steps += 1

        # --- Handle Input Phase ---
        if self.game_phase == "INPUT":
            # --- Handle Cursor Movement ---
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

            cx, cy = self.cursor_pos

            if self.selected_crystal is None:
                # --- Handle Crystal Selection ---
                if space_held and self.grid[cy, cx] > 0:
                    self.selected_crystal = (cx, cy)
                    # sfx: select_crystal.wav
            else:
                # --- Handle Swap or Cancel ---
                if space_held:  # Cancel selection
                    self.selected_crystal = None
                    # sfx: cancel.wav
                elif movement != 0:  # Attempt a swap
                    sx, sy = self.selected_crystal
                    nx, ny = sx, sy
                    # The move is relative to the cursor, not the selected crystal
                    nx, ny = self.cursor_pos
                    
                    # Check if the target is adjacent to the selected crystal
                    is_adjacent = abs(sx - nx) + abs(sy - ny) == 1

                    if is_adjacent and 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == 0:
                        # Valid move: swap crystal with empty space
                        self.grid[ny, nx] = self.grid[sy, sx]
                        self.grid[sy, sx] = 0
                        self.selected_crystal = None
                        self.moves_left -= 1
                        reward -= 1  # Small penalty for using a move
                        self.game_phase = "RESOLVE"
                        # sfx: swap.wav
                    else:
                        # Invalid move (not empty or out of bounds)
                        self.selected_crystal = None  # Deselect on failed attempt
                        # sfx: error.wav

        # --- Handle Resolution and Cascades ---
        if self.game_phase == "RESOLVE":
            chain = 0
            while True:
                matches = self._find_all_matches()
                if not matches:
                    break

                chain += 1
                num_matched = len(matches)
                self.crystals_collected += num_matched
                reward += num_matched * chain  # Reward combos more
                self.score += num_matched * 10 * chain

                if num_matched >= 5:
                    reward += 5  # Bonus for large clusters

                # sfx: match_clear.wav
                for x, y in matches:
                    self._create_particles(x, y, self.grid[y, x])
                    self.grid[y, x] = 0

                self._apply_gravity()
                self._refill_board()

        # --- Update Particles and Animations (even though we don't have long ones) ---
        self._update_particles()

        # --- Check for Game Over Conditions ---
        if not self.game_over:
            win = self.crystals_collected >= self.TARGET_CRYSTALS
            lose_moves = self.moves_left <= 0
            lose_stuck = not self._check_for_possible_moves() and self.game_phase == "INPUT"

            if win:
                reward += 100
                self.game_over = True
                terminated = True
                # sfx: win_jingle.wav
            elif lose_moves or lose_stuck:
                reward -= 100
                self.game_over = True
                terminated = True
                # sfx: lose_fanfare.wav

        # After a move resolves, return to input phase
        if self.game_phase == "RESOLVE":
            self.game_phase = "INPUT"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_screen(0, r)
            p2 = self._iso_to_screen(self.GRID_WIDTH, r)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_screen(c, 0)
            p2 = self._iso_to_screen(c, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # Draw crystals
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    self._draw_crystal(c, r, self.grid[r, c])

        # Draw selection and cursor
        self._draw_cursor_and_selection()

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

    def _draw_cursor_and_selection(self):
        # Draw selected crystal highlight
        if self.selected_crystal:
            sx, sy = self.selected_crystal
            self._draw_crystal_outline(sx, sy, (255, 255, 255), 4)

            # Draw valid move indicators on adjacent empty cells
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == 0:
                    center_pos = self._iso_to_screen(nx + 0.5, ny + 0.5)
                    pygame.draw.circle(self.screen, (255, 255, 255, 100), center_pos, 8)

        # Draw cursor
        cx, cy = self.cursor_pos
        self._draw_crystal_outline(cx, cy, (255, 255, 0), 2)

    def _render_ui(self):
        # Collected Crystals
        collect_text = self.font_main.render(f"CRYSTALS: {self.crystals_collected}/{self.TARGET_CRYSTALS}", True,
                                             (255, 255, 255))
        self.screen.blit(collect_text, (20, 10))

        # Moves Left
        moves_text = self.font_main.render(f"MOVES: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_text, (self.screen.get_width() - moves_text.get_width() - 20, 10))

        # Score
        score_text = self.font_info.render(f"SCORE: {self.score}", True, (200, 200, 200))
        score_rect = score_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() - 20))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.crystals_collected >= self.TARGET_CRYSTALS:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            end_text = self.font_gameover.render(msg, True, color)
            end_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_collected": self.crystals_collected,
            "moves_left": self.moves_left,
            "game_phase": self.game_phase,
        }

    # --- Helper Functions ---

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.origin_y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _draw_crystal(self, x, y, color_index):
        center_x, center_y = self._iso_to_screen(x + 0.5, y + 0.5)
        color = self.CRYSTAL_COLORS[color_index - 1]

        # Glow effect
        glow_color = (*color, 50)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 18, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 14, glow_color)

        # Crystal shape (rhombus)
        points = [
            self._iso_to_screen(x, y + 0.5),
            self._iso_to_screen(x + 0.5, y),
            self._iso_to_screen(x + 1, y + 0.5),
            self._iso_to_screen(x + 0.5, y + 1)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Highlight
        highlight_points = [points[0], points[1], self._iso_to_screen(x + 0.5, y + 0.5)]
        highlight_color = (min(255, color[0] + 60), min(255, color[1] + 60), min(255, color[2] + 60))
        pygame.gfxdraw.filled_polygon(self.screen, highlight_points, highlight_color)
        pygame.gfxdraw.aapolygon(self.screen, highlight_points, highlight_color)

    def _draw_crystal_outline(self, x, y, color, width):
        points = [
            self._iso_to_screen(x, y),
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1),
            self._iso_to_screen(x, y + 1)
        ]
        pygame.draw.lines(self.screen, color, True, points, width)

    def _generate_board(self):
        # Create a board with a mix of crystals and empty spaces (e.g., 20% empty).
        num_cells = self.GRID_HEIGHT * self.GRID_WIDTH
        num_empty = int(num_cells * 0.2)
        num_crystals = num_cells - num_empty

        # Generate the required number of crystals and empty tiles
        crystals = self.np_random.integers(1, self.NUM_COLORS + 1, size=num_crystals)
        empties = np.zeros(num_empty, dtype=int)

        # Combine, shuffle, and reshape to form the grid
        flat_grid = np.concatenate((crystals, empties))
        self.np_random.shuffle(flat_grid)
        self.grid = flat_grid.reshape((self.GRID_HEIGHT, self.GRID_WIDTH))

        # Ensure no initial matches
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            for x, y in matches:
                # Replace the matched crystal with a new random one. This preserves
                # the number of empty spaces on the board.
                self.grid[y, x] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r, c] > 0 and self.grid[r, c] == self.grid[r, c + 1] == self.grid[r, c + 2]:
                    matches.update([(c, r), (c + 1, r), (c + 2, r)])
        # Vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r, c] > 0 and self.grid[r, c] == self.grid[r + 1, c] == self.grid[r + 2, c]:
                    matches.update([(c, r), (c, r + 1), (c, r + 2)])
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] > 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _refill_board(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _check_for_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:  # If it's a crystal
                    # Check moving to adjacent empty spaces
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nc, nr = c + dx, r + dy
                        if 0 <= nc < self.GRID_WIDTH and 0 <= nr < self.GRID_HEIGHT and self.grid[nr, nc] == 0:
                            # Simulate the move
                            temp_grid = np.copy(self.grid)
                            temp_grid[nr, nc], temp_grid[r, c] = temp_grid[r, c], temp_grid[nr, nc]
                            # Check if this move creates a match
                            if self._check_match_at(temp_grid, nc, nr):
                                return True
        return False

    def _check_match_at(self, grid, c, r):
        color = grid[r, c]
        if color == 0: return False
        # Horizontal check
        h_count = 1
        # Check left
        for i in range(1, 3):
            if c - i >= 0 and grid[r, c - i] == color:
                h_count += 1
            else:
                break
        # Check right
        for i in range(1, 3):
            if c + i < self.GRID_WIDTH and grid[r, c + i] == color:
                h_count += 1
            else:
                break
        if h_count >= 3: return True
        
        # Vertical check
        v_count = 1
        # Check up
        for i in range(1, 3):
            if r - i >= 0 and grid[r - i, c] == color:
                v_count += 1
            else:
                break
        # Check down
        for i in range(1, 3):
            if r + i < self.GRID_HEIGHT and grid[r + i, c] == color:
                v_count += 1
            else:
                break
        if v_count >= 3: return True
        
        return False

    def _create_particles(self, x, y, color_index):
        center_pos = self._iso_to_screen(x + 0.5, y + 0.5)
        base_color = self.CRYSTAL_COLORS[color_index - 1]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(center_pos),
                'vel': vel,
                'size': random.uniform(2, 5),
                'lifespan': random.randint(20, 40),
                'color': base_color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()

    obs, info = env.reset()
    done = False

    # Pygame setup for human play
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Action mapping for human play ---
        movement = 0  # none
        space = 0
        shift = 0  # unused in this implementation

        # Use keydown events for single presses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_r:  # Reset game
                    obs, info = env.reset()
                    done = False

        action = [movement, space, shift]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Reward: {reward}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)  # Limit frame rate

    env.close()