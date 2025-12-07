import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a gem. "
        "Move the cursor to an adjacent gem and press space again to swap. "
        "Press shift to deselect."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Create cascades and "
        "big matches to maximize your score. Reach 1000 points in 20 moves to win!"
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 60)

        # --- Game Constants ---
        self.GRID_COLS, self.GRID_ROWS = 8, 8
        self.NUM_GEM_TYPES = 6
        self.GEM_SIZE = 48
        self.GRID_START_X = (640 - self.GRID_COLS * self.GEM_SIZE) // 2
        self.GRID_START_Y = (400 - self.GRID_ROWS * self.GEM_SIZE) + 10

        self.TARGET_SCORE = 1000
        self.INITIAL_MOVES = 20
        self.ANIMATION_SPEED = 0.15  # Progress per frame

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 0, 100)
        self.COLOR_SELECT = (255, 255, 255, 150)
        self.GEM_COLORS = [
            (255, 80, 80),  # Red
            (80, 255, 80),  # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 160, 80),  # Orange
        ]

        # --- State Variables ---
        self.grid = None
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.game_won = False
        self.terminal_reward_given = False
        self.cursor_pos = (0, 0)  # Use tuple for hashability
        self.selected_gem = None
        self.particles = []

        # Game Phase State Machine
        self.game_phase = "IDLE"  # IDLE, SWAP, MATCH, CLEAR, FALL, RESHUFFLE
        self.animation_progress = 0
        self.animation_data = {}
        self.pending_reward = 0

        # self.reset() is called by the wrapper, but we need to init for validation
        # self.validate_implementation() # This would fail as RNG is not seeded yet.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_remaining = self.INITIAL_MOVES
        self.game_over = False
        self.game_won = False
        self.terminal_reward_given = False
        self.cursor_pos = (self.GRID_ROWS // 2, self.GRID_COLS // 2)  # Use tuple
        self.selected_gem = None
        self.particles = []
        self.game_phase = "IDLE"
        self.animation_progress = 0
        self.animation_data = {}
        self.pending_reward = 0

        self._initialize_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self.pending_reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.game_phase == "IDLE":
            self._handle_input(movement, space_held, shift_held)
        else:
            self._update_animations()

        terminated = self.game_over
        if terminated and not self.terminal_reward_given:
            if self.game_won:
                self.pending_reward += 100
            else:
                self.pending_reward += -10
            self.terminal_reward_given = True

        return (
            self._get_observation(),
            self.pending_reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space, shift):
        # Only process input when idle and not game over
        if self.game_phase != "IDLE" or self.game_over:
            return

        # Shift to deselect
        if shift and self.selected_gem:
            self.selected_gem = None
            # sfx: deselect sound

        # Cursor movement
        if movement > 0:
            prev_pos = self.cursor_pos
            r, c = self.cursor_pos
            if movement == 1:
                self.cursor_pos = (max(0, r - 1), c)
            elif movement == 2:
                self.cursor_pos = (min(self.GRID_ROWS - 1, r + 1), c)
            elif movement == 3:
                self.cursor_pos = (r, max(0, c - 1))
            elif movement == 4:
                self.cursor_pos = (r, min(self.GRID_COLS - 1, c + 1))
            if prev_pos != self.cursor_pos:
                pass  # sfx: cursor move sound

        # Space to select/swap
        if space:
            r, c = self.cursor_pos
            if not self.selected_gem:
                self.selected_gem = (r, c)
                # sfx: select sound
            else:
                sr, sc = self.selected_gem
                # Check for adjacency
                if abs(r - sr) + abs(c - sc) == 1:
                    self._start_swap(self.selected_gem, self.cursor_pos)
                    self.selected_gem = None
                else:  # Non-adjacent click, treat as new selection
                    self.selected_gem = (r, c)
                    # sfx: select sound

    def _start_swap(self, pos1, pos2, is_revert=False):
        self.game_phase = "SWAP"
        self.animation_progress = 0
        self.animation_data = {
            "pos1": pos1,
            "pos2": pos2,
            "revert_on_fail": not is_revert,
        }
        if not is_revert:
            self.moves_remaining -= 1
        # sfx: swap sound

        # Perform swap on grid model
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _update_animations(self):
        self.animation_progress = min(1.0, self.animation_progress + self.ANIMATION_SPEED)

        if self.game_phase == "SWAP":
            if self.animation_progress >= 1.0:
                matches = self._find_matches()
                if matches and self.animation_data.get("revert_on_fail", False):
                    self.animation_data = {"matches": matches}
                    self.game_phase = "CLEAR"
                    self.animation_progress = 0
                elif self.animation_data.get("revert_on_fail", False):
                    self.pending_reward += -0.1
                    self._start_swap(
                        self.animation_data["pos1"],
                        self.animation_data["pos2"],
                        is_revert=True,
                    )
                    # sfx: invalid swap sound
                else:  # Revert swap finished
                    self.game_phase = "IDLE"
                    self._check_game_state()

        elif self.game_phase == "CLEAR":
            if self.animation_progress >= 1.0:
                matches = self.animation_data["matches"]
                # Calculate score and create particles
                num_cleared = len(matches)
                self.pending_reward += num_cleared
                if num_cleared == 4:
                    self.pending_reward += 5
                if num_cleared >= 5:
                    self.pending_reward += 10
                self.score += self.pending_reward

                for r, c in matches:
                    self._create_particles((r, c), self.GEM_COLORS[self.grid[r, c] - 1])
                    self.grid[r, c] = 0  # Mark as empty
                # sfx: match sound

                self.game_phase = "FALL"
                self.animation_progress = 0
                self._apply_gravity_and_find_falls()

        elif self.game_phase == "FALL":
            if self.animation_progress >= 1.0:
                # After falling, check for new matches (cascades)
                matches = self._find_matches()
                if matches:
                    self.animation_data = {"matches": matches}
                    self.game_phase = "CLEAR"
                    self.animation_progress = 0
                    # sfx: cascade match sound
                else:
                    self.game_phase = "IDLE"
                    self._check_game_state()

        elif self.game_phase == "RESHUFFLE":
            if self.animation_progress >= 1.0:
                self._initialize_board()
                self.game_phase = "IDLE"

    def _check_game_state(self):
        # Check for win/loss conditions
        if self.score >= self.TARGET_SCORE:
            self.game_over = True
            self.game_won = True
        elif self.moves_remaining <= 0:
            self.game_over = True
            self.game_won = False

        # Check for no more moves
        if not self.game_over and not self._find_all_possible_swaps():
            self.game_phase = "RESHUFFLE"
            self.animation_progress = 0
            self.animation_data = {}  # Gems fly out anim
            # sfx: reshuffle whoosh

    def _apply_gravity_and_find_falls(self):
        falls = {}
        for c in range(self.GRID_COLS):
            write_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != write_row:
                        falls[(r, c)] = write_row - r
                        self.grid[write_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    write_row -= 1

            # Fill empty top rows with new gems
            new_gem_start_row = -1
            for r in range(write_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
                falls[(new_gem_start_row, c)] = write_row - new_gem_start_row
                new_gem_start_row -= 1

        self.animation_data = {"falls": falls}
        if not falls:
            self.game_phase = "IDLE"
            self._check_game_state()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_bg()
        self._render_gems()
        self._render_particles()
        self._render_cursor_and_selection()
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "moves_remaining": self.moves_remaining}

    def _initialize_board(self):
        while True:
            self.grid = self.np_random.integers(
                1, self.NUM_GEM_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS)
            )

            # Remove initial matches
            while True:
                matches = self._find_matches()
                if not matches:
                    break
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

            # Ensure at least one move is possible
            if self._find_all_possible_swaps():
                break

    def _find_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if (
                    self.grid[r, c] != 0
                    and self.grid[r, c] == self.grid[r, c + 1] == self.grid[r, c + 2]
                ):
                    matches.update([(r, c), (r, c + 1), (r, c + 2)])
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if (
                    self.grid[r, c] != 0
                    and self.grid[r, c] == self.grid[r + 1, c] == self.grid[r + 2, c]
                ):
                    matches.update([(r, c), (r + 1, c), (r + 2, c)])
        return list(matches)

    def _find_all_possible_swaps(self):
        swaps = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c]
                    if self._find_matches():
                        swaps.append(((r, c), (r, c + 1)))
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c]  # Swap back
                # Swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c]
                    if self._find_matches():
                        swaps.append(((r, c), (r + 1, c)))
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c]  # Swap back
        return swaps

    def _create_particles(self, pos, color):
        r, c = pos
        center_x = self.GRID_START_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.GRID_START_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.randint(15, 30)
            self.particles.append([center_x, center_y, vx, vy, life, color])

    def _render_grid_bg(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(
                    self.GRID_START_X + c * self.GEM_SIZE,
                    self.GRID_START_Y + r * self.GEM_SIZE,
                    self.GEM_SIZE,
                    self.GEM_SIZE,
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_gems(self):
        rendered_gems = set()

        # Handle animated gems first
        if self.game_phase == "SWAP":
            p1, p2 = self.animation_data["pos1"], self.animation_data["pos2"]
            rendered_gems.add(p1)
            rendered_gems.add(p2)

            gem1_type = self.grid[p2[0], p2[1]]  # Grid is already swapped
            gem2_type = self.grid[p1[0], p1[1]]

            self._draw_gem_at_interp_pos(p1, p2, self.animation_progress, gem1_type)
            self._draw_gem_at_interp_pos(p2, p1, self.animation_progress, gem2_type)

        elif self.game_phase == "CLEAR":
            matches = self.animation_data["matches"]
            for r, c in matches:
                scale = 1.0 - self.animation_progress
                self._draw_gem(r, c, self.grid[r, c], scale)
                rendered_gems.add((r, c))

        elif self.game_phase == "FALL":
            for (start_r, start_c), dist in self.animation_data["falls"].items():
                end_r = start_r + dist
                gem_type = self.grid[end_r, start_c]

                y_offset = (1.0 - self.animation_progress) * dist * self.GEM_SIZE
                self._draw_gem(end_r, start_c, gem_type, 1.0, y_offset_pixels=-y_offset)

                # Mark all cells in the column as rendered to avoid double drawing
                for r_idx in range(self.GRID_ROWS):
                    rendered_gems.add((r_idx, start_c))

        elif self.game_phase == "RESHUFFLE":
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    scale = 1.0 - self.animation_progress
                    offset_x = (c - self.GRID_COLS / 2) * 80 * self.animation_progress
                    offset_y = (r - self.GRID_ROWS / 2) * 80 * self.animation_progress
                    self._draw_gem(r, c, self.grid[r, c], scale, offset_x, offset_y)
            return  # All gems are animated

        # Render static gems
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (r, c) not in rendered_gems and self.grid[r, c] != 0:
                    self._draw_gem(r, c, self.grid[r, c])

    def _draw_gem_at_interp_pos(self, pos_start, pos_end, progress, gem_type):
        r1, c1 = pos_start
        r2, c2 = pos_end

        x1 = self.GRID_START_X + c1 * self.GEM_SIZE
        y1 = self.GRID_START_Y + r1 * self.GEM_SIZE
        x2 = self.GRID_START_X + c2 * self.GEM_SIZE
        y2 = self.GRID_START_Y + r2 * self.GEM_SIZE

        ix = x1 + (x2 - x1) * progress
        iy = y1 + (y2 - y1) * progress

        self._draw_gem_pixel(ix, iy, gem_type)

    def _draw_gem(self, r, c, gem_type, scale=1.0, x_offset_pixels=0, y_offset_pixels=0):
        if gem_type == 0:
            return

        size = int(self.GEM_SIZE * scale)
        margin = (self.GEM_SIZE - size) // 2

        x = self.GRID_START_X + c * self.GEM_SIZE + margin + x_offset_pixels
        y = self.GRID_START_Y + r * self.GEM_SIZE + margin + y_offset_pixels

        self._draw_gem_pixel(x, y, gem_type, size)

    def _draw_gem_pixel(self, x, y, gem_type, size=None):
        if size is None:
            size = self.GEM_SIZE
        if size <= 0:
            return

        color = self.GEM_COLORS[gem_type - 1]

        padding = int(size * 0.1)
        inner_size = size - padding * 2

        rect = pygame.Rect(int(x + padding), int(y + padding), inner_size, inner_size)

        # Using gfxdraw for antialiasing
        pygame.gfxdraw.box(self.screen, rect, color)

        # Highlight
        highlight_color = (255, 255, 255, 90)
        highlight_rect = pygame.Rect(
            rect.left + int(inner_size * 0.1),
            rect.top + int(inner_size * 0.1),
            int(inner_size * 0.5),
            int(inner_size * 0.2),
        )
        pygame.gfxdraw.box(self.screen, highlight_rect, highlight_color)

    def _render_particles(self):
        for p in self.particles[:]:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1  # life -= 1
            if p[4] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p[4] / 20.0))))
                color = p[5] + (alpha,)
                size = max(1, int(p[4] / 6))
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), size, color)

    def _render_cursor_and_selection(self):
        if self.game_over:
            return

        # Draw selection
        if self.selected_gem:
            r, c = self.selected_gem
            rect = pygame.Rect(
                self.GRID_START_X + c * self.GEM_SIZE,
                self.GRID_START_Y + r * self.GEM_SIZE,
                self.GEM_SIZE,
                self.GEM_SIZE,
            )
            pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_SELECT)
            pygame.gfxdraw.rectangle(self.screen, rect.inflate(2, 2), self.COLOR_SELECT)

        # Draw cursor
        r, c = self.cursor_pos
        cursor_surf = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
        cursor_surf.fill(self.COLOR_CURSOR)
        self.screen.blit(
            cursor_surf,
            (
                self.GRID_START_X + c * self.GEM_SIZE,
                self.GRID_START_Y + r * self.GEM_SIZE,
            ),
        )

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        moves_text = self.font_ui.render(
            f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT
        )
        self.screen.blit(moves_text, (620 - moves_text.get_width(), 10))

    def _render_game_over(self):
        overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        message = "YOU WIN!" if self.game_won else "GAME OVER"
        text = self.font_game_over.render(message, True, (255, 255, 100))
        text_rect = text.get_rect(center=(320, 200))

        overlay.blit(text, text_rect)
        self.screen.blit(overlay, (0, 0))

    def close(self):
        pygame.quit()