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
        "Controls: Use arrow keys to move the cursor. Press space to select a tile. "
        "Move to an adjacent tile and press space again to swap. Press shift to deselect."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of three or more. "
        "Clear the entire board before you run out of moves!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    NUM_TILE_TYPES = 5
    BOARD_DIM = 360
    TILE_SIZE = BOARD_DIM // GRID_SIZE
    BOARD_OFFSET_X = 40
    BOARD_OFFSET_Y = (SCREEN_HEIGHT - BOARD_DIM) // 2

    MAX_MOVES = 15
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_UI_BG = (30, 40, 50)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_ACCENT = (100, 200, 255)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED_GLOW = (255, 255, 100)
    TILE_COLORS = [
        (255, 80, 80),  # Red
        (80, 255, 80),  # Green
        (80, 150, 255),  # Blue
        (255, 150, 50),  # Orange
        (200, 80, 255),  # Purple
    ]

    # --- Game Phases ---
    PHASE_INPUT = "INPUT"
    PHASE_SWAP_ANIM = "SWAP_ANIM"
    PHASE_MATCH_CHECK = "MATCH_CHECK"
    PHASE_CLEAR_ANIM = "CLEAR_ANIM"
    PHASE_GRAVITY = "GRAVITY"
    PHASE_REFILL = "REFILL"
    PHASE_GAME_OVER = "GAME_OVER"

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        self.np_random = None
        self.game_phase = self.PHASE_INPUT
        self.board = None
        self.cursor_pos = None
        self.selected_tile = None
        self.animations = []
        self.particles = []
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.reward_this_step = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.turn_score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False

        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.selected_tile = None

        self.animations = []
        self.particles = []

        self._generate_valid_board()

        self.game_phase = self.PHASE_INPUT
        self.reward_this_step = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.reward_this_step = 0

        self._update_game_state(movement, space_held, shift_held)

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:  # Max steps reached
            self.reward_this_step -= 50  # Penalty for timeout

        reward = self.reward_this_step

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, movement, space_held, shift_held):
        if self.game_phase == self.PHASE_INPUT:
            self._process_input(movement, space_held, shift_held)
        elif self.game_phase == self.PHASE_SWAP_ANIM:
            if not self.animations:
                self.game_phase = self.PHASE_MATCH_CHECK
        elif self.game_phase == self.PHASE_MATCH_CHECK:
            self._handle_match_check()
        elif self.game_phase == self.PHASE_CLEAR_ANIM:
            if not self.animations:
                self._handle_gravity()
                self.game_phase = self.PHASE_GRAVITY
        elif self.game_phase == self.PHASE_GRAVITY:
            if not self.animations:
                self._refill_board()
                self.game_phase = self.PHASE_REFILL
        elif self.game_phase == self.PHASE_REFILL:
            if not self.animations:
                self.game_phase = self.PHASE_MATCH_CHECK  # Check for cascades

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _process_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        if movement > 0:
            if movement == 1: self.cursor_pos[1] -= 1
            elif movement == 2: self.cursor_pos[1] += 1
            elif movement == 3: self.cursor_pos[0] -= 1
            elif movement == 4: self.cursor_pos[0] += 1
            self.cursor_pos[0] %= self.GRID_SIZE
            self.cursor_pos[1] %= self.GRID_SIZE

        # --- Deselection ---
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.selected_tile is not None:
            self.selected_tile = None
            # sfx: deselect_sound

        # --- Selection / Swap ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            if self.selected_tile is None:
                self.selected_tile = tuple(self.cursor_pos)
                # sfx: select_sound
            else:
                dist = abs(self.selected_tile[0] - self.cursor_pos[0]) + abs(self.selected_tile[1] - self.cursor_pos[1])
                if dist == 1:  # Adjacent tile
                    self._initiate_swap(self.selected_tile, tuple(self.cursor_pos))
                else:  # Not adjacent, re-select
                    self.selected_tile = tuple(self.cursor_pos)
                    # sfx: select_sound

    def _initiate_swap(self, pos1, pos2):
        self.moves_left -= 1
        self.turn_score = 0

        p1_x, p1_y = pos1
        p2_x, p2_y = pos2

        # Swap in the data model
        val1 = self.board[p1_y, p1_x]
        val2 = self.board[p2_y, p2_x]
        self.board[p1_y, p1_x] = val2
        self.board[p2_y, p2_x] = val1

        # Check if this swap creates a match
        is_match = self._check_for_match_at(p1_x, p1_y) or self._check_for_match_at(p2_x, p2_y)

        # Create swap animation
        self.animations.append({
            "type": "swap", "pos1": pos1, "pos2": pos2, "val1": val1, "val2": val2,
            "duration": 10, "timer": 10, "is_match": is_match
        })

        self.game_phase = self.PHASE_SWAP_ANIM
        self.selected_tile = None
        # sfx: swap_sound

    def _handle_match_check(self):
        matches = self._find_all_matches()
        if matches:
            # sfx: match_found_sound
            score_gain = 0
            for match in matches:
                if len(match) == 3: score_gain += 1
                elif len(match) == 4: score_gain += 2
                else: score_gain += 3

            self.turn_score += score_gain

            flat_matches = {pos for match in matches for pos in match}
            self.animations.append({"type": "clear", "tiles": flat_matches, "duration": 15, "timer": 15})

            for x, y in flat_matches:
                self._spawn_particles(x, y, self.board[y, x])
                self.board[y, x] = 0  # Mark for removal

            self.game_phase = self.PHASE_CLEAR_ANIM
        else:
            # Swap resulted in no match, swap back if it was a player move
            last_anim = self.animations[-1] if self.animations and self.animations[-1]['type'] == 'swap' else None
            if last_anim and not last_anim['is_match']:
                # sfx: invalid_swap_sound
                p1, p2 = last_anim['pos1'], last_anim['pos2']
                val1, val2 = self.board[p1[1], p1[0]], self.board[p2[1], p2[0]]
                self.board[p1[1], p1[0]], self.board[p2[1], p2[0]] = val2, val1
                self.reward_this_step = -0.1
                self._end_turn()
            else:
                self._end_turn()

    def _end_turn(self):
        self.score += self.turn_score
        self.reward_this_step += self.turn_score

        if np.all(self.board == 0):
            self.reward_this_step += 100
            self.game_over = True
            self.game_phase = self.PHASE_GAME_OVER
            # sfx: win_sound
        elif self.moves_left <= 0:
            self.reward_this_step -= 50
            self.game_over = True
            self.game_phase = self.PHASE_GAME_OVER
            # sfx: lose_sound
        elif not self._find_possible_swaps():
            # sfx: reshuffle_sound
            self._reshuffle_board()
            self.game_phase = self.PHASE_INPUT
        else:
            self.game_phase = self.PHASE_INPUT

    def _handle_gravity(self):
        cols_with_gaps = np.any(self.board == 0, axis=0)
        for x in np.where(cols_with_gaps)[0]:
            col = self.board[:, x]
            empty_cells = np.where(col == 0)[0]
            non_empty_cells = np.where(col != 0)[0]

            if not len(non_empty_cells): continue

            for y_empty in reversed(empty_cells):
                above_tiles = non_empty_cells[non_empty_cells < y_empty]
                if len(above_tiles):
                    y_fall_from = max(above_tiles)
                    val = self.board[y_fall_from, x]
                    self.board[y_empty, x] = val
                    self.board[y_fall_from, x] = 0

                    self.animations.append({
                        "type": "fall", "pos_from": (x, y_fall_from), "pos_to": (x, y_empty),
                        "val": val, "duration": 8, "timer": 8
                    })
                    # Update non_empty_cells for next iteration in same column
                    non_empty_cells = np.where(self.board[:, x] != 0)[0]

    def _refill_board(self):
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.board[y, x] == 0:
                    val = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
                    self.board[y, x] = val
                    self.animations.append({
                        "type": "fall", "pos_from": (x, y - self.GRID_SIZE), "pos_to": (x, y),
                        "val": val, "duration": 10, "timer": 10
                    })

    def _get_observation(self):
        # Update animations and particles
        self.animations = [anim for anim in self.animations if anim["timer"] > 0]
        for anim in self.animations: anim["timer"] -= 1

        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1

        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_ui_background()
        self._render_grid()
        self._render_tiles()
        self._render_selection()
        self._render_particles()
        self._render_animations()
        self._render_ui_text()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.BOARD_OFFSET_X + i * self.TILE_SIZE, self.BOARD_OFFSET_Y)
            end_pos = (self.BOARD_OFFSET_X + i * self.TILE_SIZE, self.BOARD_OFFSET_Y + self.BOARD_DIM)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y + i * self.TILE_SIZE)
            end_pos = (self.BOARD_OFFSET_X + self.BOARD_DIM, self.BOARD_OFFSET_Y + i * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_tiles(self):
        animating_tiles = set()
        for anim in self.animations:
            if anim['type'] == 'fall':
                animating_tiles.add(anim['pos_to'])
            elif anim['type'] == 'swap':
                animating_tiles.add(anim['pos1'])
                animating_tiles.add(anim['pos2'])

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) in animating_tiles: continue
                val = self.board[y, x]
                if val > 0:
                    self._draw_tile(x, y, val)

    def _render_selection(self):
        # Cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.BOARD_OFFSET_X + cx * self.TILE_SIZE,
            self.BOARD_OFFSET_Y + cy * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        alpha = 100 + 50 * math.sin(pygame.time.get_ticks() * 0.01)
        pygame.gfxdraw.box(self.screen, cursor_rect, (*self.COLOR_CURSOR, alpha))

        # Selected tile glow
        if self.selected_tile:
            sx, sy = self.selected_tile
            glow_size = self.TILE_SIZE + 8
            glow_rect = pygame.Rect(
                self.BOARD_OFFSET_X + sx * self.TILE_SIZE - (glow_size - self.TILE_SIZE) // 2,
                self.BOARD_OFFSET_Y + sy * self.TILE_SIZE - (glow_size - self.TILE_SIZE) // 2,
                glow_size, glow_size
            )
            alpha = 150 + 80 * math.sin(pygame.time.get_ticks() * 0.008)
            pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_SELECTED_GLOW, alpha))

    def _render_animations(self):
        for anim in self.animations:
            progress = 1 - (anim["timer"] / anim["duration"])
            if anim["type"] == "swap":
                x1, y1 = anim["pos1"]
                x2, y2 = anim["pos2"]
                px1 = x1 + (x2 - x1) * progress
                py1 = y1 + (y2 - y1) * progress
                px2 = x2 + (x1 - x2) * progress
                py2 = y2 + (y1 - y2) * progress
                self._draw_tile(px1, py1, anim["val1"])
                self._draw_tile(px2, py2, anim["val2"])
            elif anim["type"] == "clear":
                alpha = 255 * (anim["timer"] / anim["duration"])
                for x, y in anim["tiles"]:
                    rect = pygame.Rect(self.BOARD_OFFSET_X + x * self.TILE_SIZE,
                                       self.BOARD_OFFSET_Y + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                    pygame.gfxdraw.box(self.screen, rect, (255, 255, 255, alpha))
            elif anim["type"] == "fall":
                x_from, y_from = anim["pos_from"]
                x_to, y_to = anim["pos_to"]
                px = x_from + (x_to - x_from) * progress
                py = y_from + (y_to - y_from) * progress
                self._draw_tile(px, py, anim["val"])

    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p["life"] / p["max_life"])
            color = (*p["color"], alpha)
            size = p["size"] * (p["life"] / p["max_life"])
            rect = pygame.Rect(p["x"] - size / 2, p["y"] - size / 2, size, size)
            pygame.gfxdraw.box(self.screen, rect, color)

    def _render_ui_background(self):
        ui_rect = pygame.Rect(self.BOARD_OFFSET_X + self.BOARD_DIM, 0,
                                self.SCREEN_WIDTH - (self.BOARD_OFFSET_X + self.BOARD_DIM), self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        line_pos = self.BOARD_OFFSET_X + self.BOARD_DIM
        pygame.draw.line(self.screen, self.COLOR_GRID, (line_pos, 0), (line_pos, self.SCREEN_HEIGHT), 2)

    def _render_ui_text(self):
        ui_x = self.BOARD_OFFSET_X + self.BOARD_DIM + 20

        # Score
        score_label = self.font_medium.render("SCORE", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(score_label, (ui_x, 40))
        score_value = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_value, (ui_x, 70))

        # Moves
        moves_label = self.font_medium.render("MOVES LEFT", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(moves_label, (ui_x, 150))
        moves_value = self.font_large.render(f"{self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_value, (ui_x, 180))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            win_text = "BOARD CLEARED!" if np.all(self.board == 0) else "GAME OVER"
            text_surf = self.font_large.render(win_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0, 0))

    def _draw_tile(self, x, y, val):
        color = self.TILE_COLORS[val - 1]
        padding = self.TILE_SIZE * 0.1
        screen_x = self.BOARD_OFFSET_X + x * self.TILE_SIZE
        screen_y = self.BOARD_OFFSET_Y + y * self.TILE_SIZE

        # Outer shadow/base
        rect_base = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, tuple(c * 0.6 for c in color), rect_base, border_radius=int(self.TILE_SIZE * 0.2))

        # Inner gem
        rect_gem = pygame.Rect(
            screen_x + padding, screen_y + padding,
            self.TILE_SIZE - 2 * padding, self.TILE_SIZE - 2 * padding
        )
        pygame.draw.rect(self.screen, color, rect_gem, border_radius=int(self.TILE_SIZE * 0.15))

        # Highlight
        highlight_rect = pygame.Rect(
            screen_x + padding * 1.5, screen_y + padding * 1.5,
            (self.TILE_SIZE - 2 * padding) * 0.5, (self.TILE_SIZE - 2 * padding) * 0.5
        )
        pygame.gfxdraw.box(self.screen, highlight_rect, (255, 255, 255, 80))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _generate_valid_board(self):
        while True:
            self.board = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._find_possible_swaps():
                # Clear any initial matches
                while self._find_and_clear_initial_matches():
                    self._handle_gravity_instant()
                    self._refill_board_instant()

                # Final check if moves are still possible
                if self._find_possible_swaps():
                    break

    def _find_and_clear_initial_matches(self):
        matches = self._find_all_matches()
        if not matches:
            return False
        for match in matches:
            for x, y in match:
                self.board[y, x] = 0
        return True

    def _handle_gravity_instant(self):
        for x in range(self.GRID_SIZE):
            col = self.board[:, x]
            non_zeros = col[col != 0]
            zeros = col[col == 0]
            self.board[:, x] = np.concatenate((np.zeros(len(zeros)), non_zeros))

    def _refill_board_instant(self):
        zero_indices = np.where(self.board == 0)
        num_zeros = len(zero_indices[0])
        if num_zeros > 0:
            new_tiles = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=num_zeros)
            self.board[zero_indices] = new_tiles

    def _find_possible_swaps(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Swap right
                if x < self.GRID_SIZE - 1:
                    self.board[y, x], self.board[y, x + 1] = self.board[y, x + 1], self.board[y, x]
                    if self._check_for_match_at(x, y) or self._check_for_match_at(x + 1, y):
                        self.board[y, x], self.board[y, x + 1] = self.board[y, x + 1], self.board[y, x]
                        return True
                    self.board[y, x], self.board[y, x + 1] = self.board[y, x + 1], self.board[y, x]
                # Swap down
                if y < self.GRID_SIZE - 1:
                    self.board[y, x], self.board[y + 1, x] = self.board[y + 1, x], self.board[y, x]
                    if self._check_for_match_at(x, y) or self._check_for_match_at(x, y + 1):
                        self.board[y, x], self.board[y + 1, x] = self.board[y + 1, x], self.board[y, x]
                        return True
                    self.board[y, x], self.board[y + 1, x] = self.board[y + 1, x], self.board[y, x]
        return False

    def _check_for_match_at(self, x, y):
        val = self.board[y, x]
        if val == 0: return False
        # Horizontal
        h_count = 1
        for i in range(1, 3):
            if x - i >= 0 and self.board[y, x - i] == val: h_count += 1
            else: break
        for i in range(1, 3):
            if x + i < self.GRID_SIZE and self.board[y, x + i] == val: h_count += 1
            else: break
        if h_count >= 3: return True
        # Vertical
        v_count = 1
        for i in range(1, 3):
            if y - i >= 0 and self.board[y - i, x] == val: v_count += 1
            else: break
        for i in range(1, 3):
            if y + i < self.GRID_SIZE and self.board[y + i, x] == val: v_count += 1
            else: break
        if v_count >= 3: return True
        return False

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r, c + 1] == self.board[r, c + 2]:
                    match = frozenset([(c, r), (c + 1, r), (c + 2, r)])
                    matches.add(match)
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r + 1, c] == self.board[r + 2, c]:
                    match = frozenset([(c, r), (c, r + 1), (c, r + 2)])
                    matches.add(match)

        # Coalesce overlapping matches
        if not matches:
            return []

        coalesced = []
        while matches:
            current_match = set(matches.pop())
            while True:
                found_overlap = False
                remaining_matches = set()
                for other_match in matches:
                    if not current_match.isdisjoint(other_match):
                        current_match.update(other_match)
                        found_overlap = True
                    else:
                        remaining_matches.add(other_match)
                matches = remaining_matches
                if not found_overlap:
                    break
            coalesced.append(current_match)

        return coalesced

    def _reshuffle_board(self):
        flat_board = self.board.flatten()
        self.np_random.shuffle(flat_board)
        self.board = flat_board.reshape((self.GRID_SIZE, self.GRID_SIZE))
        if not self._find_possible_swaps():
            self._generate_valid_board()  # Failsafe

    def _spawn_particles(self, grid_x, grid_y, val):
        if val == 0: return
        color = self.TILE_COLORS[val - 1]
        center_x = self.BOARD_OFFSET_X + (grid_x + 0.5) * self.TILE_SIZE
        center_y = self.BOARD_OFFSET_Y + (grid_y + 0.5) * self.TILE_SIZE
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(15, 30)
            self.particles.append({
                "x": center_x, "y": center_y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "life": life, "max_life": life,
                "color": color, "size": random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # To play the game manually, you need a display.
    # The environment itself runs headlessly.
    # To run this block, comment out the `os.environ` line at the top.
    # For example: # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    try:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()

        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Match-3 Gymnasium Environment")
        clock = pygame.time.Clock()

        running = True
        while running:
            movement = 0
            space_held = False
            shift_held = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            if keys[pygame.K_SPACE]: space_held = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True

            action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
            obs, reward, terminated, truncated, info = env.step(action)

            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print("Game Over!")
                pygame.time.wait(2000)
                obs, info = env.reset()

            clock.tick(30)  # Run at 30 FPS

        pygame.quit()
    except pygame.error as e:
        print("\nPygame display error. This is expected if you are running in a headless environment.")
        print("To play manually, you need a display. Comment out the line 'os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")' at the top of the file.")
        # Test the headless version
        print("\nRunning a short headless test...")
        env = GameEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (400, 640, 3)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (400, 640, 3)
        print("Headless test passed.")