
# Generated: 2025-08-28T07:00:03.714321
# Source Brief: brief_03109.md
# Brief Index: 3109

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a tile. "
        "Select an adjacent tile to swap. Shift to deselect."
    )

    game_description = (
        "Swap adjacent tiles to match 3 or more of the same color. "
        "Clear the entire board before the timer runs out!"
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)

        # --- Game Constants ---
        self.GRID_SIZE = 10
        self.GRID_OFFSET_X = (self.screen_width - self.screen_height) // 2
        self.GRID_OFFSET_Y = 0
        self.TILE_SIZE = self.screen_height // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.INITIAL_TIME = 60.0
        self.TIME_PER_STEP = 0.05 # Time penalty for thinking

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_EMPTY = (30, 35, 60)
        self.TILE_COLORS = {
            1: (220, 50, 50),   # Red
            2: (50, 220, 50),   # Green
            3: (50, 100, 220),  # Blue
            4: (220, 220, 50),  # Yellow
            5: (150, 50, 220)   # Purple
        }
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 165, 0)

        # --- Game State (initialized in reset) ---
        self.board = None
        self.cursor_pos = None
        self.selected_tile = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.timer = None
        self.space_pressed_last_frame = None
        self.shift_pressed_last_frame = None
        self.animations = None
        self.combo_multiplier = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.board = self._generate_board()
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.timer = self.INITIAL_TIME
        self.space_pressed_last_frame = True
        self.shift_pressed_last_frame = True
        self.animations = []
        self.combo_multiplier = 1

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= self.TIME_PER_STEP
        reward = -0.01  # Small penalty for each step taken

        self._update_animations()

        if not self.animations:
            self._handle_input(action)

        match_data = self._find_and_process_matches()
        if match_data:
            num_cleared = match_data["num_cleared"]
            reward += num_cleared * self.combo_multiplier
            if num_cleared == 4: reward += 5
            if num_cleared >= 5: reward += 10
            self.score += int(reward)
            self.combo_multiplier += 0.5
        elif not self.animations:
             self.combo_multiplier = 1 # Reset combo if no new matches and animations are done

        terminated = self._check_termination()
        if terminated:
            if self._is_board_clear():
                reward += 100 # Win bonus
                self.score += 100
            else:
                reward -= 100 # Loss penalty
                self.score -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Treat held as a single press event
        space_pressed = space_held and not self.space_pressed_last_frame
        shift_pressed = shift_held and not self.shift_pressed_last_frame
        self.space_pressed_last_frame = space_held
        self.shift_pressed_last_frame = shift_held

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_SIZE
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_SIZE
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE

        # --- Shift (Deselect) ---
        if shift_pressed and self.selected_tile:
            self.selected_tile = None
            # sound effect: deselect

        # --- Space (Select/Swap) ---
        if space_pressed:
            x, y = self.cursor_pos
            if self.board[y, x] == 0: # Cannot select empty tile
                self.selected_tile = None
                return

            if not self.selected_tile:
                self.selected_tile = (x, y)
                # sound effect: select
            else:
                sx, sy = self.selected_tile
                dist = abs(sx - x) + abs(sy - y)
                if dist == 1: # Is adjacent
                    self._initiate_swap((sx, sy), (x, y))
                else: # Not adjacent, treat as new selection
                    self.selected_tile = (x, y)
                    # sound effect: select

    def _initiate_swap(self, pos1, pos2):
        self.animations.append({
            "type": "swap", "pos1": pos1, "pos2": pos2, "progress": 0, "duration": 10,
        })
        self.selected_tile = None
        # sound effect: swap

    def _update_animations(self):
        if not self.animations:
            return

        # Process the first animation in the queue
        anim = self.animations[0]
        anim["progress"] += 1

        if anim["progress"] >= anim["duration"]:
            # Finish animation
            if anim["type"] == "swap":
                x1, y1 = anim["pos1"]
                x2, y2 = anim["pos2"]
                # Perform the swap on the board
                self.board[y1, x1], self.board[y2, y2] = self.board[y2, y2], self.board[y1, x1]

                # Check if this swap creates a match
                matches1 = self._find_matches_at(x1, y1)
                matches2 = self._find_matches_at(x2, y2)
                if not matches1 and not matches2:
                    # Invalid swap, swap back
                    self.board[y1, x1], self.board[y2, y2] = self.board[y2, y2], self.board[y1, x1]
                    # sound effect: invalid_swap
                else:
                    # Valid swap, will be processed by match logic
                    # sound effect: match_found
                    pass

            elif anim["type"] == "clear":
                for x, y in anim["tiles"]:
                    self.board[y, x] = 0 # Set to empty

            elif anim["type"] == "fall":
                self._apply_gravity()
                self._refill_board()

            self.animations.pop(0) # Remove completed animation

    def _find_and_process_matches(self):
        # This function runs even if there are animations, to chain combos
        if any(a["type"] == "clear" for a in self.animations):
            return None # Don't look for new matches while clearing

        all_matches = self._find_all_matches()
        if all_matches:
            self.animations.append({
                "type": "clear", "tiles": all_matches, "progress": 0, "duration": 15,
            })
            self.animations.append({
                "type": "fall", "progress": 0, "duration": 10,
            })
            return {"num_cleared": len(all_matches)}
        return None

    def _apply_gravity(self):
        for x in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.board[y, x] != 0:
                    if y != empty_row:
                        self.board[empty_row, x] = self.board[y, x]
                        self.board[y, x] = 0
                    empty_row -= 1

    def _refill_board(self):
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.board[y, x] == 0:
                    self.board[y, x] = self.np_random.integers(1, len(self.TILE_COLORS) + 1)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.timer <= 0 or self.steps >= self.MAX_STEPS or self._is_board_clear():
            self.game_over = True
            return True
        return False

    def _is_board_clear(self):
        return np.all(self.board == 0)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer}

    def _render_game(self):
        self._render_grid()
        self._render_tiles()
        self._render_cursor()

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_SIZE * self.TILE_SIZE, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_tiles(self):
        swap_anim = next((a for a in self.animations if a["type"] == "swap"), None)

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_id = self.board[y, x]
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.TILE_SIZE,
                    self.GRID_OFFSET_Y + y * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                
                # Handle animations affecting this tile
                pos_x, pos_y = rect.topleft
                size = self.TILE_SIZE
                
                # Check for clearing animation
                clear_anim = next((a for a in self.animations if a["type"] == "clear" and (x, y) in a["tiles"]), None)
                if clear_anim:
                    progress = clear_anim["progress"] / clear_anim["duration"]
                    size = int(self.TILE_SIZE * (1 - progress))
                    pos_x += (self.TILE_SIZE - size) // 2
                    pos_y += (self.TILE_SIZE - size) // 2
                    # particle effect

                # Handle swap animation
                if swap_anim:
                    p = swap_anim["progress"] / swap_anim["duration"]
                    x1, y1 = swap_anim["pos1"]
                    x2, y2 = swap_anim["pos2"]
                    if (x, y) == (x1, y1):
                        pos_x = int(pygame.math.lerp(rect.x, self.GRID_OFFSET_X + x2 * self.TILE_SIZE, p))
                        pos_y = int(pygame.math.lerp(rect.y, self.GRID_OFFSET_Y + y2 * self.TILE_SIZE, p))
                    elif (x, y) == (x2, y2):
                        pos_x = int(pygame.math.lerp(rect.x, self.GRID_OFFSET_X + x1 * self.TILE_SIZE, p))
                        pos_y = int(pygame.math.lerp(rect.y, self.GRID_OFFSET_Y + y1 * self.TILE_SIZE, p))

                anim_rect = pygame.Rect(pos_x, pos_y, size, size)

                if color_id == 0:
                    pygame.draw.rect(self.screen, self.COLOR_EMPTY, anim_rect.inflate(-1, -1))
                else:
                    color = self.TILE_COLORS[color_id]
                    pygame.gfxdraw.box(self.screen, anim_rect.inflate(-4, -4), color)
                    # Add a subtle highlight
                    highlight_color = tuple(min(255, c + 40) for c in color)
                    pygame.draw.rect(self.screen, highlight_color, anim_rect.inflate(-4, -4), 2, border_radius=4)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cx * self.TILE_SIZE,
            self.GRID_OFFSET_Y + cy * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        
        # Pulsing effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        line_width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width, border_radius=3)

        if self.selected_tile:
            sx, sy = self.selected_tile
            selected_rect = pygame.Rect(
                self.GRID_OFFSET_X + sx * self.TILE_SIZE,
                self.GRID_OFFSET_Y + sy * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, selected_rect, 4, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        timer_text = self.font_large.render(f"Time: {max(0, int(self.timer))}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.screen_width - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_msg = "BOARD CLEARED!" if self._is_board_clear() else "TIME'S UP!"
            msg_render = self.font_large.render(win_msg, True, self.COLOR_SELECTED)
            msg_rect = msg_render.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 20))
            self.screen.blit(msg_render, msg_rect)

            final_score_render = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_render.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 20))
            self.screen.blit(final_score_render, final_score_rect)


    # --- Board Generation and Match Logic ---

    def _generate_board(self):
        while True:
            board = self.np_random.integers(1, len(self.TILE_COLORS) + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            # Remove initial matches
            while self._find_all_matches(board):
                matches = self._find_all_matches(board)
                for x, y in matches:
                    board[y, x] = self.np_random.integers(1, len(self.TILE_COLORS) + 1)
            
            # Check for possible moves
            if self._has_possible_moves(board):
                return board

    def _has_possible_moves(self, board):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Try swapping right
                if x < self.GRID_SIZE - 1:
                    temp_board = board.copy()
                    temp_board[y, x], temp_board[y, x + 1] = temp_board[y, x + 1], temp_board[y, x]
                    if self._find_matches_at(x, y, temp_board) or self._find_matches_at(x + 1, y, temp_board):
                        return True
                # Try swapping down
                if y < self.GRID_SIZE - 1:
                    temp_board = board.copy()
                    temp_board[y, x], temp_board[y + 1, x] = temp_board[y + 1, x], temp_board[y, x]
                    if self._find_matches_at(x, y, temp_board) or self._find_matches_at(x, y + 1, temp_board):
                        return True
        return False

    def _find_all_matches(self, board=None):
        if board is None:
            board = self.board
        
        matches = set()
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if board[y, x] == 0: continue
                # Horizontal
                if x < self.GRID_SIZE - 2 and board[y, x] == board[y, x + 1] == board[y, x + 2]:
                    matches.add((x, y)); matches.add((x + 1, y)); matches.add((x + 2, y))
                # Vertical
                if y < self.GRID_SIZE - 2 and board[y, x] == board[y + 1, x] == board[y + 2, x]:
                    matches.add((x, y)); matches.add((x, y + 1)); matches.add((x, y + 2))
        return list(matches)

    def _find_matches_at(self, x, y, board=None):
        if board is None:
            board = self.board
        
        if board[y, x] == 0: return []
        
        color = board[y, x]
        h_matches, v_matches = { (x, y) }, { (x, y) }

        # Horizontal check
        for i in range(1, self.GRID_SIZE):
            if x - i >= 0 and board[y, x - i] == color: h_matches.add((x - i, y))
            else: break
        for i in range(1, self.GRID_SIZE):
            if x + i < self.GRID_SIZE and board[y, x + i] == color: h_matches.add((x + i, y))
            else: break

        # Vertical check
        for i in range(1, self.GRID_SIZE):
            if y - i >= 0 and board[y - i, x] == color: v_matches.add((x, y - i))
            else: break
        for i in range(1, self.GRID_SIZE):
            if y + i < self.GRID_SIZE and board[y + i, x] == color: v_matches.add((x, y + i))
            else: break
            
        found_matches = set()
        if len(h_matches) >= 3: found_matches.update(h_matches)
        if len(v_matches) >= 3: found_matches.update(v_matches)
        
        return list(found_matches)

    def validate_implementation(self):
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