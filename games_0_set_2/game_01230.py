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
        "Controls: Arrow keys to move the cursor. Press space to select a tile, "
        "then move to an adjacent tile and press space again to swap. "
        "Match 3 or more to score!"
    )

    game_description = (
        "A fast-paced match-3 puzzle game. Swap adjacent gems to create lines of "
        "three or more. Create combos and cascades to maximize your score before the timer runs out!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    BOARD_ROWS, BOARD_COLS = 8, 8
    TILE_SIZE = 40
    BOARD_OFFSET_X = (SCREEN_WIDTH - BOARD_COLS * TILE_SIZE) // 2
    BOARD_OFFSET_Y = (SCREEN_HEIGHT - BOARD_ROWS * TILE_SIZE) // 2
    NUM_TILE_TYPES = 6
    TARGET_SCORE = 5000
    MAX_TIME = 60.0  # seconds
    MAX_STEPS = 30 * 60 # 30fps * 60s
    ANIMATION_SPEED = 0.2  # Progress per frame, so 5 frames for a full animation

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (100, 200, 255)
    TILE_COLORS = [
        (255, 50, 50),   # Red
        (50, 150, 255),  # Blue
        (50, 255, 50),   # Green
        (255, 255, 50),  # Yellow
        (200, 50, 255),  # Purple
        (255, 150, 50),  # Orange
    ]

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.game_fsm_state = "IDLE" # IDLE, SWAPPING, REVERSING, MATCHING, FALLING
        self.board = None
        self.cursor_pos = [0, 0]
        self.selected_tile = None
        self.animations = []
        self.particles = []
        self.last_space_held = False
        self.last_movement_action = 0
        self.steps_since_move = 0
        self.combo_multiplier = 1
        self.last_swap_pos = None

        self.reset()
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        
        self.game_fsm_state = "IDLE"
        self.cursor_pos = [self.BOARD_COLS // 2, self.BOARD_ROWS // 2]
        self.selected_tile = None
        self.animations = []
        self.particles = []
        self.last_space_held = False
        self.last_movement_action = 0
        self.steps_since_move = 0
        self.combo_multiplier = 1
        self.last_swap_pos = None

        self._generate_stable_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        self.steps += 1
        
        # --- Time and Termination Update ---
        time_delta = self.clock.tick(30) / 1000.0
        if not self.game_over:
            self.timer = max(0, self.timer - time_delta)

        if self.timer <= 0 or self.score >= self.TARGET_SCORE or self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            if self.score >= self.TARGET_SCORE:
                reward += 100 # Goal-oriented reward
            else:
                reward += -10 # Penalty for running out of time

        # --- Update Game State Machine and Animations ---
        if not self.game_over:
            self._update_animations()
            self._update_particles()

            if self.game_fsm_state == "IDLE":
                reward += self._handle_input(action)
            elif not self.animations: # If animations finished
                self._resolve_fsm_state()
        
        # --- Return observation and info ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Cooldown for cursor movement
        self.steps_since_move += 1
        if movement != 0 and self.steps_since_move > 4: # 4 frames cooldown
            self.steps_since_move = 0
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.BOARD_ROWS - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.BOARD_COLS - 1, self.cursor_pos[0] + 1)

        # Handle space press (rising edge detection)
        if space_held and not self.last_space_held:
            # sfx: select_sound
            if self.selected_tile is None:
                self.selected_tile = tuple(self.cursor_pos)
            else:
                x1, y1 = self.selected_tile
                x2, y2 = self.cursor_pos
                
                if (x1, y1) == (x2, y2): # Deselect
                    self.selected_tile = None
                elif abs(x1 - x2) + abs(y1 - y2) == 1: # Is adjacent
                    self._initiate_swap((x1, y1), (x2, y2))
                else: # Select new tile
                    self.selected_tile = tuple(self.cursor_pos)
        
        self.last_space_held = space_held
        return reward

    def _initiate_swap(self, pos1, pos2, is_reversing=False):
        x1, y1 = pos1
        x2, y2 = pos2
        
        if not is_reversing:
            self.last_swap_pos = (pos1, pos2)

        self.game_fsm_state = "REVERSING" if is_reversing else "SWAPPING"
        self.animations.append({
            "type": "SWAP", "pos": pos1, "target_pos": pos2, "progress": 0,
            "tile_type": self.board[y1][x1]
        })
        self.animations.append({
            "type": "SWAP", "pos": pos2, "target_pos": pos1, "progress": 0,
            "tile_type": self.board[y2][x2]
        })
        self.board[y1][x1], self.board[y2][x2] = 0, 0 # Temporarily remove for drawing
        self.selected_tile = None

    def _resolve_fsm_state(self):
        if self.game_fsm_state == "SWAPPING":
            matches = self._find_all_matches()
            if matches:
                self.combo_multiplier = 1
                self._handle_matches(matches)
            else:
                # sfx: invalid_swap_sound
                # Undo the swap in the board array
                x1, y1 = self.last_swap_pos[0]
                x2, y2 = self.last_swap_pos[1]
                self.board[y1][x1], self.board[y2][x2] = self.board[y2][x2], self.board[y1][x1]
                self._initiate_swap(self.last_swap_pos[0], self.last_swap_pos[1], is_reversing=True)
        
        elif self.game_fsm_state == "REVERSING":
            self.game_fsm_state = "IDLE"
        
        elif self.game_fsm_state == "MATCHING":
            self._apply_gravity_and_refill()
            self.game_fsm_state = "FALLING"

        elif self.game_fsm_state == "FALLING":
            matches = self._find_all_matches()
            if matches:
                self.combo_multiplier += 1
                self._handle_matches(matches) # Cascade!
            else:
                self.game_fsm_state = "IDLE" # Board is stable

    def _update_animations(self):
        if not self.animations:
            return

        for anim in self.animations[:]:
            anim["progress"] = min(1.0, anim["progress"] + self.ANIMATION_SPEED)
            if anim["progress"] >= 1.0:
                # On animation complete
                if anim["type"] == "SWAP":
                    # Put tiles back into the board at their new locations
                    target_x, target_y = anim["target_pos"]
                    self.board[target_y][target_x] = anim["tile_type"]
                elif anim["type"] == "FALL":
                    target_x, target_y = anim["target_pos"]
                    self.board[target_y][target_x] = anim["tile_type"]
                self.animations.remove(anim)

    def _handle_matches(self, matches):
        # sfx: match_sound
        self.game_fsm_state = "MATCHING"
        score_this_turn = 0
        
        # Check for 4 and 5 matches to add bonus rewards
        # This is a simplified check; a more robust system would analyze match structures
        if len(matches) == 4: score_this_turn += 10
        if len(matches) >= 5: score_this_turn += 20

        for r, c in matches:
            score_this_turn += 1 * self.combo_multiplier # Base score + combo
            self._spawn_particles((c, r), self.TILE_COLORS[self.board[r][c] - 1])
            self.board[r][c] = 0 # Mark for removal
        
        self.score += score_this_turn
    
    def _apply_gravity_and_refill(self):
        for c in range(self.BOARD_COLS):
            empty_row = self.BOARD_ROWS - 1
            for r in range(self.BOARD_ROWS - 1, -1, -1):
                if self.board[r][c] != 0:
                    if r != empty_row:
                        # Animate fall
                        self.animations.append({
                            "type": "FALL", "pos": (c, r), "target_pos": (c, empty_row),
                            "progress": 0, "tile_type": self.board[r][c]
                        })
                        self.board[empty_row][c] = self.board[r][c]
                        self.board[r][c] = 0
                    empty_row -= 1
            
            # Refill top
            for r in range(empty_row, -1, -1):
                new_tile = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
                self.board[r][c] = new_tile
                # Animate new tiles falling in from above
                self.animations.append({
                    "type": "FALL", "pos": (c, r - (empty_row + 1)), "target_pos": (c, r),
                    "progress": 0, "tile_type": new_tile
                })
                self.board[r][c] = 0 # Temporarily clear for animation

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS - 2):
                if self.board[r][c] != 0 and self.board[r][c] == self.board[r][c+1] == self.board[r][c+2]:
                    match_len = 3
                    while c + match_len < self.BOARD_COLS and self.board[r][c] == self.board[r][c + match_len]:
                        match_len += 1
                    for i in range(match_len):
                        matches.add((r, c + i))
        # Vertical
        for c in range(self.BOARD_COLS):
            for r in range(self.BOARD_ROWS - 2):
                if self.board[r][c] != 0 and self.board[r][c] == self.board[r+1][c] == self.board[r+2][c]:
                    match_len = 3
                    while r + match_len < self.BOARD_ROWS and self.board[r][c] == self.board[r + match_len][c]:
                        match_len += 1
                    for i in range(match_len):
                        matches.add((r + i, c))
        return matches

    def _generate_stable_board(self):
        self.board = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.BOARD_ROWS, self.BOARD_COLS))
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            # In-place removal and refill without animation
            for r, c in matches:
                self.board[r][c] = 0
            for c in range(self.BOARD_COLS):
                empty_row = self.BOARD_ROWS - 1
                for r in range(self.BOARD_ROWS - 1, -1, -1):
                    if self.board[r][c] != 0:
                        if r != empty_row:
                           self.board[empty_row][c], self.board[r][c] = self.board[r][c], self.board[empty_row][c]
                        empty_row -= 1
                for r in range(empty_row, -1, -1):
                    self.board[r][c] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_static_tiles()
        self._draw_animated_tiles()
        self._draw_cursor()
        self._draw_particles()

    def _draw_grid(self):
        for r in range(self.BOARD_ROWS + 1):
            y = self.BOARD_OFFSET_Y + r * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_OFFSET_X, y), (self.BOARD_OFFSET_X + self.BOARD_COLS * self.TILE_SIZE, y))
        for c in range(self.BOARD_COLS + 1):
            x = self.BOARD_OFFSET_X + c * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_OFFSET_Y), (x, self.BOARD_OFFSET_Y + self.BOARD_ROWS * self.TILE_SIZE))

    def _draw_static_tiles(self):
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                tile_type = self.board[r][c]
                if tile_type != 0:
                    self._draw_tile(c, r, tile_type)

    def _draw_animated_tiles(self):
        for anim in self.animations:
            prog = anim["progress"]
            start_x, start_y = anim["pos"]
            end_x, end_y = anim["target_pos"]
            
            # Lerp position
            draw_x = start_x + (end_x - start_x) * prog
            draw_y = start_y + (end_y - start_y) * prog
            
            self._draw_tile(draw_x, draw_y, anim["tile_type"])

    def _draw_tile(self, c, r, tile_type, alpha=255):
        if tile_type == 0: return
        
        center_x = int(self.BOARD_OFFSET_X + (c + 0.5) * self.TILE_SIZE)
        center_y = int(self.BOARD_OFFSET_Y + (r + 0.5) * self.TILE_SIZE)
        color = self.TILE_COLORS[tile_type - 1]
        radius = int(self.TILE_SIZE * 0.4)
        
        # Use gfxdraw for antialiasing
        if tile_type == 1: # Circle
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        elif tile_type == 2: # Square
            rect = pygame.Rect(center_x - radius, center_y - radius, radius*2, radius*2)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
        elif tile_type == 3: # Triangle
            points = [(center_x, center_y - radius), (center_x - radius, center_y + radius*0.7), (center_x + radius, center_y + radius*0.7)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif tile_type == 4: # Diamond
            points = [(center_x, center_y - radius), (center_x + radius, center_y), (center_x, center_y + radius), (center_x - radius, center_y)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif tile_type == 5: # Hexagon
            points = [(center_x + radius * math.cos(math.radians(angle)), center_y + radius * math.sin(math.radians(angle))) for angle in range(30, 391, 60)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif tile_type == 6: # Star
            points = []
            for i in range(10):
                angle = math.radians(i * 36)
                r_val = radius if i % 2 == 0 else radius * 0.5
                points.append((center_x + r_val * math.cos(angle), center_y + r_val * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_cursor(self):
        c, r = self.cursor_pos
        rect = pygame.Rect(self.BOARD_OFFSET_X + c * self.TILE_SIZE, self.BOARD_OFFSET_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        
        # Pulsating alpha for cursor
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        cursor_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), cursor_surface.get_rect(), 3, border_radius=4)
        self.screen.blit(cursor_surface, rect.topleft)

        if self.selected_tile:
            sel_c, sel_r = self.selected_tile
            sel_rect = pygame.Rect(self.BOARD_OFFSET_X + sel_c * self.TILE_SIZE, self.BOARD_OFFSET_Y + sel_r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, sel_rect, 3, border_radius=4)

    def _spawn_particles(self, board_pos, color):
        c, r = board_pos
        center_x = self.BOARD_OFFSET_X + (c + 0.5) * self.TILE_SIZE
        center_y = self.BOARD_OFFSET_Y + (r + 0.5) * self.TILE_SIZE
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifetime": random.randint(15, 30),
                "color": color,
                "radius": random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30.0))
            color_with_alpha = (*p["color"], alpha)
            surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color_with_alpha, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Timer Bar
        timer_bar_width = 200
        timer_bar_height = 20
        timer_x = self.SCREEN_WIDTH - timer_bar_width - 10
        timer_y = 10
        
        time_ratio = self.timer / self.MAX_TIME
        current_width = int(timer_bar_width * time_ratio)
        
        bar_color = (80, 200, 80)
        if time_ratio < 0.25:
            bar_color = (200, 80, 80)
        elif time_ratio < 0.5:
            bar_color = (200, 200, 80)

        pygame.draw.rect(self.screen, self.COLOR_GRID, (timer_x, timer_y, timer_bar_width, timer_bar_height))
        pygame.draw.rect(self.screen, bar_color, (timer_x, timer_y, current_width, timer_bar_height))
        pygame.draw.rect(self.screen, (255,255,255), (timer_x, timer_y, timer_bar_width, timer_bar_height), 1)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY!" if self.score >= self.TARGET_SCORE else "TIME'S UP!"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")