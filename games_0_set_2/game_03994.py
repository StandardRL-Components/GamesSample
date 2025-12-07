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
        "Controls: Use arrow keys to slide the selected crystal. "
        "Spacebar selects the next crystal, Shift selects the previous. "
        "Match 3 or more of the same color to clear them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Slide crystals to create matches of three or more. "
        "Clear the entire board before you run out of moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 10
    MOVE_LIMIT = 30
    NUM_CRYSTALS_INITIAL = 50

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 50, 62)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 255)
    CRYSTAL_PALETTE = {
        1: {"base": (255, 80, 100), "light": (255, 150, 160), "dark": (180, 50, 70)},  # Red
        2: {"base": (80, 255, 120), "light": (150, 255, 170), "dark": (50, 180, 80)},  # Green
        3: {"base": (80, 150, 255), "light": (150, 200, 255), "dark": (50, 100, 180)},  # Blue
        4: {"base": (255, 240, 80), "light": (255, 250, 150), "dark": (180, 160, 50)},  # Yellow
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)

        # Isometric projection constants
        self.tile_width_half = 22
        self.tile_height_half = 11
        self.origin_x = self.WIDTH // 2
        self.origin_y = 100

        # Initialize attributes that are set in reset()
        self.board = None
        self.np_random = None
        
        # self.reset() # This is called by the wrapper, no need to call it here.
        # self.validate_implementation() # This is for debugging, not part of the final env.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None: # First reset
            self.np_random = np.random.default_rng(seed)

        self.board = self._generate_board()
        self._update_crystal_list()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MOVE_LIMIT
        self.selected_crystal_index = 0 if self.crystal_locations else -1
        self.last_move_info = {}
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.last_move_info = {}  # Clear effects from the previous step

        # --- 1. Handle Selection ---
        num_crystals = len(self.crystal_locations)
        if num_crystals > 0:
            if space_pressed and not shift_pressed:
                self.selected_crystal_index = (self.selected_crystal_index + 1) % num_crystals
            elif shift_pressed and not space_pressed:
                self.selected_crystal_index = (self.selected_crystal_index - 1 + num_crystals) % num_crystals

        # --- 2. Handle Movement ---
        is_move_action = movement != 0 and num_crystals > 0
        if is_move_action:
            self.moves_remaining -= 1
            reward -= 0.1

            move_successful, start_pos, end_pos, moved_crystal_color = self._perform_slide(movement)

            if move_successful:
                self.last_move_info = {"start": start_pos, "end": end_pos, "color": moved_crystal_color}
                
                # --- 3. Handle Matching and Chain Reactions ---
                while True:
                    matches = self._find_matches()
                    if not matches:
                        break
                    
                    if 'cleared' not in self.last_move_info: self.last_move_info['cleared'] = []
                    self.last_move_info['cleared'].extend([(x, y, self.board[y][x]) for x, y in matches])
                    
                    num_cleared = len(matches)
                    reward += num_cleared * 1.0
                    if num_cleared > 3: reward += 5.0

                    for x, y in matches: self.board[y][x] = 0

                self._update_crystal_list()
                if not self.crystal_locations:
                    self.selected_crystal_index = -1
                else:
                    self.selected_crystal_index = min(self.selected_crystal_index, len(self.crystal_locations) - 1)
        
        # --- 4. Check for Termination ---
        terminated = False
        if not self.crystal_locations:
            terminated = True
            reward += 100
            self.game_over = True
            self.last_move_info['win'] = True
        elif self.moves_remaining <= 0:
            terminated = True
            reward -= 100
            self.game_over = True
            self.last_move_info['loss'] = True
        
        self.steps += 1
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _perform_slide(self, movement):
        if self.selected_crystal_index == -1: return False, None, None, None

        dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        dx, dy = dirs[movement]
        
        start_x, start_y = self.crystal_locations[self.selected_crystal_index]
        crystal_color = self.board[start_y][start_x]

        nx, ny = start_x + dx, start_y + dy
        
        while 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.board[ny][nx] == 0:
            nx += dx
            ny += dy
        
        end_x, end_y = nx - dx, ny - dy

        if (end_x, end_y) != (start_x, start_y):
            self.board[end_y][end_x] = crystal_color
            self.board[start_y][start_x] = 0
            return True, (start_x, start_y), (end_x, end_y), crystal_color
        
        return False, (start_x, start_y), (end_x, end_y), crystal_color

    def _find_matches(self):
        to_remove = set()
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = self.board[y][x]
                if color == 0: continue

                h_match = [(x, y)]
                for i in range(1, self.GRID_SIZE):
                    if x + i < self.GRID_SIZE and self.board[y][x+i] == color: h_match.append((x+i, y))
                    else: break
                if len(h_match) >= 3: to_remove.update(h_match)

                v_match = [(x, y)]
                for i in range(1, self.GRID_SIZE):
                    if y + i < self.GRID_SIZE and self.board[y+i][x] == color: v_match.append((x, y+i))
                    else: break
                if len(v_match) >= 3: to_remove.update(v_match)
        return to_remove

    def _generate_board(self):
        while True:
            board = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
            empty_cells = list(np.ndindex(board.shape))
            self.np_random.shuffle(empty_cells)
            
            num_crystals = min(self.NUM_CRYSTALS_INITIAL, len(empty_cells))
            for i in range(num_crystals):
                y, x = empty_cells[i]
                board[y][x] = self.np_random.integers(1, len(self.CRYSTAL_PALETTE) + 1)
            
            # Temporarily set self.board to check the new board for matches
            original_board = self.board
            self.board = board
            matches = self._find_matches()
            self.board = original_board # Restore
            
            if not matches:
                return board # Found a valid board with no initial matches

    def _update_crystal_list(self):
        self.crystal_locations = []
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.board[y][x] != 0:
                    self.crystal_locations.append((x, y))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "crystals_remaining": len(self.crystal_locations),
        }

    def _world_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * self.tile_width_half
        iso_y = self.origin_y + (x + y) * self.tile_height_half
        return int(iso_x), int(iso_y)

    def _render_game(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                self._render_tile(x, y)
        
        if self.board is not None:
            for y in range(self.GRID_SIZE):
                for x in range(self.GRID_SIZE):
                    color_id = self.board[y][x]
                    if color_id != 0:
                        self._render_crystal(x, y, color_id)
        
        if self.selected_crystal_index != -1 and not self.game_over:
            sx, sy = self.crystal_locations[self.selected_crystal_index]
            self._render_cursor(sx, sy)

    def _render_tile(self, x, y):
        cx, cy = self._world_to_iso(x, y)
        points = [
            (cx, cy - self.tile_height_half),
            (cx + self.tile_width_half, cy),
            (cx, cy + self.tile_height_half),
            (cx - self.tile_width_half, cy),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _render_crystal(self, x, y, color_id, y_offset=0):
        cx, cy = self._world_to_iso(x, y)
        cy += y_offset
        
        palette = self.CRYSTAL_PALETTE[color_id]
        h, w = self.tile_height_half, self.tile_width_half
        
        top_points = [(cx, cy - h), (cx + w, cy), (cx, cy + h), (cx - w, cy)]
        pygame.draw.polygon(self.screen, palette["base"], top_points)
        
        left_points = [(cx - w, cy), (cx, cy + h), (cx, cy + h * 2), (cx - w, cy + h)]
        pygame.draw.polygon(self.screen, palette["dark"], left_points)
        
        right_points = [(cx + w, cy), (cx, cy + h), (cx, cy + h * 2), (cx + w, cy + h)]
        pygame.draw.polygon(self.screen, palette["light"], right_points)

        pygame.draw.aalines(self.screen, (0,0,0,50), True, top_points)
        pygame.draw.aalines(self.screen, (0,0,0,50), True, left_points)
        pygame.draw.aalines(self.screen, (0,0,0,50), True, right_points)

    def _render_cursor(self, x, y):
        cx, cy = self._world_to_iso(x, y)
        anim_offset = -35 + abs(math.sin(self.steps * 0.2) * 5)
        cy += anim_offset
        
        points = [
            (cx, cy),
            (cx - 6, cy - 8),
            (cx + 6, cy - 8),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points)

    def _render_effects(self):
        if 'start' in self.last_move_info and 'end' in self.last_move_info:
            start_iso = self._world_to_iso(*self.last_move_info['start'])
            end_iso = self._world_to_iso(*self.last_move_info['end'])
            color = self.CRYSTAL_PALETTE[self.last_move_info['color']]['base']
            pygame.draw.line(self.screen, color, start_iso, end_iso, 3)

        if 'cleared' in self.last_move_info:
            for x, y, color_id in self.last_move_info['cleared']:
                cx, cy = self._world_to_iso(x, y)
                cy += self.tile_height_half
                color = self.CRYSTAL_PALETTE[color_id]['light']
                for i in range(8):
                    angle = i * (math.pi / 4)
                    sx = cx + math.cos(angle) * 10
                    ex = cx + math.cos(angle) * 20
                    sy = cy + math.sin(angle) * 10
                    ey = cy + math.sin(angle) * 20
                    pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), 2)
                    
    def _render_ui(self):
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 20))

        crystals_rem = len(self.crystal_locations) if self.crystal_locations is not None else 0
        crystals_text = self.font_large.render(f"Crystals: {crystals_rem}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystals_text, (self.WIDTH - crystals_text.get_width() - 20, 20))

        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 55))

        if 'win' in self.last_move_info:
            self._render_centered_text("BOARD CLEARED!", self.CRYSTAL_PALETTE[4]['light'])
        elif 'loss' in self.last_move_info:
            self._render_centered_text("OUT OF MOVES", self.CRYSTAL_PALETTE[1]['light'])

    def _render_centered_text(self, text, color):
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 100))
        
        bg_rect = text_rect.inflate(20, 20)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(s, bg_rect)
        
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()