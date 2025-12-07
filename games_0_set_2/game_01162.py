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
        "Controls: Arrows to move cursor. Space to swap right, Shift to swap down."
    )

    game_description = (
        "Match cascading fruits in a grid-based puzzle game to reach a target score."
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_FRUIT_TYPES = 3
    WIN_SCORE = 1000
    MAX_STEPS = 1000
    INITIAL_MOVES = 50

    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    
    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (50, 60, 70)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_GAMEOVER_TEXT = (255, 50, 50)
    COLOR_WIN_TEXT = (50, 255, 50)
    CURSOR_COLOR = (255, 255, 0)
    FRUIT_COLORS = [
        (220, 50, 50),   # Red
        (50, 220, 50),   # Green
        (50, 100, 220),  # Blue
    ]
    FRUIT_HIGHLIGHTS = [
        (255, 120, 120),
        (120, 255, 120),
        (120, 170, 255),
    ]

    # Animation
    ANIMATION_SPEED = 0.2  # Lower is faster

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
        
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("Arial", 48, bold=True)

        self.grid_rect = pygame.Rect(0, 0, 320, 320)
        self.grid_rect.center = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20)
        self.tile_size = self.grid_rect.width // self.GRID_WIDTH
        
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.game_won = False
        self.grid = None
        self.cursor_pos = [0,0]
        self.animation_state = "IDLE"
        self.animation_progress = 0
        self.animation_data = {}
        self.accumulated_reward = 0
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.INITIAL_MOVES
        self.game_over = False
        self.game_won = False
        
        self.grid = self._generate_board()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.animation_state = "IDLE"
        self.animation_progress = 0
        self.animation_data = {}
        self.accumulated_reward = 0
        
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0.0

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        if self.animation_state != "IDLE":
            self._update_animations()
        else:
            self._handle_input(action)
        
        if self.animation_state == "IDLE":
            reward = self.accumulated_reward
            self.accumulated_reward = 0
            
            # Check for termination after a turn resolves
            no_more_moves = not self._find_all_possible_moves()
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                self.game_won = True
                reward += 100
            elif self.moves_remaining <= 0 or no_more_moves:
                self.game_over = True
                reward -= 50
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Swap Action ---
        if space_held or shift_held:
            if self.moves_remaining <= 0: return

            cx, cy = self.cursor_pos
            if space_held:  # Swap Right
                nx, ny = cx + 1, cy
            else:  # Swap Down
                nx, ny = cx, cy + 1

            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                self.moves_remaining -= 1
                
                # Perform swap
                self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]
                
                matches1 = self._find_matches_at(cx, cy)
                matches2 = self._find_matches_at(nx, ny)
                all_matches = matches1.union(matches2)

                if all_matches:
                    self.animation_state = "SWAP_VALID"
                    self.animation_data = {"from": (cx, cy), "to": (nx, ny), "matches": all_matches}
                else:
                    # Invalid swap, swap back
                    self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]
                    self.animation_state = "SWAP_INVALID"
                    self.animation_data = {"from": (cx, cy), "to": (nx, ny)}
                    self.accumulated_reward = -0.1
                
                self.animation_progress = 0

    def _update_animations(self):
        self.animation_progress += self.ANIMATION_SPEED
        if self.animation_progress >= 1:
            self.animation_progress = 1
            
            # --- State Transitions ---
            if self.animation_state == "SWAP_VALID":
                self._start_match_animation(self.animation_data["matches"])
            elif self.animation_state == "SWAP_INVALID":
                self.animation_state = "IDLE"
            elif self.animation_state == "MATCHING":
                self._start_cascade_animation()
            elif self.animation_state == "CASCADING":
                new_matches = self._find_matches()
                if new_matches:
                    self._start_match_animation(new_matches) # Chain reaction
                else:
                    self.animation_state = "IDLE"
            
            if self.animation_progress >= 1 and self.animation_state != "IDLE":
                self.animation_progress = 0

    def _start_match_animation(self, matches):
        self.animation_state = "MATCHING"
        self.animation_data = {"matches": list(matches)}
        
        # Calculate reward
        num_matched = len(matches)
        self.score += num_matched
        self.accumulated_reward += num_matched
        if num_matched >= 5:
            self.accumulated_reward += 10

        # Create particles and remove fruits
        for r, c in matches:
            fruit_type = self.grid[r, c]
            if fruit_type > 0:
                self._create_particles(c, r, self.FRUIT_COLORS[fruit_type - 1])
            self.grid[r, c] = 0

    def _start_cascade_animation(self):
        moved_fruits = {} # from (r,c) to (r,c)
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        moved_fruits[(r,c)] = (empty_row, c)
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
        
        # Refill top rows
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)
                    # FIX: The original key `(-1, c), r` was not a 2-tuple, causing errors.
                    # This new key `(-r-1, c)` is a unique 2-tuple for each new fruit.
                    moved_fruits[(-r - 1, c)] = (r, c) # Animate from off-screen

        if moved_fruits:
            self.animation_state = "CASCADING"
            self.animation_data = {"moved_fruits": moved_fruits}
        else:
            self.animation_state = "IDLE"

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), we need (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, self.grid_rect, border_radius=5)

        # Draw fruits
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                fruit_type = self.grid[r, c]
                if fruit_type == 0:
                    continue

                pos_x, pos_y = self._get_pixel_pos(c, r)
                
                # Handle animations
                if self.animation_state in ["SWAP_VALID", "SWAP_INVALID"]:
                    (fx, fy), (tx, ty) = self.animation_data["from"], self.animation_data["to"]
                    if (c, r) == (fx, fy):
                        pos_x, pos_y = self._lerp_pos(fx, fy, tx, ty, self.animation_progress)
                    elif (c, r) == (tx, ty):
                        pos_x, pos_y = self._lerp_pos(tx, ty, fx, fy, self.animation_progress)
                elif self.animation_state == "MATCHING":
                    if (r, c) in self.animation_data["matches"]:
                        scale = 1.0 - self.animation_progress
                        self._draw_fruit(pos_x, pos_y, fruit_type, scale)
                        continue
                elif self.animation_state == "CASCADING":
                    is_moving = False
                    for (start_r, start_c), (end_r, end_c) in self.animation_data["moved_fruits"].items():
                        if (r, c) == (end_r, end_c):
                            start_x, start_y = self._get_pixel_pos(start_c, start_r)
                            end_x, end_y = self._get_pixel_pos(end_c, end_r)
                            pos_x, pos_y = self._lerp(start_x, end_x, self.animation_progress), self._lerp(start_y, end_y, self.animation_progress)
                            is_moving = True
                            break
                    if is_moving:
                         self._draw_fruit(pos_x, pos_y, fruit_type)
                         continue

                self._draw_fruit(pos_x, pos_y, fruit_type)

        # Draw cursor
        cursor_pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        alpha = int(150 + cursor_pulse * 105)
        cx, cy = self._get_pixel_pos(self.cursor_pos[0], self.cursor_pos[1])
        cursor_rect = pygame.Rect(cx - self.tile_size/2, cy - self.tile_size/2, self.tile_size, self.tile_size)
        
        s = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.CURSOR_COLOR, alpha), s.get_rect(), 5, border_radius=5)
        self.screen.blit(s, cursor_rect.topleft)

    def _draw_fruit(self, x, y, fruit_type, scale=1.0):
        radius = int(self.tile_size * 0.38 * scale)
        if radius <= 0: return

        color = self.FRUIT_COLORS[fruit_type - 1]
        highlight = self.FRUIT_HIGHLIGHTS[fruit_type - 1]
        
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), radius, highlight)
        
        highlight_x = int(x - radius * 0.3)
        highlight_y = int(y - radius * 0.3)
        pygame.gfxdraw.filled_circle(self.screen, highlight_x, highlight_y, int(radius * 0.3), highlight)
        pygame.gfxdraw.aacircle(self.screen, highlight_x, highlight_y, int(radius * 0.3), highlight)

    def _render_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
                self.screen.blit(temp_surf, (p["pos"][0] - p["size"], p["pos"][1] - p["size"]))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        moves_text = self.font_main.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)
        
        if self.game_over:
            if self.game_won:
                msg = "YOU WIN!"
                color = self.COLOR_WIN_TEXT
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER_TEXT
            
            over_text = self.font_gameover.render(msg, True, color)
            over_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 40))
            self.screen.blit(over_text, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
        }

    # --- Game Logic Helpers ---
    def _generate_board(self):
        while True:
            grid = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            # FIX: Use walrus operator to avoid calling _find_matches twice.
            while (matches := self._find_matches(grid)):
                for r, c in matches:
                    grid[r, c] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)
            
            if len(self._find_all_possible_moves(grid)) >= 5:
                return grid

    def _find_matches(self, grid=None):
        if grid is None: grid = self.grid
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if grid[r,c] == 0: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_matches_at(self, c, r):
        if self.grid[r, c] == 0: return set()
        
        matches = set()
        fruit = self.grid[r, c]
        
        # Horizontal
        h_line = [ (r, i) for i in range(self.GRID_WIDTH) if self.grid[r, i] == fruit ]
        for i in range(len(h_line) - 2):
            if h_line[i][1] + 1 == h_line[i+1][1] and h_line[i+1][1] + 1 == h_line[i+2][1]:
                matches.update(h_line[i:i+3])

        # Vertical
        v_line = [ (i, c) for i in range(self.GRID_HEIGHT) if self.grid[i, c] == fruit ]
        for i in range(len(v_line) - 2):
            if v_line[i][0] + 1 == v_line[i+1][0] and v_line[i+1][0] + 1 == v_line[i+2][0]:
                matches.update(v_line[i:i+3])
        
        return matches

    def _find_all_possible_moves(self, grid=None):
        if grid is None: grid = self.grid
        possible_moves = []
        temp_grid = grid.copy()
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Swap right
                if c < self.GRID_WIDTH - 1:
                    temp_grid[r,c], temp_grid[r,c+1] = temp_grid[r,c+1], temp_grid[r,c]
                    if self._find_matches(temp_grid):
                        possible_moves.append(((r,c), 'right'))
                    temp_grid[r,c], temp_grid[r,c+1] = temp_grid[r,c+1], temp_grid[r,c] # Swap back
                # Swap down
                if r < self.GRID_HEIGHT - 1:
                    temp_grid[r,c], temp_grid[r+1,c] = temp_grid[r+1,c], temp_grid[r,c]
                    if self._find_matches(temp_grid):
                        possible_moves.append(((r,c), 'down'))
                    temp_grid[r,c], temp_grid[r+1,c] = temp_grid[r+1,c], temp_grid[r,c] # Swap back
        return possible_moves

    def _create_particles(self, c, r, color):
        px, py = self._get_pixel_pos(c, r)
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": [px, py],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(15, 30),
                "max_life": 30,
                "color": color,
                "size": random.randint(2, 5)
            })

    # --- Math/Positioning Helpers ---
    def _get_pixel_pos(self, c, r):
        x = self.grid_rect.left + c * self.tile_size + self.tile_size / 2
        y = self.grid_rect.top + r * self.tile_size + self.tile_size / 2
        return x, y

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _lerp_pos(self, c1, r1, c2, r2, t):
        x1, y1 = self._get_pixel_pos(c1, r1)
        x2, y2 = self._get_pixel_pos(c2, r2)
        return self._lerp(x1, x2, t), self._lerp(y1, y2, t)