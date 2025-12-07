import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Arrows to move cursor, Shift to cycle crystal type, Space to place crystal."

    # Must be a short, user-facing description of the game:
    game_description = "Place light-bending crystals in a cavern to illuminate all target gems before time or crystals run out."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 24
    GRID_HEIGHT = 16
    TILE_WIDTH_HALF = 16
    TILE_HEIGHT_HALF = 8
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80

    MAX_STEPS = 1000
    TIME_LIMIT_SECONDS = 60
    MAX_CRYSTALS = 10
    MAX_BOUNCES = 20

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (60, 70, 90)
    COLOR_WALL_TOP = (80, 90, 110)

    COLOR_LIGHT_SOURCE = (255, 255, 150)
    COLOR_LIGHT_BEAM = (255, 255, 200)
    COLOR_LIGHT_GLOW = (255, 255, 200, 50)

    COLOR_GEM_INACTIVE = (100, 100, 120)
    COLOR_GEM_ACTIVE = {
        0: (255, 80, 80), 1: (80, 255, 80), 2: (80, 80, 255),
        3: (255, 255, 80), 4: (80, 255, 255)
    }

    CRYSTAL_TYPES = {
        0: {"name": "Reflector", "color": (255, 50, 50), "angle": 45},
        1: {"name": "Refractor", "color": (50, 255, 50), "angle": 90},
        2: {"name": "Bender", "color": (50, 50, 255), "angle": -45}
    }
    NUM_CRYSTAL_TYPES = len(CRYSTAL_TYPES)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        
        # State variables initialized in reset()
        self.grid = None
        self.placed_crystals = None
        self.target_gems = None
        self.light_source_pos = None
        self.light_source_dir = None
        self.light_path = None
        self.cursor_pos = None
        self.selected_crystal_type = None
        self.crystals_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.prev_illuminated_count = 0
        self.rng = None
        self._last_shift_press = False
        self._last_space_press = False
        
        # Initialize state variables
        # self.reset() is called by the wrapper/runner, not needed in __init__
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crystals_remaining = self.MAX_CRYSTALS
        self.cursor_pos = pygame.math.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_crystal_type = 0
        self.placed_crystals = []
        self.light_path = []
        self.prev_illuminated_count = 0
        self._last_shift_press = False
        self._last_space_press = False

        self._setup_level()
        self._calculate_light_path()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0
        crystal_placed_this_step = False

        # Handle actions with simple debounce for presses
        if shift_press and not self._last_shift_press:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % self.NUM_CRYSTAL_TYPES
            # sfx: ui_switch

        if movement != 0:
            new_pos = self.cursor_pos.copy()
            if movement == 1: new_pos.y -= 1    # Up
            elif movement == 2: new_pos.y += 1    # Down
            elif movement == 3: new_pos.x -= 1    # Left
            elif movement == 4: new_pos.x += 1    # Right
            if 0 <= new_pos.x < self.GRID_WIDTH and 0 <= new_pos.y < self.GRID_HEIGHT:
                self.cursor_pos = new_pos

        if space_press and not self._last_space_press:
            if self._can_place_crystal(self.cursor_pos):
                self.placed_crystals.append({"pos": self.cursor_pos.copy(), "type": self.selected_crystal_type})
                self.crystals_remaining -= 1
                crystal_placed_this_step = True
                # sfx: crystal_place
                reward -= 0.1

        self._last_shift_press = shift_press
        self._last_space_press = space_press

        if crystal_placed_this_step:
            self._calculate_light_path()
            current_illuminated_count = sum(1 for gem in self.target_gems if gem["illuminated"])
            newly_illuminated = current_illuminated_count - self.prev_illuminated_count
            if newly_illuminated > 0:
                reward += newly_illuminated * 1.0
                # sfx: gem_activate
            self.prev_illuminated_count = current_illuminated_count

        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            win = all(gem["illuminated"] for gem in self.target_gems)
            if win:
                reward += 50
                # sfx: win_jingle
            else:
                reward -= 50
                # sfx: lose_sound
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is always False in this environment
            self._get_info()
        )

    def _check_termination(self):
        time_elapsed = self.steps * (self.TIME_LIMIT_SECONDS / self.MAX_STEPS)
        win = all(gem["illuminated"] for gem in self.target_gems)
        
        if win or self.crystals_remaining <= 0 or time_elapsed >= self.TIME_LIMIT_SECONDS or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
            "crystals_remaining": self.crystals_remaining,
            "time_remaining": self.TIME_LIMIT_SECONDS - self.steps * (self.TIME_LIMIT_SECONDS / self.MAX_STEPS),
            "gems_lit": sum(1 for gem in self.target_gems if gem["illuminated"])
        }

    def _setup_level(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.grid[0, :] = 1; self.grid[-1, :] = 1
        self.grid[:, 0] = 1; self.grid[:, -1] = 1
        
        self.grid[5:10, 5] = 1; self.grid[15, 3:8] = 1
        self.grid[8, 10:14] = 1; self.grid[18:22, 11] = 1

        self.light_source_pos = pygame.math.Vector2(2, self.GRID_HEIGHT // 2)
        self.light_source_dir = pygame.math.Vector2(1, 0)

        self.target_gems = [
            {"pos": pygame.math.Vector2(4, 4), "illuminated": False, "color_idx": 0},
            {"pos": pygame.math.Vector2(18, 6), "illuminated": False, "color_idx": 1},
            {"pos": pygame.math.Vector2(6, 12), "illuminated": False, "color_idx": 2},
            {"pos": pygame.math.Vector2(20, 13), "illuminated": False, "color_idx": 3},
            {"pos": pygame.math.Vector2(16, 1), "illuminated": False, "color_idx": 4},
        ]

    def _can_place_crystal(self, pos):
        if self.crystals_remaining <= 0: return False
        if self.grid[int(pos.x), int(pos.y)] == 1: return False
        if any(c["pos"] == pos for c in self.placed_crystals): return False
        if any(g["pos"] == pos for g in self.target_gems): return False
        if self.light_source_pos == pos: return False
        return True

    def _calculate_light_path(self):
        for gem in self.target_gems: gem["illuminated"] = False

        pos = self.light_source_pos.copy() * 10 + pygame.math.Vector2(5, 5)
        direction = self.light_source_dir.copy()
        path_points = [self._grid_to_iso(self.light_source_pos.x, self.light_source_pos.y)]
        
        for bounce in range(self.MAX_BOUNCES):
            hit = False
            for _ in range(self.GRID_WIDTH * 20): # Max ray length per bounce
                pos += direction * 0.5
                grid_x, grid_y = int(pos.x / 10), int(pos.y / 10)

                if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                    hit = True; break
                
                if self.grid[grid_x, grid_y] == 1:
                    path_points.append(self._grid_to_iso(pos.x/10, pos.y/10))
                    prev_grid_x = int((pos.x - direction.x * 0.6) / 10)
                    prev_grid_y = int((pos.y - direction.y * 0.6) / 10)
                    if grid_x != prev_grid_x: direction.x *= -1
                    if grid_y != prev_grid_y: direction.y *= -1
                    pos += direction * 1.0
                    hit = True; break

                for crystal in self.placed_crystals:
                    if crystal["pos"].x == grid_x and crystal["pos"].y == grid_y:
                        path_points.append(self._grid_to_iso(pos.x/10, pos.y/10))
                        angle = self.CRYSTAL_TYPES[crystal["type"]]["angle"]
                        direction = direction.rotate(angle)
                        pos = (crystal["pos"] * 10 + pygame.math.Vector2(5, 5)) + direction * 6
                        hit = True; break
                if hit: break

                for gem in self.target_gems:
                    if gem["pos"].x == grid_x and gem["pos"].y == grid_y:
                        gem["illuminated"] = True
            
            if not hit: break
        
        path_points.append(self._grid_to_iso(pos.x/10, pos.y/10))
        self.light_path = path_points

    def _grid_to_iso(self, x, y):
        return (
            int(self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF),
            int(self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF)
        )

    def _render_iso_cube(self, surface, x, y, color_top, color_side, height=16):
        px, py = self._grid_to_iso(x, y)
        points_top = [self._grid_to_iso(x, y), self._grid_to_iso(x + 1, y), self._grid_to_iso(x + 1, y + 1), self._grid_to_iso(x, y + 1)]
        px_bottom, py_bottom = self._grid_to_iso(x, y + 1)
        
        points_left = [self._grid_to_iso(x, y), self._grid_to_iso(x, y+1), (px_bottom, py_bottom + height), (px, py+height)]
        px_right, py_right = self._grid_to_iso(x+1, y)
        points_right = [self._grid_to_iso(x, y), (px_right, py_right), (px_right, py_right+height), (px, py+height)]

        color_side_dark = tuple(max(0, c - 20) for c in color_side)

        pygame.gfxdraw.filled_polygon(surface, points_top, color_top)
        pygame.gfxdraw.aapolygon(surface, points_top, color_top)
        pygame.gfxdraw.filled_polygon(surface, points_right, color_side)
        pygame.gfxdraw.aapolygon(surface, points_right, color_side)
        pygame.gfxdraw.filled_polygon(surface, points_left, color_side_dark)
        pygame.gfxdraw.aapolygon(surface, points_left, color_side_dark)

    def _render_game(self):
        # Render floor grid for context
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 0:
                     p1 = self._grid_to_iso(x, y); p2 = self._grid_to_iso(x + 1, y)
                     p3 = self._grid_to_iso(x, y + 1)
                     pygame.draw.aaline(self.screen, (30,35,50), p1, p2)
                     pygame.draw.aaline(self.screen, (30,35,50), p1, p3)

        # Render elements from back to front
        render_queue = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 1:
                    render_queue.append(('wall', x, y))
                for gem in self.target_gems:
                    if gem['pos'] == (x,y): render_queue.append(('gem', gem))
                for crystal in self.placed_crystals:
                    if crystal['pos'] == (x,y): render_queue.append(('crystal', crystal))
                if self.light_source_pos == (x,y): render_queue.append(('source', x, y))
        
        for item in render_queue:
            if item[0] == 'wall': self._render_iso_cube(self.screen, item[1], item[2], self.COLOR_WALL_TOP, self.COLOR_WALL)
            elif item[0] == 'gem':
                gem = item[1]
                px, py = self._grid_to_iso(gem["pos"].x, gem["pos"].y)
                py += 8
                color = self.COLOR_GEM_ACTIVE[gem["color_idx"]] if gem["illuminated"] else self.COLOR_GEM_INACTIVE
                points = [(px, py-8), (px+6, py), (px, py+8), (px-6, py)]
                if gem["illuminated"]:
                    pygame.gfxdraw.filled_circle(self.screen, px, py, 12, (*color, 50))
                    pygame.gfxdraw.filled_circle(self.screen, px, py, 8, (*color, 70))
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, color)
            elif item[0] == 'crystal':
                crystal = item[1]
                c_info = self.CRYSTAL_TYPES[crystal["type"]]
                self._render_iso_cube(self.screen, crystal["pos"].x, crystal["pos"].y, c_info["color"], tuple(max(0, c-40) for c in c_info["color"]), height=12)
            elif item[0] == 'source':
                self._render_iso_cube(self.screen, item[1], item[2], self.COLOR_LIGHT_SOURCE, tuple(max(0, c-40) for c in self.COLOR_LIGHT_SOURCE), height=12)

        if len(self.light_path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_LIGHT_GLOW, False, self.light_path, 15)
            pygame.draw.lines(self.screen, self.COLOR_LIGHT_GLOW, False, self.light_path, 8)
            pygame.draw.lines(self.screen, self.COLOR_LIGHT_BEAM, False, self.light_path, 3)

        cx, cy = int(self.cursor_pos.x), int(self.cursor_pos.y)
        points_floor = [self._grid_to_iso(cx, cy), self._grid_to_iso(cx + 1, cy), self._grid_to_iso(cx + 1, cy + 1), self._grid_to_iso(cx, cy + 1)]
        cursor_color = self.CRYSTAL_TYPES[self.selected_crystal_type]["color"]
        pygame.draw.polygon(self.screen, cursor_color, points_floor, 2)

    def _render_ui(self):
        crystal_text = f"Crystals: {self.crystals_remaining}/{self.MAX_CRYSTALS}"
        text_surf = self.font_ui.render(crystal_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (10, 10))

        time_left = max(0, self.TIME_LIMIT_SECONDS - self.steps * (self.TIME_LIMIT_SECONDS / self.MAX_STEPS))
        timer_text = f"Time: {time_left:.1f}s"
        text_surf = self.font_ui.render(timer_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))
        
        score_text = f"Score: {self.score:.1f}"
        text_surf = self.font_ui.render(score_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, 10))

        c_info = self.CRYSTAL_TYPES[self.selected_crystal_type]
        selected_text = f"Selected: {c_info['name']}"
        text_surf = self.font_ui.render(selected_text, True, c_info["color"])
        self.screen.blit(text_surf, (10, 35))

    def close(self):
        pygame.quit()