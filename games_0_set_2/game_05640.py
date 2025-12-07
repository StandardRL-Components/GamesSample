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
    """
    An isometric puzzle game where the player places crystals to redirect light beams.

    The goal is to illuminate all targets on the map using a limited number of crystals.
    The game is turn-based, with the state only changing when the player performs an action.

    Action Space: MultiDiscrete([5, 2, 2])
    - Movement (0=none, 1=up, 2=down, 3=left, 4=right): Moves the placement cursor.
    - Space (0=released, 1=held): Places the selected crystal.
    - Shift (0=released, 1=held): Cycles through available crystal types.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a crystal. Shift to cycle crystal type."
    )

    game_description = (
        "A puzzle game of light and reflection. Place crystals to bend and split light beams, "
        "illuminating all the targets to win. You have a limited number of crystals, so place them wisely!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.FONT_UI = pygame.font.SysFont("monospace", 16, bold=True)
        self.FONT_MSG = pygame.font.SysFont("sans-serif", 50, bold=True)
        
        # --- Visual & Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 28, 18
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 18, 9
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 60

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_WALL = (40, 50, 70)
        self.COLOR_WALL_OUTLINE = (60, 70, 90)
        self.COLOR_GRID = (25, 30, 45)
        self.COLOR_TARGET_OFF = (180, 50, 50)
        self.COLOR_TARGET_ON = (50, 220, 50)
        self.COLOR_LIGHT_BEAM = (255, 255, 200)
        self.COLOR_LIGHT_GLOW = (255, 255, 200, 30)
        self.COLOR_CURSOR = (0, 200, 255)
        self.CRYSTAL_COLORS = {
            "refractor": (255, 80, 80),
            "splitter": (80, 80, 255),
        }

        # Game parameters
        self.MAX_STEPS = 1000
        self.NUM_TARGETS = 5
        self.MAX_BOUNCES = 15
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.cursor_pos = None
        self.walls = None
        self.targets = None
        self.light_source = None
        self.placed_crystals = None
        self.crystal_inventory = None
        self.selected_crystal_idx = None
        self.available_crystal_types = None
        self.light_paths = None
        self.illuminated_targets = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.grid_map = None

        # self.reset() is called here to ensure the environment is ready after __init__
        # However, to avoid double-initialization issues if a user calls reset() again,
        # we let the first call be from the user or wrapper.
        # For validation purposes, we can call it.
        # self.validate_implementation() calls reset, so we don't need to call it here.
    
    def _generate_level(self):
        self.walls = set()
        self.targets = set()
        
        # Create a boundary wall
        for i in range(self.GRID_WIDTH):
            self.walls.add((i, -1))
            self.walls.add((i, self.GRID_HEIGHT))
        for i in range(self.GRID_HEIGHT):
            self.walls.add((-1, i))
            self.walls.add((self.GRID_WIDTH, i))

        # Place light source
        self.light_source = {"pos": (self.GRID_WIDTH // 2, 0), "dir": (0, 1)}

        # Place targets
        while len(self.targets) < self.NUM_TARGETS:
            pos = (
                self.np_random.integers(1, self.GRID_WIDTH - 2),
                self.np_random.integers(self.GRID_HEIGHT // 2, self.GRID_HEIGHT - 2)
            )
            if pos != self.light_source["pos"] and pos not in self.targets:
                self.targets.add(pos)
        
        # Place some random wall blocks
        for _ in range(self.np_random.integers(5, 15)):
            pos = (
                self.np_random.integers(1, self.GRID_WIDTH - 2),
                self.np_random.integers(1, self.GRID_HEIGHT - 2)
            )
            if pos != self.light_source["pos"] and pos not in self.targets:
                self.walls.add(pos)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self._generate_level()
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.placed_crystals = {}
        self.crystal_inventory = {"refractor": 5, "splitter": 3}
        self.available_crystal_types = list(self.crystal_inventory.keys())
        self.selected_crystal_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Initialize light path related state variables before they are used
        self.light_paths = []
        self.illuminated_targets = set()

        self._rebuild_grid_map()
        self._recalculate_light_path()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- Handle Actions ---
        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # 2. Cycle Crystal Type (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % len(self.available_crystal_types)

        # 3. Place Crystal (on press)
        crystal_placed = False
        cursor_tuple = tuple(self.cursor_pos)
        selected_type = self.available_crystal_types[self.selected_crystal_idx]

        if space_held and not self.prev_space_held and self.crystal_inventory[selected_type] > 0:
            if cursor_tuple not in self.grid_map:
                self.placed_crystals[cursor_tuple] = selected_type
                self.crystal_inventory[selected_type] -= 1
                self._rebuild_grid_map()
                crystal_placed = True
                reward -= 0.1

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State & Rewards ---
        if crystal_placed:
            prev_illuminated = self.illuminated_targets.copy()
            self._recalculate_light_path()
            newly_lit = self.illuminated_targets - prev_illuminated
            if newly_lit:
                reward += len(newly_lit) * 5.0

        reward += len(self.illuminated_targets) * 1.0

        self.steps += 1
        self.score += reward
        terminated = self._check_termination()

        if terminated:
            if self.win:
                reward += 50.0 # Win bonus
            elif sum(self.crystal_inventory.values()) == 0 and not self.win:
                reward -= 50.0 # Lose penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _rebuild_grid_map(self):
        self.grid_map = {}
        for pos in self.walls: self.grid_map[pos] = "wall"
        for pos in self.targets: self.grid_map[pos] = "target"
        for pos, type in self.placed_crystals.items(): self.grid_map[pos] = type
        self.grid_map[self.light_source["pos"]] = "source"

    def _recalculate_light_path(self):
        self.light_paths = []
        self.illuminated_targets.clear()
        
        beams_to_process = [(self.light_source["pos"], self.light_source["dir"], self.MAX_BOUNCES)]
        
        processed_beams = set()

        while beams_to_process:
            pos, direction, bounces = beams_to_process.pop(0)
            
            beam_key = (pos, direction, bounces)
            if beam_key in processed_beams: continue
            processed_beams.add(beam_key)

            path_segment = [pos]
            current_pos = pos

            for _ in range(self.GRID_WIDTH + self.GRID_HEIGHT):
                next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                path_segment.append(next_pos)
                
                obj_type = self.grid_map.get(next_pos)

                if obj_type == "target":
                    self.illuminated_targets.add(next_pos)
                
                if obj_type in ["wall", "refractor", "splitter", "source"]:
                    if bounces > 0:
                        if obj_type == "refractor":
                            # 45-degree clockwise bend
                            new_dir = (direction[0] - direction[1], direction[0] + direction[1])
                            beams_to_process.append((next_pos, new_dir, bounces - 1))
                        elif obj_type == "splitter":
                            # Split +/- 45 degrees
                            dir1 = (direction[0] - direction[1], direction[0] + direction[1])
                            dir2 = (direction[0] + direction[1], -direction[0] + direction[1])
                            beams_to_process.append((next_pos, dir1, bounces - 1))
                            beams_to_process.append((next_pos, dir2, bounces - 1))
                    break # Stop current segment
                
                current_pos = next_pos
            
            self.light_paths.append(path_segment)

    def _check_termination(self):
        if len(self.illuminated_targets) == self.NUM_TARGETS:
            self.game_over = True
            self.win = True
            return True
        if sum(self.crystal_inventory.values()) == 0 and not self.illuminated_targets == self.NUM_TARGETS:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
    def _grid_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_WIDTH + 1):
            start = self._grid_to_screen(i, 0)
            end = self._grid_to_screen(i, self.GRID_HEIGHT -1)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.GRID_HEIGHT):
            start = self._grid_to_screen(0, i)
            end = self._grid_to_screen(self.GRID_WIDTH - 1, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw walls, targets, crystals
        sorted_items = sorted(self.grid_map.items(), key=lambda item: item[0][0] + item[0][1])
        
        for pos, item_type in sorted_items:
            if item_type == "wall": self._draw_iso_cube(pos, self.COLOR_WALL, self.COLOR_WALL_OUTLINE)
            elif item_type == "target": self._draw_target(pos)
            elif item_type == "source": self._draw_source(pos)
            elif item_type in self.CRYSTAL_COLORS: self._draw_crystal(pos, item_type)

        # Draw light beams
        for path in self.light_paths:
            if len(path) > 1:
                screen_points = [self._grid_to_screen(p[0], p[1]) for p in path]
                # Glow effect
                pygame.draw.lines(self.screen, self.COLOR_LIGHT_GLOW, False, screen_points, 7)
                pygame.draw.lines(self.screen, self.COLOR_LIGHT_GLOW, False, screen_points, 5)
                # Core beam
                pygame.draw.lines(self.screen, self.COLOR_LIGHT_BEAM, False, screen_points, 2)
        
        # Draw cursor
        self._draw_cursor()

    def _draw_iso_cube(self, pos, color, outline_color):
        x, y = pos
        center_x, center_y = self._grid_to_screen(x, y)
        points = [
            (center_x, center_y - self.TILE_HEIGHT_HALF),
            (center_x + self.TILE_WIDTH_HALF, center_y),
            (center_x, center_y + self.TILE_HEIGHT_HALF),
            (center_x - self.TILE_WIDTH_HALF, center_y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _draw_target(self, pos):
        is_lit = pos in self.illuminated_targets
        color = self.COLOR_TARGET_ON if is_lit else self.COLOR_TARGET_OFF
        center_x, center_y = self._grid_to_screen(pos[0], pos[1])
        
        if is_lit: # Add glow effect when lit
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 9, (*color, 50))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 7, (*color, 100))
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 5, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 5, (255, 255, 255))

    def _draw_source(self, pos):
        center_x, center_y = self._grid_to_screen(pos[0], pos[1])
        color = self.COLOR_LIGHT_BEAM
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 8, (*color, 50))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 6, (*color, 100))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 4, color)

    def _draw_crystal(self, pos, type):
        color = self.CRYSTAL_COLORS[type]
        center_x, center_y = self._grid_to_screen(pos[0], pos[1])
        
        # Glow
        pygame.gfxdraw.filled_polygon(self.screen, self._get_diamond_points(center_x, center_y, 10), (*color, 100))
        # Body
        pygame.gfxdraw.filled_polygon(self.screen, self._get_diamond_points(center_x, center_y, 7), color)
        pygame.gfxdraw.aapolygon(self.screen, self._get_diamond_points(center_x, center_y, 7), (255, 255, 255))

    def _get_diamond_points(self, cx, cy, size):
        return [ (cx, cy - size), (cx + size//2, cy), (cx, cy + size), (cx - size//2, cy) ]

    def _draw_cursor(self):
        center_x, center_y = self._grid_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        points = [
            (center_x - self.TILE_WIDTH_HALF, center_y), (center_x, center_y - self.TILE_HEIGHT_HALF),
            (center_x + self.TILE_WIDTH_HALF, center_y), (center_x, center_y + self.TILE_HEIGHT_HALF),
        ]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)
        
    def _render_ui(self):
        # --- Crystal Inventory ---
        y_offset = 20
        for i, c_type in enumerate(self.available_crystal_types):
            color = self.CRYSTAL_COLORS[c_type]
            count = self.crystal_inventory[c_type]
            text = f"{c_type.upper()}: {count}"
            
            # Draw crystal preview
            diamond_pts = self._get_diamond_points(30, y_offset, 10)
            pygame.gfxdraw.filled_polygon(self.screen, diamond_pts, color)
            pygame.gfxdraw.aapolygon(self.screen, diamond_pts, (255,255,255))
            
            # Draw text
            txt_surf = self.FONT_UI.render(text, True, (255, 255, 255))
            self.screen.blit(txt_surf, (50, y_offset - txt_surf.get_height() // 2))

            # Draw selector
            if i == self.selected_crystal_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (10, y_offset-12, 150, 24), 2)
            
            y_offset += 40

        # --- Targets Lit ---
        targets_text = f"TARGETS: {len(self.illuminated_targets)} / {self.NUM_TARGETS}"
        txt_surf = self.FONT_UI.render(targets_text, True, self.COLOR_TARGET_ON)
        self.screen.blit(txt_surf, (self.WIDTH - txt_surf.get_width() - 20, 20))

        # --- Game Over Message ---
        if self.game_over:
            msg = "VICTORY!" if self.win else "OUT OF CRYSTALS"
            if self.steps >= self.MAX_STEPS: msg = "TIME UP"
            color = self.COLOR_TARGET_ON if self.win else self.COLOR_TARGET_OFF
            
            msg_surf = self.FONT_MSG.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = msg_rect.inflate(40, 40)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(msg_surf, msg_rect)

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
            "crystals_left": sum(self.crystal_inventory.values()),
            "targets_lit": len(self.illuminated_targets),
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # This is a helper function to check if the implementation follows the Gymnasium API
        print("✓ Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    try:
        env = GameEnv()
        env.validate_implementation()
        obs, info = env.reset()
        done = False
        
        # --- Key mapping for manual play ---
        key_to_action = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }

        # Use a separate screen for rendering if playing manually
        render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Crystal Cavern")
        clock = pygame.time.Clock()

        print(GameEnv.user_guide)
        print(GameEnv.game_description)

        # Debounce for shift and space
        last_space = False
        last_shift = False

        while not done:
            # --- Action gathering ---
            movement = 0
            space = 0
            shift = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            for key, move_action in key_to_action.items():
                if keys[key]:
                    movement = move_action
                    break # only one movement at a time
            
            current_space = keys[pygame.K_SPACE]
            if current_space and not last_space:
                space = 1
            last_space = current_space
            
            current_shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            if current_shift and not last_shift:
                shift = 1
            last_shift = current_shift

            if keys[pygame.K_r]: # Add a reset key for convenience
                obs, info = env.reset()
                last_space = False
                last_shift = False
                continue

            action = [movement, space, shift]
            
            # --- Step the environment ---
            # Only step if an action is taken for this turn-based game
            if any(action):
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # --- Render to the display ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(20) # Limit frame rate for manual play

        print(f"Game Over. Final Info: {info}")
        
        # Keep the final screen visible for a few seconds
        pygame.time.wait(3000)

        env.close()
    except pygame.error as e:
        print(f"\nPygame error: '{e}'. This is expected in a headless environment. Manual play is not available.")
        print("The environment class is still valid for training.")