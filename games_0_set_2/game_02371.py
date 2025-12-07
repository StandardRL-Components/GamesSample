
# Generated: 2025-08-28T04:35:23.271095
# Source Brief: brief_02371.md
# Brief Index: 2371

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys to move the placement cursor. Press space to place a reflection crystal."

    # Must be a short, user-facing description of the game:
    game_description = "An isometric puzzle game. Place crystals to reflect a light beam and illuminate all the targets before you run out of crystals."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Grid and world
    GRID_WIDTH = 20
    GRID_HEIGHT = 16
    TILE_WIDTH = 36
    TILE_HEIGHT = 18
    MAX_STEPS = 1000
    INITIAL_CRYSTALS = 20
    NUM_TARGETS = 12
    MAX_BEAM_LENGTH = GRID_WIDTH * GRID_HEIGHT

    # Grid cell types
    CELL_EMPTY = 0
    CELL_WALL = 1
    CELL_SOURCE = 2
    CELL_TARGET = 3
    CELL_PLACED = 4

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_WALL = (40, 40, 50)
    COLOR_SOURCE = (255, 255, 100)
    COLOR_SOURCE_GLOW = (80, 80, 20)
    COLOR_TARGET_UNLIT = (50, 80, 180)
    COLOR_TARGET_LIT = (100, 200, 255)
    COLOR_TARGET_LIT_GLOW = (30, 60, 80)
    COLOR_PLACED = (200, 100, 255)
    COLOR_PLACED_GLOW = (60, 30, 80)
    COLOR_BEAM = (255, 255, 180)
    COLOR_CURSOR_VALID = (200, 100, 255, 100)
    COLOR_CURSOR_INVALID = (255, 50, 50, 100)
    COLOR_UI_TEXT = (220, 220, 220)

    # Fixed level layout
    LEVEL_LAYOUT = [
        "WWWWWWWWWWWWWWWWWWWW",
        "W..................W",
        "W.T................W",
        "W................T.W",
        "W..W...........W...W",
        "W..W.T.......T.W...W",
        "W..W...........W...W",
        "W.S....T...T.......W",
        "W..W...........W...W",
        "W..W.T.......T.W...W",
        "W..W...........W...W",
        "W.T................W",
        "W................T.W",
        "W...T..........T...W",
        "WWWWWWWWWWWWWWWWWWWW",
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 48, bold=True)
        
        # World offsets for centering isometric grid
        self.offset_x = 640 // 2
        self.offset_y = (400 - (self.GRID_HEIGHT * self.TILE_HEIGHT / 2)) // 2 + 30

        # State variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.light_source_pos = None
        self.target_crystal_pos = None
        self.placed_crystal_pos = None
        self.beam_path = None
        self.illuminated_targets = None
        self.crystals_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.crystals_remaining = self.INITIAL_CRYSTALS
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.placed_crystal_pos = []
        self.target_crystal_pos = []
        
        self._generate_level()
        self._calculate_light_path()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), self.CELL_EMPTY, dtype=int)
        for y, row in enumerate(self.LEVEL_LAYOUT):
            for x, char in enumerate(row):
                if char == 'W':
                    self.grid[y, x] = self.CELL_WALL
                elif char == 'S':
                    self.grid[y, x] = self.CELL_SOURCE
                    self.light_source_pos = (x, y)
                elif char == 'T':
                    self.grid[y, x] = self.CELL_TARGET
                    self.target_crystal_pos.append((x, y))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        place_crystal_action = action[1] == 1  # Boolean
        
        # 1. Handle cursor movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_cursor_x = self.cursor_pos[0] + dx
        new_cursor_y = self.cursor_pos[1] + dy
        
        if 0 <= new_cursor_x < self.GRID_WIDTH and 0 <= new_cursor_y < self.GRID_HEIGHT:
            self.cursor_pos = (new_cursor_x, new_cursor_y)
        
        # 2. Handle crystal placement
        if place_crystal_action:
            if self._is_valid_placement(self.cursor_pos):
                # Place the crystal
                self.grid[self.cursor_pos[1], self.cursor_pos[0]] = self.CELL_PLACED
                self.placed_crystal_pos.append(self.cursor_pos)
                self.crystals_remaining -= 1
                
                # Recalculate light and get reward
                lit_before = len(self.illuminated_targets)
                self._calculate_light_path()
                lit_after = len(self.illuminated_targets)
                
                newly_lit = lit_after - lit_before
                if newly_lit > 0:
                    reward += newly_lit  # +1 for each new crystal lit
                
                # Check for first-time completion bonus
                if lit_after == self.NUM_TARGETS and lit_before < self.NUM_TARGETS:
                    reward += 10
        
        # 3. Update state and check for termination
        self.steps += 1
        
        terminated = False
        if len(self.illuminated_targets) == self.NUM_TARGETS:
            terminated = True
            self.win = True
            self.game_over = True
            reward += 100  # Win bonus
        elif self.crystals_remaining <= 0 and not self.win:
            terminated = True
            self.game_over = True
            reward -= 100  # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward -= 100 # Loss penalty
        
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _is_valid_placement(self, pos):
        if self.crystals_remaining <= 0:
            return False
        if self.grid[pos[1], pos[0]] != self.CELL_EMPTY:
            return False
        return True

    def _calculate_light_path(self):
        self.beam_path = []
        self.illuminated_targets = set()
        
        pos = self.light_source_pos
        direction = (1, 0) # Fixed to right from source
        
        path_points = [pos]
        
        for _ in range(self.MAX_BEAM_LENGTH):
            pos = (pos[0] + direction[0], pos[1] + direction[1])
            
            if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT):
                break
            
            path_points.append(pos)
            cell_type = self.grid[pos[1], pos[0]]
            
            if cell_type == self.CELL_WALL:
                break
            elif cell_type == self.CELL_TARGET:
                self.illuminated_targets.add(pos)
                # Beam passes through
            elif cell_type == self.CELL_PLACED:
                # 90-degree clockwise reflection
                dx, dy = direction
                direction = (dy, -dx)
        
        self.beam_path = path_points

    def _iso_to_screen(self, x, y):
        screen_x = self.offset_x + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.offset_y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid elements
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._iso_to_screen(x, y)
                cell_type = self.grid[y, x]
                
                if cell_type == self.CELL_WALL:
                    self._draw_iso_cube(screen_pos, self.COLOR_WALL)
                elif cell_type == self.CELL_SOURCE:
                    self._draw_glowing_circle(screen_pos, self.COLOR_SOURCE, self.COLOR_SOURCE_GLOW, 10)
                elif cell_type == self.CELL_TARGET:
                    is_lit = (x, y) in self.illuminated_targets
                    color = self.COLOR_TARGET_LIT if is_lit else self.COLOR_TARGET_UNLIT
                    glow_color = self.COLOR_TARGET_LIT_GLOW if is_lit else None
                    self._draw_glowing_diamond(screen_pos, color, glow_color, 8)
                elif cell_type == self.CELL_PLACED:
                    self._draw_glowing_diamond(screen_pos, self.COLOR_PLACED, self.COLOR_PLACED_GLOW, 8)
        
        # Draw light beam
        if self.beam_path and len(self.beam_path) > 1:
            screen_path = [self._iso_to_screen(p[0], p[1]) for p in self.beam_path]
            # Use a separate surface for transparency
            beam_surface = self.screen.copy()
            beam_surface.set_colorkey((0,0,0))
            beam_surface.fill((0,0,0))
            pygame.draw.lines(beam_surface, self.COLOR_BEAM, False, screen_path, 3)
            beam_surface.set_alpha(180)
            self.screen.blit(beam_surface, (0,0))

        # Draw cursor
        cursor_screen_pos = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        is_valid = self._is_valid_placement(self.cursor_pos)
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        self._draw_iso_rect_outline(cursor_screen_pos, cursor_color)

    def _render_ui(self):
        # Crystals remaining
        text_surf = self.font_ui.render(f"Crystals: {self.crystals_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Targets lit
        text_surf = self.font_ui.render(f"Targets Lit: {len(self.illuminated_targets)}/{self.NUM_TARGETS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 30))

        # Game over message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (150, 255, 150) if self.win else (255, 100, 100)
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(640 // 2, 400 // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _draw_iso_cube(self, pos, color):
        x, y = pos
        w, h = self.TILE_WIDTH, self.TILE_HEIGHT
        points = [
            (x, y - h/2), (x + w/2, y), (x, y + h/2), (x - w/2, y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_glowing_circle(self, pos, color, glow_color, radius):
        if glow_color:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 1.8), glow_color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
        
    def _draw_glowing_diamond(self, pos, color, glow_color, size):
        x, y = pos
        points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
        if glow_color:
            glow_points = [(x, y - int(size*1.8)), (x + int(size*1.8), y), (x, y + int(size*1.8)), (x - int(size*1.8), y)]
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, glow_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_iso_rect_outline(self, pos, color):
        x, y = pos
        w, h = self.TILE_WIDTH, self.TILE_HEIGHT
        points = [
            (x, y - h/2), (x + w/2, y), (x, y + h/2), (x - w/2, y)
        ]
        
        temp_surf = pygame.Surface((640, 400), pygame.SRCALPHA)
        pygame.draw.lines(temp_surf, color, True, points, 2)
        self.screen.blit(temp_surf, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_remaining": self.crystals_remaining,
            "illuminated_targets": len(self.illuminated_targets),
            "cursor_pos": self.cursor_pos,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment for visualization
if __name__ == '__main__':
    import os
    # Set a dummy video driver if not running with a display
    if "SDL_VIDEODRIVER" not in os.environ:
         os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # Re-initialize pygame for display
    pygame.display.init()
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Crystal Cavern")
    
    terminated = False
    running = True
    clock = pygame.time.Clock()
    
    # Track key presses for single action per press
    last_space_press = False

    print(env.user_guide)
    print(env.game_description)

    while running:
        action = np.array([0, 0, 0])  # Default no-op
        should_step = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    env.reset()
                    terminated = False
                
                # For turn-based, we only step on a key press
                if not terminated:
                    movement = 0
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    
                    space = 1 if event.key == pygame.K_SPACE else 0

                    if movement > 0 or space > 0:
                        action = np.array([movement, space, 0])
                        should_step = True
        
        if not terminated and should_step:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) 

    env.close()