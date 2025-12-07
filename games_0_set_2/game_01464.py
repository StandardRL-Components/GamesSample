
# Generated: 2025-08-27T17:14:44.582160
# Source Brief: brief_01464.md
# Brief Index: 1464

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle crystal type. Press Space to place a crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically place light-bending crystals in an isometric cavern to illuminate all target areas."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    # Colors
    COLOR_BG = (15, 18, 26)
    COLOR_GRID = (30, 35, 50)
    COLOR_WALL = (50, 55, 70)
    COLOR_WALL_TOP = (70, 75, 90)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    
    CRYSTAL_COLORS = {
        0: {"main": (255, 50, 50), "glow": (180, 20, 20)},  # Red
        1: {"main": (50, 255, 50), "glow": (20, 180, 20)},  # Green
        2: {"main": (80, 80, 255), "glow": (30, 30, 180)},  # Blue
    }
    TARGET_COLOR_OFF = (180, 140, 20)
    TARGET_COLOR_ON = (255, 215, 0)
    BEAM_COLOR = (255, 255, 255)
    BEAM_GLOW_COLOR = (200, 200, 255)

    # Grid & Isometric Projection
    GRID_W, GRID_H = 16, 16
    TILE_W, TILE_H = 40, 20
    ORIGIN_X, ORIGIN_Y = 320, 60

    # Directions (Hex-style on a square grid)
    # E, SE, SW, W, NW, NE
    DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

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
        
        self.font_main = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 48)
        
        self._setup_layout()
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def _setup_layout(self):
        """Initializes fixed game elements like walls, targets, and light source."""
        self.walls = set()
        for i in range(-1, self.GRID_W + 1):
            self.walls.add((i, -1))
            self.walls.add((i, self.GRID_H))
        for i in range(-1, self.GRID_H + 1):
            self.walls.add((-1, i))
            self.walls.add((self.GRID_W, i))
        
        self.targets = [(3, 3), (12, 3), (3, 12), (12, 12), (7, 8)]
        self.light_source = {'pos': (self.GRID_W // 2, 0), 'dir_idx': 1} # Start at top-center, facing SE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.cursor_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.selected_crystal_type = 0  # 0: Red, 1: Green, 2: Blue
        self.crystals_placed = {}  # {(x, y): type}
        self.crystals_remaining = 20
        
        self.targets_lit_status = [False] * len(self.targets)
        self.light_path_segments = []
        
        self.last_space_held = 0
        self.last_shift_held = 0

        self._update_game_state()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        self.steps += 1
        reward = 0
        crystal_placed_this_step = False

        self._handle_input(movement, space_held, shift_held)

        # Handle crystal placement on press
        if space_held and not self.last_space_held:
            if self.crystals_remaining > 0 and self.cursor_pos not in self.walls and self.cursor_pos not in self.crystals_placed and self.cursor_pos not in self.targets and self.cursor_pos != self.light_source['pos']:
                # sfx: crystal_place.wav
                self.crystals_placed[self.cursor_pos] = self.selected_crystal_type
                self.crystals_remaining -= 1
                crystal_placed_this_step = True
                reward = -0.1

        self.last_space_held, self.last_shift_held = space_held, shift_held

        prev_lit_count = sum(self.targets_lit_status)
        if crystal_placed_this_step:
            self._update_game_state()
            newly_lit_count = sum(self.targets_lit_status) - prev_lit_count
            if newly_lit_count > 0:
                # sfx: target_lit.wav
                reward += newly_lit_count * 10
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_x = max(0, min(self.GRID_W - 1, self.cursor_pos[0] + dx))
        new_y = max(0, min(self.GRID_H - 1, self.cursor_pos[1] + dy))
        self.cursor_pos = (new_x, new_y)

        # Handle crystal type cycle on press
        if shift_held and not self.last_shift_held:
            # sfx: ui_cycle.wav
            self.selected_crystal_type = (self.selected_crystal_type + 1) % 3

    def _update_game_state(self):
        """Recalculates the light path and which targets are lit."""
        self.light_path_segments = self._calculate_light_path()
        self._update_target_status()

    def _calculate_light_path(self):
        path_segments = []
        pos = self.light_source['pos']
        dir_idx = self.light_source['dir_idx']
        
        for _ in range(30): # Max bounces to prevent infinite loops
            start_pos = pos
            current_segment = [start_pos]

            next_pos = start_pos
            for i in range(1, max(self.GRID_W, self.GRID_H) + 5):
                next_pos = (start_pos[0] + self.DIRECTIONS[dir_idx][0] * i,
                            start_pos[1] + self.DIRECTIONS[dir_idx][1] * i)

                if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                    current_segment.append(next_pos)
                    path_segments.append(current_segment)
                    return path_segments

                if next_pos in self.crystals_placed:
                    # sfx: beam_refract.wav
                    current_segment.append(next_pos)
                    path_segments.append(current_segment)
                    pos = next_pos
                    crystal_type = self.crystals_placed[next_pos]
                    dir_idx = (dir_idx + (crystal_type + 1)) % 6
                    break
            else:
                current_segment.append(next_pos)
                path_segments.append(current_segment)
                return path_segments
        
        return path_segments

    def _update_target_status(self):
        path_points = set()
        for segment in self.light_path_segments:
            if len(segment) < 2: continue
            for i in range(len(segment) - 1):
                x1, y1 = segment[i]
                x2, y2 = segment[i+1]
                dx, dy = abs(x2 - x1), -abs(y2 - y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx + dy
                while True:
                    path_points.add((x1, y1))
                    if x1 == x2 and y1 == y2: break
                    e2 = 2 * err
                    if e2 >= dy:
                        err += dy
                        x1 += sx
                    if e2 <= dx:
                        err += dx
                        y1 += sy
        
        for i, target_pos in enumerate(self.targets):
            self.targets_lit_status[i] = target_pos in path_points

    def _check_termination(self):
        reward = 0
        terminated = False
        if all(self.targets_lit_status):
            # sfx: win.wav
            self.win = True
            self.game_over = True
            terminated = True
            reward = 100
        elif self.crystals_remaining <= 0:
            # sfx: lose.wav
            self.game_over = True
            terminated = True
            reward = -100
        elif self.steps >= 1000:
            self.game_over = True
            terminated = True
        
        return terminated, reward

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H / 2
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_and_walls()
        self._render_targets()
        self._render_light_source()
        self._render_crystals()
        self._render_light_path()
        self._render_cursor()

    def _render_grid_and_walls(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

    def _render_targets(self):
        for i, pos in enumerate(self.targets):
            screen_pos = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
            is_lit = self.targets_lit_status[i]
            color = self.TARGET_COLOR_ON if is_lit else self.TARGET_COLOR_OFF
            radius = int(self.TILE_H * 0.6)
            
            if is_lit:
                pulse_radius = radius * (1 + 0.1 * math.sin(self.steps * 0.2))
                glow_color = (*color, 50)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(pulse_radius), glow_color)
            
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, color)

    def _render_light_source(self):
        pos = self.light_source['pos']
        screen_pos = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
        radius = int(self.TILE_H * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, self.BEAM_GLOW_COLOR)
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, self.BEAM_COLOR)
        
    def _render_crystals(self):
        for pos, type in self.crystals_placed.items():
            center_x, center_y = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
            colors = self.CRYSTAL_COLORS[type]
            size = self.TILE_H * 0.6
            
            points = [
                (center_x, center_y - size),
                (center_x + size * 0.7, center_y),
                (center_x, center_y + size),
                (center_x - size * 0.7, center_y)
            ]
            
            pygame.gfxdraw.filled_polygon(self.screen, points, (*colors['glow'], 100))
            pygame.gfxdraw.aapolygon(self.screen, points, colors['glow'])
            pygame.gfxdraw.filled_polygon(self.screen, points, colors['main'])
            pygame.gfxdraw.aapolygon(self.screen, points, colors['main'])

    def _render_light_path(self):
        for segment in self.light_path_segments:
            if len(segment) < 2: continue
            
            points_screen = [self._iso_to_screen(p[0] + 0.5, p[1] + 0.5) for p in segment]
            if len(points_screen) > 1:
                pygame.draw.aalines(self.screen, self.BEAM_GLOW_COLOR, False, points_screen, 3)
                pygame.draw.aalines(self.screen, self.BEAM_COLOR, False, points_screen, 1)

    def _render_cursor(self):
        x, y = self.cursor_pos
        p1 = self._iso_to_screen(x, y)
        p2 = self._iso_to_screen(x + 1, y)
        p3 = self._iso_to_screen(x + 1, y + 1)
        p4 = self._iso_to_screen(x, y + 1)
        
        color = self.CRYSTAL_COLORS[self.selected_crystal_type]['main']
        points = [p1, p2, p3, p4]
        
        alpha = int(128 + 127 * math.sin(pygame.time.get_ticks() * 0.005))
        pygame.gfxdraw.filled_polygon(self.screen, points, (*color, alpha // 4))
        pygame.draw.lines(self.screen, (*color, alpha), True, points, 2)

    def _render_ui(self):
        text_surf = self.font_main.render(f"Crystals: {self.crystals_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        text_surf = self.font_main.render("Selected:", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 40))
        
        colors = self.CRYSTAL_COLORS[self.selected_crystal_type]
        pygame.draw.rect(self.screen, colors['main'], (90, 42, 20, 15))
        pygame.draw.rect(self.screen, colors['glow'], (90, 42, 20, 15), 1)

        if self.game_over:
            msg = "YOU WIN!" if self.win else "OUT OF CRYSTALS"
            color = self.TARGET_COLOR_ON if self.win else self.CRYSTAL_COLORS[0]['main']
            text_surf = self.font_title.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_BG, 200))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_remaining": self.crystals_remaining,
            "targets_lit": sum(self.targets_lit_status)
        }
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((640, 400))
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            # Wait for a key press to reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    obs, info = env.reset()
                    terminated = False
        else:
            # Map pygame keys to gymnasium actions
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Pygame specific event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for human play
        
    pygame.quit()