
# Generated: 2025-08-27T13:22:51.269880
# Source Brief: brief_00342.md
# Brief Index: 342

        
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
        "Controls: Arrow keys to move cursor. Space to place a crystal. Shift to cycle crystal type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a crystal cavern, placing light-refracting crystals to illuminate all targets."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Game world
    GRID_WIDTH = 24
    GRID_HEIGHT = 16
    CELL_SIZE_X = 28
    CELL_SIZE_Y = 14
    ISO_OFFSET_X = 320
    ISO_OFFSET_Y = 50
    WALL_HEIGHT = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 40, 60)
    COLOR_WALL = (70, 80, 100)
    COLOR_WALL_TOP = (90, 100, 120)

    COLOR_TARGET_UNLIT = (80, 80, 80)
    COLOR_TARGET_LIT = (255, 255, 220)
    COLOR_TARGET_GLOW = (255, 255, 180)

    COLOR_CURSOR = (255, 255, 0)
    
    CRYSTAL_COLORS = {
        1: (255, 50, 50),   # Red: '/' mirror
        2: (50, 255, 50),   # Green: '\' mirror
        3: (80, 80, 255),   # Blue: Retro-reflector
    }
    
    LIGHT_COLOR = (255, 255, 255)
    LIGHT_GLOW_COLOR = (200, 200, 100)

    # Directions (dx, dy)
    DIR_MAP = {
        "N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)
    }
    
    # Crystal type IDs
    CRYSTAL_RED = 1
    CRYSTAL_GREEN = 2
    CRYSTAL_BLUE = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 15)
        self.font_large = pygame.font.SysFont("monospace", 20, bold=True)

        self.grid = None
        self.cursor_pos = None
        self.targets = None
        self.light_source_pos = None
        self.light_source_dir = None
        self.light_path = None
        self.placed_crystals = None
        self.crystal_inventory = None
        self.selected_crystal_type = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._procedural_generation()

        self.selected_crystal_type = self.CRYSTAL_RED
        self.light_path = []
        self._calculate_light_path()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1

        # 1. Handle Shift: Cycle selected crystal
        if shift_pressed:
            self.selected_crystal_type += 1
            if self.selected_crystal_type > self.CRYSTAL_BLUE:
                self.selected_crystal_type = self.CRYSTAL_RED
            # sfx: UI_crystal_cycle.wav
        
        # 2. Handle Movement: Move cursor
        if movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            new_x, new_y = self.cursor_pos[0] + dx, self.cursor_pos[1] + dy
            if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT:
                self.cursor_pos = (new_x, new_y)
            # sfx: UI_cursor_move.wav

        # 3. Handle Space: Place crystal
        if space_pressed:
            x, y = self.cursor_pos
            can_place = (
                self.grid[y][x] == 0 and
                self.crystal_inventory[self.selected_crystal_type] > 0
            )
            if can_place:
                # sfx: place_crystal.wav
                self.grid[y][x] = self.selected_crystal_type
                self.placed_crystals[(x, y)] = self.selected_crystal_type
                self.crystal_inventory[self.selected_crystal_type] -= 1
                
                reward -= 0.1
                self.score -= 0.1

                targets_lit_before = sum(t['lit'] for t in self.targets)
                self._calculate_light_path()
                targets_lit_after = sum(t['lit'] for t in self.targets)
                
                newly_lit = targets_lit_after - targets_lit_before
                if newly_lit > 0:
                    # sfx: target_lit.wav
                    reward += newly_lit
                    self.score += newly_lit
            else:
                # sfx: action_fail.wav
                pass
        
        # 4. Check for termination
        terminated = False
        all_targets_lit = all(t['lit'] for t in self.targets)
        no_crystals_left = sum(self.crystal_inventory.values()) == 0

        if all_targets_lit:
            # sfx: win_jingle.wav
            reward += 50
            self.score += 50
            terminated = True
            self.game_over = True
        elif no_crystals_left and not self._can_still_win():
            # sfx: lose_jingle.wav
            reward -= 10
            self.score -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _procedural_generation(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        # Walls
        self.grid[0, :] = -1
        self.grid[-1, :] = -1
        self.grid[:, 0] = -1
        self.grid[:, -1] = -1
        for _ in range(int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.1)):
            x, y = self.np_random.integers(1, [self.GRID_WIDTH-1, self.GRID_HEIGHT-1])
            self.grid[y, x] = -1

        # Light Source
        self.light_source_pos = (1, self.np_random.integers(1, self.GRID_HEIGHT - 1))
        self.light_source_dir = self.DIR_MAP["E"]
        self.grid[self.light_source_pos[1]][self.light_source_pos[0]] = 0 # Ensure start is clear

        # Targets
        self.targets = []
        possible_targets = []
        for y in range(1, self.GRID_HEIGHT - 1):
            for x in range(1, self.GRID_WIDTH - 1):
                if self.grid[y][x] == 0 and (x,y) != self.light_source_pos:
                    possible_targets.append({'pos': (x, y), 'lit': False})
        
        # Ensure one easy target
        easy_y = self.light_source_pos[1]
        for x in range(self.light_source_pos[0] + 3, self.GRID_WIDTH - 2):
            if all(self.grid[easy_y][i] == 0 for i in range(self.light_source_pos[0], x + 1)):
                if {'pos': (x, easy_y), 'lit': False} in possible_targets:
                    self.targets.append({'pos': (x, easy_y), 'lit': False})
                    possible_targets.remove({'pos': (x, easy_y), 'lit': False})
                    self.grid[easy_y][x] = -2 # Mark as target
                    break
        
        # Add more random targets
        num_targets = self.np_random.integers(3, 6)
        if len(possible_targets) > 0 and len(self.targets) < num_targets:
            self.np_random.shuffle(possible_targets)
            for t in possible_targets[:num_targets - len(self.targets)]:
                 self.targets.append(t)
                 self.grid[t['pos'][1]][t['pos'][0]] = -2 # Mark as target
        
        # Clear walls that block targets
        for t in self.targets:
            tx, ty = t['pos']
            if self.grid[ty, tx-1] == -1: self.grid[ty, tx-1] = 0
            if self.grid[ty, tx+1] == -1: self.grid[ty, tx+1] = 0
            if self.grid[ty-1, tx] == -1: self.grid[ty-1, tx] = 0
            if self.grid[ty+1, tx] == -1: self.grid[ty+1, tx] = 0


        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        if self.grid[self.cursor_pos[1]][self.cursor_pos[0]] != 0:
            self.cursor_pos = (self.light_source_pos[0]+1, self.light_source_pos[1])

        self.crystal_inventory = {self.CRYSTAL_RED: 5, self.CRYSTAL_GREEN: 5, self.CRYSTAL_BLUE: 5}
        self.placed_crystals = {}

    def _calculate_light_path(self):
        # Reset targets and path
        for t in self.targets:
            t['lit'] = False
        self.light_path = []

        pos = self.light_source_pos
        direction = self.light_source_dir
        path_segment = [pos]

        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT): # Safety break
            next_pos = (pos[0] + direction[0], pos[1] + direction[1])

            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                break # Hit edge of map

            path_segment.append(next_pos)
            cell_content = self.grid[next_pos[1]][next_pos[0]]

            if cell_content == -1: # Wall
                break
            
            if cell_content == -2: # Target
                 for t in self.targets:
                    if t['pos'] == next_pos:
                        t['lit'] = True
            
            if cell_content > 0: # Crystal
                crystal_type = cell_content
                dx, dy = direction
                
                if crystal_type == self.CRYSTAL_RED: # '/' mirror
                    direction = (-dy, -dx)
                elif crystal_type == self.CRYSTAL_GREEN: # '\' mirror
                    direction = (dy, dx)
                elif crystal_type == self.CRYSTAL_BLUE: # Retro-reflector
                    direction = (-dx, -dy)

                self.light_path.append(path_segment)
                path_segment = [next_pos]
            
            pos = next_pos
        
        self.light_path.append(path_segment)

    def _can_still_win(self):
        # This is a complex check. For this implementation, we simply assume a solution
        # might exist until all crystals are used, preventing premature termination.
        return True

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
            "cursor_pos": self.cursor_pos,
            "targets_lit": sum(t['lit'] for t in self.targets),
            "total_targets": len(self.targets),
            "crystals_left": self.crystal_inventory.copy()
        }

    def _world_to_iso(self, x, y):
        iso_x = self.ISO_OFFSET_X + (x - y) * self.CELL_SIZE_X
        iso_y = self.ISO_OFFSET_Y + (x + y) * self.CELL_SIZE_Y
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, x, y, color, top_color, height):
        px, py = self._world_to_iso(x, y)
        
        points_top = [
            (px, py - height),
            (px + self.CELL_SIZE_X, py + self.CELL_SIZE_Y - height),
            (px, py + 2 * self.CELL_SIZE_Y - height),
            (px - self.CELL_SIZE_X, py + self.CELL_SIZE_Y - height),
        ]
        
        points_bottom = [
            (px, py),
            (px + self.CELL_SIZE_X, py + self.CELL_SIZE_Y),
            (px, py + 2 * self.CELL_SIZE_Y),
            (px - self.CELL_SIZE_X, py + self.CELL_SIZE_Y),
        ]

        pygame.gfxdraw.filled_polygon(surface, points_top, top_color)
        pygame.gfxdraw.aapolygon(surface, points_top, top_color)

        darker_color = tuple(max(0, c - 20) for c in color)
        pygame.draw.polygon(surface, darker_color, [points_top[1], points_bottom[1], points_bottom[2], points_top[2]])
        pygame.draw.polygon(surface, color, [points_top[2], points_bottom[2], points_bottom[3], points_top[3]])

    def _draw_glowing_circle(self, surface, center, radius, color, glow_color):
        glow_radius = int(radius * 2.5)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color + (30,), (glow_radius, glow_radius), glow_radius)
        pygame.draw.circle(temp_surf, glow_color + (50,), (glow_radius, glow_radius), int(glow_radius * 0.7))
        surface.blit(temp_surf, (center[0] - glow_radius, center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)

    def _render_game(self):
        # Draw grid, walls, targets
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell_type = self.grid[y][x]
                iso_x, iso_y = self._world_to_iso(x, y)
                
                points = [
                    (iso_x, iso_y),
                    (iso_x + self.CELL_SIZE_X, iso_y + self.CELL_SIZE_Y),
                    (iso_x, iso_y + 2 * self.CELL_SIZE_Y),
                    (iso_x - self.CELL_SIZE_X, iso_y + self.CELL_SIZE_Y),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

                if cell_type == -1: # Wall
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_WALL, self.COLOR_WALL_TOP, self.WALL_HEIGHT)
                elif cell_type == -2: # Target
                    target = next((t for t in self.targets if t['pos'] == (x,y)), None)
                    if target:
                        color = self.COLOR_TARGET_LIT if target['lit'] else self.COLOR_TARGET_UNLIT
                        glow_color = self.COLOR_TARGET_GLOW if target['lit'] else self.COLOR_TARGET_UNLIT
                        center_y = iso_y + self.CELL_SIZE_Y
                        self._draw_glowing_circle(self.screen, (iso_x, center_y), 8, color, glow_color)

        # Draw light source
        ls_x, ls_y = self.light_source_pos
        self._draw_iso_cube(self.screen, ls_x, ls_y, (150,150,0), (255,255,100), 5)

        # Draw light path
        for segment in self.light_path:
            if len(segment) > 1:
                iso_points = []
                for x, y in segment:
                    iso_x, iso_y = self._world_to_iso(x, y)
                    iso_points.append((iso_x, iso_y + self.CELL_SIZE_Y))
                
                pygame.draw.lines(self.screen, self.LIGHT_GLOW_COLOR, False, iso_points, 7)
                pygame.draw.lines(self.screen, self.LIGHT_GLOW_COLOR, False, iso_points, 11)
                pygame.draw.lines(self.screen, self.LIGHT_COLOR, False, iso_points, 3)

        # Draw placed crystals
        for (x, y), crystal_type in self.placed_crystals.items():
            iso_x, iso_y = self._world_to_iso(x, y)
            center_y = iso_y + self.CELL_SIZE_Y
            color = self.CRYSTAL_COLORS[crystal_type]
            points = [
                (iso_x, center_y - 8), (iso_x + 8, center_y),
                (iso_x, center_y + 8), (iso_x - 8, center_y)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Draw cursor
        cur_x, cur_y = self.cursor_pos
        if self.grid[cur_y][cur_x] == 0:
            iso_x, iso_y = self._world_to_iso(cur_x, cur_y)
            points = [
                (iso_x, iso_y),
                (iso_x + self.CELL_SIZE_X, iso_y + self.CELL_SIZE_Y),
                (iso_x, iso_y + 2 * self.CELL_SIZE_Y),
                (iso_x - self.CELL_SIZE_X, iso_y + self.CELL_SIZE_Y),
            ]
            pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 2)
    
    def _render_ui(self):
        ui_x, ui_y = 10, 10
        for c_type in sorted(self.crystal_inventory.keys()):
            color = self.CRYSTAL_COLORS[c_type]
            count = self.crystal_inventory[c_type]
            
            if c_type == self.selected_crystal_type:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (ui_x-2, ui_y-2, 34, 34), 2)
            
            pygame.draw.rect(self.screen, color, (ui_x, ui_y, 30, 30))
            text = self.font_large.render(str(count), True, (255, 255, 255))
            self.screen.blit(text, (ui_x + 35, ui_y + 5))
            ui_x += 80

        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, (200, 200, 200))
        self.screen.blit(score_text, (630 - score_text.get_width(), 10))
        
        targets_lit = sum(t['lit'] for t in self.targets)
        total_targets = len(self.targets)
        target_text = self.font_small.render(f"TARGETS: {targets_lit}/{total_targets}", True, (200, 200, 200))
        self.screen.blit(target_text, (630 - target_text.get_width(), 30))
        
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, (200, 200, 200))
        self.screen.blit(steps_text, (630 - steps_text.get_width(), 50))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((640, 400))
    
    terminated = False
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
        
        if action_taken:
            action = [movement, space, shift]
            obs, reward, term, truncated, info = env.step(action)
            terminated = term
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)
        
    print(f"Game Over. Final Score: {info['score']:.2f}")
    pygame.quit()