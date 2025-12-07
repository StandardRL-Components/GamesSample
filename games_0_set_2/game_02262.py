
# Generated: 2025-08-27T19:47:28.460222
# Source Brief: brief_02262.md
# Brief Index: 2262

        
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
        "Controls: Arrow keys to move cursor. Space to place crystal. Shift to cycle crystal type."
    )

    game_description = (
        "Redirect a laser through a crystal-filled cavern to hit a target. Manage your limited energy and time. "
        "Blue crystals reflect, Yellow split, and Purple conserve energy."
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_WALL = (70, 80, 100)
    COLOR_WALL_TOP = (90, 100, 120)
    
    COLOR_TARGET = (0, 255, 120)
    COLOR_TARGET_GLOW = (0, 255, 120, 50)
    
    COLOR_SOURCE = (255, 100, 0)
    COLOR_SOURCE_GLOW = (255, 100, 0, 50)

    COLOR_LASER = (255, 20, 20)
    COLOR_LASER_GLOW = (255, 20, 20, 100)

    CRYSTAL_COLORS = [
        ((50, 150, 255), (50, 150, 255, 50)),   # Blue: Reflect
        ((255, 220, 50), (255, 220, 50, 50)),  # Yellow: Split
        ((200, 50, 255), (200, 50, 255, 50)),  # Purple: Conserve
    ]
    CRYSTAL_NAMES = ["REFLECT", "SPLIT", "CONSERVE"]

    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (35, 40, 60, 200)

    # Game Parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    MAX_STEPS = 6000  # 60 seconds at 100 steps/sec (since auto_advance=False, this is more like an action limit)
    INITIAL_ENERGY = 250
    ENERGY_PER_BOUNCE = 5
    MAX_BOUNCES = 50

    # Isometric projection
    TILE_WIDTH_HALF = 20
    TILE_HEIGHT_HALF = 10
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80
    
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
        self.font_small = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 48)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.placed_crystals = []
        self.laser_path = []
        self.laser_beams = []
        self.cursor_pos = [0, 0]
        self.selected_crystal_type = 0
        self.laser_source = (0, 0, (1, 1))
        self.target_pos = (0, 0)
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.time_left = self.MAX_STEPS
        self.energy = self.INITIAL_ENERGY
        
        self.placed_crystals = []
        self.selected_crystal_type = 0
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self._generate_level()
        self._recalculate_laser_path()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        action_taken = False
        if shift_pressed:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_COLORS)
            action_taken = True
            # sfx: UI_CYCLE_SOUND
        
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)
            action_taken = True
        
        if space_pressed:
            self._place_crystal()
            action_taken = True

        self.steps += 1
        self.time_left -= 1
        
        reward = -0.01  # Small penalty for taking a step
        
        terminated = self._check_termination()
        
        if terminated:
            if self.game_won:
                # Reward is 100 for winning, scaled by remaining energy.
                reward += 100 * max(0, self.energy / self.INITIAL_ENERGY)
                # sfx: WIN_SOUND
            else:
                # Penalty for losing
                reward -= 100
                # sfx: LOSE_SOUND
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.grid.fill(0)
        # Create a border of walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # Add some internal walls
        self.grid[5, 2:6] = 1
        self.grid[10, 4:8] = 1
        self.grid[3:8, 7] = 1

        self.laser_source = (1, 4, (1, 0)) # pos_x, pos_y, dir_x, dir_y
        self.target_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT - 3)

        # Ensure source and target are not walls
        self.grid[self.laser_source[0], self.laser_source[1]] = 0
        self.grid[self.target_pos[0], self.target_pos[1]] = 0

    def _place_crystal(self):
        pos = tuple(self.cursor_pos)
        if self.grid[pos] == 1: return # Can't place on wall
        if pos == self.laser_source[:2]: return # Can't place on source
        if pos == self.target_pos: return # Can't place on target
        if any(c['pos'] == pos for c in self.placed_crystals): return # Can't place on another crystal
        
        self.placed_crystals.append({
            'pos': pos,
            'type': self.selected_crystal_type
        })
        # sfx: CRYSTAL_PLACE_SOUND
        self._recalculate_laser_path()

    def _recalculate_laser_path(self):
        self.laser_path = []
        self.game_won = False
        self.energy = self.INITIAL_ENERGY
        
        beams = [{'pos': self.laser_source[:2], 'dir': self.laser_source[2], 'bounces': 0, 'energized': False}]
        
        processed_beams = 0
        while beams and processed_beams < 100: # Safety break for complex splits
            beam = beams.pop(0)
            processed_beams += 1
            
            start_pos = beam['pos']
            current_pos = list(start_pos)
            
            for _ in range(max(self.GRID_WIDTH, self.GRID_HEIGHT) * 2): # Trace until hit or edge
                next_pos = (current_pos[0] + beam['dir'][0], current_pos[1] + beam['dir'][1])
                
                # Check for out of bounds
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                    self.laser_path.append((start_pos, tuple(current_pos), beam['energized']))
                    break
                
                # Check for hit
                hit = False
                # Hit Target
                if next_pos == self.target_pos:
                    self.laser_path.append((start_pos, next_pos, beam['energized']))
                    self.game_won = True
                    return # Game is won, stop all calculations
                
                # Hit Crystal
                for crystal in self.placed_crystals:
                    if crystal['pos'] == next_pos:
                        self.laser_path.append((start_pos, next_pos, beam['energized']))
                        if not beam['energized']: self.energy -= self.ENERGY_PER_BOUNCE
                        else: beam['energized'] = False # Consume energize buff
                        
                        if beam['bounces'] + 1 > self.MAX_BOUNCES: break

                        # Apply crystal effect
                        # Type 0: Reflect (90 deg turn)
                        if crystal['type'] == 0:
                            new_dir = (-beam['dir'][1], beam['dir'][0])
                            beams.append({'pos': next_pos, 'dir': new_dir, 'bounces': beam['bounces'] + 1, 'energized': beam['energized']})
                        # Type 1: Split
                        elif crystal['type'] == 1:
                            dir1 = (-beam['dir'][1], beam['dir'][0])
                            dir2 = (beam['dir'][1], -beam['dir'][0])
                            beams.append({'pos': next_pos, 'dir': dir1, 'bounces': beam['bounces'] + 1, 'energized': beam['energized']})
                            beams.append({'pos': next_pos, 'dir': dir2, 'bounces': beam['bounces'] + 1, 'energized': beam['energized']})
                        # Type 2: Conserve (pass through, energize next bounce)
                        elif crystal['type'] == 2:
                            beams.append({'pos': next_pos, 'dir': beam['dir'], 'bounces': beam['bounces'], 'energized': True})
                        
                        hit = True
                        break
                if hit: break
                
                # Hit Wall
                if self.grid[next_pos] == 1:
                    self.laser_path.append((start_pos, tuple(current_pos), beam['energized']))
                    if not beam['energized']: self.energy -= self.ENERGY_PER_BOUNCE
                    else: beam['energized'] = False

                    if beam['bounces'] + 1 > self.MAX_BOUNCES: break

                    # Reflect off wall
                    if self.grid[current_pos[0] + beam['dir'][0], current_pos[1]] == 1: # Vertical wall
                        new_dir = (-beam['dir'][0], beam['dir'][1])
                    elif self.grid[current_pos[0], current_pos[1] + beam['dir'][1]] == 1: # Horizontal wall
                        new_dir = (beam['dir'][0], -beam['dir'][1])
                    else: # Corner case
                        new_dir = (-beam['dir'][0], -beam['dir'][1])
                        
                    beams.append({'pos': tuple(current_pos), 'dir': new_dir, 'bounces': beam['bounces'] + 1, 'energized': beam['energized']})
                    hit = True
                    break
                
                current_pos[0], current_pos[1] = next_pos[0], next_pos[1]

    def _check_termination(self):
        if self.game_won:
            self.game_over = True
            return True
        if self.time_left <= 0 or self.energy <= 0:
            self.game_over = True
            return True
        return False

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_iso_cube(self, x, y, color_top, color_side, height=1.0):
        tile_height = self.TILE_HEIGHT_HALF * 2
        px, py = self._iso_to_screen(x, y)
        
        top_points = [
            (px, py - tile_height * (height-1)),
            (px + self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF - tile_height * (height-1)),
            (px, py + self.TILE_HEIGHT_HALF * 2 - tile_height * (height-1)),
            (px - self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF - tile_height * (height-1)),
        ]
        
        left_side = [top_points[3], top_points[2], (top_points[2][0], top_points[2][1] + tile_height), (top_points[3][0], top_points[3][1] + tile_height)]
        right_side = [top_points[2], top_points[1], (top_points[1][0], top_points[1][1] + tile_height), (top_points[2][0], top_points[2][1] + tile_height)]

        pygame.gfxdraw.filled_polygon(self.screen, left_side, color_side)
        pygame.gfxdraw.filled_polygon(self.screen, right_side, color_side)
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color_top)
        pygame.gfxdraw.aapolygon(self.screen, top_points, color_top)
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # Draw walls, target, source
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 1:
                    self._render_iso_cube(x, y, self.COLOR_WALL_TOP, self.COLOR_WALL)

        # Draw Target
        tx, ty = self.target_pos
        t_px, t_py = self._iso_to_screen(tx, ty)
        t_py += self.TILE_HEIGHT_HALF
        pygame.gfxdraw.filled_circle(self.screen, t_px, t_py, 12, self.COLOR_TARGET_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, t_px, t_py, 8, self.COLOR_TARGET)
        pygame.gfxdraw.aacircle(self.screen, t_px, t_py, 8, self.COLOR_TARGET)

        # Draw Source
        sx, sy = self.laser_source[:2]
        s_px, s_py = self._iso_to_screen(sx, sy)
        s_py += self.TILE_HEIGHT_HALF
        pygame.gfxdraw.filled_circle(self.screen, s_px, s_py, 12, self.COLOR_SOURCE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, s_px, s_py, 8, self.COLOR_SOURCE)
        pygame.gfxdraw.aacircle(self.screen, s_px, s_py, 8, self.COLOR_SOURCE)

        # Draw placed crystals
        for crystal in self.placed_crystals:
            cx, cy = crystal['pos']
            c_px, c_py = self._iso_to_screen(cx, cy)
            c_py += self.TILE_HEIGHT_HALF
            color, glow_color = self.CRYSTAL_COLORS[crystal['type']]
            
            points = [
                (c_px, c_py - 8), (c_px + 8, c_py),
                (c_px, c_py + 8), (c_px - 8, c_py)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, glow_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Draw laser path
        laser_brightness = 0.5 + 0.5 * (max(0, self.energy) / self.INITIAL_ENERGY)
        glow_color = (self.COLOR_LASER_GLOW[0], self.COLOR_LASER_GLOW[1], self.COLOR_LASER_GLOW[2], int(self.COLOR_LASER_GLOW[3] * laser_brightness))

        for start_grid, end_grid, energized in self.laser_path:
            start_px, start_py = self._iso_to_screen(start_grid[0], start_grid[1])
            end_px, end_py = self._iso_to_screen(end_grid[0], end_grid[1])
            center_offset = self.TILE_HEIGHT_HALF
            p1 = (start_px, start_py + center_offset)
            p2 = (end_px, end_py + center_offset)
            pygame.draw.line(self.screen, glow_color, p1, p2, 5)
            pygame.draw.aaline(self.screen, self.COLOR_LASER, p1, p2)
            if energized:
                pygame.draw.line(self.screen, (255,255,255,150), p1, p2, 2)

        # Draw cursor
        cur_x, cur_y = self.cursor_pos
        cur_px, cur_py = self._iso_to_screen(cur_x, cur_y)
        points = [
            (cur_px, cur_py),
            (cur_px + self.TILE_WIDTH_HALF, cur_py + self.TILE_HEIGHT_HALF),
            (cur_px, cur_py + self.TILE_HEIGHT_HALF * 2),
            (cur_px - self.TILE_WIDTH_HALF, cur_py + self.TILE_HEIGHT_HALF),
        ]
        pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, points, 1)

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)

        # Time
        time_text = self.font_small.render(f"TIME: {self.time_left / 100:.1f}s", True, self.COLOR_TEXT)
        ui_surf.blit(time_text, (15, 15))

        # Energy
        energy_text = self.font_small.render(f"ENERGY: {self.energy}", True, self.COLOR_TEXT)
        ui_surf.blit(energy_text, (150, 15))

        # Selected Crystal
        sel_text = self.font_small.render("SELECTED:", True, self.COLOR_TEXT)
        ui_surf.blit(sel_text, (self.SCREEN_WIDTH - 220, 15))

        c_px, c_py = self.SCREEN_WIDTH - 120, 25
        color, _ = self.CRYSTAL_COLORS[self.selected_crystal_type]
        points = [
                (c_px, c_py - 8), (c_px + 8, c_py),
                (c_px, c_py + 8), (c_px - 8, c_py)
            ]
        pygame.gfxdraw.filled_polygon(ui_surf, points, color)
        pygame.gfxdraw.aapolygon(ui_surf, points, color)
        
        name_text = self.font_small.render(self.CRYSTAL_NAMES[self.selected_crystal_type], True, self.COLOR_TEXT)
        ui_surf.blit(name_text, (self.SCREEN_WIDTH - 95, 15))

        self.screen.blit(ui_surf, (0, 0))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            if self.game_won:
                msg = "TARGET HIT"
                color = self.COLOR_TARGET
            else:
                msg = "FAILED"
                color = self.COLOR_LASER
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "energy": self.energy,
            "game_won": self.game_won,
        }
    
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Cavern Laser")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_q:
                    running = False
                if terminated: continue

                if event.key == pygame.K_SPACE:
                    space = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
        
        # In manual play, an action is only sent if a key is pressed
        if movement or space or shift or not GameEnv.auto_advance:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit human play speed

    pygame.quit()