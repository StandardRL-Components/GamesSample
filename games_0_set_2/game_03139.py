
# Generated: 2025-08-28T07:06:35.418637
# Source Brief: brief_03139.md
# Brief Index: 3139

        
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
        "Controls: Arrows to move cursor. Shift to cycle crystal type. Space to place crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Redirect lasers to their targets by placing reflective crystals on the grid before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 20, 12
    CELL_SIZE = 30
    GRID_MARGIN_X = (WIDTH - GRID_COLS * CELL_SIZE) // 2
    GRID_MARGIN_Y = (HEIGHT - GRID_ROWS * CELL_SIZE) // 2
    
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 50, 80)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_UI_TEXT = (220, 220, 240)
    
    LASER_COLORS = [
        (255, 0, 128),   # Magenta
        (0, 255, 255),   # Cyan
        (255, 128, 0),   # Orange
        (0, 255, 0),     # Green
        (255, 0, 255),   # Fuchsia
    ]
    
    NUM_LASERS = 4
    MAX_STEPS = 600
    INITIAL_CRYSTALS = 25
    MAX_LASER_LENGTH = GRID_COLS * GRID_ROWS

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
        
        try:
            self.font_main = pygame.font.Font(None, 24)
            self.font_title = pygame.font.Font(None, 36)
        except IOError:
            self.font_main = pygame.font.SysFont("sans-serif", 24)
            self.font_title = pygame.font.SysFont("sans-serif", 36)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        
        self.grid = None
        self.lasers = None
        self.targets = None
        self.laser_paths = None
        
        self.cursor_pos = [0, 0]
        self.selected_crystal_type = 0
        self.remaining_crystals = 0
        self.num_targets_hit = 0
        
        self.crystal_types = {
            1: {"symbol": "/", "color": (200, 200, 255)},
            2: {"symbol": "\\", "color": (255, 200, 200)}
        }
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_crystal_type = 1
        self.remaining_crystals = self.INITIAL_CRYSTALS
        
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        
        self._generate_puzzle()
        self._calculate_all_laser_paths()
        self.num_targets_hit = self._count_hits()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        self.lasers = []
        self.targets = []
        occupied_cells = set()

        for i in range(self.NUM_LASERS):
            color = self.LASER_COLORS[i % len(self.LASER_COLORS)]
            
            while True:
                edge = self.np_random.integers(4)
                if edge == 0:
                    pos = (self.np_random.integers(self.GRID_COLS), -1)
                    direction = (0, 1)
                elif edge == 1:
                    pos = (self.np_random.integers(self.GRID_COLS), self.GRID_ROWS)
                    direction = (0, -1)
                elif edge == 2:
                    pos = (-1, self.np_random.integers(self.GRID_ROWS))
                    direction = (1, 0)
                else:
                    pos = (self.GRID_COLS, self.np_random.integers(self.GRID_ROWS))
                    direction = (-1, 0)
                
                if pos not in [l['pos'] for l in self.lasers]:
                    self.lasers.append({"pos": pos, "dir": direction, "color": color})
                    break

            while True:
                target_pos = (
                    self.np_random.integers(1, self.GRID_COLS - 1),
                    self.np_random.integers(1, self.GRID_ROWS - 1)
                )
                if target_pos not in occupied_cells:
                    self.targets.append({"pos": target_pos, "color": color, "hit": False})
                    occupied_cells.add(target_pos)
                    break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        crystal_placed_this_step = False
        targets_hit_before = self._count_hits()

        if shift_pressed:
            self.selected_crystal_type = (self.selected_crystal_type % len(self.crystal_types)) + 1
            # sfx: ui_chime.wav
        elif space_pressed:
            cx, cy = self.cursor_pos
            if self.grid[cx, cy] == 0 and self.remaining_crystals > 0:
                self.grid[cx, cy] = self.selected_crystal_type
                self.remaining_crystals -= 1
                crystal_placed_this_step = True
                # sfx: crystal_place.wav
        elif movement > 0:
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_ROWS - 1)
            # sfx: cursor_move.wav
            
        if crystal_placed_this_step:
            self._calculate_all_laser_paths()
            targets_hit_after = self._count_hits()
            reward -= 0.01
            newly_hit_count = targets_hit_after - targets_hit_before
            if newly_hit_count > 0:
                reward += newly_hit_count * 5.0
                # sfx: target_hit_new.wav
        
        self.num_targets_hit = self._count_hits()
        reward += self.num_targets_hit * 0.1
        
        self.steps += 1
        self.score += reward
        
        terminated = False
        win = self.num_targets_hit == self.NUM_LASERS
        
        if win:
            terminated = True
            self.game_over = True
            reward += 100
            # sfx: level_win.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward -= 100
            # sfx: time_out.wav
        elif self.remaining_crystals <= 0 and not crystal_placed_this_step:
            if self._count_hits() < self.NUM_LASERS:
                terminated = True
                self.game_over = True
                reward -= 100
                # sfx: lose_crystals.wav

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_all_laser_paths(self):
        self.laser_paths = []
        for laser in self.lasers:
            self.laser_paths.append(self._trace_laser_path(laser))
        
        for target in self.targets:
            target['hit'] = False
        for path in self.laser_paths:
            if path['hit_target_pos']:
                for target in self.targets:
                    if target['pos'] == path['hit_target_pos'] and target['color'] == path['color']:
                        target['hit'] = True
                        # sfx: target_hit_sustained.wav

    def _trace_laser_path(self, laser):
        path_points = []
        pos = np.array(laser['pos'])
        direction = np.array(laser['dir'])
        
        path_points.append(self._grid_to_pixel(pos + direction))

        for _ in range(self.MAX_LASER_LENGTH):
            pos += direction
            
            if not (0 <= pos[0] < self.GRID_COLS and 0 <= pos[1] < self.GRID_ROWS):
                break

            crystal_type = self.grid[pos[0], pos[1]]
            if crystal_type > 0:
                path_points.append(self._grid_to_pixel(pos))
                if crystal_type == 1: # Mirror /
                    direction = np.array([-direction[1], -direction[0]])
                elif crystal_type == 2: # Mirror \
                    direction = np.array([direction[1], direction[0]])
                path_points.append(self._grid_to_pixel(pos))
            
            for target in self.targets:
                if tuple(pos) == target['pos'] and target['color'] == laser['color']:
                    path_points.append(self._grid_to_pixel(pos))
                    return {"points": path_points, "color": laser['color'], "hit_target_pos": tuple(pos)}
        
        path_points.append(self._grid_to_pixel(pos))
        return {"points": path_points, "color": laser['color'], "hit_target_pos": None}
    
    def _count_hits(self):
        return sum(1 for t in self.targets if t['hit'])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_MARGIN_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_MARGIN_Y), (px, self.GRID_MARGIN_Y + self.GRID_ROWS * self.CELL_SIZE))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_MARGIN_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN_X, py), (self.GRID_MARGIN_X + self.GRID_COLS * self.CELL_SIZE, py))

        for target in self.targets:
            px, py = self._grid_to_pixel(target['pos'])
            radius = self.CELL_SIZE // 3
            color = target['color']
            if target['hit']:
                pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
                pygame.gfxdraw.aacircle(self.screen, px, py, radius, (255,255,255))
                pulse_radius = radius + int(abs(math.sin(self.steps * 0.2)) * 5)
                pulse_alpha = 100 - int(abs(math.sin(self.steps * 0.2)) * 100)
                try:
                    pygame.gfxdraw.aacircle(self.screen, px, py, pulse_radius, (*color, pulse_alpha))
                except TypeError: # Some pygame versions have issues with this
                    pass
            else:
                pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)
                pygame.gfxdraw.aacircle(self.screen, px, py, radius-1, color)

        for laser in self.lasers:
            px, py = self._grid_to_pixel(laser['pos'])
            pygame.draw.rect(self.screen, laser['color'], (px - 5, py - 5, 10, 10))
            pygame.draw.rect(self.screen, (255,255,255), (px - 5, py - 5, 10, 10), 1)

        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if self.grid[x, y] > 0:
                    px, py = self._grid_to_pixel((x,y))
                    details = self.crystal_types[self.grid[x, y]]
                    half_cell = self.CELL_SIZE // 2
                    if details['symbol'] == '/':
                        pygame.draw.line(self.screen, details['color'], (px - half_cell + 4, py + half_cell - 4), (px + half_cell - 4, py - half_cell + 4), 3)
                    elif details['symbol'] == '\\':
                        pygame.draw.line(self.screen, details['color'], (px - half_cell + 4, py - half_cell + 4), (px + half_cell - 4, py + half_cell - 4), 3)

        for path in self.laser_paths:
            if len(path['points']) > 1:
                color = path['color']
                pulse = 2 + abs(math.sin(self.steps * 0.3)) * 3
                glow_color = (*color, 50)
                pygame.draw.lines(self.screen, glow_color, False, path['points'], int(pulse*2.5))
                pygame.draw.lines(self.screen, color, False, path['points'], int(pulse))
                pygame.draw.lines(self.screen, (255,255,255), False, path['points'], 1)

        cx, cy = self.cursor_pos
        px = self.GRID_MARGIN_X + cx * self.CELL_SIZE
        py = self.GRID_MARGIN_Y + cy * self.CELL_SIZE
        cursor_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, (px, py))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR[:3], cursor_rect, 2)
        
        if self.grid[cx, cy] == 0:
            details = self.crystal_types[self.selected_crystal_type]
            half_cell = self.CELL_SIZE // 2
            center_x, center_y = cursor_rect.center
            if details['symbol'] == '/':
                pygame.draw.line(self.screen, details['color'], (center_x - half_cell + 8, center_y + half_cell - 8), (center_x + half_cell - 8, center_y - half_cell + 8), 2)
            elif details['symbol'] == '\\':
                pygame.draw.line(self.screen, details['color'], (center_x - half_cell + 8, center_y - half_cell + 8), (center_x + half_cell - 8, center_y + half_cell - 8), 2)

    def _render_ui(self):
        time_ratio = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        time_bar_width = self.WIDTH * time_ratio
        pygame.draw.rect(self.screen, (0, 128, 255), (0, 0, time_bar_width, 8))
        
        score_text = self.font_main.render(f"Score: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 15))

        crystal_text = self.font_main.render(f"Crystals: {self.remaining_crystals}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (self.WIDTH - crystal_text.get_width() - 10, 15))

        target_text = self.font_main.render(f"Targets: {self.num_targets_hit} / {self.NUM_LASERS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_text, (self.WIDTH//2 - target_text.get_width()//2, 15))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "PUZZLE SOLVED" if self._count_hits() == self.NUM_LASERS else "GAME OVER"
            color = (100, 255, 150) if self._count_hits() == self.NUM_LASERS else (255, 100, 100)
            end_text = self.font_title.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - end_text.get_height()//2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_crystals": self.remaining_crystals,
            "targets_hit": self.num_targets_hit,
        }
    
    def _grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        px = self.GRID_MARGIN_X + int((x + 0.5) * self.CELL_SIZE)
        py = self.GRID_MARGIN_Y + int((y + 0.5) * self.CELL_SIZE)
        return px, py

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Laser Grid")
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        action_to_take = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: running = False
                if event.key == pygame.K_r: 
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- NEW GAME ---")
                    action_to_take = [0,0,0] # No-op to redraw
                
                if not env.game_over:
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    elif event.key == pygame.K_SPACE: space = 1
                    elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
                    
                    if any([movement, space, shift]):
                        action_to_take = [movement, space, shift]

        if action_to_take:
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action_to_take}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

            if terminated:
                print("--- GAME OVER ---")
                
        env._get_observation()
        pygame.display.flip()
        env.clock.tick(30)

    env.close()