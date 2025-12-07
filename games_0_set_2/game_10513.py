import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:32:35.837646
# Source Brief: brief_00513.md
# Brief Index: 513
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    Chrono-Corp Data Center Environment

    In this environment, the agent acts as a network administrator for Chrono-Corp.
    The goal is to manage data flow across a grid of network sectors to prevent
    catastrophic overloads. The agent can select different data streams and
    manipulate their speed. The game is lost if all sectors crash, and won if
    the system remains stable for 1000 time units.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) - Up/Down cycles stream selection.
    - action[1]: Space button (0=released, 1=held) - Speeds up the selected stream.
    - action[2]: Shift button (0=released, 1=held) - Toggles pause.

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage data flow across a network grid to prevent sector overloads. "
        "Select data streams and manipulate their speed to keep the system stable."
    )
    user_guide = (
        "Use ↑/↓ arrow keys to cycle through data streams. Hold space to speed up the selected stream. "
        "Press shift to pause the simulation."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_overlay = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.GRID_COLS, self.GRID_ROWS = 10, 6
        self.SECTOR_MAX_LOAD = 100
        self.NEW_STREAM_INTERVAL = 50
        self.SPEED_UP_MULTIPLIER = 3.0

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_GRID = (30, 45, 75)
        self.COLOR_TEXT = (220, 230, 255)
        self.COLOR_TIME_MANIP = (0, 192, 255)
        self.COLOR_SELECTION = (255, 255, 255)
        self.SECTOR_COLORS = {
            'normal': (20, 80, 50),
            'warn': (120, 110, 30),
            'danger': (140, 40, 40),
            'crashed': (40, 40, 40)
        }
        self.STREAM_COLORS = [
            (100, 255, 100), (255, 100, 100), (100, 100, 255),
            (255, 255, 100), (100, 255, 255), (255, 100, 255)
        ]

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.is_paused = False
        self.sectors = []
        self.streams = []
        self.particles = []
        self.selected_stream_index = 0
        self.prev_movement = 0
        self.prev_shift_held = False
        self.predefined_paths = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.predefined_paths:
            self.predefined_paths = self._generate_paths()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.is_paused = False
        self.particles.clear()

        self.prev_movement = 0
        self.prev_shift_held = False

        self._init_sectors()
        self._init_streams()
        self.selected_stream_index = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)

        newly_crashed_sectors = 0
        reward = 0
        if not self.is_paused:
            newly_crashed_sectors = self._update_game_state()
            reward = self._calculate_reward(newly_crashed_sectors)
            self.score += reward

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.steps >= self.MAX_STEPS:
                win_bonus = 50
                self.score += win_bonus
                reward += win_bonus

        truncated = False # This environment does not truncate based on time limits
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if shift_held and not self.prev_shift_held:
            self.is_paused = not self.is_paused

        if not self.is_paused and len(self.streams) > 0:
            up_pressed = (movement == 1 and self.prev_movement != 1)
            down_pressed = (movement == 2 and self.prev_movement != 2)

            if up_pressed:
                self.selected_stream_index = (self.selected_stream_index - 1 + len(self.streams)) % len(self.streams)
            if down_pressed:
                self.selected_stream_index = (self.selected_stream_index + 1) % len(self.streams)

            selected_stream = self.streams[self.selected_stream_index]
            if space_held:
                selected_stream['speed'] = selected_stream['base_speed'] * self.SPEED_UP_MULTIPLIER
                if self.np_random.random() < 0.6:
                    self._spawn_particles(1, selected_stream['pixel_pos'], self.COLOR_TIME_MANIP, 2, 20)
            else:
                selected_stream['speed'] = selected_stream['base_speed']

        self.prev_movement = movement
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        self.steps += 1
        self._update_streams()
        self._update_sector_loads()
        newly_crashed_this_step = self._check_sector_crashes()
        self._spawn_new_streams()
        self._update_particles()
        return newly_crashed_this_step

    def _calculate_reward(self, newly_crashed_sectors):
        reward = 1  # Base reward for surviving a step
        reward -= 5 * newly_crashed_sectors
        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self.sectors and all(s['crashed'] for s in self.sectors):
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_sectors()
        self._render_stream_paths()
        self._render_streams()
        self._render_particles()
        self._render_ui()

        if self.is_paused: self._render_overlay("PAUSED")
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                self._render_overlay("SYSTEM STABLE", self.STREAM_COLORS[0])
            else:
                self._render_overlay("SYSTEM CRASH", self.SECTOR_COLORS['danger'])

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Initialization Methods ---
    def _init_sectors(self):
        self.sectors.clear()
        self.sector_width = self.WIDTH // self.GRID_COLS
        self.sector_height = self.HEIGHT // self.GRID_ROWS
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(c * self.sector_width, r * self.sector_height, self.sector_width, self.sector_height)
                self.sectors.append({
                    'rect': rect,
                    'load': 0,
                    'max_load': self.SECTOR_MAX_LOAD,
                    'crashed': False,
                    'grid_pos': (c, r),
                    'center_pos': rect.center
                })

    def _init_streams(self):
        self.streams.clear()
        self._add_stream()
        self._add_stream()

    def _add_stream(self):
        path = self.predefined_paths[self.np_random.integers(len(self.predefined_paths))]
        path_pixels = [self.sectors[c + r * self.GRID_COLS]['center_pos'] for c, r in path]
        base_speed = self.np_random.uniform(0.015, 0.03)
        self.streams.append({
            'path': path,
            'path_pixels': path_pixels,
            'current_pos_on_path': 0.0,
            'speed': base_speed,
            'base_speed': base_speed,
            'load_value': self.np_random.integers(20, 41),
            'color': self.STREAM_COLORS[self.np_random.integers(len(self.STREAM_COLORS))],
            'pixel_pos': path_pixels[0],
            'current_segment': 0
        })

    def _generate_paths(self):
        paths = []
        for _ in range(20):
            path = []
            start_row = self.np_random.integers(self.GRID_ROWS)
            path.append((0, start_row))
            current_pos = [0, start_row]
            while current_pos[0] < self.GRID_COLS - 1:
                move_dir = self.np_random.choice(['R', 'U', 'D'], p=[0.6, 0.2, 0.2])
                if move_dir == 'R': current_pos[0] += 1
                if move_dir == 'U': current_pos[1] = max(0, current_pos[1] - 1)
                if move_dir == 'D': current_pos[1] = min(self.GRID_ROWS - 1, current_pos[1] + 1)
                if tuple(current_pos) != path[-1]:
                    path.append(tuple(current_pos))
            paths.append(path)
        return paths

    # --- Update Methods ---
    def _update_streams(self):
        for stream in self.streams:
            stream['current_pos_on_path'] += stream['speed']
            if stream['current_pos_on_path'] >= len(stream['path']) - 1:
                stream['current_pos_on_path'] = 0 # Loop stream

            segment_idx = int(stream['current_pos_on_path'])
            progress = stream['current_pos_on_path'] - segment_idx
            
            start_node = stream['path_pixels'][segment_idx]
            end_node = stream['path_pixels'][segment_idx + 1]

            stream['pixel_pos'] = (
                int(start_node[0] + (end_node[0] - start_node[0]) * progress),
                int(start_node[1] + (end_node[1] - start_node[1]) * progress)
            )
            stream['current_segment'] = segment_idx

    def _update_sector_loads(self):
        for sector in self.sectors:
            sector['load'] = 0
        for stream in self.streams:
            c, r = stream['path'][stream['current_segment']]
            sector_idx = c + r * self.GRID_COLS
            if not self.sectors[sector_idx]['crashed']:
                self.sectors[sector_idx]['load'] += stream['load_value']

    def _check_sector_crashes(self):
        newly_crashed = 0
        for sector in self.sectors:
            if not sector['crashed'] and sector['load'] > sector['max_load']:
                sector['crashed'] = True
                newly_crashed += 1
                self._spawn_particles(30, sector['center_pos'], self.SECTOR_COLORS['danger'], 4, 40, 1.5)
        return newly_crashed

    def _spawn_new_streams(self):
        if self.steps > 0 and self.steps % self.NEW_STREAM_INTERVAL == 0:
            self._add_stream()

    def _spawn_particles(self, count, pos, color, max_radius, max_life, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.0) * speed_mult
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(max_life // 2, max_life),
                'max_life': max_life,
                'color': color,
                'radius': self.np_random.integers(1, max_radius)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    # --- Rendering Methods ---
    def _render_grid(self):
        for c in range(1, self.GRID_COLS):
            x = c * self.sector_width
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for r in range(1, self.GRID_ROWS):
            y = r * self.sector_height
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_sectors(self):
        for sector in self.sectors:
            if sector['crashed']:
                color = self.SECTOR_COLORS['crashed']
            else:
                load_ratio = min(1, sector['load'] / sector['max_load'])
                if load_ratio < 0.6:
                    color = self.SECTOR_COLORS['normal']
                elif load_ratio < 0.9:
                    color = self.SECTOR_COLORS['warn']
                else:
                    color = self.SECTOR_COLORS['danger']
            
            if color == self.SECTOR_COLORS['danger'] and (self.steps // 3) % 2 == 0:
                color = tuple(min(255, c + 40) for c in color)
            if sector['crashed'] and (self.steps // 5) % 2 == 0:
                 color = tuple(min(255, c + 15) for c in color)
            
            pygame.draw.rect(self.screen, color, sector['rect'].inflate(-4, -4))

    def _render_stream_paths(self):
        if not self.streams:
            return
        for i, stream in enumerate(self.streams):
            is_selected = (i == self.selected_stream_index and not self.is_paused)
            color = self.COLOR_SELECTION if is_selected else self.COLOR_GRID
            width = 2 if is_selected else 1
            pygame.draw.aalines(self.screen, color, False, stream['path_pixels'], width)

    def _render_streams(self):
        for stream in self.streams:
            thickness = int(4 + (stream['load_value'] / 40) * 8)
            pos = (int(stream['pixel_pos'][0]), int(stream['pixel_pos'][1]))
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], thickness + 3, (*stream['color'], 60))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], thickness + 3, (*stream['color'], 60))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], thickness, stream['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], thickness, stream['color'])

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(p['radius'] * life_ratio)
            if radius > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                alpha = int(150 * life_ratio)
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
    
    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_text = self.font_ui.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

    def _render_overlay(self, text, color=None):
        if color is None:
            color = self.COLOR_TEXT
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text_surf = self.font_overlay.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    
    # Un-dummy the video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Chrono-Corp Data Center")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Hold state for actions
    _action = [0, 0, 0]
    
    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            
            # Handle single-press actions (like pause)
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT):
                 _action[2] = 1
            if event.type == pygame.KEYUP and (event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT):
                 _action[2] = 0

        # Handle held-down actions
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            _action[0] = 1 # up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            _action[0] = 2 # down
        else:
            _action[0] = 0
            
        _action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        obs, reward, terminated, truncated, info = env.step(_action)
        total_reward += reward
        
        # Reset single-press part of action after step
        _action[2] = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}. Score: {info['score']}. Steps: {info['steps']}")
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()