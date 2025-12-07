import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:30:07.460862
# Source Brief: brief_02897.md
# Brief Index: 2897
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player positions a reflector to guide a beam
    from an emitter to a target, avoiding obstacles. The goal is to achieve 10
    successful reflections in an episode.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Position a reflector to guide a beam from an emitter to a target, avoiding obstacles along the way."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the reflector. Press space to fire the beam and test your alignment."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (20, 40, 70)
    COLOR_REFLECTOR = (255, 255, 255)
    COLOR_REFLECTOR_GLOW = (200, 200, 255)
    COLOR_EMITTER = (0, 255, 255)
    COLOR_EMITTER_GLOW = (0, 150, 150)
    COLOR_TARGET = (255, 255, 0)
    COLOR_TARGET_GLOW = (200, 200, 0)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (180, 40, 40)
    COLOR_BEAM_SUCCESS = (0, 255, 255)
    COLOR_BEAM_FAIL = (255, 100, 100)
    COLOR_SHOCKWAVE = (255, 150, 0)
    COLOR_TEXT = (220, 220, 240)

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
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Persistent state across episodes
        self.initial_obstacle_count = 2
        self.successful_episodes_in_a_row = 0
        
        # Initialize state variables (will be properly set in reset)
        self.reflector_pos = np.array([0, 0])
        self.emitter_pos = np.array([0, 0])
        self.target_pos = np.array([0, 0])
        self.obstacles = []
        self.particles = []
        self.shockwaves = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.beam_path_cache = []
        self.beam_hit_type = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.particles.clear()
        self.shockwaves.clear()
        self.beam_path_cache = []
        self.beam_hit_type = None

        # Place game elements
        occupied_cells = set()

        self.emitter_pos = np.array([3, self.GRID_HEIGHT // 2])
        occupied_cells.add(tuple(self.emitter_pos))

        self.reflector_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        if tuple(self.reflector_pos) in occupied_cells:
             self.reflector_pos[0] += 1
        
        # Place obstacles
        self.obstacles = []
        for _ in range(self.initial_obstacle_count):
            pos = self._get_random_empty_cell(occupied_cells)
            self.obstacles.append(pos)
            occupied_cells.add(tuple(pos))
        
        # Place target
        self.target_pos = self._get_random_empty_cell(occupied_cells)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        self.game_over = False
        reward = 0
        self.beam_path_cache = []
        self.beam_hit_type = None

        # 1. Handle Movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = self.reflector_pos + np.array([dx, dy])
            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                if tuple(new_pos) != tuple(self.emitter_pos) and tuple(new_pos) not in [tuple(o) for o in self.obstacles]:
                     self.reflector_pos = new_pos

        # 2. Handle Firing the Beam
        if space_pressed:
            # sfx_beam_fire()
            
            # Trace from emitter to reflector
            path1, hit1_type, hit1_pos = self._trace_line(self.emitter_pos, self.reflector_pos)
            self.beam_path_cache.extend(path1)

            if hit1_type == 'target' and np.array_equal(hit1_pos, self.reflector_pos): # Direct hit on reflector
                dir_in = self.reflector_pos - self.emitter_pos
                # Simple 45-degree reflection logic, assuming a '/' mirror
                dir_out = np.array([dir_in[1], -dir_in[0]]) if dir_in[0] * dir_in[1] >= 0 else np.array([-dir_in[1], dir_in[0]])
                
                # Trace from reflector outwards
                path2, hit2_type, _ = self._trace_line(self.reflector_pos, self.reflector_pos + dir_out * max(self.GRID_WIDTH, self.GRID_HEIGHT))
                self.beam_path_cache.extend(path2)

                if hit2_type == 'target':
                    # SUCCESS
                    # sfx_success_chime()
                    self.score += 1
                    reward = 1.1 # 0.1 for reflection, 1.0 for success
                    self.beam_hit_type = 'success'
                    self._create_particles(self._grid_to_pixel(self.target_pos), self.COLOR_TARGET)

                    if self.score >= 10:
                        # VICTORY
                        # sfx_victory_fanfare()
                        reward += 100
                        self.game_over = True
                    else:
                        self._respawn_target_and_add_obstacle()
                else:
                    # FAIL (hit obstacle or edge after reflection)
                    # sfx_beam_fail_explode()
                    reward = -10
                    self.game_over = True
                    self.beam_hit_type = 'fail'
                    if hit2_type == 'obstacle':
                        self._create_shockwave(self._grid_to_pixel(self.beam_path_cache[-1]), self.COLOR_SHOCKWAVE)
            else:
                # FAIL (beam from emitter blocked before reflector)
                # sfx_beam_fail_explode()
                reward = -10
                self.game_over = True
                self.beam_hit_type = 'fail'
                if hit1_type == 'obstacle':
                    self._create_shockwave(self._grid_to_pixel(hit1_pos), self.COLOR_SHOCKWAVE)

        self.steps += 1
        if self.steps >= 1000:
            self.game_over = True
        
        if self.game_over:
            if self.score >= 10:
                self.successful_episodes_in_a_row += 1
                if self.successful_episodes_in_a_row > 0 and self.successful_episodes_in_a_row % 2 == 0:
                    self.initial_obstacle_count = min(15, self.initial_obstacle_count + 1)
            else:
                self.successful_episodes_in_a_row = 0

        terminated = self.game_over
        truncated = False # This environment does not truncate based on time limits, only terminates
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_effects()
        self._render_grid()
        self._render_game_objects()
        self._render_beam()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "initial_obstacles": self.initial_obstacle_count}

    def _get_random_empty_cell(self, occupied_cells):
        while True:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if pos not in occupied_cells:
                return np.array(pos)

    def _respawn_target_and_add_obstacle(self):
        occupied = {tuple(o) for o in self.obstacles}
        occupied.add(tuple(self.emitter_pos))
        occupied.add(tuple(self.reflector_pos))
        
        self.target_pos = self._get_random_empty_cell(occupied)
        occupied.add(tuple(self.target_pos))

        if len(self.obstacles) < (self.GRID_WIDTH * self.GRID_HEIGHT) / 4: # Cap obstacles
            new_obstacle_pos = self._get_random_empty_cell(occupied)
            self.obstacles.append(new_obstacle_pos)

    def _trace_line(self, start_pos, end_pos):
        path = []
        x1, y1 = start_pos.astype(int)
        x2, y2 = end_pos.astype(int)
        dx, dy = abs(x2 - x1), -abs(y2 - y1)
        sx, sy = 1 if x1 < x2 else -1, 1 if y1 < y2 else -1
        err = dx + dy
        
        x, y = x1, y1
        obstacle_set = {tuple(o) for o in self.obstacles}
        while True:
            if not (x == x1 and y == y1):
                pos = np.array([x, y])
                path.append(pos)
                if np.array_equal(pos, self.target_pos):
                    return path, 'target', pos
                if tuple(pos) in obstacle_set:
                    return path, 'obstacle', pos
                if np.array_equal(pos, self.reflector_pos):
                    return path, 'target', pos
            
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            if e2 >= dy:
                if x == x2: break
                err += dy
                x += sx
            if e2 <= dx:
                if y == y2: break
                err += dx
                y += sy
        
        return path, 'edge', end_pos

    def _grid_to_pixel(self, grid_pos):
        return (int(grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2), 
                int(grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2))

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_glowing_circle(self, pos, color, glow_color, radius):
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 1.8), (*glow_color, 60))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 1.3), (*glow_color, 120))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_game_objects(self):
        self._draw_glowing_circle(self._grid_to_pixel(self.emitter_pos), self.COLOR_EMITTER, self.COLOR_EMITTER_GLOW, self.GRID_SIZE // 3)
        self._draw_glowing_circle(self._grid_to_pixel(self.target_pos), self.COLOR_TARGET, self.COLOR_TARGET_GLOW, self.GRID_SIZE // 2)
        for obs_pos in self.obstacles:
            px, py = self._grid_to_pixel(obs_pos)
            size = self.GRID_SIZE // 2
            rect = pygame.Rect(px - size, py - size, size * 2, size * 2)
            glow_rect = rect.inflate(size, size)
            pygame.draw.rect(self.screen, (*self.COLOR_OBSTACLE_GLOW, 100), glow_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
        px, py = self._grid_to_pixel(self.reflector_pos)
        size = self.GRID_SIZE * 0.7
        p1 = (int(px - size // 2), int(py + size // 2))
        p2 = (int(px + size // 2), int(py - size // 2))
        pygame.draw.aaline(self.screen, (*self.COLOR_REFLECTOR_GLOW, 150), p1, p2, 6)
        pygame.draw.aaline(self.screen, self.COLOR_REFLECTOR, p1, p2, 2)
    
    def _render_beam(self):
        if not self.beam_path_cache:
            return
        color = self.COLOR_BEAM_SUCCESS if self.beam_hit_type == 'success' else self.COLOR_BEAM_FAIL
        points = [self._grid_to_pixel(self.emitter_pos)] + [self._grid_to_pixel(p) for p in self.beam_path_cache]
        if len(points) > 1:
            pygame.draw.lines(self.screen, (*color, 80), False, points, 10)
            pygame.draw.lines(self.screen, (*color, 150), False, points, 5)
            pygame.draw.lines(self.screen, (255, 255, 255), False, points, 2)

    def _render_ui(self):
        score_text = self.font.render(f"Reflections: {self.score} / 10", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.small_font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        obs_text = self.small_font.render(f"Obstacles: {self.initial_obstacle_count}", True, self.COLOR_TEXT)
        self.screen.blit(obs_text, (self.SCREEN_WIDTH - obs_text.get_width() - 10, 35))

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': 1.0, 'color': color})

    def _create_shockwave(self, pos, color):
        self.shockwaves.append({'pos': pos, 'radius': 0, 'max_radius': 80, 'life': 1.0, 'color': color})

    def _update_and_draw_effects(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.04
            if p['life'] <= 0: self.particles.remove(p)
            else:
                alpha = max(0, int(p['life'] * 255))
                pygame.draw.circle(self.screen, (*p['color'], alpha), [int(c) for c in p['pos']], 2)
        for sw in self.shockwaves[:]:
            sw['life'] -= 0.05
            if sw['life'] <= 0: self.shockwaves.remove(sw)
            else:
                current_radius = int(sw['max_radius'] * (1.0 - sw['life']))
                alpha = max(0, int(sw['life'] * 150))
                pos_i = (int(sw['pos'][0]), int(sw['pos'][1]))
                pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], current_radius, (*sw['color'], alpha))
                if current_radius > 2:
                    pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], current_radius - 2, (*sw['color'], alpha))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and debugging.
    # It is not part of the required Gymnasium interface.
    # Set the SDL_VIDEODRIVER to a real driver to see the window.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    key_to_action = {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}
    
    pygame.display.set_caption("Beam Reflector Gym Environment")
    window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while running:
        movement_action = 0
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    terminated = False
                    obs, info = env.reset()
                if event.key == pygame.K_q: running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            # Handle one-off key presses for movement and firing
            # This is more suitable for a turn-based game like this
            action_taken = False
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key in key_to_action:
                    movement_action = key_to_action[event.key]
                    action_taken = True
                elif event.key == pygame.K_SPACE:
                    space_action = 1
                    action_taken = True
            
            # If any action was taken, step the environment
            if action_taken:
                action = [movement_action, space_action, 0]
                obs, reward, terminated, truncated, info = env.step(action)
                
                if space_action == 1:
                    print(f"Fired! Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # Always render the current state
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        window.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata["render_fps"])

    env.close()