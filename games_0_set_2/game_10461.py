import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:27:05.150438
# Source Brief: brief_00461.md
# Brief Index: 461
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a real-time puzzle game.
    The player must guide a light beam to an exit by rotating mirrors.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Guide a laser beam to the exit by rotating an array of mirrors before time runs out."
    user_guide = "Controls: Use ↑→ to select the next mirror and ↓← for the previous. Press space to rotate clockwise and shift to rotate counter-clockwise."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 90

    MAZE_W, MAZE_H = 16, 10
    CELL_SIZE = 40

    COLOR_BG = (26, 26, 46)  # #1a1a2e
    COLOR_WALL = (60, 60, 80)  # #3c3c50
    COLOR_MIRROR = (204, 204, 204)  # #cccccc
    COLOR_MIRROR_SELECTED = (0, 255, 255)  # #00ffff
    COLOR_BEAM = (255, 255, 0)  # #ffff00
    COLOR_EXIT = (0, 255, 128)  # #00ff80
    COLOR_TEXT = (240, 240, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('monospace', 18, bold=True)
        self.font_game_over = pygame.font.SysFont('sans-serif', 50, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        self.level = 0
        
        # Game state variables are initialized in reset()
        self.mirrors = []
        self.beam_start_pos = (0, 0)
        self.beam_start_dir = (0, 0)
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.selected_mirror_idx = 0
        self.beam_path = []
        self.hit_mirrors_this_step = set()
        self.hit_mirrors_last_step = set()
        self.last_dist_to_exit = float('inf')
        self.timer = 0
        self.score = 0
        self.game_over = False
        self.level_complete = False
        self.empty_cells = []
        self.num_mirrors = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level += 1
        self.num_mirrors = 30 + (self.level - 1) * 2

        self.steps = 0
        self.score = 0
        self.timer = self.GAME_DURATION_SECONDS
        self.game_over = False
        self.level_complete = False

        self._prepare_grid()
        self._place_game_elements()

        self.selected_mirror_idx = 0
        self.hit_mirrors_last_step = set()
        
        self._update_beam_path()
        self.last_dist_to_exit = self._get_min_dist_to_exit()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()
            
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        action_taken = False

        # Handle discrete actions (presses, not holds)
        if movement in [1, 4]:  # Up/Right -> Next mirror
            self.selected_mirror_idx = (self.selected_mirror_idx + 1) % len(self.mirrors)
        elif movement in [2, 3]:  # Down/Left -> Previous mirror
            self.selected_mirror_idx = (self.selected_mirror_idx - 1 + len(self.mirrors)) % len(self.mirrors)

        if space_pressed:  # Rotate CW
            # SFX: Mirror rotate click
            self.mirrors[self.selected_mirror_idx]['angle'] = (self.mirrors[self.selected_mirror_idx]['angle'] + 45) % 360
            action_taken = True
        
        if shift_pressed:  # Rotate CCW
            # SFX: Mirror rotate click
            self.mirrors[self.selected_mirror_idx]['angle'] = (self.mirrors[self.selected_mirror_idx]['angle'] - 45 + 360) % 360
            action_taken = True
        
        self.steps += 1
        self.timer -= 1 / self.FPS
        
        if action_taken:
            self._update_beam_path()

        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _calculate_reward(self):
        reward = 0.0
        
        current_dist = self._get_min_dist_to_exit()
        dist_change = self.last_dist_to_exit - current_dist
        if dist_change > 1:
            reward += 0.1
        elif dist_change < -1:
            reward -= 0.01
        self.last_dist_to_exit = current_dist

        newly_hit = self.hit_mirrors_this_step - self.hit_mirrors_last_step
        reward += len(newly_hit) * 5.0
        self.hit_mirrors_last_step = self.hit_mirrors_this_step

        if self.level_complete:
            reward += 100.0
        elif self.timer <= 0:
            reward -= 50.0

        self.score += reward
        return reward

    def _check_termination(self):
        if self.level_complete:
            self.game_over = True
            # SFX: Level complete fanfare
        elif self.timer <= 0:
            self.timer = 0
            self.game_over = True
            # SFX: Timeout buzzer
        
        if self.steps >= self.GAME_DURATION_SECONDS * self.FPS:
            self.game_over = True

        return self.game_over

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
            "level": self.level,
            "time_left": self.timer,
            "level_complete": self.level_complete
        }

    def _prepare_grid(self):
        self.empty_cells = []
        for x in range(self.MAZE_W):
            for y in range(self.MAZE_H):
                self.empty_cells.append((x, y))

    def _place_game_elements(self):
        self.beam_start_pos = (2, 2)
        self.beam_start_dir = self._normalize((1, 0.2))

        available_cells = self.empty_cells[:]
        self.np_random.shuffle(available_cells)

        exit_cell_x, exit_cell_y = available_cells.pop()
        exit_center_x = (exit_cell_x + 0.5) * self.CELL_SIZE
        exit_center_y = (exit_cell_y + 0.5) * self.CELL_SIZE
        self.exit_rect = pygame.Rect(exit_center_x - 10, exit_center_y - 10, 20, 20)
        
        self.mirrors = []
        for _ in range(min(self.num_mirrors, len(available_cells))):
            cell_x, cell_y = available_cells.pop()
            self.mirrors.append({
                'pos': ((cell_x + 0.5) * self.CELL_SIZE, (cell_y + 0.5) * self.CELL_SIZE),
                'angle': self.np_random.integers(0, 8) * 45,
                'size': self.CELL_SIZE * 0.7
            })
        
        self.mirrors.sort(key=lambda m: (m['pos'][1], m['pos'][0]))

    def _update_beam_path(self):
        self.beam_path = [self.beam_start_pos]
        self.hit_mirrors_this_step = set()
        
        pos = np.array(self.beam_start_pos, dtype=float)
        direction = np.array(self.beam_start_dir, dtype=float)
        
        max_reflections = len(self.mirrors) + 5
        for _ in range(max_reflections):
            intersections = []
            
            for idx, mirror in enumerate(self.mirrors):
                m_pos = np.array(mirror['pos'])
                m_angle_rad = math.radians(mirror['angle'])
                m_size = mirror['size']
                p1 = m_pos + np.array([math.cos(m_angle_rad), math.sin(m_angle_rad)]) * m_size / 2
                p2 = m_pos - np.array([math.cos(m_angle_rad), math.sin(m_angle_rad)]) * m_size / 2
                t, u = self._get_line_intersection(pos, direction, p1, p2-p1)
                if t is not None and t > 1e-6 and 0 <= u <= 1:
                    intersections.append({'t': t, 'type': 'mirror', 'idx': idx, 'p1': p1, 'p2': p2})
            
            wall_points = [
                (np.array([0,0]), np.array([self.SCREEN_WIDTH, 0])),
                (np.array([self.SCREEN_WIDTH, 0]), np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])),
                (np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT]), np.array([0, self.SCREEN_HEIGHT])),
                (np.array([0, self.SCREEN_HEIGHT]), np.array([0, 0])),
            ]
            for p1, p2 in wall_points:
                t, u = self._get_line_intersection(pos, direction, p1, p2-p1)
                if t is not None and t > 1e-6 and 0 <= u <= 1:
                     intersections.append({'t': t, 'type': 'wall'})

            for p1, p2 in self._rect_to_lines(self.exit_rect):
                t, u = self._get_line_intersection(pos, direction, np.array(p1), np.array(p2)-np.array(p1))
                if t is not None and t > 1e-6 and 0 <= u <= 1:
                    intersections.append({'t': t, 'type': 'exit'})

            if not intersections: break

            closest = min(intersections, key=lambda x: x['t'])
            hit_point = pos + closest['t'] * direction
            self.beam_path.append(tuple(hit_point))

            if closest['type'] == 'wall': break
            if closest['type'] == 'exit':
                self.level_complete = True
                break
            if closest['type'] == 'mirror':
                self.hit_mirrors_this_step.add(closest['idx'])
                p1, p2 = closest['p1'], closest['p2']
                surface_vec = self._normalize(p2 - p1)
                normal = np.array([-surface_vec[1], surface_vec[0]])
                direction = self._normalize(direction - 2 * np.dot(direction, normal) * normal)
                pos = hit_point
        
    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)
        pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in self.COLOR_EXIT), self.exit_rect, 3)

        for i, mirror in enumerate(self.mirrors):
            color = self.COLOR_MIRROR_SELECTED if i == self.selected_mirror_idx else self.COLOR_MIRROR
            pos, angle_rad, size = mirror['pos'], math.radians(mirror['angle']), mirror['size']
            p1 = (pos[0] + math.cos(angle_rad) * size / 2, pos[1] + math.sin(angle_rad) * size / 2)
            p2 = (pos[0] - math.cos(angle_rad) * size / 2, pos[1] - math.sin(angle_rad) * size / 2)
            pygame.draw.line(self.screen, color, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 4)
    
        if len(self.beam_path) > 1:
            for i in range(len(self.beam_path) - 1):
                self._render_glow_line(self.screen, self.COLOR_BEAM, self.beam_path[i], self.beam_path[i+1], 2)

    def _render_ui(self):
        timer_text = f"TIME: {max(0, self.timer):.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 10))

        level_text = f"LEVEL: {self.level}"
        level_surf = self.font_ui.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (10, 10))

        score_text = f"SCORE: {self.score:.0f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH // 2 - score_surf.get_width() // 2, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "LEVEL COMPLETE" if self.level_complete else "TIME OUT"
            msg_surf = self.font_game_over.render(msg, True, self.COLOR_EXIT if self.level_complete else self.COLOR_TEXT)
            self.screen.blit(msg_surf, (self.SCREEN_WIDTH // 2 - msg_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_surf.get_height() // 2))

    def _render_glow_line(self, surf, color, start, end, width):
        start_i, end_i = (int(start[0]), int(start[1])), (int(end[0]), int(end[1]))
        try:
            pygame.draw.aaline(surf, color, start_i, end_i)
            pygame.draw.line(surf, (255, 255, 220), start_i, end_i, width)
        except TypeError: # Can happen if coordinates are out of bounds
            pass

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else v

    def _get_line_intersection(self, p1, v1, p2, v2):
        v1_v2_cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(v1_v2_cross) < 1e-8: return None, None
        dp = p2 - p1
        t = (dp[0] * v2[1] - dp[1] * v2[0]) / v1_v2_cross
        u = (dp[0] * v1[1] - dp[1] * v1[0]) / v1_v2_cross
        return t, u
    
    def _rect_to_lines(self, rect):
        return [
            (rect.topleft, rect.topright), (rect.topright, rect.bottomright),
            (rect.bottomright, rect.bottomleft), (rect.bottomleft, rect.topleft)
        ]

    def _get_min_dist_to_exit(self):
        if not self.beam_path or len(self.beam_path) < 2:
            return math.dist(self.beam_start_pos, self.exit_rect.center)
        min_dist = float('inf')
        exit_center = np.array(self.exit_rect.center)
        for i in range(len(self.beam_path) - 1):
            p1, p2 = np.array(self.beam_path[i]), np.array(self.beam_path[i+1])
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0.0: dist = np.linalg.norm(exit_center - p1)
            else:
                t = max(0, min(1, np.dot(exit_center - p1, p2 - p1) / l2))
                dist = np.linalg.norm(exit_center - (p1 + t * (p2 - p1)))
            min_dist = min(min_dist, dist)
        return min_dist

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the environment directly for testing.
    # It will use a visible pygame window.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Maze Environment")
    clock = pygame.time.Clock()
    running = True
    
    while running:
        action = [0, 0, 0]
        action_taken_this_frame = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                # Use keydown for single-press actions
                if not action_taken_this_frame:
                    if event.key == pygame.K_UP or event.key == pygame.K_RIGHT:
                        action[0] = 1
                        action_taken_this_frame = True
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_LEFT:
                        action[0] = 2
                        action_taken_this_frame = True
                    elif event.key == pygame.K_SPACE:
                        action[1] = 1
                        action_taken_this_frame = True
                    elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                        action[2] = 1
                        action_taken_this_frame = True

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(3000)
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()