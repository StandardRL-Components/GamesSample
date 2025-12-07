
# Generated: 2025-08-28T02:56:50.288521
# Source Brief: brief_04614.md
# Brief Index: 4614

        
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


class Robot:
    """A helper class to store state for each robot."""
    def __init__(self, robot_id, pos, color, tile_size):
        self.id = robot_id
        self.pos = np.array(pos, dtype=int)
        self.color = color
        self.escaped = False
        self.tile_size = tile_size
        
        # For smooth visual interpolation
        self.pixel_pos = self.pos * self.tile_size + self.tile_size / 2
        self.target_pixel_pos = self.pixel_pos.copy()

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use arrow keys (↑↓←→) to move the selected robot. An action with no movement cycles selection."
    )

    game_description = (
        "Guide a squad of robots through a maze to their escape zone before you run out of moves."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.TILE_SIZE = 20
        self.NUM_ROBOTS = 4
        self.MAX_MOVES = 25
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (60, 70, 80)
        self.COLOR_PATH = (30, 35, 40)
        self.COLOR_ESCAPE = (80, 220, 120)
        self.ROBOT_COLORS = [
            (255, 87, 87), (87, 134, 255), (255, 240, 87), (87, 255, 150)
        ]
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SELECTION = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 48)
        
        # --- Game State (initialized in reset) ---
        self.maze = None
        self.robots = []
        self.escape_zone = None
        self.selected_robot_idx = 0
        self.moves_left = 0
        self.escaped_robots_count = 0
        self.game_over = False
        self.win = False
        self.score = 0
        self.steps = 0
        self.particles = []

        self.reset()
        self.validate_implementation()

    def _generate_maze(self, width, height):
        maze = np.ones((height, width), dtype=np.uint8)
        
        def is_valid(x, y):
            return 0 <= x < width and 0 <= y < height

        def carve(x, y):
            maze[y, x] = 0
            directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
            self.np_random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and maze[ny, nx] == 1:
                    maze[y + dy // 2, x + dx // 2] = 0
                    carve(nx, ny)

        start_x = self.np_random.integers(0, width // 2) * 2 + 1
        start_y = self.np_random.integers(0, height // 2) * 2 + 1
        
        carve(start_x, start_y)
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.MAX_MOVES
        self.escaped_robots_count = 0
        self.selected_robot_idx = 0
        self.particles = []

        self.maze = self._generate_maze(self.GRID_W, self.GRID_H)
        valid_positions = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(valid_positions)

        ez_size = 3
        ez_pos = valid_positions.pop(self.np_random.integers(len(valid_positions) // 2))
        self.escape_zone = pygame.Rect(ez_pos[1] - ez_size//2, ez_pos[0] - ez_size//2, ez_size, ez_size)
        
        self.robots = []
        for i in range(self.NUM_ROBOTS):
            pos = valid_positions.pop(self.np_random.integers(len(valid_positions)))
            robot = Robot(i, [pos[1], pos[0]], self.ROBOT_COLORS[i % len(self.ROBOT_COLORS)], self.TILE_SIZE)
            self.robots.append(robot)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        movement = action[0]
        
        selected_robot = self.robots[self.selected_robot_idx]

        if movement == 0: # Cycle selection
            self._cycle_selection()
        else: # Attempt to move
            self.moves_left -= 1
            
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1  # Right
            
            target_pos = selected_robot.pos + np.array([dx, dy])
            
            is_valid_move = True
            if not (0 <= target_pos[0] < self.GRID_W and 0 <= target_pos[1] < self.GRID_H):
                is_valid_move = False
            elif self.maze[target_pos[1], target_pos[0]] == 1:
                is_valid_move = False
            else:
                for r in self.robots:
                    if not r.escaped and r.id != selected_robot.id and np.array_equal(r.pos, target_pos):
                        is_valid_move = False
                        break
            
            if is_valid_move:
                dist_before = self._dist_to_escape(selected_robot.pos)
                dist_after = self._dist_to_escape(target_pos)
                
                if dist_after < dist_before: reward += 0.5
                else: reward -= 0.2
                
                selected_robot.pos = target_pos
                # sfx: footstep
                self._create_particles(selected_robot.pixel_pos, selected_robot.color, 5)
                
                if self.escape_zone.collidepoint(selected_robot.pos[0], selected_robot.pos[1]):
                    if not selected_robot.escaped:
                        selected_robot.escaped = True
                        self.escaped_robots_count += 1
                        reward += 10
                        # sfx: success chime
                        self._create_particles(selected_robot.pixel_pos, self.COLOR_ESCAPE, 20)
                        self._cycle_selection()
            else:
                reward -= 0.1 # Penalty for invalid move
                # sfx: bump

        self.score += reward
        
        terminated = False
        if self.escaped_robots_count == self.NUM_ROBOTS:
            self.win = True
            self.game_over = True
            terminated = True
            final_bonus = 50
            reward += final_bonus
            self.score += final_bonus
        elif self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _cycle_selection(self):
        if self.escaped_robots_count == self.NUM_ROBOTS:
            return
        
        start_idx = self.selected_robot_idx
        while True:
            self.selected_robot_idx = (self.selected_robot_idx + 1) % self.NUM_ROBOTS
            if not self.robots[self.selected_robot_idx].escaped:
                break
            if self.selected_robot_idx == start_idx:
                break

    def _dist_to_escape(self, pos):
        center_escape = np.array(self.escape_zone.center, dtype=float)
        return np.linalg.norm(pos.astype(float) - center_escape, ord=1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_maze()
        self._render_escape_zone()
        self._update_and_render_particles()
        self._render_robots()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_maze(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

    def _render_escape_zone(self):
        ez_rect_px = self.escape_zone.copy()
        ez_rect_px.x *= self.TILE_SIZE
        ez_rect_px.y *= self.TILE_SIZE
        ez_rect_px.w *= self.TILE_SIZE
        ez_rect_px.h *= self.TILE_SIZE
        
        s = pygame.Surface((ez_rect_px.width, ez_rect_px.height), pygame.SRCALPHA)
        s.fill((*self.COLOR_ESCAPE, 60))
        self.screen.blit(s, ez_rect_px.topleft)

    def _render_robots(self):
        lerp_rate = 0.4
        for r in self.robots:
            r.target_pixel_pos = r.pos * self.TILE_SIZE + self.TILE_SIZE / 2
            r.pixel_pos += (r.target_pixel_pos - r.pixel_pos) * lerp_rate

            pos_x, pos_y = int(r.pixel_pos[0]), int(r.pixel_pos[1])
            radius = self.TILE_SIZE // 2 - 3
            
            color = r.color
            if r.escaped:
                color = tuple(c * 0.5 for c in r.color)

            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, color)
        
        if not self.game_over and self.escaped_robots_count < self.NUM_ROBOTS:
            selected_robot = self.robots[self.selected_robot_idx]
            pos_x, pos_y = int(selected_robot.pixel_pos[0]), int(selected_robot.pixel_pos[1])
            pulse_radius = self.TILE_SIZE // 2 + abs(2 * math.sin(self.steps * 0.2))
            alpha = int(100 + 155 * abs(math.sin(self.steps * 0.2)))
            
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, int(pulse_radius), (*self.COLOR_SELECTION, alpha))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                alpha = int(255 * (p['life'] / 20))
                color = (*p['color'], alpha)
                pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), 1)
                active_particles.append(p)
        self.particles = active_particles


    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - 140, 20))

        escaped_text = self.font_main.render(f"Escaped: {self.escaped_robots_count}/{self.NUM_ROBOTS}", True, self.COLOR_TEXT)
        self.screen.blit(escaped_text, (self.WIDTH - 140, 50))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY!" if self.win else "OUT OF MOVES"
            color = self.COLOR_ESCAPE if self.win else self.ROBOT_COLORS[0]
            end_text = self.font_title.render(end_text_str, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "escaped_robots": self.escaped_robots_count,
        }

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    key_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }

    while running:
        action_movement = None 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r and terminated:
                    obs, info = env.reset()
                    terminated = False
                elif not terminated:
                    if event.key in key_map:
                        action_movement = key_map[event.key]
                    else:
                        action_movement = 0 # Any other key is a cycle action
        
        if action_movement is not None and not terminated:
            full_action = [action_movement, 0, 0]
            obs, reward, terminated, truncated, info = env.step(full_action)
            print(f"Action: {action_movement}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        else:
            # If no action, we still need to get the latest observation for rendering
            obs = env._get_observation()

        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(display_screen, frame)
        pygame.display.flip()
        
        clock.tick(60)

    pygame.quit()