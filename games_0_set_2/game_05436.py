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



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selected robot. "
        "Press nothing (no-op) to cycle which robot is selected."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide three stranded robots to their escape pods in a procedurally generated "
        "isometric maze within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        # Set headless mode for server execution
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # --- Visuals ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_WALL = (70, 80, 90)
        self.COLOR_WALL_TOP = (90, 100, 110)
        self.COLOR_FLOOR = (45, 50, 60)
        self.COLOR_ROBOT = [(255, 60, 60), (60, 255, 60), (60, 120, 255)]
        self.COLOR_POD = self.COLOR_ROBOT
        self.COLOR_SELECTED = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.TILE_W = 24
        self.TILE_H = 12
        self.WALL_HEIGHT = 18

        # --- Game State ---
        self.maze_dim = 7
        self.stage = 1
        self.stage_wins = 0
        self.max_steps = 1000

        # Initialize state variables to be populated in reset()
        self.maze = None
        self.robots = None
        self.pods = None
        self.robots_home = None
        self.selected_robot_idx = 0
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        
        # self.reset() is called by the environment wrapper, no need to call it here.
        # self._validate_implementation() # This is a helper for development, not needed in final version
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over and self.win_message == "STAGE CLEAR!":
            self.stage_wins += 1
            if self.stage_wins >= 3:
                self.stage_wins = 0
                self.stage += 1
                self.maze_dim = min(25, 7 + (self.stage - 1) * 2)
        elif self.game_over and self.win_message != "STAGE CLEAR!":
             # On loss, reset stage progress
             self.stage_wins = 0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        self.selected_robot_idx = 0

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.maze = self._generate_maze(self.maze_dim, self.maze_dim)
        floor_tiles = np.argwhere(self.maze == 0)
        
        if len(floor_tiles) < 6:
            # Fallback if maze generation is too dense
            self.maze_dim = 7
            self.stage = 1
            self.stage_wins = 0
            self.maze = self._generate_maze(self.maze_dim, self.maze_dim)
            floor_tiles = np.argwhere(self.maze == 0)

        placements = self.np_random.choice(
            np.arange(len(floor_tiles)), size=6, replace=False
        )
        
        # Note: maze uses (y, x), but we store positions as (x, y)
        self.robots = [tuple(floor_tiles[i][::-1]) for i in placements[:3]]
        self.pods = [tuple(floor_tiles[i][::-1]) for i in placements[3:]]
        self.robots_home = [False, False, False]
        
        for i in range(3):
            if self.robots[i] == self.pods[i]:
                self.robots_home[i] = True

        self.moves_remaining = self.maze_dim * self.maze_dim // 2 + self.maze_dim

    def _generate_maze(self, width, height):
        maze = np.ones((height, width), dtype=np.uint8)
        # Ensure start is on an odd coordinate for the carving algorithm
        start_x, start_y = self.np_random.integers(0, (width//2, height//2)) * 2 + 1
        
        maze[start_y, start_x] = 0
        stack = [(start_x, start_y)]
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < width-1 and 0 < ny < height-1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Use np_random for reproducibility
                idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[idx]
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        
        self.particles = [p for p in self.particles if p.update()]

        if movement == 0:  # Cycle selected robot
            self.selected_robot_idx = (self.selected_robot_idx + 1) % 3
        else: # Move action
            self.moves_remaining -= 1
            
            robot_idx = self.selected_robot_idx
            if self.robots_home[robot_idx]: # Can't move a robot that's home
                pass
            else:
                x, y = self.robots[robot_idx]
                pod_x, pod_y = self.pods[robot_idx]

                dx, dy = 0, 0
                if movement == 1: dy = -1  # Up
                elif movement == 2: dy = 1   # Down
                elif movement == 3: dx = -1  # Left
                elif movement == 4: dx = 1   # Right

                nx, ny = x + dx, y + dy

                is_in_bounds = 0 <= nx < self.maze_dim and 0 <= ny < self.maze_dim
                is_wall = is_in_bounds and self.maze[ny, nx] == 1
                is_occupied = (nx, ny) in self.robots

                if is_in_bounds and not is_wall and not is_occupied:
                    dist_before = abs(x - pod_x) + abs(y - pod_y)
                    dist_after = abs(nx - pod_x) + abs(ny - pod_y)
                    reward += (dist_before - dist_after) # +1 if closer, -1 if further

                    self.robots[robot_idx] = (nx, ny)
                    # sfx: robot_move

                    if (nx, ny) == (pod_x, pod_y):
                        self.robots_home[robot_idx] = True
                        reward += 10
                        # sfx: robot_home
                        self._create_particles((nx, ny))

        self.score += reward
        self.steps += 1
        
        if all(self.robots_home):
            terminated = True
            self.game_over = True
            self.win_message = "STAGE CLEAR!"
            reward += 100
            # sfx: level_win
        elif self.moves_remaining <= 0:
            terminated = True
            self.game_over = True
            self.win_message = "OUT OF MOVES"
            reward -= 100
            # sfx: level_loss
        elif self.steps >= self.max_steps:
            terminated = True
            self.game_over = True
            self.win_message = "TIME LIMIT"

        truncated = False # This environment does not truncate based on time limit
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        maze_pixel_w = self.maze_dim * self.TILE_W
        maze_pixel_h = self.maze_dim * self.TILE_H
        offset_x = int((self.screen.get_width() - maze_pixel_w) / 2)
        offset_y = int((self.screen.get_height() - maze_pixel_h) / 2 + 30)

        for y in range(self.maze_dim):
            for x in range(self.maze_dim):
                is_wall = self.maze[y, x] == 1
                if is_wall:
                    self._draw_iso_cube((x, y), offset_x, offset_y, self.WALL_HEIGHT, self.COLOR_WALL, self.COLOR_WALL_TOP)
                else:
                    iso_pts = self._get_iso_points((x, y), offset_x, offset_y, 0)
                    pygame.gfxdraw.filled_polygon(self.screen, iso_pts, self.COLOR_FLOOR)
                    pygame.gfxdraw.aapolygon(self.screen, iso_pts, self.COLOR_FLOOR)

        for i, (px, py) in enumerate(self.pods):
            iso_pts = self._get_iso_points((px, py), offset_x, offset_y, 0)
            center_x = iso_pts[0][0] + self.TILE_W // 2
            center_y = iso_pts[0][1] + self.TILE_H // 2
            
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.002 + i))
            radius = int(self.TILE_H * 0.6 + pulse * 2)
            color = self.COLOR_POD[i]
            if not self.robots_home[i]:
                 pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*color, 100))
                 pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (*color, 150))

        for p in self.particles:
            p.draw(self.screen)
            
        for i, (rx, ry) in enumerate(self.robots):
            height = 4
            
            if i == self.selected_robot_idx and not self.robots_home[i] and not self.game_over:
                base_pts = self._get_iso_points((rx, ry), offset_x, offset_y, 0)
                center_x = base_pts[0][0] + self.TILE_W // 2
                center_y = base_pts[0][1] + self.TILE_H // 2
                pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
                radius = int(self.TILE_W * 0.6 + pulse * 3)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*self.COLOR_SELECTED, 50))
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_SELECTED)
            
            color = self.COLOR_ROBOT[i]
            self._draw_iso_cube((rx, ry), offset_x, offset_y, height, color, tuple(min(255, c+40) for c in color))

            if self.robots_home[i]:
                iso_pts = self._get_iso_points((rx, ry), offset_x, offset_y, 0)
                center_x = iso_pts[0][0] + self.TILE_W // 2
                center_y = iso_pts[0][1] + self.TILE_H // 2 - height
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 3, (255,255,255))

    def _to_iso(self, x, y):
        iso_x = (x - y) * (self.TILE_W / 2)
        iso_y = (x + y) * (self.TILE_H / 2)
        return int(iso_x), int(iso_y)

    def _get_iso_points(self, pos, offset_x, offset_y, z_offset):
        x, y = pos
        p1 = self._to_iso(x, y)
        p2 = self._to_iso(x + 1, y)
        p3 = self._to_iso(x + 1, y + 1)
        p4 = self._to_iso(x, y + 1)
        return [
            (p1[0] + offset_x, p1[1] + offset_y - z_offset),
            (p2[0] + offset_x, p2[1] + offset_y - z_offset),
            (p3[0] + offset_x, p3[1] + offset_y - z_offset),
            (p4[0] + offset_x, p4[1] + offset_y - z_offset),
        ]

    def _draw_iso_cube(self, pos, offset_x, offset_y, height, side_color, top_color):
        top_pts = self._get_iso_points(pos, offset_x, offset_y, height)
        
        pygame.gfxdraw.filled_polygon(self.screen, top_pts, top_color)
        pygame.gfxdraw.aapolygon(self.screen, top_pts, top_color)

        x, y = pos
        p_top_1 = self._to_iso(x + 1, y)
        p_top_2 = self._to_iso(x + 1, y + 1)
        p_top_3 = self._to_iso(x, y + 1)
        
        right_face = [
            (p_top_1[0] + offset_x, p_top_1[1] + offset_y),
            (p_top_2[0] + offset_x, p_top_2[1] + offset_y),
            (p_top_2[0] + offset_x, p_top_2[1] + offset_y - height),
            (p_top_1[0] + offset_x, p_top_1[1] + offset_y - height),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, right_face, side_color)
        
        left_face = [
            (p_top_2[0] + offset_x, p_top_2[1] + offset_y),
            (p_top_3[0] + offset_x, p_top_3[1] + offset_y),
            (p_top_3[0] + offset_x, p_top_3[1] + offset_y - height),
            (p_top_2[0] + offset_x, p_top_2[1] + offset_y - height),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, left_face, side_color)

    def _render_ui(self):
        moves_text = self.font_small.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        stage_text = self.font_small.render(f"Stage: {self.stage}-{self.stage_wins+1}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.screen.get_width() - stage_text.get_width() - 10, 10))
        
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 35))

        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_SELECTED)
            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "stage": self.stage,
            "robots_home": sum(self.robots_home),
        }
        
    def _create_particles(self, pos):
        maze_pixel_w = self.maze_dim * self.TILE_W
        maze_pixel_h = self.maze_dim * self.TILE_H
        offset_x = int((self.screen.get_width() - maze_pixel_w) / 2)
        offset_y = int((self.screen.get_height() - maze_pixel_h) / 2 + 30)
        
        iso_pts = self._get_iso_points(pos, offset_x, offset_y, 0)
        center_x = iso_pts[0][0] + self.TILE_W // 2
        center_y = iso_pts[0][1] + self.TILE_H // 2
        
        for _ in range(20):
            self.particles.append(Particle(center_x, center_y, self.np_random))

    def _validate_implementation(self):
        # This is a helper for development, not needed in final version
        # but useful for verification during the fix.
        print("✓ Validating implementation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

class Particle:
    def __init__(self, x, y, np_random):
        self.x = x
        self.y = y
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = 1.0
        self.color = (255, 255, 100)
        self.radius = np_random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 0.05 # Fast decay for turn-based game
        self.radius *= 0.95
        return self.lifespan > 0 and self.radius > 0.5

    def draw(self, surface):
        if self.lifespan <= 0:
            return
        alpha = int(255 * self.lifespan)
        color = (*self.color, alpha)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), color)

if __name__ == '__main__':
    # The environment is designed to be run headless, this is for testing
    env = GameEnv()
    env._validate_implementation()
    
    obs, info = env.reset()
    print("Initial state:")
    print(f"Info: {info}")

    terminated = False
    total_reward = 0
    for i in range(50):
        if terminated:
            print("Episode terminated. Resetting.")
            obs, info = env.reset()
            total_reward = 0
            terminated = False # Reset termination flag
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}, Info: {info}")