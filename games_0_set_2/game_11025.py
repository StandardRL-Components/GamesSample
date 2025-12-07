import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:22:28.737709
# Source Brief: brief_01025.md
# Brief Index: 1025
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import heapq

class GameEnv(gym.Env):
    """
    Cyberpunk Hacking Environment

    In this game, the player controls a cursor to hack into a corporate network.
    The goal is to reroute colored data streams to matching access points by
    "terraforming" the network grid. Rerouting streams correctly unlocks
    access points. Unlock all of them to win. Avoid security programs that
    patrol the network, as proximity will lower your health.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Match attempt (0=released, 1=pressed)
    - actions[2]: Terraform (0=released, 1=pressed)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Hack into a corporate network by rerouting data streams to their matching access points. "
        "Terraform the grid to create pathways, but avoid the patrolling security programs that threaten your connection."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'shift' to terraform a grid tile, creating a path for data streams. "
        "Press 'space' over an access point to attempt a match."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = 20
    MAX_STEPS = 2000
    MAX_HEALTH = 100.0

    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (20, 30, 50)
    COLOR_WALL = (50, 60, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_GLOW = (0, 192, 255)
    COLOR_TERRAFORM = (0, 255, 255)
    COLOR_TERRAFORM_GLOW = (0, 128, 128)
    COLOR_SECURITY = (255, 0, 80)
    COLOR_SECURITY_GLOW = (128, 0, 40)
    COLOR_ACCESS_LOCKED = (100, 100, 120)
    COLOR_ACCESS_UNLOCKED = (0, 255, 128)
    STREAM_COLORS = {
        "blue": (0, 150, 255),
        "yellow": (255, 255, 0),
        "purple": (200, 0, 255),
        "orange": (255, 165, 0),
        "pink": (255, 105, 180),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_obj = pygame.font.SysFont("Consolas", 16)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.health = self.MAX_HEALTH
        self.cursor_pos = [0, 0]
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int) # 0: empty, 1: wall
        self.terraformed_tiles = set()
        self.access_points = []
        self.data_streams = []
        self.security_programs = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.game_over = False
        self.victory = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.health = self.MAX_HEALTH
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.terraformed_tiles.clear()
        self.particles.clear()
        self.game_over = False
        self.victory = False
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_level()
        self._recalculate_all_stream_paths()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid.fill(0)
        self.access_points.clear()
        self.data_streams.clear()
        self.security_programs.clear()
        
        # Add some random walls
        for _ in range(30):
            x, y = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)
            self.grid[x, y] = 1

        # Define access points and corresponding streams
        ap_positions = [(3, 3), (28, 16), (4, 15)]
        stream_starts = [(28, 3), (3, 16), (16, 1)]
        stream_color_keys = ["blue", "yellow", "purple"]
        
        for i in range(3):
            ap_pos = ap_positions[i]
            stream_start = stream_starts[i]
            color_key = stream_color_keys[i]

            self.grid[ap_pos] = 0 # Ensure not a wall
            self.grid[stream_start] = 0 # Ensure not a wall
            
            self.access_points.append({
                "pos": ap_pos,
                "required_colors": {color_key},
                "unlocked": False,
                "current_streams": set()
            })
            self.data_streams.append({
                "start_pos": stream_start,
                "target_ap_idx": i,
                "color": self.STREAM_COLORS[color_key],
                "color_key": color_key,
                "path": [],
                "particles": []
            })
            
        # Final access point (requires all streams)
        final_ap_pos = (16, 10)
        self.grid[final_ap_pos] = 0
        self.access_points.append({
            "pos": final_ap_pos,
            "required_colors": set(stream_color_keys),
            "unlocked": False,
            "current_streams": set()
        })

        # Security programs
        self.security_programs.append({"path": [(2, 10), (10, 18), (10, 2)], "path_idx": 0, "pos": [2.0, 10.0], "speed": 0.05})
        self.security_programs.append({"path": [(29, 10), (20, 2), (20, 18)], "path_idx": 0, "pos": [29.0, 10.0], "speed": 0.05})


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions ---
        self._handle_movement(movement)
        
        if shift_held and not self.prev_shift_held:
            # Terraform on key press
            reward += self._handle_terraform()
        
        if space_held and not self.prev_space_held:
            # Match attempt on key press
            reward += self._handle_match_attempt()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self._update_security_programs()
        self._update_stream_particles()
        self.particles = [p for p in self.particles if p["life"] > 0]

        # --- Calculate Rewards & Penalties ---
        security_penalty = self._calculate_security_penalty()
        self.health += security_penalty # penalty is negative
        reward += security_penalty

        # --- Check Termination Conditions ---
        if self.health <= 0:
            self.health = 0
            self.game_over = True
            reward = -100.0
            # sfx: player_detected_alarm

        if all(ap["unlocked"] for ap in self.access_points):
            self.game_over = True
            self.victory = True
            reward = 100.0
            # sfx: victory_fanfare

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            reward -= 20 # Penalty for timeout

        terminated = self.game_over
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            for sec in self.security_programs:
                sec['speed'] = min(0.2, sec['speed'] + 0.01)

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Action Handlers ---
    def _handle_movement(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1  # Right
        
        new_x = max(0, min(self.GRID_COLS - 1, self.cursor_pos[0] + dx))
        new_y = max(0, min(self.GRID_ROWS - 1, self.cursor_pos[1] + dy))
        self.cursor_pos = [new_x, new_y]

    def _handle_terraform(self):
        pos = tuple(self.cursor_pos)
        if self.grid[pos] == 1: return 0 # Cannot terraform walls
        is_ap = any(tuple(ap["pos"]) == pos for ap in self.access_points)
        is_stream_start = any(tuple(s["start_pos"]) == pos for s in self.data_streams)
        if is_ap or is_stream_start: return 0 # Cannot terraform on special nodes

        if pos not in self.terraformed_tiles:
            self.terraformed_tiles.add(pos)
            self._recalculate_all_stream_paths()
            # sfx: terraform_activate
            self._create_particles(pos, self.COLOR_TERRAFORM, 20, 1.5)
            return -0.01 # Small cost for terraforming
        return 0

    def _handle_match_attempt(self):
        pos = tuple(self.cursor_pos)
        reward = 0
        for ap in self.access_points:
            if tuple(ap["pos"]) == pos and not ap["unlocked"]:
                # sfx: match_attempt
                matched_colors = ap["current_streams"]
                required_colors = ap["required_colors"]
                
                if matched_colors == required_colors:
                    ap["unlocked"] = True
                    self.score += 100
                    reward += 5.0
                    # sfx: access_point_unlocked
                    self._create_particles(pos, self.COLOR_ACCESS_UNLOCKED, 50, 3.0)
                else:
                    # Partial reward for correct streams
                    correct_matches = len(matched_colors.intersection(required_colors))
                    reward += correct_matches * 0.1
                    # sfx: match_fail
                    self._create_particles(pos, self.COLOR_SECURITY, 20, 1.0)
        return reward

    # --- State Update and Calculations ---
    def _update_security_programs(self):
        for sec in self.security_programs:
            target_node = sec["path"][sec["path_idx"]]
            current_pos = np.array(sec["pos"])
            target_pos = np.array(target_node)
            
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)

            if distance < sec["speed"]:
                sec["pos"] = list(target_pos)
                sec["path_idx"] = (sec["path_idx"] + 1) % len(sec["path"])
            else:
                move_vec = (direction / distance) * sec["speed"]
                sec["pos"] = list(current_pos + move_vec)

    def _update_stream_particles(self):
        for stream in self.data_streams:
            if not stream["path"]: continue
            
            # Add new particle
            if self.np_random.random() < 0.5: # Spawn rate
                stream["particles"].append({"progress": 0.0, "life": 1.0})

            # Update existing particles
            new_particles = []
            for p in stream["particles"]:
                p["progress"] += 0.03 # Particle speed
                if p["progress"] < 1.0:
                    new_particles.append(p)
            stream["particles"] = new_particles

    def _calculate_security_penalty(self):
        penalty = 0
        for sec in self.security_programs:
            dist = math.hypot(self.cursor_pos[0] - sec["pos"][0], self.cursor_pos[1] - sec["pos"][1])
            if dist < 3.0:
                penalty -= 0.5 * (1.0 - dist / 3.0) # Proximity-based penalty
        return penalty

    def _recalculate_all_stream_paths(self):
        # Reset stream routings
        for ap in self.access_points:
            ap["current_streams"].clear()
            
        for stream in self.data_streams:
            target_ap = self.access_points[stream["target_ap_idx"]]
            path = self._find_path(stream["start_pos"], target_ap["pos"])
            stream["path"] = path
            
            # If a path was found, find which AP it terminates at and update it
            if path:
                end_pos = path[-1]
                for ap in self.access_points:
                    if tuple(ap["pos"]) == end_pos:
                        ap["current_streams"].add(stream["color_key"])
                        break

    def _find_path(self, start, end):
        # Dijkstra's algorithm to find path favoring terraformed tiles
        pq = [(0, start, [start])] # (cost, pos, path)
        visited = {start}

        while pq:
            cost, current_pos, path = heapq.heappop(pq)

            if current_pos == end:
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if 0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS:
                    if self.grid[next_pos] == 0 and next_pos not in visited:
                        visited.add(next_pos)
                        tile_cost = 1 if next_pos in self.terraformed_tiles else 5
                        heapq.heappush(pq, (cost + tile_cost, next_pos, path + [next_pos]))
        return []

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_walls()
        self._draw_terraformed_tiles()
        self._draw_data_streams()
        self._draw_access_points()
        self._draw_security_programs()
        self._draw_particles()
        self._draw_cursor()

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_walls(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[c, r] == 1:
                    rect = pygame.Rect(c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

    def _draw_terraformed_tiles(self):
        for pos in self.terraformed_tiles:
            center_x = int((pos[0] + 0.5) * self.CELL_SIZE)
            center_y = int((pos[1] + 0.5) * self.CELL_SIZE)
            self._draw_glow_circle(center_x, center_y, self.CELL_SIZE // 3, self.COLOR_TERRAFORM, self.COLOR_TERRAFORM_GLOW)

    def _draw_data_streams(self):
        for stream in self.data_streams:
            if len(stream["path"]) > 1:
                # Draw path line
                points = [(int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE)) for x, y in stream["path"]]
                pygame.draw.lines(self.screen, stream["color"], False, points, 2)

                # Draw particles along path
                path_len = len(stream["path"]) - 1
                for p in stream["particles"]:
                    idx_float = p["progress"] * path_len
                    idx0 = int(idx_float)
                    idx1 = min(idx0 + 1, path_len)
                    interp = idx_float - idx0
                    
                    pos0 = np.array(points[idx0])
                    pos1 = np.array(points[idx1])
                    p_pos = pos0 + (pos1 - pos0) * interp
                    
                    pygame.gfxdraw.filled_circle(self.screen, int(p_pos[0]), int(p_pos[1]), 3, stream["color"])
                    pygame.gfxdraw.aacircle(self.screen, int(p_pos[0]), int(p_pos[1]), 3, stream["color"])

    def _draw_access_points(self):
        for ap in self.access_points:
            center_x = int((ap["pos"][0] + 0.5) * self.CELL_SIZE)
            center_y = int((ap["pos"][1] + 0.5) * self.CELL_SIZE)
            color = self.COLOR_ACCESS_UNLOCKED if ap["unlocked"] else self.COLOR_ACCESS_LOCKED
            
            pygame.draw.circle(self.screen, color, (center_x, center_y), self.CELL_SIZE // 2 - 2, 2)
            
            # Draw required color icons
            num_req = len(ap["required_colors"])
            for i, color_key in enumerate(sorted(list(ap["required_colors"]))):
                angle = (i / num_req) * 2 * math.pi
                icon_x = center_x + int(math.cos(angle) * self.CELL_SIZE * 0.25)
                icon_y = center_y + int(math.sin(angle) * self.CELL_SIZE * 0.25)
                pygame.draw.circle(self.screen, self.STREAM_COLORS[color_key], (icon_x, icon_y), 3)

    def _draw_security_programs(self):
        for sec in self.security_programs:
            center_x = int((sec["pos"][0] + 0.5) * self.CELL_SIZE)
            center_y = int((sec["pos"][1] + 0.5) * self.CELL_SIZE)
            self._draw_glow_circle(center_x, center_y, self.CELL_SIZE // 3, self.COLOR_SECURITY, self.COLOR_SECURITY_GLOW)
            
            # Pulsing effect
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
            s = pygame.Surface((self.CELL_SIZE * 3, self.CELL_SIZE * 3), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_SECURITY, int(pulse * 50)), (s.get_width()//2, s.get_height()//2), int(pulse * self.CELL_SIZE * 1.5))
            self.screen.blit(s, (center_x - s.get_width()//2, center_y - s.get_height()//2))

    def _draw_cursor(self):
        x, y = self.cursor_pos
        center_x = int((x + 0.5) * self.CELL_SIZE)
        center_y = int((y + 0.5) * self.CELL_SIZE)
        size = self.CELL_SIZE // 2
        
        self._draw_glow_circle(center_x, center_y, size // 2, self.COLOR_CURSOR, self.COLOR_CURSOR_GLOW)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (center_x - size, center_y), (center_x + size, center_y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (center_x, center_y - size), (center_x, center_y + size), 1)

    def _draw_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 0.05
            p["vel"][1] += 0.05 # Gravity
            
            alpha = max(0, int(p["life"] * 255))
            color = (*p["color"], alpha)
            radius = int(p["life"] * p["size"])
            if radius > 0:
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (p["pos"][0]-radius, p["pos"][1]-radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.health / self.MAX_HEALTH
        health_color = (int(255 * (1 - health_ratio)), int(255 * health_ratio), 0)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, health_color, (10, 10, 200 * health_ratio, 20))
        health_text = self.font_ui.render("DETECTION RISK", True, (200, 200, 200))
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (200, 200, 200))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Objective
        unlocked_aps = sum(1 for ap in self.access_points if ap["unlocked"])
        total_aps = len(self.access_points)
        obj_text = self.font_obj.render(f"ACCESS POINTS: {unlocked_aps}/{total_aps}", True, self.COLOR_TERRAFORM)
        self.screen.blit(obj_text, (self.SCREEN_WIDTH - obj_text.get_width() - 10, 35))
        
        if self.victory:
            win_text = self.font_ui.render("CORPORATE SECRETS STOLEN", True, self.COLOR_ACCESS_UNLOCKED)
            self.screen.blit(win_text, (self.SCREEN_WIDTH//2 - win_text.get_width()//2, self.SCREEN_HEIGHT//2 - win_text.get_height()//2))
        elif self.game_over and self.health <= 0:
            lose_text = self.font_ui.render("CONNECTION TERMINATED", True, self.COLOR_SECURITY)
            self.screen.blit(lose_text, (self.SCREEN_WIDTH//2 - lose_text.get_width()//2, self.SCREEN_HEIGHT//2 - lose_text.get_height()//2))

    # --- Helper Methods ---
    def _create_particles(self, grid_pos, color, count, max_speed):
        px, py = (grid_pos[0] + 0.5) * self.CELL_SIZE, (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            self.particles.append({
                "pos": [px, py],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": 1.0,
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })

    def _draw_glow_circle(self, x, y, radius, color, glow_color):
        s = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        center = (s.get_width()//2, s.get_height()//2)
        pygame.draw.circle(s, (*glow_color, 64), center, int(radius * 1.8))
        pygame.draw.circle(s, (*glow_color, 96), center, int(radius * 1.4))
        self.screen.blit(s, (x - center[0], y - center[1]), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.health, "victory": self.victory}

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # This method is for development and can be removed.
        try:
            # Test action space
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            # Test observation space  
            test_obs = self._get_observation()
            assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert test_obs.dtype == np.uint8
            
            # Test reset
            obs, info = self.reset()
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(info, dict)
            
            # Test step
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert not trunc
            assert isinstance(info, dict)
            
            # Test game-specific assertions
            assert 0 <= self.health <= self.MAX_HEALTH
            assert len(self.data_streams) <= 5
        except AssertionError as e:
            print(f"Validation failed: {e}")


if __name__ == "__main__":
    # --- Manual Play Testing ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override Pygame display for direct rendering
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cyberpunk Hacking Sim")
    clock = pygame.time.Clock()

    while not done:
        # --- Action Mapping for Human Player ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over. Final Info: {info}")
    env.close()