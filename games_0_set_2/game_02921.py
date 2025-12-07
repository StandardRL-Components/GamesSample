
# Generated: 2025-08-28T06:24:27.371458
# Source Brief: brief_02921.md
# Brief Index: 2921

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a dot, then move to its matching partner and press space again to connect them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Connect all matching colored dots on the grid by drawing non-overlapping paths. Plan your routes carefully to solve the puzzle before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 12, 8
        self.NUM_PAIRS = 5
        self.MAX_MOVES = 25
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (45, 55, 65)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 0)
        self.DOT_COLORS = [
            (255, 71, 87),    # Red
            (46, 213, 115),   # Green
            (30, 144, 255),   # Blue
            (255, 165, 2),    # Orange
            (247, 143, 179),  # Pink
            (177, 112, 255)   # Purple
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Grid and Cell Dimensions ---
        self.cell_width = self.WIDTH // self.GRID_COLS
        self.cell_height = self.HEIGHT // self.GRID_ROWS
        self.dot_radius = int(min(self.cell_width, self.cell_height) * 0.3)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.remaining_moves = 0
        self.last_space_state = 0
        self.cursor_pos = [0, 0]
        self.dots = []
        self.dot_map = {}
        self.paths = []
        self.selected_dot = None
        self.particles = []
        self.last_reward_text = ""

        # Initialize state
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.remaining_moves = self.MAX_MOVES
        self.last_space_state = 0
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_dot = None
        self.paths = []
        self.particles = []
        self.last_reward_text = ""

        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        self.dots = []
        self.dot_map = {}
        
        all_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_positions)

        for i in range(self.NUM_PAIRS):
            color = self.DOT_COLORS[i % len(self.DOT_COLORS)]
            
            pos1 = all_positions.pop()
            dot1 = {"id": i * 2, "pos": pos1, "color": color, "pair_id": i, "connected": False}
            self.dots.append(dot1)
            self.dot_map[pos1] = dot1

            pos2 = all_positions.pop()
            dot2 = {"id": i * 2 + 1, "pos": pos2, "color": color, "pair_id": i, "connected": False}
            self.dots.append(dot2)
            self.dot_map[pos2] = dot2

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        self.steps += 1
        self.last_reward_text = ""

        movement, space_action, _ = action
        space_pressed = (space_action == 1 and self.last_space_state == 0)
        self.last_space_state = space_action

        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        if space_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            dot_at_cursor = self.dot_map.get(cursor_tuple)

            if self.selected_dot is None:
                if dot_at_cursor and not dot_at_cursor["connected"]:
                    self.selected_dot = dot_at_cursor
                else:
                    reward = -0.1
                    self.last_reward_text = "-0.1 Invalid Selection"
            else:
                self.remaining_moves -= 1
                if dot_at_cursor and not dot_at_cursor["connected"] and \
                   dot_at_cursor["pair_id"] == self.selected_dot["pair_id"] and \
                   dot_at_cursor["id"] != self.selected_dot["id"]:
                    
                    path = self._find_path(self.selected_dot["pos"], dot_at_cursor["pos"])
                    if path:
                        reward += 1.0
                        self.last_reward_text = "+1.0 Connection"
                        
                        if not self._check_crossings(path):
                            reward += 5.0
                            self.last_reward_text += " +5.0 No Crossings!"
                        
                        self.paths.append({"path": path, "color": self.selected_dot["color"]})
                        self.selected_dot["connected"] = True
                        dot_at_cursor["connected"] = True
                        self._create_particles(self.selected_dot["pos"], self.selected_dot["color"])
                        self._create_particles(dot_at_cursor["pos"], dot_at_cursor["color"])
                        # sfx_connect.play()
                    else:
                        reward = -0.1
                        self.last_reward_text = "-0.1 Path Blocked"
                        # sfx_error.play()
                else:
                    reward = -0.1
                    self.last_reward_text = "-0.1 Invalid Target"
                    # sfx_error.play()

                self.selected_dot = None

        self._update_particles()
        
        all_connected = all(d["connected"] for d in self.dots)
        if all_connected:
            reward += 50.0
            self.last_reward_text = "+50.0 YOU WIN!"
            terminated = True
            self.game_over = True
            self.win_state = True
        elif self.remaining_moves <= 0:
            reward -= 10.0
            self.last_reward_text = "-10.0 Out of Moves"
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _find_path(self, start_pos, end_pos):
        q = deque([(start_pos, [start_pos])])
        visited = {start_pos}
        
        obstacles = {
            d["pos"] for d in self.dots 
            if not d["connected"] and d["pos"] != start_pos and d["pos"] != end_pos
        }

        while q:
            (vx, vy), path = q.popleft()

            if (vx, vy) == end_pos:
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = vx + dx, vy + dy
                
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    neighbor = (nx, ny)
                    if neighbor not in visited and neighbor not in obstacles:
                        visited.add(neighbor)
                        new_path = list(path)
                        new_path.append(neighbor)
                        q.append((neighbor, new_path))
        return None

    def _check_crossings(self, new_path):
        # A simple check for grid-based paths: do they share any non-endpoint cells?
        new_path_body = set(new_path[1:-1])
        if not new_path_body: # Path is only 2 cells long
            return False
            
        for old_path_data in self.paths:
            old_path_body = set(old_path_data["path"][1:-1])
            if not new_path_body.isdisjoint(old_path_body):
                return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(self.GRID_COLS + 1):
            px = x * self.cell_width
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = y * self.cell_height
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

        for p in self.paths:
            pixel_path = [self._grid_to_pixel_center(pos) for pos in p["path"]]
            if len(pixel_path) > 1:
                pygame.draw.lines(self.screen, p["color"], False, pixel_path, width=max(1, self.dot_radius // 2))

        for dot in self.dots:
            if not dot["connected"]:
                center_px = self._grid_to_pixel_center(dot["pos"])
                radius = self.dot_radius
                
                if self.selected_dot and self.selected_dot["id"] == dot["id"]:
                    radius = int(self.dot_radius * (1.2 + 0.1 * math.sin(self.steps * 0.5)))
                
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius + 3, (*dot["color"], 60))
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius, dot["color"])
                pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], radius, dot["color"])

        cursor_rect = pygame.Rect(
            self.cursor_pos[0] * self.cell_width,
            self.cursor_pos[1] * self.cell_height,
            self.cell_width,
            self.cell_height
        )
        cursor_color = self.selected_dot["color"] if self.selected_dot else self.COLOR_CURSOR
        pygame.draw.rect(self.screen, cursor_color, cursor_rect, 3, border_radius=4)
        
        if self.selected_dot:
            start_px = self._grid_to_pixel_center(self.selected_dot["pos"])
            end_px = self._grid_to_pixel_center(self.cursor_pos)
            pygame.draw.line(self.screen, self.selected_dot["color"], start_px, end_px, 2)

        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["x"]), int(p["y"])), int(p["size"]))

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        moves_text = self.font_main.render(f"Moves: {self.remaining_moves}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(moves_text, moves_rect)

        if self.last_reward_text:
            reward_surf = self.font_small.render(self.last_reward_text, True, self.COLOR_UI_TEXT)
            reward_rect = reward_surf.get_rect(center=(self.WIDTH // 2, 20))
            self.screen.blit(reward_surf, reward_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_state else "GAME OVER"
            color = self.DOT_COLORS[1] if self.win_state else self.DOT_COLORS[0]
            
            end_text = self.font_main.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "pairs_connected": sum(1 for d in self.dots if d["connected"]) // 2,
        }

    def _grid_to_pixel_center(self, grid_pos):
        x = int((grid_pos[0] + 0.5) * self.cell_width)
        y = int((grid_pos[1] + 0.5) * self.cell_height)
        return x, y

    def _create_particles(self, grid_pos, color):
        center_px = self._grid_to_pixel_center(grid_pos)
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "x": center_px[0],
                "y": center_px[1],
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "size": self.np_random.uniform(2, 5),
                "life": 30,
                "color": color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.1
            p["size"] *= 0.95
            p["life"] -= 1
            if p["life"] > 0 and p["size"] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dot Connect")
    clock = pygame.time.Clock()
    running = True

    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            # Render final state before reset
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Slower tick for turn-based human play

    env.close()