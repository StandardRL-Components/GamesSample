
# Generated: 2025-08-27T17:40:15.520219
# Source Brief: brief_01604.md
# Brief Index: 1604

        
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
        "Controls: Use arrow keys to move the selector. Press Space to select/place a robot. Press Shift to cancel selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A tactical puzzle game. Guide all stranded robots to their charging stations using a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE
        self.MAX_MOVES = 15
        self.NUM_ROBOTS = 4

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_WALL = (80, 80, 90)
        self.COLOR_STATION = (60, 220, 120)
        self.COLOR_ROBOTS = [
            (255, 80, 80),   # Red
            (80, 150, 255),  # Blue
            (255, 255, 100), # Yellow
            (220, 100, 255)  # Magenta
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State Variables ---
        self.game_phase = "SELECT"  # 'SELECT' or 'MOVE'
        self.cursor_pos = [0, 0]
        self.selected_robot_idx = None
        self.robots = []
        self.stations = []
        self.walls = []
        self.particles = []
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.np_random = None

        self._define_levels()
        self.reset()
        
        # self.validate_implementation() # Optional: uncomment for self-testing

    def _define_levels(self):
        self.levels = [
            {
                "robots": [(1, 2), (1, 7), (14, 2), (14, 7)],
                "stations": [(7, 4), (7, 5), (8, 4), (8, 5)],
                "walls": [(i, 0) for i in range(16)] + [(i, 9) for i in range(16)] +
                         [(0, i) for i in range(1, 9)] + [(15, i) for i in range(1, 9)] +
                         [(5, 3), (5, 4), (5, 5), (5, 6), (10, 3), (10, 4), (10, 5), (10, 6)]
            },
            {
                "robots": [(2, 1), (2, 8), (13, 1), (13, 8)],
                "stations": [(4, 4), (4, 5), (11, 4), (11, 5)],
                "walls": [(i, 0) for i in range(16)] + [(i, 9) for i in range(16)] +
                         [(0, i) for i in range(1, 9)] + [(15, i) for i in range(1, 9)] +
                         [(i, 3) for i in range(3, 13)] + [(i, 6) for i in range(3, 13)]
            },
            {
                "robots": [(1, 1), (3, 1), (1, 3), (3, 3)],
                "stations": [(12, 6), (14, 6), (12, 8), (14, 8)],
                "walls": [(i, 0) for i in range(16)] + [(i, 9) for i in range(16)] +
                         [(0, i) for i in range(1, 9)] + [(15, i) for i in range(1, 9)] +
                         [(i, 5) for i in range(0, 8)] + [(7, i) for i in range(5, 10)]
            }
        ]

    def _generate_puzzle(self):
        level_idx = self.np_random.integers(0, len(self.levels))
        level = self.levels[level_idx]

        self.robots = []
        for i, pos in enumerate(level["robots"]):
            self.robots.append({
                "id": i,
                "pos": list(pos),
                "color": self.COLOR_ROBOTS[i],
                "charged": False
            })

        self.stations = [{"pos": list(pos)} for pos in level["stations"]]
        self.walls = [list(pos) for pos in level["walls"]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.particles = []
        
        self._generate_puzzle()
        
        self.game_phase = "SELECT"
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_robot_idx = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        
        dpad = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1

        # --- Handle Cursor Movement ---
        if dpad == 1: self.cursor_pos[1] -= 1  # Up
        if dpad == 2: self.cursor_pos[1] += 1  # Down
        if dpad == 3: self.cursor_pos[0] -= 1  # Left
        if dpad == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if self.game_phase == "SELECT":
            if space_pressed:
                # Try to select a robot
                for i, robot in enumerate(self.robots):
                    if robot["pos"] == self.cursor_pos and not robot["charged"]:
                        self.selected_robot_idx = i
                        self.game_phase = "MOVE"
                        # sfx: select_robot.wav
                        break
        
        elif self.game_phase == "MOVE":
            if shift_pressed:
                # Cancel selection
                self.selected_robot_idx = None
                self.game_phase = "SELECT"
                # sfx: cancel.wav
            elif space_pressed:
                # Try to place the robot
                robot = self.robots[self.selected_robot_idx]
                is_move_valid = self._is_valid_move(robot, self.cursor_pos)

                if is_move_valid:
                    # Execute the move
                    old_pos = robot["pos"]
                    new_pos = self.cursor_pos
                    
                    self._create_particles(self._grid_to_pixel(old_pos), robot["color"], 10, "out")
                    robot["pos"] = list(new_pos)
                    self._create_particles(self._grid_to_pixel(new_pos), robot["color"], 20, "in")
                    
                    self.moves_left -= 1
                    reward -= 1 # Cost of moving
                    # sfx: robot_move.wav

                    # Check for charging
                    if any(station["pos"] == new_pos for station in self.stations):
                        if not robot["charged"]:
                            robot["charged"] = True
                            reward += 10 # Reward for charging a robot
                            self.score += 10
                            self._create_particles(self._grid_to_pixel(new_pos), self.COLOR_STATION, 40)
                            # sfx: charge_complete.wav
                    
                    # Reset phase
                    self.selected_robot_idx = None
                    self.game_phase = "SELECT"
                else:
                    # Invalid move
                    # sfx: error.wav
                    pass

        terminated = self._check_termination()
        if terminated:
            if all(r['charged'] for r in self.robots):
                reward += 50 # Win bonus
                self.score += 50
                # sfx: level_win.wav
            else:
                # sfx: level_lose.wav
                pass
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_valid_move(self, robot, target_pos):
        # Check if target is adjacent
        dx = abs(robot["pos"][0] - target_pos[0])
        dy = abs(robot["pos"][1] - target_pos[1])
        if not (dx + dy == 1):
            return False

        # Check for walls
        if list(target_pos) in self.walls:
            return False
            
        # Check for other robots
        for other_robot in self.robots:
            if other_robot["id"] != robot["id"] and other_robot["pos"] == list(target_pos):
                return False
        
        return True

    def _check_termination(self):
        if self.game_over:
            return True
        
        win_condition = all(r['charged'] for r in self.robots)
        lose_condition = self.moves_left <= 0
        
        if win_condition or lose_condition:
            self.game_over = True
            return True
            
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "robots_charged": sum(1 for r in self.robots if r['charged'])
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_walls()
        self._render_stations()
        self._update_and_render_particles()
        self._render_robots()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, grid_pos):
        return (
            int(grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
            int(grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        )

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_walls(self):
        for wall_pos in self.walls:
            px, py = self._grid_to_pixel(wall_pos)
            rect = pygame.Rect(px - self.CELL_SIZE/2, py - self.CELL_SIZE/2, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_stations(self):
        for station in self.stations:
            px, py = self._grid_to_pixel(station["pos"])
            radius = int(self.CELL_SIZE * 0.4)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, (*self.COLOR_STATION, 50))
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_STATION)
            # Power symbol
            pts = [
                (px - 5, py), (px + 5, py - 8), (px, py), (px + 10, py), (px, py + 8)
            ]
            pygame.draw.lines(self.screen, self.COLOR_STATION, False, pts, 2)

    def _render_robots(self):
        for i, robot in enumerate(self.robots):
            px, py = self._grid_to_pixel(robot["pos"])
            radius = int(self.CELL_SIZE * 0.35)
            
            color = robot["color"]
            if robot["charged"]:
                color = self.COLOR_STATION
            
            # Glow effect
            glow_radius = int(radius * (1.5 + 0.2 * math.sin(pygame.time.get_ticks() / 200 + i)))
            if i == self.selected_robot_idx:
                glow_radius = int(radius * 2.0)
            glow_color = (*color, 20)
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, glow_color)
            
            # Robot body
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, tuple(np.clip(np.array(color) * 1.2, 0, 255)))
            
            # "Eye"
            eye_color = self.COLOR_BG
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(radius * 0.3), eye_color)

    def _render_cursor(self):
        px, py = self._grid_to_pixel(self.cursor_pos)
        size = self.CELL_SIZE
        rect = pygame.Rect(px - size/2, py - size/2, size, size)
        
        color = self.COLOR_CURSOR
        if self.game_phase == "MOVE":
            color = self.robots[self.selected_robot_idx]["color"]
            
        pygame.draw.rect(self.screen, (*color, 50), rect)
        pygame.draw.rect(self.screen, color, rect, 2)

    def _render_ui(self):
        # Moves Left
        text_str = f"Moves Left: {self.moves_left}"
        text_surf = self.font_main.render(text_str, True, self.COLOR_TEXT)
        shadow_surf = self.font_main.render(text_str, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (12, 12))
        self.screen.blit(text_surf, (10, 10))

        # Robots Charged
        charged_count = sum(1 for r in self.robots if r['charged'])
        text_str = f"Robots Charged: {charged_count}/{len(self.robots)}"
        text_surf = self.font_main.render(text_str, True, self.COLOR_TEXT)
        shadow_surf = self.font_main.render(text_str, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        shadow_rect = shadow_surf.get_rect(topright=(self.SCREEN_WIDTH - 8, 12))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if all(r['charged'] for r in self.robots):
                msg = "MISSION COMPLETE"
                color = self.COLOR_STATION
            else:
                msg = "OUT OF MOVES"
                color = self.COLOR_ROBOTS[0]
                
            text_surf = self.font_main.render(msg, True, color)
            shadow_surf = self.font_main.render(msg, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH/2 + 2, self.SCREEN_HEIGHT/2 + 2))
            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, color, count, direction="out"):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            if direction == "out":
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else: # in
                vel = [-math.cos(angle) * speed, -math.sin(angle) * speed]

            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "lifespan": random.randint(20, 40),
                "color": color
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["lifespan"] / 40))
                pygame.draw.circle(self.screen, (*p["color"], alpha), p["pos"], int(p["lifespan"] / 10))
                
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Rescue")
    clock = pygame.time.Clock()

    running = True
    while running:
        dpad, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: dpad = 1
        elif keys[pygame.K_DOWN]: dpad = 2
        elif keys[pygame.K_LEFT]: dpad = 3
        elif keys[pygame.K_RIGHT]: dpad = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [dpad, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Since auto_advance is False, we only step once per frame for manual play
        # The environment state only changes when we call step()
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        clock.tick(30) # Limit manual play to 30 FPS

    env.close()