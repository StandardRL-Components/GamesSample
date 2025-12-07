
# Generated: 2025-08-28T04:53:54.871638
# Source Brief: brief_05398.md
# Brief Index: 5398

        
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
        "Controls: ←→ to steer. Hold space for a speed boost, hold shift to brake."
    )

    game_description = (
        "Race through a neon-drenched procedural track. Dodge obstacles, complete three laps, "
        "and set a record time in this fast-paced arcade racer."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000  # Extended for 3 laps

        # Game constants
        self.LAP_LENGTH = 10000
        self.NUM_LAPS = 3
        self.TRACK_WIDTH = 250
        self.PLAYER_Y_POS = self.HEIGHT * 0.8
        self.INITIAL_SPEED = 6.0
        self.SPEED_INCREMENT_PER_LAP = 1.0

        # Player physics
        self.PLAYER_ACCEL = 1.2
        self.PLAYER_FRICTION = 0.90
        self.PLAYER_MAX_VX = 12.0
        self.BOOST_MULTIPLIER = 1.6
        self.BRAKE_MULTIPLIER = 0.6

        # Colors (Neon/Tron aesthetic)
        self.COLOR_BG = (5, 0, 15)
        self.COLOR_GRID = (20, 5, 40)
        self.COLOR_PLAYER = (0, 192, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 150)
        self.COLOR_TRACK = (180, 0, 255)
        self.COLOR_TRACK_GLOW = (80, 0, 120)
        self.COLOR_OBSTACLE = (255, 20, 20)
        self.COLOR_OBSTACLE_GLOW = (150, 10, 10)
        self.COLOR_FINISH = (0, 255, 100)
        self.COLOR_FINISH_GLOW = (0, 150, 60)
        self.COLOR_TEXT = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.Font(None, 28)
        self.end_font = pygame.font.Font(None, 64)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laps_completed = 0
        self.track_progress = 0.0
        self.player_x_offset = 0.0
        self.player_vx = 0.0
        self.base_speed = 0.0
        self.particles = []
        self.obstacles = []
        self.track_nodes = []
        self.finish_line_visible = False
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laps_completed = 0
        self.track_progress = 0.0
        self.player_x_offset = 0.0
        self.player_vx = 0.0
        self.base_speed = self.INITIAL_SPEED
        self.particles = []
        self.finish_line_visible = False

        self._generate_track()
        self._populate_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        if not self.game_over:
            # Base survival reward
            reward += 0.01

            # Unpack actions
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            # Update player physics
            self._handle_input(movement)
            self._update_player_position()

            # Update world state
            scroll_speed = self.base_speed
            if space_held:
                scroll_speed *= self.BOOST_MULTIPLIER
                # Add engine boost particles
                if self.np_random.random() < 0.8:
                    self._create_player_trail(2, (255, 200, 0))
            if shift_held:
                scroll_speed *= self.BRAKE_MULTIPLIER
            self.track_progress += scroll_speed

            # Add regular engine trail
            if self.np_random.random() < 0.5:
                self._create_player_trail(1, self.COLOR_PLAYER)

            # Check for events (collisions, laps)
            event_reward, terminated_by_event = self._check_events()
            reward += event_reward
            if terminated_by_event:
                self.game_over = True
        
        self._update_particles()
        self.steps += 1
        self.score += reward

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.player_vx -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vx += self.PLAYER_ACCEL

    def _update_player_position(self):
        self.player_vx *= self.PLAYER_FRICTION
        self.player_vx = np.clip(self.player_vx, -self.PLAYER_MAX_VX, self.PLAYER_MAX_VX)
        self.player_x_offset += self.player_vx

        # Prevent player from going too far off-center
        max_offset = self.WIDTH / 2
        self.player_x_offset = np.clip(self.player_x_offset, -max_offset, max_offset)

    def _check_events(self):
        reward = 0
        terminated = False
        player_world_y = self.track_progress + self.PLAYER_Y_POS
        player_screen_x = self.WIDTH / 2 + self.player_x_offset
        player_rect = pygame.Rect(player_screen_x - 3, self.PLAYER_Y_POS, 6, 20)

        # --- Obstacle Interaction ---
        for obs in self.obstacles:
            if not obs["passed"] and obs["y"] < player_world_y:
                # Reward for passing an obstacle
                reward += 1.0
                obs["passed"] = True
            
            # Collision check
            obs_screen_y = self.PLAYER_Y_POS - (obs["y"] - self.track_progress)
            track_center_at_obs = self._get_track_center_x_at(obs["y"])
            obs_screen_x = track_center_at_obs + obs["offset"]
            
            if abs(obs_screen_y - self.PLAYER_Y_POS) < 50: # Optimization
                obs_rect = pygame.Rect(obs_screen_x - obs["w"]/2, obs_screen_y - obs["h"]/2, obs["w"], obs["h"])
                if player_rect.colliderect(obs_rect):
                    reward = -100.0 # Collision penalty
                    terminated = True
                    self._create_explosion(player_rect.center, 50, self.COLOR_OBSTACLE)
                    # sfx: explosion
                    return reward, terminated

        # --- Track Wall Collision ---
        track_center_at_player = self._get_track_center_x_at(player_world_y)
        half_track = self.TRACK_WIDTH / 2
        if not (track_center_at_player - half_track < player_screen_x < track_center_at_player + half_track):
            reward = -100.0 # Wall collision penalty
            terminated = True
            self._create_explosion(player_rect.center, 50, self.COLOR_TRACK)
            # sfx: crash
            return reward, terminated

        # --- Lap Completion ---
        if self.track_progress >= (self.laps_completed + 1) * self.LAP_LENGTH:
            self.laps_completed += 1
            self.base_speed += self.SPEED_INCREMENT_PER_LAP
            if self.laps_completed >= self.NUM_LAPS:
                reward += 100.0  # Win reward
                terminated = True
            else:
                reward += 50.0  # Lap reward
            # sfx: lap_complete
        
        self.finish_line_visible = (self.laps_completed + 1) * self.LAP_LENGTH - self.track_progress < self.HEIGHT

        return reward, terminated

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] - p["decay"])

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        self._render_grid()

        # --- Game Elements ---
        self._render_track()
        if self.finish_line_visible:
            self._render_finish_line()
        self._render_obstacles()
        self._render_particles()
        if not self.game_over:
            self._render_player()

        # --- UI ---
        self._render_ui()
        if self.game_over:
            self._render_end_message()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        offset = (self.track_progress * 0.2) % 40
        for i in range(self.HEIGHT // 40 + 2):
            y = i * 40 - offset
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_track(self):
        points_left, points_right = [], []
        for y_screen in range(0, self.HEIGHT + 20, 20):
            world_y = self.track_progress + y_screen
            center_x = self._get_track_center_x_at(world_y)
            points_left.append((center_x - self.TRACK_WIDTH / 2, y_screen))
            points_right.append((center_x + self.TRACK_WIDTH / 2, y_screen))
        
        if len(points_left) > 1:
            self._draw_glow_line(self.screen, self.COLOR_TRACK, self.COLOR_TRACK_GLOW, points_left, 12, 3)
            self._draw_glow_line(self.screen, self.COLOR_TRACK, self.COLOR_TRACK_GLOW, points_right, 12, 3)

    def _render_finish_line(self):
        lap_end_y = (self.laps_completed + 1) * self.LAP_LENGTH
        finish_screen_y = self.PLAYER_Y_POS - (lap_end_y - self.track_progress)
        
        if 0 < finish_screen_y < self.HEIGHT:
            track_center = self._get_track_center_x_at(lap_end_y)
            x1 = track_center - self.TRACK_WIDTH / 2
            x2 = track_center + self.TRACK_WIDTH / 2
            
            # Draw checkered pattern
            is_white = True
            for x in range(int(x1), int(x2), 20):
                color = (255, 255, 255) if is_white else self.COLOR_FINISH
                pygame.draw.rect(self.screen, color, (x, finish_screen_y - 5, 20, 10))
                is_white = not is_white

    def _render_obstacles(self):
        for obs in self.obstacles:
            obs_screen_y = self.PLAYER_Y_POS - (obs["y"] - self.track_progress)
            if -50 < obs_screen_y < self.HEIGHT + 50:
                track_center_at_obs = self._get_track_center_x_at(obs["y"])
                obs_screen_x = track_center_at_obs + obs["offset"]
                
                points = [
                    (obs_screen_x - obs["w"]/2, obs_screen_y - obs["h"]/2),
                    (obs_screen_x + obs["w"]/2, obs_screen_y - obs["h"]/2),
                    (obs_screen_x + obs["w"]/2, obs_screen_y + obs["h"]/2),
                    (obs_screen_x - obs["w"]/2, obs_screen_y + obs["h"]/2),
                ]
                
                # Glow effect
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE_GLOW)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE_GLOW)
                # Core shape
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)

    def _render_player(self):
        x = self.WIDTH / 2 + self.player_x_offset
        y = self.PLAYER_Y_POS
        player_points = [(x, y - 10), (x - 4, y + 10), (x + 4, y + 10)]
        
        # Glow
        glow_points = [(x, y - 15), (x - 8, y + 15), (x + 8, y + 15)]
        pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        # Core
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            if p["radius"] > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["x"]), int(p["y"]), int(p["radius"]), p["color"]
                )

    def _render_ui(self):
        lap_text = f"LAP: {min(self.laps_completed + 1, self.NUM_LAPS)} / {self.NUM_LAPS}"
        speed_val = self.base_speed * (self.BOOST_MULTIPLIER if self.action_space.sample()[1] else 1.0)
        speed_text = f"SPEED: {int(speed_val * 10)}"
        
        lap_surf = self.ui_font.render(lap_text, True, self.COLOR_TEXT)
        speed_surf = self.ui_font.render(speed_text, True, self.COLOR_TEXT)
        
        self.screen.blit(lap_surf, (10, 10))
        self.screen.blit(speed_surf, (self.WIDTH - speed_surf.get_width() - 10, 10))

    def _render_end_message(self):
        if self.laps_completed >= self.NUM_LAPS:
            msg = "FINISH!"
            color = self.COLOR_FINISH
        else:
            msg = "GAME OVER"
            color = self.COLOR_OBSTACLE
        
        text_surf = self.end_font.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.laps_completed + 1,
        }

    def _generate_track(self):
        self.track_nodes = []
        y = 0
        offset = 0
        amplitude = self.np_random.uniform(50, 150)
        frequency = self.np_random.uniform(0.0002, 0.0005)
        total_len = self.LAP_LENGTH * self.NUM_LAPS + self.HEIGHT
        while y < total_len:
            offset = amplitude * math.sin(y * frequency)
            self.track_nodes.append({"y": y, "offset": offset})
            y += 50  # Node density

    def _populate_obstacles(self):
        self.obstacles = []
        for y in range(500, self.LAP_LENGTH * self.NUM_LAPS, int(self.np_random.uniform(150, 400))):
            side = self.np_random.choice([-1, 1])
            offset = side * self.np_random.uniform(self.TRACK_WIDTH * 0.1, self.TRACK_WIDTH * 0.4)
            self.obstacles.append({
                "y": y,
                "offset": offset,
                "w": self.np_random.uniform(20, 40),
                "h": self.np_random.uniform(20, 40),
                "passed": False,
            })

    def _get_track_center_x_at(self, y):
        # Find two closest nodes and interpolate
        if not self.track_nodes: return self.WIDTH / 2
        
        # Simple linear search, can be optimized with binary search if needed
        node1 = self.track_nodes[0]
        node2 = self.track_nodes[0]
        for node in self.track_nodes:
            if node["y"] <= y:
                node1 = node
            if node["y"] > y:
                node2 = node
                break
        
        if node1 == node2:
            return self.WIDTH / 2 + node1["offset"]

        # Interpolate
        dy = node2["y"] - node1["y"]
        if dy == 0: return self.WIDTH / 2 + node1["offset"]
        
        p = (y - node1["y"]) / dy
        interp_offset = node1["offset"] + p * (node2["offset"] - node1["offset"])
        return self.WIDTH / 2 + interp_offset

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            self.particles.append({
                "x": pos[0], "y": pos[1],
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "radius": self.np_random.uniform(3, 8),
                "life": self.np_random.integers(20, 40),
                "decay": 0.2,
                "color": color,
            })

    def _create_player_trail(self, count, color):
        for _ in range(count):
            self.particles.append({
                "x": self.WIDTH/2 + self.player_x_offset + self.np_random.uniform(-3, 3),
                "y": self.PLAYER_Y_POS + 10,
                "vx": self.np_random.uniform(-0.5, 0.5), "vy": self.np_random.uniform(0.5, 2),
                "radius": self.np_random.uniform(2, 4),
                "life": self.np_random.integers(10, 20),
                "decay": 0.25,
                "color": color,
            })

    def _draw_glow_line(self, surface, color, glow_color, points, glow_width, core_width):
        # Antialiasing is expensive, so we use wide lines for a similar effect
        pygame.draw.lines(surface, glow_color, False, points, glow_width)
        pygame.draw.lines(surface, color, False, points, core_width)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Use a display for manual play
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1 # Not used but available
        if keys[pygame.K_DOWN]: movement = 2 # Not used but available
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Render to display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()