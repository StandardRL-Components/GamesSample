
# Generated: 2025-08-28T02:20:50.881685
# Source Brief: brief_01679.md
# Brief Index: 1679

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓ to steer. Hold space to accelerate and shift to brake."
    )

    game_description = (
        "Race a sleek neon line against time and obstacles across a vibrant, "
        "procedurally generated track in this fast-paced, side-view arcade racer."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_TRACK = (255, 0, 255)  # Magenta
    COLOR_OBSTACLE = (0, 255, 255)  # Cyan
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_PLAYER_SLOW = (0, 100, 255) # Blue
    COLOR_PLAYER_FAST = (255, 50, 50)  # Red

    # Game Mechanics
    TRACK_Y_TOP = 100
    TRACK_Y_BOTTOM = 300
    TRACK_LENGTH = 6000
    LAPS_TO_WIN = 3
    MAX_STEPS = 5000
    OBSTACLES_PER_LAP = [3, 6, 9]

    # Player Physics
    PLAYER_STEER_SPEED = 6.0
    PLAYER_ACCELERATION = 0.5
    PLAYER_BRAKING = 1.0
    PLAYER_FRICTION = 0.98
    PLAYER_MAX_VELOCITY = 15.0
    PLAYER_MIN_VELOCITY = 1.0
    PLAYER_START_X = 150
    PLAYER_RADIUS = 8


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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_speed = pygame.font.Font(None, 36)

        self.player_y = 0
        self.player_world_x = 0
        self.player_velocity = 0
        self.camera_x = 0
        self.current_lap = 0
        self.total_time_steps = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        
        # This check is for development and ensures the implementation matches the spec
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_time_steps = 0
        self.current_lap = 1

        self.player_y = (self.TRACK_Y_TOP + self.TRACK_Y_BOTTOM) / 2
        self.player_world_x = 0
        self.player_velocity = self.PLAYER_MIN_VELOCITY

        self.camera_x = 0
        self.particles = []
        self._generate_all_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False

        # --- Action Handling ---
        movement = action[0]
        accelerate_held = action[1] == 1
        brake_held = action[2] == 1

        # Vertical steering
        if movement == 1:  # Up
            self.player_y -= self.PLAYER_STEER_SPEED
        elif movement == 2:  # Down
            self.player_y += self.PLAYER_STEER_SPEED
        
        self.player_y = np.clip(self.player_y, self.TRACK_Y_TOP + self.PLAYER_RADIUS, self.TRACK_Y_BOTTOM - self.PLAYER_RADIUS)

        # Velocity and particles
        if accelerate_held and not brake_held:
            self.player_velocity += self.PLAYER_ACCELERATION
            # # Sound: Engine hum / acceleration
            if self.np_random.random() < 0.5:
                self._create_particles(1, "accelerate")
        elif brake_held:
            self.player_velocity -= self.PLAYER_BRAKING
            # # Sound: Brake screech / sparks
            if self.np_random.random() < 0.8:
                self._create_particles(3, "brake")
            reward -= 0.2
        else: # Friction
            self.player_velocity *= self.PLAYER_FRICTION
        
        self.player_velocity = np.clip(self.player_velocity, self.PLAYER_MIN_VELOCITY, self.PLAYER_MAX_VELOCITY)

        # --- Game State Update ---
        self.player_world_x += self.player_velocity
        self.steps += 1
        self.total_time_steps += 1
        self._update_particles()

        # Reward for forward movement, proportional to speed
        reward += (self.player_velocity / self.PLAYER_MAX_VELOCITY) * 0.1

        # --- Collision & Lap Detection ---
        if self._check_collision():
            terminated = True
            self.game_over = True
            # # Sound: Explosion / Crash
            self._create_particles(50, "explode")

        lap_completed = self.player_world_x >= self.current_lap * self.TRACK_LENGTH
        if lap_completed and not terminated:
            # # Sound: Lap complete fanfare
            self.current_lap += 1
            reward += 5.0
            self.score += 5
            if self.current_lap > self.LAPS_TO_WIN:
                terminated = True
                self.game_over = True
                reward += 50.0  # Big bonus for winning
                self.score += 50

        # --- Termination ---
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "lap": self.current_lap,
            "velocity": self.player_velocity
        }

    def _render_game(self):
        # Update camera to follow player smoothly
        target_camera_x = self.player_world_x - self.PLAYER_START_X
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

        # --- Render Track ---
        self._draw_glowing_line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP), (self.SCREEN_WIDTH, self.TRACK_Y_TOP), 2, 4)
        self._draw_glowing_line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_BOTTOM), (self.SCREEN_WIDTH, self.TRACK_Y_BOTTOM), 2, 4)
        
        # --- Render Obstacles ---
        pulsation = (math.sin(self.total_time_steps * 0.1) + 1) / 2 # 0 to 1
        for ox, oy, o_size in self.obstacles:
            screen_x = ox - self.camera_x
            if -o_size < screen_x < self.SCREEN_WIDTH + o_size:
                pulse_radius = o_size + pulsation * 4
                self._draw_glowing_poly(self.screen, self.COLOR_OBSTACLE, screen_x, oy, pulse_radius, 3, self.total_time_steps * 0.02)

        # --- Render Particles ---
        for p in self.particles:
            screen_x = p['pos'][0] - self.camera_x
            alpha = p['life'] / p['max_life']
            color = (*p['color'], int(alpha * 255))
            pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(p['pos'][1]), int(p['size']), color)

        # --- Render Player ---
        speed_ratio = (self.player_velocity - self.PLAYER_MIN_VELOCITY) / (self.PLAYER_MAX_VELOCITY - self.PLAYER_MIN_VELOCITY)
        player_color = tuple(int(c1 + (c2 - c1) * speed_ratio) for c1, c2 in zip(self.COLOR_PLAYER_SLOW, self.COLOR_PLAYER_FAST))
        self._draw_glowing_circle(self.screen, player_color, (self.PLAYER_START_X, self.player_y), self.PLAYER_RADIUS, 10)

    def _render_ui(self):
        # Lap Counter
        lap_text = f"LAP: {min(self.current_lap, self.LAPS_TO_WIN)} / {self.LAPS_TO_WIN}"
        lap_surf = self.font_ui.render(lap_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_surf, (15, 15))

        # Timer
        time_seconds = self.total_time_steps / self.TARGET_FPS
        time_text = f"TIME: {time_seconds:.2f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 15, 15))

        # Speed
        speed_text = f"{int(self.player_velocity * 10)} KPH"
        speed_surf = self.font_speed.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_surf, (self.SCREEN_WIDTH / 2 - speed_surf.get_width() / 2, self.SCREEN_HEIGHT - 40))

    def _generate_all_obstacles(self):
        self.obstacles = []
        for i, num in enumerate(self.OBSTACLES_PER_LAP):
            lap_idx = i + 1
            self._generate_obstacles_for_lap(num, lap_idx)
    
    def _generate_obstacles_for_lap(self, num_obstacles, lap_index):
        track_height = self.TRACK_Y_BOTTOM - self.TRACK_Y_TOP
        start_x = (lap_index - 1) * self.TRACK_LENGTH
        
        # Ensure first part of the track is clear
        safe_zone = 500
        
        for _ in range(num_obstacles):
            x = start_x + safe_zone + self.np_random.random() * (self.TRACK_LENGTH - safe_zone - 200)
            y = self.TRACK_Y_TOP + self.np_random.random() * track_height
            size = self.np_random.integers(10, 16)
            self.obstacles.append((x, y, size))

    def _check_collision(self):
        player_pos = np.array([self.PLAYER_START_X, self.player_y])
        for ox, oy, o_size in self.obstacles:
            screen_x = ox - self.camera_x
            if abs(screen_x - self.PLAYER_START_X) < self.PLAYER_RADIUS + o_size:
                dist = np.linalg.norm(player_pos - np.array([screen_x, oy]))
                if dist < self.PLAYER_RADIUS + o_size:
                    return True
        return False

    def _create_particles(self, count, p_type):
        for _ in range(count):
            if p_type == "accelerate":
                particle = {
                    'pos': [self.player_world_x - self.PLAYER_RADIUS, self.player_y + self.np_random.uniform(-5, 5)],
                    'vel': [-self.player_velocity / 2, self.np_random.uniform(-0.5, 0.5)],
                    'life': self.np_random.uniform(10, 20),
                    'max_life': 20,
                    'color': self.COLOR_PLAYER_SLOW,
                    'size': self.np_random.uniform(1, 3)
                }
            elif p_type == "brake":
                particle = {
                    'pos': [self.player_world_x + self.PLAYER_RADIUS, self.player_y + self.np_random.uniform(-self.PLAYER_RADIUS, self.PLAYER_RADIUS)],
                    'vel': [self.player_velocity / 2 + self.np_random.uniform(0, 3), self.np_random.uniform(-2, 2)],
                    'life': self.np_random.uniform(15, 25),
                    'max_life': 25,
                    'color': (255, 200, 0),
                    'size': self.np_random.uniform(1, 4)
                }
            elif p_type == "explode":
                 angle = self.np_random.uniform(0, 2 * math.pi)
                 speed = self.np_random.uniform(2, 10)
                 particle = {
                    'pos': [self.player_world_x, self.player_y],
                    'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'life': self.np_random.uniform(20, 40),
                    'max_life': 40,
                    'color': self.np_random.choice([self.COLOR_OBSTACLE, (255,255,0), (255,100,0)]),
                    'size': self.np_random.uniform(2, 5)
                }
            self.particles.append(particle)
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] *= 0.98
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _draw_glowing_circle(self, surface, color, center, radius, glow_width):
        center_int = (int(center[0]), int(center[1]))
        for i in range(glow_width, 0, -1):
            alpha = int(100 * (1 - i / glow_width))
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius + i), glow_color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)

    def _draw_glowing_line(self, surface, color, start, end, width, glow_width):
        for i in range(glow_width, -1, -1):
            alpha = int(80 * (1 - (i / glow_width)))
            pygame.draw.line(surface, (*color, alpha), start, end, width + i * 2)
        pygame.draw.aaline(surface, color, start, end)

    def _draw_glowing_poly(self, surface, color, cx, cy, radius, num_points, rotation):
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points) + rotation
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius
            points.append((int(x), int(y)))
        
        for i in range(5, 0, -1):
            alpha = int(100 * (1 - i / 5))
            glow_color = (*color, alpha)
            scaled_points = []
            for p in points:
                scaled_x = cx + (p[0] - cx) * (1 + i * 0.1)
                scaled_y = cy + (p[1] - cy) * (1 + i * 0.1)
                scaled_points.append((scaled_x, scaled_y))
            if len(scaled_points) > 2:
                 pygame.gfxdraw.aapolygon(surface, scaled_points, glow_color)

        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # For interactive play, we need to set the video driver properly
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # --- Interactive Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for display
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Racer")
    clock = pygame.time.Clock()

    while not terminated:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(GameEnv.TARGET_FPS)

    print(f"Game Over. Final Info: {info}")
    env.close()