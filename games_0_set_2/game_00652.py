
# Generated: 2025-08-27T14:21:01.535982
# Source Brief: brief_00652.md
# Brief Index: 652

        
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
    """
    An arcade physics puzzle game where the player draws a track in real-time
    to guide a physics-based rider to a finish line. The environment is designed
    for visual quality and satisfying gameplay feel, with a minimalist, high-contrast
    aesthetic.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to draw track. Hold Space for steeper inclines, Shift for steeper declines."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track in real-time to guide a physics-based rider to the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and World Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FINISH_LINE_X = 2500

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (35, 40, 60)
        self.COLOR_TRACK = (240, 240, 240)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_RIDER_GLOW = (200, 200, 255)
        self.COLOR_START = (0, 255, 128)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 255, 255)

        # Physics and Game Constants
        self.GRAVITY = pygame.Vector2(0, 0.25)
        self.RIDER_RADIUS = 10
        self.DRAW_LENGTH = 25
        self.MAX_TRACK_POINTS = 200
        self.MAX_STEPS = 1500
        self.SOFT_LOCK_STEPS = 300
        
        # State variables (initialized in reset)
        self.rider = None
        self.track_points = None
        self.particles = None
        self.camera_offset_x = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_progress_x = None
        self.steps_since_progress = None
        self.rng = None
        
        # Initialize state variables
        self.reset()

        # self.validate_implementation() # For development; comment out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.camera_offset_x = 0.0

        start_y = self.SCREEN_HEIGHT / 2
        self.rider = {
            "pos": pygame.Vector2(100, start_y - 100),
            "vel": pygame.Vector2(2, 0),
        }
        self.last_progress_x = self.rider["pos"].x
        self.steps_since_progress = 0

        self.track_points = [
            pygame.Vector2(0, start_y),
            pygame.Vector2(200, start_y)
        ]
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_drawing(action)
        self._update_rider_physics()
        self._update_particles()
        self._update_camera()

        self.steps += 1
        reward, terminated = self._calculate_reward_and_termination()
        self.score += reward
        self.game_over = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_drawing(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        last_point = self.track_points[-1]

        draw_horizon = self.rider["pos"].x + self.SCREEN_WIDTH * 0.75
        if last_point.x > draw_horizon:
            return

        direction = pygame.Vector2(0, 0)
        angle_mod = 0
        if space_held: angle_mod = -30
        if shift_held: angle_mod = 30

        if movement == 0: # None/Extend right
            direction.x = 1
        elif movement == 1: # Up
            direction = pygame.Vector2(1, -1).rotate(angle_mod)
        elif movement == 2: # Down
            direction = pygame.Vector2(1, 1).rotate(angle_mod)
        elif movement == 3: # Left
            direction.x = -0.5
        elif movement == 4: # Right
            direction.x = 1

        if direction.length_squared() > 0:
            new_point = last_point + direction.normalize() * self.DRAW_LENGTH
            new_point.y = np.clip(new_point.y, 0, self.SCREEN_HEIGHT)
            new_point.x = max(new_point.x, last_point.x + 1)
            self.track_points.append(new_point)

        min_x = self.camera_offset_x - 50
        while len(self.track_points) > 2 and self.track_points[1].x < min_x:
            self.track_points.pop(0)

    def _update_rider_physics(self):
        self.rider["vel"] += self.GRAVITY
        
        collided = False
        for i in range(len(self.track_points) - 1):
            p1, p2 = self.track_points[i], self.track_points[i+1]
            
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            point_vec = self.rider["pos"] - p1
            t = point_vec.dot(line_vec) / line_vec.length_squared()
            t = np.clip(t, 0, 1)
            
            closest_point = p1 + t * line_vec
            dist_vec = self.rider["pos"] - closest_point
            
            if dist_vec.length_squared() < self.RIDER_RADIUS ** 2:
                collided = True
                dist = dist_vec.length()
                penetration = self.RIDER_RADIUS - dist
                normal = dist_vec.normalize() if dist > 0 else pygame.Vector2(0, -1)
                
                self.rider["pos"] += normal * penetration
                
                # Reflect velocity and apply restitution/friction
                self.rider["vel"] = self.rider["vel"].reflect(normal) * 0.8
                # play_sound("scrape.wav")
                
                for _ in range(3):
                    self._create_particle(self.rider["pos"], normal)
                break

        self.rider["pos"] += self.rider["vel"]
        if not collided:
            self.rider["vel"] *= 0.998 # Air drag

    def _create_particle(self, pos, collision_normal):
        p_vel = pygame.Vector2(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1))
        p_vel += collision_normal * self.rng.uniform(1, 3)
        self.particles.append({
            "pos": pos.copy(), "vel": p_vel,
            "life": self.rng.integers(10, 20), "radius": self.rng.uniform(1, 3)
        })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _update_camera(self):
        target_cam_x = self.rider["pos"].x - self.SCREEN_WIDTH / 4
        self.camera_offset_x = self.camera_offset_x * 0.9 + target_cam_x * 0.1

    def _calculate_reward_and_termination(self):
        reward = 0.0
        terminated = False

        if self.rider["pos"].x >= self.FINISH_LINE_X:
            reward = 100.0 - (0.01 * self.steps)
            terminated = True
            return reward, terminated

        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -10.0
        elif not (0 < self.rider["pos"].y < self.SCREEN_HEIGHT + 50):
            terminated = True
            reward = -100.0
        
        if self.rider["pos"].x > self.last_progress_x + 1:
            self.last_progress_x = self.rider["pos"].x
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
        
        if self.steps_since_progress > self.SOFT_LOCK_STEPS:
            terminated = True
            reward = -50.0

        if not terminated:
            reward += np.clip(self.rider["vel"].x * 0.1, -0.1, 0.5)
            reward -= 0.01

        return reward, terminated

    def _world_to_screen(self, pos):
        return int(pos.x - self.camera_offset_x), int(pos.y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_size = 50
        start_x = int(-self.camera_offset_x % grid_size)
        for x in range(start_x, self.SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        start_screen_pos = self._world_to_screen(pygame.Vector2(0, 0))
        if start_screen_pos[0] > 0:
            pygame.draw.line(self.screen, self.COLOR_START, (start_screen_pos[0], 0), (start_screen_pos[0], self.SCREEN_HEIGHT), 3)

        finish_screen_pos = self._world_to_screen(pygame.Vector2(self.FINISH_LINE_X, 0))
        if 0 < finish_screen_pos[0] < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_pos[0], 0), (finish_screen_pos[0], self.SCREEN_HEIGHT), 5)

        if len(self.track_points) > 1:
            screen_points = [self._world_to_screen(p) for p in self.track_points]
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, screen_points, 5)
            for p in screen_points:
                pygame.gfxdraw.aacircle(self.screen, p[0], p[1], 2, self.COLOR_TRACK)
                pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 2, self.COLOR_TRACK)

        rider_screen_pos = self._world_to_screen(self.rider["pos"])
        glow_radius = int(self.RIDER_RADIUS * 1.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_RIDER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (rider_screen_pos[0] - glow_radius, rider_screen_pos[1] - glow_radius))

        pygame.gfxdraw.aacircle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)

        for p in self.particles:
            p_screen_pos = self._world_to_screen(p["pos"])
            alpha = int(255 * (p["life"] / 20.0))
            color = (*self.COLOR_PARTICLE, alpha)
            s = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (int(p["radius"]), int(p["radius"])), int(p["radius"]))
            self.screen.blit(s, (p_screen_pos[0] - int(p["radius"]), p_screen_pos[1] - int(p["radius"])))

    def _render_ui(self):
        time_text = self.font.render(f"Time: {self.steps}", True, self.COLOR_UI)
        self.screen.blit(time_text, (10, 10))

        score_text = self.font.render(f"Score: {int(self.score)}", True, self.COLOR_UI)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        progress = self.rider["pos"].x / self.FINISH_LINE_X
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 10
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.SCREEN_HEIGHT - 20, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_START, (10, self.SCREEN_HEIGHT - 20, max(0, int(bar_width * progress)), bar_height))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos_x": self.rider["pos"].x,
            "rider_pos_y": self.rider["pos"].y,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Track Rider")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    movement, space_held, shift_held = 0, 0, 0
    
    print("-" * 30)
    print("Track Rider - Human Play Mode")
    print(env.game_description)
    print(env.user_guide)
    print("Press 'R' to reset.")
    print("-" * 30)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift_held = 1
                elif event.key == pygame.K_r: terminated = True

            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    if (event.key == pygame.K_UP and movement == 1) or \
                       (event.key == pygame.K_DOWN and movement == 2) or \
                       (event.key == pygame.K_LEFT and movement == 3) or \
                       (event.key == pygame.K_RIGHT and movement == 4):
                        movement = 0
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift_held = 0

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}. Resetting...")
            pygame.time.wait(1000)
            obs, info = env.reset()
            terminated = False
            movement, space_held, shift_held = 0, 0, 0
            continue

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30)