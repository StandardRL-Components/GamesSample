
# Generated: 2025-08-27T22:33:51.067884
# Source Brief: brief_03164.md
# Brief Index: 3164

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to set the angle of the next track segment. "
        "Hold Space for an upward curve or Shift for a downward curve. "
        "A new segment is drawn automatically as the rider nears the end of the track."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a procedurally generated track for a physics-based rider to navigate to the finish line. "
        "Plan your track segments to build momentum, clear gaps, and reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 100.0, 62.5  # 100m track
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 50)
    COLOR_TRACK = (255, 255, 255)
    COLOR_RIDER = (255, 50, 50)
    COLOR_RIDER_GLOW = (255, 50, 50)
    COLOR_START_FINISH = (50, 200, 50)
    COLOR_PARTICLE = (255, 150, 0)
    COLOR_UI_TEXT = (220, 220, 220)

    # Physics
    GRAVITY = pygame.math.Vector2(0, -35.0)
    RIDER_RADIUS_WORLD = 1.0
    FRICTION = 0.02
    BOUNCE = 0.1
    INITIAL_VELOCITY = pygame.math.Vector2(10.0, 0)

    # Gameplay
    DRAW_LOOKAHEAD = 20.0  # How far ahead of the track end the rider must be to trigger a new segment
    SEGMENT_LENGTH = 4.0

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)

        self.scale_x = self.SCREEN_WIDTH / self.WORLD_WIDTH
        self.scale_y = self.SCREEN_HEIGHT / self.WORLD_HEIGHT
        self.rider_radius_screen = int(self.RIDER_RADIUS_WORLD * self.scale_y)
        
        self.attempt_number = 0
        self.game_state_initialized = False # Defer state init until first reset() is called
        
        # Attributes to be initialized in reset()
        self.rider_pos = pygame.math.Vector2(0, 0)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.track_points = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_rider_x = 0.0
        self.start_line_x = 0.0
        self.finish_line_x = 0.0

    def _world_to_screen(self, pos: pygame.math.Vector2) -> tuple[int, int]:
        x = int(pos.x * self.scale_x)
        y = int(self.SCREEN_HEIGHT - (pos.y * self.scale_y))
        return x, y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game_state_initialized = True
        self.attempt_number += 1

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        start_pos = pygame.math.Vector2(5.0, 15.0)
        self.rider_pos.update(start_pos)
        self.rider_vel.update(self.INITIAL_VELOCITY)
        self.last_rider_x = self.rider_pos.x

        self.track_points = [
            pygame.math.Vector2(0, start_pos.y - self.RIDER_RADIUS_WORLD),
            pygame.math.Vector2(start_pos.x + self.DRAW_LOOKAHEAD, start_pos.y - self.RIDER_RADIUS_WORLD)
        ]

        self.particles = []
        
        # Start/Finish line positions in world coordinates
        self.start_line_x = start_pos.x
        self.finish_line_x = self.WORLD_WIDTH - 5.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if not self.game_state_initialized:
            self.reset()

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self._update_physics()
        self._handle_input_and_drawing(action)
        self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        reward = self._calculate_reward(terminated)
        self.score += reward
        
        if terminated:
            self.game_over = True
            if not (self.rider_pos.x >= self.finish_line_x):
                 self._create_crash_particles()
                 # sfx: crash
            else:
                 pass # sfx: victory fanfare

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input_and_drawing(self, action):
        last_track_point = self.track_points[-1]
        if self.rider_pos.x > last_track_point.x - self.DRAW_LOOKAHEAD:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            angle = 0.0
            # Action priority: space > shift > movement
            if space_held: # Upward curve
                angle = math.pi / 4 # 45 degrees
                # sfx: draw_curve_up
            elif shift_held: # Downward curve
                angle = -math.pi / 4 # -45 degrees
                # sfx: draw_curve_down
            else:
                # sfx: draw_line
                if movement == 0: angle = 0.0 # Horizontal
                elif movement == 1: angle = math.pi / 6 # 30 deg up
                elif movement == 2: angle = -math.pi / 6 # 30 deg down
                elif movement == 3: angle = math.pi / 2.5 # Steep up (72 deg)
                elif movement == 4: angle = 0.05 # Slight down (for speed)

            length = self.SEGMENT_LENGTH * 1.5 if movement == 4 else self.SEGMENT_LENGTH

            new_point = last_track_point + pygame.math.Vector2(
                length * math.cos(angle),
                length * math.sin(angle)
            )

            new_point.x = max(0, min(self.WORLD_WIDTH, new_point.x))
            new_point.y = max(0.1, min(self.WORLD_HEIGHT - 0.1, new_point.y))
            
            if new_point.x > last_track_point.x + 0.1:
                self.track_points.append(new_point)

            if len(self.track_points) > 2 and self.track_points[1].x < self.rider_pos.x - self.DRAW_LOOKAHEAD * 2:
                self.track_points.pop(0)

    def _update_physics(self):
        dt = 1.0 / self.FPS
        
        self.rider_vel += self.GRAVITY * dt
        self.rider_pos += self.rider_vel * dt

        for i in range(len(self.track_points) - 1):
            p1, p2 = self.track_points[i], self.track_points[i+1]

            if p1.x <= self.rider_pos.x < p2.x:
                if abs(p2.x - p1.x) < 1e-6: continue
                t = (self.rider_pos.x - p1.x) / (p2.x - p1.x)
                track_y = p1.y + t * (p2.y - p1.y)

                if self.rider_pos.y < track_y + self.RIDER_RADIUS_WORLD:
                    self.rider_pos.y = track_y + self.RIDER_RADIUS_WORLD
                    
                    segment_vec = (p2 - p1)
                    if segment_vec.length_squared() == 0: continue
                    
                    normal_vec = pygame.math.Vector2(-segment_vec.y, segment_vec.x).normalize()
                    dot_product = self.rider_vel.dot(normal_vec)
                    
                    if dot_product < 0:
                        self.rider_vel.reflect_ip(normal_vec)
                        self.rider_vel *= (1 - self.BOUNCE)

                        tangent_vec = segment_vec.normalize()
                        friction_force = self.rider_vel.dot(tangent_vec) * self.FRICTION
                        self.rider_vel -= tangent_vec * friction_force
                break

    def _update_particles(self):
        dt = 1.0 / self.FPS
        self.particles = [
            (p[0] + p[1] * dt, p[1] + self.GRAVITY * 0.5 * dt, p[2] - 255 * dt * 0.8)
            for p in self.particles if p[2] > 0
        ]
        
    def _create_crash_particles(self):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(5, 20)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append((pygame.math.Vector2(self.rider_pos), vel, 255.0))
            
    def _calculate_reward(self, terminated: bool) -> float:
        reward = 0.0
        
        progress = self.rider_pos.x - self.last_rider_x
        reward += progress * 0.1
        self.last_rider_x = self.rider_pos.x

        if terminated:
            if self.rider_pos.x >= self.finish_line_x:
                reward += 100.0
                reward -= 0.02 * self.steps
            elif self.steps >= self.MAX_STEPS:
                 reward -= 5.0
            else:
                reward -= 10.0
        return reward

    def _check_termination(self) -> bool:
        win = self.rider_pos.x >= self.finish_line_x
        crash = not (0 <= self.rider_pos.y <= self.WORLD_HEIGHT) or not (0 <= self.rider_pos.x <= self.WORLD_WIDTH)
        timeout = self.steps >= self.MAX_STEPS
        return win or crash or timeout

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for i in range(0, self.SCREEN_WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        start_screen_x, _ = self._world_to_screen(pygame.math.Vector2(self.start_line_x, 0))
        finish_screen_x, _ = self._world_to_screen(pygame.math.Vector2(self.finish_line_x, 0))
        pygame.draw.line(self.screen, self.COLOR_START_FINISH, (start_screen_x, 0), (start_screen_x, self.SCREEN_HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_START_FINISH, (finish_screen_x, 0), (finish_screen_x, self.SCREEN_HEIGHT), 3)

        if len(self.track_points) > 1:
            screen_points = [self._world_to_screen(p) for p in self.track_points]
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, screen_points, 2)

        if self.game_state_initialized:
            rider_screen_pos = self._world_to_screen(self.rider_pos)
            glow_radius = int(self.rider_radius_screen * 2.5)
            if glow_radius > 0:
                glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*self.COLOR_RIDER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surface, (rider_screen_pos[0] - glow_radius, rider_screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            pygame.gfxdraw.aacircle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.rider_radius_screen, self.COLOR_RIDER)
            pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.rider_radius_screen, self.COLOR_RIDER)

        for pos, vel, alpha in self.particles:
            screen_pos = self._world_to_screen(pos)
            color = (*self.COLOR_PARTICLE, int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], 2, color)

    def _render_ui(self):
        attempt_text = self.font_ui.render(f"ATTEMPT: {self.attempt_number}", True, self.COLOR_UI_TEXT)
        self.screen.blit(attempt_text, (10, 10))

        distance = (self.last_rider_x / self.finish_line_x * 100) if self.finish_line_x > 0 else 0
        dist_text = self.font_ui.render(f"DIST: {max(0, distance):.1f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_text, (self.SCREEN_WIDTH - dist_text.get_width() - 10, 10))

        time_s = self.steps / self.FPS
        time_text = self.font_ui.render(f"TIME: {time_s:.2f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 - time_text.get_width() // 2, 10))

        if self.game_over:
            if self.rider_pos.x >= self.finish_line_x:
                msg, color = "FINISH!", self.COLOR_START_FINISH
            else:
                msg, color = "CRASHED!", self.COLOR_RIDER
            
            end_text = self.font_msg.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_percent": (self.last_rider_x / self.finish_line_x * 100) if self.finish_line_x > 0 else 0,
            "attempt": self.attempt_number
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    pygame.display.set_caption("Line Rider Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = [0, 0, 0]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            if any(keys):
                obs, info = env.reset()
                terminated = False

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()