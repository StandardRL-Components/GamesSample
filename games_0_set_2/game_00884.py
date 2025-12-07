
# Generated: 2025-08-27T15:05:08.433995
# Source Brief: brief_00884.md
# Brief Index: 884

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrow keys to move the drawing cursor. Hold Space to draw the track for the sledder."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track for a sledder to navigate a perilous, procedurally generated landscape. Reach the finish line before time runs out, but be careful not to crash into the terrain!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # Colors
    COLOR_BG_TOP = (40, 40, 60)
    COLOR_BG_BOTTOM = (10, 10, 20)
    COLOR_TERRAIN = (139, 69, 19) # Brown
    COLOR_TERRAIN_OUTLINE = (100, 40, 10)
    COLOR_TRACK = (0, 191, 255) # Deep Sky Blue
    COLOR_RIDER = (255, 255, 255)
    COLOR_CURSOR = (255, 255, 0, 100) # Semi-transparent yellow
    COLOR_START = (0, 255, 0, 150)
    COLOR_FINISH = (255, 0, 0, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (200, 200, 255)

    # Physics & Gameplay
    GRAVITY = 0.4
    FRICTION = 0.995
    RIDER_SPEED_ON_DRAW = 0.5
    CURSOR_SPEED = 6.0
    MIN_TRACK_NODE_DISTANCE = 10
    FINISH_DISTANCE = 8000
    DIFFICULTY_INTERVAL = 200

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self.np_random = None

        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.rider_pos = None
        self.rider_vel = None
        self.is_airborne = True
        self.airborne_frames = 0
        self.cursor_pos = None
        self.track_points = None
        self.terrain_points = None
        self.camera_x = 0
        self.terrain_freq_mod = 1.0
        self.terrain_amp_mod = 1.0
        self.particles = None
        self.last_space_press = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS

        self.rider_pos = pygame.math.Vector2(100, 150)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.is_airborne = True
        self.airborne_frames = 0
        
        self.cursor_pos = pygame.math.Vector2(200, 200)
        self.track_points = [pygame.math.Vector2(50, 200), pygame.math.Vector2(150, 200)]
        self.camera_x = 0

        self.terrain_freq_mod = 1.0
        self.terrain_amp_mod = 1.0
        self._generate_terrain()
        
        self.particles = deque(maxlen=200)
        self.last_space_press = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_press
        self.last_space_press = space_held

        prev_rider_x = self.rider_pos.x
        
        # --- Update Game Logic ---
        self._handle_input(movement, space_held)
        self._update_rider()
        self._update_particles()
        self._update_camera()
        
        self.steps += 1
        self.time_left -= 1
        
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.terrain_freq_mod *= 1.01
            self.terrain_amp_mod *= 1.01

        # --- Check Termination and Calculate Reward ---
        terminated, termination_reason = self._check_termination()
        reward = self._calculate_reward(prev_rider_x, termination_reason)
        self.score += reward
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED # Up
        if movement == 2: self.cursor_pos.y += self.CURSOR_SPEED # Down
        if movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED # Left
        if movement == 4: self.cursor_pos.x += self.CURSOR_SPEED # Right
        self.cursor_pos.x = np.clip(self.cursor_pos.x, self.camera_x, self.camera_x + self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # Draw track
        if space_held:
            last_point = self.track_points[-1]
            if self.cursor_pos.distance_to(last_point) > self.MIN_TRACK_NODE_DISTANCE:
                # Prevent drawing behind rider or into terrain
                if self.cursor_pos.x > last_point.x:
                    can_draw = True
                    for i in range(len(self.terrain_points) - 1):
                        terrain_p1 = self.terrain_points[i]
                        terrain_p2 = self.terrain_points[i+1]
                        if self.cursor_pos.x > terrain_p1[0] and self.cursor_pos.x < terrain_p2[0]:
                            terrain_y = np.interp(self.cursor_pos.x, [terrain_p1[0], terrain_p2[0]], [terrain_p1[1], terrain_p2[1]])
                            if self.cursor_pos.y > terrain_y:
                                can_draw = False
                                break
                    if can_draw:
                        self.track_points.append(pygame.math.Vector2(self.cursor_pos))
                        # `sfx: draw_line`

    def _update_rider(self):
        # Apply gravity
        self.rider_vel.y += self.GRAVITY
        self.rider_pos += self.rider_vel

        on_track = False
        landed_this_frame = False

        # Check for track collision
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            
            # Bounding box check for performance
            if not (min(p1.x, p2.x) - 10 < self.rider_pos.x < max(p1.x, p2.x) + 10):
                continue
            
            # Project rider onto line segment
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            point_vec = self.rider_pos - p1
            t = point_vec.dot(line_vec) / line_vec.length_squared()
            
            if 0 <= t <= 1:
                closest_point = p1 + t * line_vec
                dist_vec = self.rider_pos - closest_point
                
                if dist_vec.length() < 10 and self.rider_vel.dot(dist_vec) > 0:
                    on_track = True
                    self.rider_pos = closest_point
                    
                    # Adjust velocity to follow track
                    line_normal = dist_vec.normalize()
                    self.rider_vel -= self.rider_vel.project(line_normal)
                    self.rider_vel *= self.FRICTION
                    
                    # Add force along the slope
                    slope_dir = line_vec.normalize()
                    gravity_force = pygame.math.Vector2(0, self.GRAVITY).dot(slope_dir)
                    self.rider_vel += slope_dir * gravity_force * 1.5

                    if self.is_airborne:
                        landed_this_frame = True
                        # `sfx: land`
                    self.is_airborne = False
                    break
        
        if not on_track:
            self.is_airborne = True
        
        if self.is_airborne:
            self.airborne_frames += 1
        elif landed_this_frame:
            self.airborne_frames = 0
        
        # Add particles if moving
        if self.rider_vel.length() > 1.0:
            for _ in range(2):
                offset = pygame.math.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
                particle_pos = self.rider_pos + offset
                particle_vel = self.rider_vel.rotate(self.np_random.uniform(-15, 15)) * -0.1
                particle_life = self.np_random.integers(10, 20)
                self.particles.append([particle_pos, particle_vel, particle_life])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[2] -= 1 # life -= 1

    def _update_camera(self):
        target_x = self.rider_pos.x - self.SCREEN_WIDTH / 4
        self.camera_x += (target_x - self.camera_x) * 0.1

    def _check_termination(self):
        # Rider off-screen (bottom)
        if self.rider_pos.y > self.SCREEN_HEIGHT + 50:
            # `sfx: fall`
            return True, "crash"
        
        # Rider falls behind camera
        if self.rider_pos.x < self.camera_x - 50:
            return True, "crash"

        # Rider hits terrain
        for i in range(len(self.terrain_points) - 1):
            p1 = self.terrain_points[i]
            p2 = self.terrain_points[i+1]
            if p1[0] - 10 < self.rider_pos.x < p2[0] + 10:
                terrain_y = np.interp(self.rider_pos.x, [p1[0], p2[0]], [p1[1], p2[1]])
                if self.rider_pos.y > terrain_y - 5:
                    # `sfx: crash`
                    return True, "crash"

        # Time runs out
        if self.time_left <= 0:
            return True, "timeout"

        # Reached finish line
        if self.rider_pos.x >= self.FINISH_DISTANCE:
            # `sfx: victory`
            return True, "finish"

        return False, None

    def _calculate_reward(self, prev_rider_x, termination_reason):
        reward = 0
        
        # Reward for horizontal progress
        distance_traveled = self.rider_pos.x - prev_rider_x
        if distance_traveled > 0:
            reward += distance_traveled * 0.1

        # Cost for drawing track (applied in _handle_input but accounted for here)
        # Simplified: small penalty per step if drawing
        if self.last_space_press:
            reward -= 0.01

        # Reward for successful jump
        if not self.is_airborne and self.airborne_frames > 15: # Landed after being airborne
            reward += 1.0
            self.airborne_frames = 0 # Reset to prevent multiple rewards

        # Terminal rewards
        if termination_reason == "finish":
            reward += 100.0
        elif termination_reason == "crash":
            reward -= 10.0 # Small penalty for crashing to differentiate from timeout
        
        return reward

    def _generate_terrain(self):
        points = []
        x = 0
        y = self.SCREEN_HEIGHT * 0.8
        
        while x < self.FINISH_DISTANCE + self.SCREEN_WIDTH:
            points.append((x, y))
            base_freq = 0.005 * self.terrain_freq_mod
            base_amp = 70 * self.terrain_amp_mod
            
            y = self.SCREEN_HEIGHT * 0.8 \
                + math.sin(x * base_freq) * base_amp \
                + math.sin(x * base_freq * 2.1) * base_amp * 0.4 \
                + math.sin(x * base_freq * 4.7) * base_amp * 0.2
            
            # Create gaps
            if self.np_random.uniform() < 0.05:
                x += self.np_random.uniform(100, 300)
            else:
                x += self.np_random.uniform(20, 50)
        
        self.terrain_points = points

    def _get_observation(self):
        self._render_background()
        self._render_world()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_world(self):
        # Render Terrain
        screen_terrain_points = []
        for x, y in self.terrain_points:
            sx = int(x - self.camera_x)
            if -50 < sx < self.SCREEN_WIDTH + 50:
                screen_terrain_points.append((sx, int(y)))
        
        if len(screen_terrain_points) > 1:
            poly_points = [(p[0], self.SCREEN_HEIGHT) for p in reversed(screen_terrain_points)]
            poly_points.extend(screen_terrain_points)
            pygame.gfxdraw.filled_polygon(self.screen, poly_points, self.COLOR_TERRAIN)
            pygame.gfxdraw.aapolygon(self.screen, poly_points, self.COLOR_TERRAIN_OUTLINE)

        # Render Start/Finish Lines
        start_x = int(100 - self.camera_x)
        pygame.draw.line(self.screen, self.COLOR_START, (start_x, 0), (start_x, self.SCREEN_HEIGHT), 3)
        finish_x = int(self.FINISH_DISTANCE - self.camera_x)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x, 0), (finish_x, self.SCREEN_HEIGHT), 5)

        # Render Track
        if len(self.track_points) > 1:
            screen_track_points = [(int(p.x - self.camera_x), int(p.y)) for p in self.track_points]
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, screen_track_points, 3)

        # Render Particles
        for pos, vel, life in self.particles:
            if life > 0:
                sx = int(pos.x - self.camera_x)
                sy = int(pos.y)
                alpha = int(255 * (life / 20.0))
                color = (*self.COLOR_PARTICLE, alpha)
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2, 2), 2)
                self.screen.blit(temp_surf, (sx - 2, sy - 2))

        # Render Rider
        rider_sx = int(self.rider_pos.x - self.camera_x)
        rider_sy = int(self.rider_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, rider_sx, rider_sy, 7, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_sx, rider_sy, 7, (0,0,0))
        
        # Render Cursor
        cursor_sx = int(self.cursor_pos.x - self.camera_x)
        cursor_sy = int(self.cursor_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, cursor_sx, cursor_sy, 10, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, cursor_sx, cursor_sy, 10, self.COLOR_CURSOR)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Time
        time_str = f"TIME: {self.time_left / self.FPS:.1f}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Distance
        progress = self.rider_pos.x / self.FINISH_DISTANCE
        dist_str = f"PROGRESS: {progress:.1%}"
        dist_text = self.font_ui.render(dist_str, True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (10, 35))

        # Speed
        speed = self.rider_vel.length() * 5 # Arbitrary scaling for display
        speed_str = f"SPEED: {int(speed)} km/h"
        speed_text = self.font_ui.render(speed_str, True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 10, 35))
        
        if self.game_over:
            reason = self._check_termination()[1]
            msg = "GAME OVER"
            if reason == "finish": msg = "YOU WIN!"
            elif reason == "timeout": msg = "TIME UP!"
            
            end_text = self.font_big.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.rider_pos.x,
            "time_left": self.time_left,
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Sled Rider")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # --- Main Game Loop ---
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, term, trunc, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if term:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()

        clock.tick(env.FPS)
        
    env.close()