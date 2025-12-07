
# Generated: 2025-08-28T06:22:46.270252
# Source Brief: brief_02913.md
# Brief Index: 2913

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to angle the slope up, ↓ to angle the slope down. "
        "Guide the rider to the finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a sledder down a procedurally generated track by influencing its slope, "
        "aiming for a fast and stylish run."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (70, 130, 180)  # Steel Blue
    COLOR_TRACK = (255, 255, 255)
    COLOR_RIDER = (220, 20, 60)  # Crimson
    COLOR_PARTICLE = (240, 248, 255) # Alice Blue
    COLOR_UI_SPEED = (50, 205, 50) # Lime Green
    COLOR_UI_TIMER = (255, 215, 0) # Gold
    COLOR_FINISH_1 = (0, 0, 0)
    COLOR_FINISH_2 = (255, 255, 255)

    # Physics & Gameplay
    GRAVITY = 0.25
    FRICTION = 0.998
    RIDER_SIZE = 12
    SLOPE_INCREMENT = 0.02
    MAX_SLOPE = math.pi / 3  # 60 degrees
    TRACK_SEGMENT_LENGTH = 20
    MAX_STEPS = 2000
    FINISH_LINE_X = 15000 # Distance to finish

    # Rewards
    LOW_SPEED_THRESHOLD = 1.0
    HIGH_VARIANCE_THRESHOLD = 0.3 # Radians
    HIGH_VARIANCE_COOLDOWN = 60 # Steps

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables that are not reset
        self.np_random = None
        self.rider_pos = pygame.math.Vector2(0, 0)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.track_points = deque()
        self.track_slopes = deque()
        self.particles = deque()
        self.current_slope = 0.0
        self.difficulty_modifier = 0.0
        self.camera_offset_x = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.air_time = 0
        self.last_high_variance_reward_step = -self.HIGH_VARIANCE_COOLDOWN

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.rider_pos = pygame.math.Vector2(100, self.SCREEN_HEIGHT / 2)
        self.rider_vel = pygame.math.Vector2(2, 0)
        self.current_slope = 0.0
        self.difficulty_modifier = 0.0
        self.camera_offset_x = 0.0
        self.air_time = 0
        self.last_high_variance_reward_step = -self.HIGH_VARIANCE_COOLDOWN
        
        self.track_points.clear()
        self.track_slopes.clear()
        self.particles.clear()
        
        # Generate initial flat track
        start_y = self.SCREEN_HEIGHT / 2 + 50
        for i in range(-5, int(self.SCREEN_WIDTH / self.TRACK_SEGMENT_LENGTH) + 5):
            x = i * self.TRACK_SEGMENT_LENGTH
            self.track_points.append(pygame.math.Vector2(x, start_y))
            self.track_slopes.append(0.0)
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, _, _ = action
        reward = 0
        
        if not self.game_over:
            # --- Handle Input ---
            if movement == 1:  # Up -> angle slope up (rider slows)
                self.current_slope -= self.SLOPE_INCREMENT
            elif movement == 2:  # Down -> angle slope down (rider accelerates)
                self.current_slope += self.SLOPE_INCREMENT
            self.current_slope = np.clip(self.current_slope, -self.MAX_SLOPE, self.MAX_SLOPE)

            # --- Update Game Logic ---
            self._update_track()
            reward = self._update_rider_physics()
            self._update_particles()
            self._update_camera()

        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_track(self):
        # Generate new track segments if rider is near the end
        while self.rider_pos.x > self.track_points[-5].x:
            last_point = self.track_points[-1]
            
            # Difficulty scales procedural noise
            if self.steps > 0 and self.steps % 200 == 0:
                self.difficulty_modifier = min(self.difficulty_modifier + 0.05, 0.8)

            procedural_noise = self.np_random.uniform(-1, 1) * self.difficulty_modifier * 0.1
            
            # Player input has direct control, with some procedural variation
            slope_for_segment = self.current_slope + procedural_noise
            slope_for_segment = np.clip(slope_for_segment, -self.MAX_SLOPE * 1.2, self.MAX_SLOPE * 1.2)
            
            new_point = pygame.math.Vector2(
                last_point.x + self.TRACK_SEGMENT_LENGTH,
                last_point.y + math.tan(slope_for_segment) * self.TRACK_SEGMENT_LENGTH
            )
            self.track_points.append(new_point)
            self.track_slopes.append(slope_for_segment)

        # Remove old segments
        while len(self.track_points) > 2 and self.track_points[1].x < self.camera_offset_x - 50:
            self.track_points.popleft()
            self.track_slopes.popleft()

    def _update_rider_physics(self):
        reward = 0
        
        # Apply gravity
        self.rider_vel.y += self.GRAVITY
        
        # Find current track segment
        on_track = False
        segment_found = False
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            if p1.x <= self.rider_pos.x < p2.x:
                segment_found = True
                # Interpolate track height at rider's x
                ratio = (self.rider_pos.x - p1.x) / (p2.x - p1.x) if (p2.x - p1.x) != 0 else 0
                track_y = p1.y + ratio * (p2.y - p1.y)
                
                # Check for difficult section navigation
                prev_slope = self.track_slopes[i-1] if i > 0 else self.track_slopes[i]
                current_segment_slope = self.track_slopes[i]
                slope_delta = abs(current_segment_slope - prev_slope)
                if slope_delta > self.HIGH_VARIANCE_THRESHOLD and self.steps > self.last_high_variance_reward_step + self.HIGH_VARIANCE_COOLDOWN:
                    reward += 5
                    self.last_high_variance_reward_step = self.steps

                # Collision detection and response
                if self.rider_pos.y >= track_y - self.RIDER_SIZE / 2:
                    on_track = True
                    self.air_time = 0
                    self.rider_pos.y = track_y - self.RIDER_SIZE / 2
                    
                    # Track reaction force (simplified)
                    track_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
                    normal_angle = track_angle - math.pi / 2
                    
                    # Redirect velocity along the track surface
                    normal_vec = pygame.math.Vector2(math.cos(normal_angle), math.sin(normal_angle))
                    dot_product = self.rider_vel.dot(normal_vec)
                    if dot_product > 0:
                        self.rider_vel -= normal_vec * dot_product
                    
                    # Apply friction
                    self.rider_vel *= self.FRICTION
                    
                    # Emit particles (sound: snow crunch/spray)
                    if self.rider_vel.length() > 2:
                        num_particles = min(5, int(self.rider_vel.length() * abs(math.sin(track_angle)) * 0.5))
                        for _ in range(num_particles):
                            self._create_particle()
                break
        
        if not on_track:
            self.air_time += 1
            if not segment_found: # Flew off the generated track
                self.air_time += 50 

        # Update position
        self.rider_pos += self.rider_vel

        # Calculate continuous rewards
        reward += 0.1  # Reward for surviving
        reward += min(1.0, 0.2 * self.rider_vel.x) # Capped reward for forward velocity
        if self.rider_vel.x < self.LOW_SPEED_THRESHOLD and on_track:
            reward -= 1.0
            
        return reward

    def _create_particle(self):
        p_pos = self.rider_pos.copy() + pygame.math.Vector2(self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5))
        p_vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-2, 0))
        p_life = self.np_random.integers(15, 30)
        self.particles.append({'pos': p_pos, 'vel': p_vel, 'life': p_life, 'max_life': p_life})

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_camera(self):
        target_offset = self.rider_pos.x - self.SCREEN_WIDTH * 0.25
        # Smooth camera motion (lerp)
        self.camera_offset_x = self.camera_offset_x * 0.9 + target_offset * 0.1

    def _check_termination(self):
        if self.game_over:
            return True
            
        # Fall condition
        if self.air_time > 60 or self.rider_pos.y > self.SCREEN_HEIGHT + 50:
            self.game_over = True
            return True
            
        # Win condition
        if self.rider_pos.x >= self.FINISH_LINE_X:
            self.game_over = True
            self.score += 50 # Final bonus
            return True
        
        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        # --- Background Gradient ---
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # --- Game Elements ---
        # Finish Line
        finish_screen_x = self.FINISH_LINE_X - self.camera_offset_x
        if 0 < finish_screen_x < self.SCREEN_WIDTH:
            for y in range(0, self.SCREEN_HEIGHT, 20):
                pygame.draw.rect(self.screen, self.COLOR_FINISH_1, (finish_screen_x, y, 10, 10))
                pygame.draw.rect(self.screen, self.COLOR_FINISH_2, (finish_screen_x, y + 10, 10, 10))

        # Particles
        for p in self.particles:
            screen_pos = p['pos'] - pygame.math.Vector2(self.camera_offset_x, 0)
            alpha = int(255 * (p['life'] / p['max_life']))
            radius = int(3 * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), radius, (*self.COLOR_PARTICLE, alpha))
        
        # Track
        screen_points = [(p.x - self.camera_offset_x, p.y) for p in self.track_points]
        if len(screen_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, screen_points, 3)

        # Rider
        rider_screen_pos = self.rider_pos - pygame.math.Vector2(self.camera_offset_x, 0)
        angle = self.rider_vel.angle_to(pygame.math.Vector2(1, 0)) if self.rider_vel.length() > 0 else 0
        
        p1 = rider_screen_pos + pygame.math.Vector2(self.RIDER_SIZE, 0).rotate(-angle)
        p2 = rider_screen_pos + pygame.math.Vector2(-self.RIDER_SIZE/2, -self.RIDER_SIZE/2).rotate(-angle)
        p3 = rider_screen_pos + pygame.math.Vector2(-self.RIDER_SIZE/2, self.RIDER_SIZE/2).rotate(-angle)
        
        rider_points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
        pygame.gfxdraw.aapolygon(self.screen, rider_points, self.COLOR_RIDER)
        pygame.gfxdraw.filled_polygon(self.screen, rider_points, self.COLOR_RIDER)
        
        # --- UI ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Speed
        speed_text = f"SPEED: {int(self.rider_vel.x * 5)}"
        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_UI_SPEED)
        self.screen.blit(speed_surf, (10, 10))

        # Timer
        time_text = f"TIME: {self.steps / 30:.1f}s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TIMER)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        # Game Over Text
        if self.game_over:
            msg = "FINISH!" if self.rider_pos.x >= self.FINISH_LINE_X else "WIPEOUT!"
            color = (0, 255, 0) if msg == "FINISH!" else (255, 0, 0)
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_surf, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_x_pos": self.rider_pos.x,
            "rider_speed": self.rider_vel.length()
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

# Example usage for interactive play
if __name__ == '__main__':
    import os
    # Set a video driver that works in most environments, including headless ones
    try:
        os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        os.environ["SDL_VIDEODRIVER"] = "x11"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Slope Rider")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)
    
    # Main game loop
    while not terminated:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling for closing the window ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # After game over, wait for a moment before quitting
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            pygame.time.wait(2000)

        clock.tick(30) # Match the intended frame rate
        
    env.close()