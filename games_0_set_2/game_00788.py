
# Generated: 2025-08-27T14:46:23.700169
# Source Brief: brief_00788.md
# Brief Index: 788

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the drawing cursor. Hold Space to draw a normal track, or Shift to draw a boost track. The rider will follow the track you create."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle game where you draw tracks for a sledder. Guide the rider from the start to the finish by creating clever paths. Use normal lines for structure and boost lines for speed!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    
    # Colors
    COLOR_BG_TOP = (173, 216, 230) # Light Blue
    COLOR_BG_BOTTOM = (240, 248, 255) # Alice Blue
    COLOR_TRACK = (20, 20, 20)
    COLOR_BOOST_TRACK = (0, 100, 255)
    COLOR_RIDER = (220, 50, 50)
    COLOR_RIDER_SLED = (120, 60, 30)
    COLOR_START_FINISH = (50, 200, 50)
    COLOR_CHECKPOINT = (255, 165, 0)
    COLOR_UI_TEXT = (10, 10, 10)
    COLOR_CURSOR = (100, 100, 100, 150) # Semi-transparent grey
    
    # Physics
    GRAVITY = pygame.math.Vector2(0, 0.25)
    FRICTION = 0.998
    BOUNCE_DAMPENING = 0.7
    BOOST_FORCE = 0.3
    CURSOR_SPEED = 5
    RIDER_RADIUS = 6
    MAX_STEPS = 2000
    STUCK_THRESHOLD_SPEED = 0.1
    STUCK_STEPS_LIMIT = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 20, bold=True)
        self.font_title = pygame.font.SysFont("sans-serif", 28, bold=True)

        # Game state variables are initialized in reset()
        self.lines = []
        self.particles = []
        self.rider_pos = pygame.math.Vector2(0, 0)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.draw_cursor = pygame.math.Vector2(0, 0)
        self.last_draw_pos = pygame.math.Vector2(0, 0)
        self.steps = 0
        self.score = 0
        self.max_x_reached = 0
        self.checkpoint_reached = False
        self.stuck_counter = 0

        self.start_pos_x = 50
        self.finish_pos_x = self.WIDTH - 50
        self.checkpoint_pos_x = self.start_pos_x + (self.finish_pos_x - self.start_pos_x) / 2
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        initial_slope_start = (self.start_pos_x - 20, 100)
        initial_slope_end = (self.start_pos_x + 80, 150)

        self.lines = [
            (pygame.math.Vector2(initial_slope_start), pygame.math.Vector2(initial_slope_end), 'normal')
        ]

        self.rider_pos = pygame.math.Vector2(self.start_pos_x, 80)
        self.rider_vel = pygame.math.Vector2(0.5, 0)
        
        self.draw_cursor = pygame.math.Vector2(self.start_pos_x + 100, 160)
        self.last_draw_pos = self.draw_cursor.copy()
        
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.max_x_reached = self.rider_pos.x
        self.checkpoint_reached = False
        self.stuck_counter = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle player action (drawing lines)
        prev_cursor_pos = self.draw_cursor.copy()
        if movement == 1: self.draw_cursor.y -= self.CURSOR_SPEED
        elif movement == 2: self.draw_cursor.y += self.CURSOR_SPEED
        elif movement == 3: self.draw_cursor.x -= self.CURSOR_SPEED
        elif movement == 4: self.draw_cursor.x += self.CURSOR_SPEED
        
        self.draw_cursor.x = np.clip(self.draw_cursor.x, 0, self.WIDTH)
        self.draw_cursor.y = np.clip(self.draw_cursor.y, 0, self.HEIGHT)

        # If a button is held, draw a line from the last point to the new cursor position
        if (space_held or shift_held) and self.last_draw_pos is not None:
            line_type = 'boost' if shift_held else 'normal'
            # Prevent zero-length lines
            if prev_cursor_pos.distance_to(self.draw_cursor) > 1:
                self.lines.append((prev_cursor_pos, self.draw_cursor.copy(), line_type))
        
        # Update the "pen down" position. If no movement, pen is "lifted".
        if movement != 0:
            if not space_held and not shift_held:
                self.last_draw_pos = self.draw_cursor.copy()
        else: # No movement action
             self.last_draw_pos = self.draw_cursor.copy()


        # 2. Update physics
        self._update_rider_physics()
        self._update_particles()

        # 3. Calculate reward
        reward = self._calculate_reward()
        self.score += reward

        # 4. Check for termination
        terminated = self._check_termination()
        
        self.steps += 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_rider_physics(self):
        # Apply gravity
        self.rider_vel += self.GRAVITY
        self.rider_pos += self.rider_vel

        collided_this_frame = False
        for p1, p2, line_type in self.lines:
            # Find the closest point on the line segment to the rider
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            rider_vec = self.rider_pos - p1
            t = rider_vec.dot(line_vec) / line_vec.length_squared()
            t_clamped = max(0, min(1, t))
            closest_point = p1 + t_clamped * line_vec

            dist_vec = self.rider_pos - closest_point
            if dist_vec.length() < self.RIDER_RADIUS:
                collided_this_frame = True
                
                # Resolve penetration
                penetration_depth = self.RIDER_RADIUS - dist_vec.length()
                if dist_vec.length() > 0:
                    self.rider_pos += dist_vec.normalize() * penetration_depth
                
                # Collision response
                normal = dist_vec.normalize()
                
                # Reflect velocity and apply dampening
                # sound: "sled_scrape.wav"
                self.rider_vel = self.rider_vel.reflect(normal) * self.BOUNCE_DAMPENING
                
                # Apply friction
                self.rider_vel *= self.FRICTION

                # Apply boost if it's a boost line
                if line_type == 'boost':
                    # sound: "boost_whoosh.wav"
                    boost_dir = line_vec.normalize()
                    self.rider_vel += boost_dir * self.BOOST_FORCE

                # Add impact particles
                for _ in range(3):
                    self.particles.append(Particle(closest_point, self.COLOR_RIDER))

        # Check if stuck
        if self.rider_vel.length() < self.STUCK_THRESHOLD_SPEED:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _calculate_reward(self):
        reward = 0.0
        
        # Reward for forward progress
        progress = self.rider_pos.x - self.max_x_reached
        if progress > 0:
            reward += progress * 0.1
            self.max_x_reached = self.rider_pos.x

        # Penalty for time passing
        reward -= 0.01

        # Checkpoint reward
        if not self.checkpoint_reached and self.rider_pos.x > self.checkpoint_pos_x:
            reward += 5.0
            self.checkpoint_reached = True
            # sound: "checkpoint.wav"
            for _ in range(20):
                self.particles.append(Particle(self.rider_pos, self.COLOR_CHECKPOINT))

        return reward

    def _check_termination(self):
        crashed = (
            self.rider_pos.x < 0 or self.rider_pos.x > self.WIDTH or
            self.rider_pos.y < 0 or self.rider_pos.y > self.HEIGHT or
            self.stuck_counter > self.STUCK_STEPS_LIMIT
        )
        finished = self.rider_pos.x >= self.finish_pos_x

        if finished:
            self.score += 100 # Final bonus
            # sound: "victory.wav"
            return True
        if crashed:
            self.score -= 50 # Crash penalty
            # sound: "crash.wav"
            return True
        if self.steps >= self.MAX_STEPS:
            return True
            
        return False

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            color_ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[0] * color_ratio),
                int(self.COLOR_BG_TOP[1] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[1] * color_ratio),
                int(self.COLOR_BG_TOP[2] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[2] * color_ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw start, finish, checkpoint lines
        pygame.draw.line(self.screen, self.COLOR_START_FINISH, (self.start_pos_x, 0), (self.start_pos_x, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_START_FINISH, (self.finish_pos_x, 0), (self.finish_pos_x, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_CHECKPOINT, (self.checkpoint_pos_x, 0), (self.checkpoint_pos_x, self.HEIGHT), 2)
        
        # Draw all tracks
        for p1, p2, line_type in self.lines:
            color = self.COLOR_BOOST_TRACK if line_type == 'boost' else self.COLOR_TRACK
            pygame.draw.aaline(self.screen, color, p1, p2, 3)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw rider
        rider_x, rider_y = int(self.rider_pos.x), int(self.rider_pos.y)
        sled_angle = self.rider_vel.angle_to(pygame.math.Vector2(1, 0))
        sled_p1 = self.rider_pos + pygame.math.Vector2(self.RIDER_RADIUS, 0).rotate(-sled_angle)
        sled_p2 = self.rider_pos + pygame.math.Vector2(-self.RIDER_RADIUS, 0).rotate(-sled_angle)
        pygame.draw.aaline(self.screen, self.COLOR_RIDER_SLED, sled_p1, sled_p2, 3)
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y - 2, self.RIDER_RADIUS - 2, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y - 2, self.RIDER_RADIUS - 2, self.COLOR_RIDER)

        # Draw cursor
        cursor_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.line(cursor_surf, self.COLOR_CURSOR, (10, 0), (10, 20), 2)
        pygame.draw.line(cursor_surf, self.COLOR_CURSOR, (0, 10), (20, 10), 2)
        self.screen.blit(cursor_surf, (self.draw_cursor.x - 10, self.draw_cursor.y - 10))

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_ui.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        speed_text = self.font_ui.render(f"Speed: {self.rider_vel.length():.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (10, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "rider_vel": (self.rider_vel.x, self.rider_vel.y),
            "max_x_reached": self.max_x_reached,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

class Particle:
    def __init__(self, pos, color):
        self.pos = pos.copy()
        self.vel = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        self.lifespan = random.randint(10, 25)
        self.size = random.randint(2, 4)
        self.color = color

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)
        return self.lifespan > 0

    def draw(self, surface):
        if self.size > 0:
            pygame.draw.rect(surface, self.color, (self.pos.x, self.pos.y, int(self.size), int(self.size)))


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run in a window
    # os.environ['SDL_VIDEODRIVER'] = 'x11'

    env = GameEnv(render_mode="rgb_array")
    
    # --- To run with a human player ---
    pygame.display.set_caption("Line Rider Gym Environment")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # Human controls
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        env.clock.tick(60) # Limit frame rate for human playability
        
    env.close()