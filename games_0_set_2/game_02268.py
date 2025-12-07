
# Generated: 2025-08-27T19:50:07.586828
# Source Brief: brief_02268.md
# Brief Index: 2268

        
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
        "Controls: Use arrow keys to aim the track segment. Hold Space for a boost track, or Shift for a brake track. "
        "The goal is to draw a path for the rider to reach the green finish line."
    )

    game_description = (
        "A physics-based puzzle game where you draw tracks for a sledder. Use different track types to "
        "control the rider's speed and guide them to the finish line while avoiding obstacles."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_RIDER_GLOW = (200, 200, 255)
        self.COLOR_TRACK_BLUE = (60, 160, 255)
        self.COLOR_TRACK_GREEN = (80, 255, 150)
        self.COLOR_TRACK_RED = (255, 100, 100)
        self.COLOR_OBSTACLE = (20, 20, 20)
        self.COLOR_FINISH = (80, 255, 150, 150)
        self.COLOR_UI = (230, 230, 230)

        # Game constants
        self.GRAVITY = 0.12
        self.RIDER_RADIUS = 7
        self.MAX_SPEED = 12
        self.PHYSICS_SUBSTEPS = 10
        self.MAX_STEPS = 500
        self.TRACK_WIDTH = 5

        # State variables (initialized in reset)
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rider_pos = None
        self.rider_vel = None
        self.track_segments = None
        self.finish_line = None
        self.obstacles = None
        self.particles = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.rider_pos = pygame.Vector2(80, 150)
        self.rider_vel = pygame.Vector2(2, 0)
        
        self.track_segments = []
        # Add initial flat track segment
        self._add_track_segment(pygame.Vector2(40, 170), pygame.Vector2(120, 170), 'blue')

        self.finish_line = pygame.Rect(self.WIDTH - 40, self.HEIGHT - 150, 10, 100)
        self.obstacles = [pygame.Rect(300, 280, 80, 20)]
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. ACTION PHASE: Place a new track segment
        placement_reward = self._place_track(action)

        # 2. PHYSICS PHASE: Simulate rider movement
        terminated = False
        terminal_reward = 0
        was_on_track = False
        
        for _ in range(self.PHYSICS_SUBSTEPS):
            if terminated:
                break
            
            on_track_this_substep = self._update_rider_physics()
            if on_track_this_substep:
                was_on_track = True

            self._update_particles()

            # Check for termination conditions
            if not (0 < self.rider_pos.x < self.WIDTH and 0 < self.rider_pos.y < self.HEIGHT):
                terminated = True
                terminal_reward = -50  # Crash reward
                # sfx: fall_whistle, crash
            elif self.finish_line.collidepoint(self.rider_pos.x, self.rider_pos.y):
                terminated = True
                terminal_reward = 100  # Win reward
                # sfx: win_jingle
            for obs in self.obstacles:
                if obs.collidepoint(self.rider_pos.x, self.rider_pos.y):
                    terminated = True
                    terminal_reward = -50  # Crash reward
                    # sfx: thud
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        # 3. REWARD CALCULATION
        continuous_reward = 0.1 if was_on_track else 0
        total_reward = placement_reward + continuous_reward + terminal_reward
        self.score += total_reward
        
        # Assert reward scale
        assert -51 <= total_reward <= 110.1

        return (
            self._get_observation(),
            total_reward,
            terminated,
            False,
            self._get_info()
        )

    def _place_track(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        track_type = 'blue'
        placement_reward = 0
        if space_held:
            track_type = 'green'
            placement_reward = 10
        elif shift_held:
            track_type = 'red'
            placement_reward = -1

        length = 40
        start_pos = self.rider_pos.copy()
        
        angle_map = {
            0: 0,     # none -> horizontal right
            1: 60,    # up -> steep up-right
            2: -60,   # down -> steep down-right
            3: 150,   # left -> gentle up-left
            4: -30    # right -> gentle down-right
        }
        angle = math.radians(angle_map.get(movement, 0))
        
        end_pos = start_pos + pygame.Vector2(length, 0).rotate_rad(-angle) # Pygame rotation is counter-clockwise

        self._add_track_segment(start_pos, end_pos, track_type)
        return placement_reward

    def _update_rider_physics(self):
        self.rider_vel.y += self.GRAVITY
        if self.rider_vel.length() > self.MAX_SPEED:
            self.rider_vel.scale_to_length(self.MAX_SPEED)
        assert self.rider_vel.length() <= 100 # Validation benchmark

        self.rider_pos += self.rider_vel
        
        collided_this_substep = False
        for seg in self.track_segments:
            p1, p2, seg_type = seg['start'], seg['end'], seg['type']
            line_vec = p2 - p1
            
            if line_vec.length_squared() == 0: continue

            p1_to_rider = self.rider_pos - p1
            t = line_vec.dot(p1_to_rider) / line_vec.length_squared()
            t = max(0, min(1, t))
            
            closest_point = p1 + t * line_vec
            dist_vec = self.rider_pos - closest_point
            
            if dist_vec.length() < self.RIDER_RADIUS:
                collided_this_substep = True
                
                # Resolve penetration
                if dist_vec.length() > 0:
                    penetration_depth = self.RIDER_RADIUS - dist_vec.length()
                    self.rider_pos += dist_vec.normalize() * (penetration_depth + 0.01)
                
                # Calculate response
                track_normal = dist_vec.normalize()
                friction, bounce = 0.99, 0.4
                
                vel_normal_component = self.rider_vel.dot(track_normal)
                vel_normal = track_normal * vel_normal_component
                vel_tangent = self.rider_vel - vel_normal
                
                self.rider_vel = (vel_tangent * friction) - (vel_normal * bounce)
                
                # Apply track type effect
                speed_change = 0.0
                if seg_type == 'green':
                    speed_change = 0.25
                    self._create_particles(self.rider_pos, 'green')
                    # sfx: boost_sound
                elif seg_type == 'red':
                    speed_change = -0.25
                    self._create_particles(self.rider_pos, 'red')
                    # sfx: brake_sound
                
                new_speed = self.rider_vel.length() + speed_change
                if new_speed > 0:
                    self.rider_vel.scale_to_length(max(0.5, new_speed))
                
                break # Handle one collision per substep
        return collided_this_substep

    def _add_track_segment(self, start_pos, end_pos, type):
        self.track_segments.append({'start': start_pos, 'end': end_pos, 'type': type})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw finish line
        finish_surface = pygame.Surface((self.finish_line.width, self.finish_line.height), pygame.SRCALPHA)
        finish_surface.fill(self.COLOR_FINISH)
        self.screen.blit(finish_surface, self.finish_line.topleft)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)
            pygame.draw.rect(self.screen, self.COLOR_GRID, obs, 1)

        # Draw track segments
        for seg in self.track_segments:
            color = self.COLOR_TRACK_BLUE
            if seg['type'] == 'green': color = self.COLOR_TRACK_GREEN
            elif seg['type'] == 'red': color = self.COLOR_TRACK_RED
            pygame.draw.line(self.screen, color, seg['start'], seg['end'], self.TRACK_WIDTH)
        
        # Draw particles
        self._draw_particles()

        # Draw rider
        rider_x, rider_y = int(self.rider_pos.x), int(self.rider_pos.y)
        glow_radius = int(self.RIDER_RADIUS * (1.5 + self.rider_vel.length() / self.MAX_SPEED))
        
        # Simple glow effect
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = self.COLOR_RIDER_GLOW
        alpha = 60
        pygame.draw.circle(glow_surf, (glow_color[0], glow_color[1], glow_color[2], alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (rider_x - glow_radius, rider_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)

    def _render_ui(self):
        speed_text = self.font_small.render(f"Speed: {self.rider_vel.length():.1f}", True, self.COLOR_UI)
        self.screen.blit(speed_text, (10, 10))

        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

        if self.game_over:
            result_text_str = "FINISH!" if self.finish_line.collidepoint(self.rider_pos.x, self.rider_pos.y) else "CRASHED"
            result_text = self.font_large.render(result_text_str, True, self.COLOR_UI)
            self.screen.blit(result_text, (self.WIDTH // 2 - result_text.get_width() // 2, self.HEIGHT // 2 - result_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_speed": self.rider_vel.length() if self.rider_vel else 0,
        }

    def _create_particles(self, pos, p_type):
        color = self.COLOR_TRACK_GREEN if p_type == 'green' else self.COLOR_TRACK_RED
        for _ in range(3):
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)) * 2
            lifetime = self.np_random.integers(10, 20)
            self.particles.append([pos.copy(), vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _draw_particles(self):
        for p in self.particles:
            pos, _, lifetime, color = p
            size = max(0, int(lifetime / 4))
            pygame.draw.rect(self.screen, color, (int(pos.x - size/2), int(pos.y - size/2), size, size))

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment with a human player or a random agent.
    # To enable human play, uncomment the "Manual Play" section and comment out the "Agent Simulation" section.

    # --- Agent Simulation ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    total_reward = 0
    print("--- Running Random Agent Simulation ---")
    for i in range(150):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
    env.close()
    
    # --- Manual Play ---
    # print("\n--- Starting Manual Play ---")
    # print(GameEnv.user_guide)
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # done = False
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Line Rider Gym")
    # game_loop_active = True
    # while game_loop_active:
    #     action = [0, 0, 0] 
    #     keys = pygame.key.get_pressed()
        
    #     # Map keys to movement part of action
    #     if keys[pygame.K_UP]: action[0] = 1
    #     elif keys[pygame.K_DOWN]: action[0] = 2
    #     elif keys[pygame.K_LEFT]: action[0] = 3
    #     elif keys[pygame.K_RIGHT]: action[0] = 4
        
    #     # Map space/shift to other parts
    #     if keys[pygame.K_SPACE]: action[1] = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             game_loop_active = False
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_r: # Reset on 'r'
    #                 print("--- Resetting Environment ---")
    #                 obs, info = env.reset()
    #                 done = False
    #             # Any key other than 'r' triggers a step if the game is not over
    #             elif not done:
    #                 obs, reward, terminated, truncated, info = env.step(action)
    #                 print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}")
    #                 done = terminated or truncated
    #                 if done:
    #                     print(f"--- Episode Finished --- Final Score: {info['score']:.2f}")


    #     # Draw the observation to the display
    #     frame = env._get_observation()
    #     frame = np.transpose(frame, (1, 0, 2))
    #     surf = pygame.surfarray.make_surface(frame)
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    # pygame.quit()