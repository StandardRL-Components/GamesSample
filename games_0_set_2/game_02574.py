
# Generated: 2025-08-27T20:47:07.192584
# Source Brief: brief_02574.md
# Brief Index: 2574

        
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
        "Controls: Arrows set line angle. Space/Shift change line length."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle game. Draw track segments to guide a sled to the finish line on the right."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FINISH_X = WIDTH - 40
    MAX_STEPS = 500

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_TRACK = (200, 200, 210)
    COLOR_SLED = (255, 80, 80)
    COLOR_SLED_GLOW = (255, 120, 120)
    COLOR_START = (80, 255, 80)
    COLOR_FINISH = (255, 80, 80)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PARTICLE = (255, 255, 255)

    # Physics
    GRAVITY = 0.1
    FRICTION = 0.995
    BOUNCE_DAMPENING = 0.7
    PHYSICS_SUBSTEPS = 8
    SLED_RADIUS = 6
    STOPPED_THRESHOLD = 0.1
    STOPPED_FRAMES_LIMIT = 10

    # Track Drawing
    BASE_LINE_LENGTH = 30
    LENGTH_MOD_PLUS = 20
    LENGTH_MOD_MINUS = 15
    ANGLE_MAP = {
        0: 0,          # Horizontal
        1: math.pi / 6,  # 30 deg up
        2: -math.pi / 6, # 30 deg down
        3: math.pi / 3,  # 60 deg up
        4: -math.pi / 3, # 60 deg down
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Internal state variables are initialized in reset()
        self.sled_pos = None
        self.sled_vel = None
        self.track_lines = None
        self.last_line_end = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.stopped_frames = 0
        self.game_over = False

        self.reset()
        
        # Self-check to ensure API compliance
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stopped_frames = 0

        # Sled state
        self.sled_pos = np.array([50.0, 150.0])
        self.sled_vel = np.array([1.0, 0.0])

        # Track state
        start_platform = (np.array([10.0, 200.0]), np.array([100.0, 200.0]))
        self.track_lines = [start_platform]
        self.last_line_end = start_platform[1].copy()

        # Effects
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Action: Place new track segment ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        line_length = self.BASE_LINE_LENGTH
        if space_held:
            line_length += self.LENGTH_MOD_PLUS
        if shift_held:
            line_length -= self.LENGTH_MOD_MINUS
        line_length = max(5, line_length)

        angle = self.ANGLE_MAP[movement]
        
        new_end_x = self.last_line_end[0] + line_length * math.cos(angle)
        new_end_y = self.last_line_end[1] + line_length * math.sin(angle)
        
        # Clamp line to screen bounds
        new_end_x = np.clip(new_end_x, 0, self.WIDTH)
        new_end_y = np.clip(new_end_y, 0, self.HEIGHT)
        
        new_line_end = np.array([new_end_x, new_end_y])
        
        # Add new line segment if it has length
        if np.linalg.norm(new_line_end - self.last_line_end) > 1:
            self.track_lines.append((self.last_line_end.copy(), new_line_end))
            self.last_line_end = new_line_end

        # --- 2. Simulate Physics ---
        old_pos = self.sled_pos.copy()
        for _ in range(self.PHYSICS_SUBSTEPS):
            self._handle_physics()
        
        self._update_particles()
        
        # --- 3. Calculate Reward ---
        reward = 0
        # Reward for horizontal progress
        dx = self.sled_pos[0] - old_pos[0]
        reward += dx * 0.1
        # Small penalty for vertical movement to encourage smooth tracks
        dy = self.sled_pos[1] - old_pos[1]
        reward -= abs(dy) * 0.01

        # --- 4. Check Termination ---
        self.steps += 1
        terminated = False
        
        # Win condition
        if self.sled_pos[0] >= self.FINISH_X:
            terminated = True
            self.game_over = True
            reward += 100 # Large bonus for winning
            self.score += 100

        # Crash conditions
        if not (0 < self.sled_pos[1] < self.HEIGHT):
            terminated = True
            self.game_over = True
            reward -= 10 # Penalty for crashing out of bounds
            self.score -= 10

        # Stoppage condition
        speed = np.linalg.norm(self.sled_vel)
        if speed < self.STOPPED_THRESHOLD:
            self.stopped_frames += 1
        else:
            self.stopped_frames = 0
            
        if self.stopped_frames > self.STOPPED_FRAMES_LIMIT:
            terminated = True
            self.game_over = True
            reward -= 10 # Penalty for getting stuck
            self.score -= 10

        # Step limit
        if self.steps >= self.MAX_STEPS:
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

    def _handle_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY / self.PHYSICS_SUBSTEPS

        # Apply friction
        self.sled_vel *= self.FRICTION

        # Update position
        self.sled_pos += self.sled_vel / self.PHYSICS_SUBSTEPS

        # Collision detection and response
        for p1, p2 in self.track_lines:
            # Vector from line start to sled
            line_vec = p2 - p1
            p1_to_sled = self.sled_pos - p1
            
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            # Project sled position onto the line
            t = np.dot(p1_to_sled, line_vec) / line_len_sq
            t = np.clip(t, 0, 1) # Clamp to segment

            closest_point = p1 + t * line_vec
            dist_vec = self.sled_pos - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)

            if dist_sq < self.SLED_RADIUS ** 2:
                # Collision detected
                # sfx: scrape or clank
                dist = math.sqrt(dist_sq)
                penetration = self.SLED_RADIUS - dist
                
                # Correction vector to push sled out of the line
                if dist > 0:
                    correction_vec = (dist_vec / dist) * penetration
                    self.sled_pos += correction_vec
                else: # Sled is exactly on the line, push it along the normal
                    normal = np.array([-line_vec[1], line_vec[0]])
                    normal /= np.linalg.norm(normal)
                    self.sled_pos += normal * penetration

                # Calculate response velocity
                normal = dist_vec / dist if dist > 0 else np.array([0,1])
                
                # Reflect velocity
                vel_dot_normal = np.dot(self.sled_vel, normal)
                if vel_dot_normal < 0:
                    # Spawn particles on hard impact
                    if abs(vel_dot_normal) > 1.0:
                        self._spawn_particles(self.sled_pos, 5, abs(vel_dot_normal))

                    # Reflect velocity vector
                    self.sled_vel -= 2 * vel_dot_normal * normal
                    self.sled_vel *= self.BOUNCE_DAMPENING
                break # Handle one collision per substep

    def _spawn_particles(self, pos, count, speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            particle_speed = self.np_random.uniform(0.5, 1.5) * speed * 0.5
            vel = np.array([math.cos(angle), math.sin(angle)]) * particle_speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([pos.copy(), vel, lifetime])

    def _update_particles(self):
        # sfx: fizzle
        for p in self.particles:
            p[0] += p[1] # Update position
            p[1] *= 0.95 # Dampen velocity
            p[2] -= 1 # Decrease lifetime
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Draw start/finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (10, 0), (10, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_X, 0), (self.FINISH_X, self.HEIGHT), 3)

        # Draw track
        for p1, p2 in self.track_lines:
            pygame.draw.line(self.screen, self.COLOR_TRACK, p1.astype(int), p2.astype(int), 3)

        # Draw particles
        for pos, _, lifetime in self.particles:
            alpha = max(0, min(255, int(255 * (lifetime / 30.0))))
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos[0]), int(pos[1]), 2, (*self.COLOR_PARTICLE, alpha)
            )

        # Draw sled
        pos_int = self.sled_pos.astype(int)
        # Glow effect
        pygame.gfxdraw.filled_circle(
            self.screen, pos_int[0], pos_int[1], int(self.SLED_RADIUS * 1.5), (*self.COLOR_SLED_GLOW, 50)
        )
        pygame.gfxdraw.aacircle(
            self.screen, pos_int[0], pos_int[1], self.SLED_RADIUS, self.COLOR_SLED
        )
        pygame.gfxdraw.filled_circle(
            self.screen, pos_int[0], pos_int[1], self.SLED_RADIUS, self.COLOR_SLED
        )

        # Draw UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        speed = np.linalg.norm(self.sled_vel)
        score_text = self.font.render(f"Score: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        speed_text = self.font.render(f"Speed: {speed * 10:.1f}", True, self.COLOR_UI_TEXT)
        steps_text = self.font.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)

        self.screen.blit(score_text, (15, self.HEIGHT - 30))
        self.screen.blit(speed_text, (15, self.HEIGHT - 55))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 15, self.HEIGHT - 30))
        
        if self.game_over:
            status_text_str = "GOAL!" if self.sled_pos[0] >= self.FINISH_X else "CRASHED"
            status_text = pygame.font.Font(None, 72).render(status_text_str, True, self.COLOR_UI_TEXT)
            text_rect = status_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(status_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": self.sled_pos.tolist(),
            "sled_vel": self.sled_vel.tolist(),
        }

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Line Rider Gym")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    # Instructions
    print(env.game_description)
    print(env.user_guide)
    print("Manual Controls:")
    print("  - 1, 2, 3, 4, 5: Select line angle (corresponds to action[0])")
    print("  - SPACE: Make line longer (action[1]=1)")
    print("  - SHIFT: Make line shorter (action[2]=1)")
    print("  - ENTER: Submit action and advance frame")

    action = env.action_space.sample() # Start with a default action
    action[0] = 0 # Horizontal line
    action[1] = 0 # Normal length
    action[2] = 0 # Normal length

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    # Submit the action on Enter key
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
                    
                    # Reset action for next turn
                    action = np.array([0, 0, 0])

                    if terminated or truncated:
                        print("Episode Finished!")
                        print(f"Final Score: {info['score']:.2f}")
                        obs, info = env.reset()
                        total_reward = 0

        # Update action based on held keys
        keys = pygame.key.get_pressed()
        # Angle selection
        if keys[pygame.K_1]: action[0] = 0
        if keys[pygame.K_2]: action[0] = 1
        if keys[pygame.K_3]: action[0] = 2
        if keys[pygame.K_4]: action[0] = 3
        if keys[pygame.K_5]: action[0] = 4
        # Length modification
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Render the current state
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Limit FPS for human play

    env.close()