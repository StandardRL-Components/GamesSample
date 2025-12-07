
# Generated: 2025-08-28T03:09:26.833462
# Source Brief: brief_04834.md
# Brief Index: 4834

        
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
        "Controls: Arrow keys draw track segments. Spacebar draws a line in the direction of travel. Shift draws a line straight down."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track for your sled to ride. Reach the finish line as fast as possible, but don't crash!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and World Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 160
        self.WORLD_HEIGHT = 100
        self.WORLD_SCALE = self.SCREEN_HEIGHT / self.WORLD_HEIGHT
        self.X_OFFSET = (self.SCREEN_WIDTH - self.WORLD_WIDTH * self.WORLD_SCALE) / 2

        # Game constants
        self.MAX_STEPS = 1500
        self.FINISH_LINE_X = self.WORLD_WIDTH - 10
        self.MAX_SEGMENTS = 100
        self.LINE_LENGTH = 8.0

        # Physics constants
        self.GRAVITY = 0.02
        self.FRICTION = 0.005
        self.SLED_WIDTH = 4
        self.SLED_HEIGHT = 2
        
        # Color Palette
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TRACK = (220, 220, 220)
        self.COLOR_SLED = (255, 60, 60)
        self.COLOR_SLED_ACCENT = (255, 150, 150)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_START = (80, 200, 120)
        self.COLOR_FINISH = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.sled_pos = None
        self.sled_vel = None
        self.track_segments = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.stationary_steps = None
        self.terminated_reason = ""
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.sled_pos = np.array([10.0, 20.0])
        self.sled_vel = np.array([1.0, 0.0])
        
        # Initial flat track to start
        self.track_segments = [
            (np.array([2.0, 25.0]), np.array([25.0, 25.0]))
        ]
        
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stationary_steps = 0
        self.terminated_reason = ""
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        self._handle_action(movement, space_held, shift_held)
        
        old_pos_x = self.sled_pos[0]
        self._update_physics()
        self._update_particles()
        
        self.steps += 1
        reward = self._calculate_reward(old_pos_x)
        self.score += reward
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_action(self, movement, space_held, shift_held):
        start_point = self.sled_pos
        end_point = None

        if space_held:
            # Draw in direction of sled's velocity
            if np.linalg.norm(self.sled_vel) > 0.1:
                direction = self.sled_vel / np.linalg.norm(self.sled_vel)
            else:
                direction = np.array([1.0, 0.0]) # Default forward
            end_point = start_point + direction * self.LINE_LENGTH
        elif shift_held:
            # Draw straight down
            end_point = start_point + np.array([0, self.LINE_LENGTH])
        elif movement > 0:
            directions = {
                1: np.array([0, -1]),  # Up
                2: np.array([0, 1]),   # Down
                3: np.array([-1, 0]),  # Left
                4: np.array([1, 0]),   # Right
            }
            end_point = start_point + directions[movement] * self.LINE_LENGTH
        
        if end_point is not None:
            # Add new segment and maintain max length
            self.track_segments.append((start_point.copy(), end_point.copy()))
            if len(self.track_segments) > self.MAX_SEGMENTS:
                self.track_segments.pop(0)

    def _update_physics(self):
        # Apply gravity and predict next position
        self.sled_vel[1] += self.GRAVITY
        potential_pos = self.sled_pos + self.sled_vel

        best_support_point = None
        best_segment = None
        min_dist_to_support = float('inf')

        # Find the best support line below the sled
        for seg_start, seg_end in self.track_segments:
            x1, y1 = seg_start
            x2, y2 = seg_end
            if x1 > x2: x1, y1, x2, y2 = x2, y2, x1, y1 # Sort by x

            sled_x = potential_pos[0]
            if x1 <= sled_x <= x2:
                # Interpolate the track's Y at the sled's X
                line_y = y1 + ((sled_x - x1) / (x2 - x1) if x2 > x1 else 0) * (y2 - y1)
                
                # Check if the track is below the sled and within a reasonable distance
                if line_y >= potential_pos[1] and (line_y - potential_pos[1]) < min_dist_to_support:
                    min_dist_to_support = line_y - potential_pos[1]
                    best_support_point = np.array([sled_x, line_y])
                    best_segment = (np.array([x1, y1]), np.array([x2, y2]))

        # Collision response if supported
        if best_segment is not None and min_dist_to_support < self.SLED_HEIGHT:
            self.stationary_steps = 0
            
            # Snap position to the track surface
            self.sled_pos = best_support_point.copy()
            self.sled_pos[1] -= self.SLED_HEIGHT / 2

            # Align velocity with the track
            seg_start, seg_end = best_segment
            line_vec = seg_end - seg_start
            if np.linalg.norm(line_vec) > 0:
                tangent = line_vec / np.linalg.norm(line_vec)
                gravity_vec = np.array([0, self.GRAVITY])
                
                # Acceleration along the line due to gravity
                accel_on_line = np.dot(gravity_vec, tangent) * tangent
                self.sled_vel += accel_on_line
                
                # Project velocity onto tangent to enforce sliding
                self.sled_vel = np.dot(self.sled_vel, tangent) * tangent
                self.sled_vel *= (1.0 - self.FRICTION) # Apply friction
        else:
            # Freefall
            self.sled_pos = potential_pos

        # Update stationary counter
        if np.linalg.norm(self.sled_vel) < 0.01:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0

        # Add particle for trail
        if self.steps % 2 == 0:
            self._add_particle()

    def _add_particle(self):
        p_pos = self.sled_pos.copy()
        p_life = random.uniform(20, 40)
        p_size = random.uniform(self.WORLD_SCALE, self.WORLD_SCALE * 2.5)
        self.particles.append({'pos': p_pos, 'life': p_life, 'max_life': p_life, 'size': p_size})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1

    def _calculate_reward(self, old_pos_x):
        reward = -0.01 # Cost of time
        
        # Reward for forward progress
        progress = self.sled_pos[0] - old_pos_x
        if progress > 0:
            reward += 0.1 * progress

        return reward

    def _check_termination(self):
        if self.game_over:
            return True

        # Win condition
        if self.sled_pos[0] >= self.FINISH_LINE_X:
            self.score += 10.0
            self.game_over = True
            self.terminated_reason = "Finished!"
            return True

        # Crash conditions
        if not (0 < self.sled_pos[0] < self.WORLD_WIDTH and 0 < self.sled_pos[1] < self.WORLD_HEIGHT):
            self.score -= 5.0
            self.game_over = True
            self.terminated_reason = "Out of Bounds"
            return True
        if self.stationary_steps >= 5:
            self.score -= 5.0
            self.game_over = True
            self.terminated_reason = "Stalled"
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.terminated_reason = "Time Up"
            return True

        return False

    def _world_to_screen(self, pos):
        x = self.X_OFFSET + pos[0] * self.WORLD_SCALE
        y = pos[1] * self.WORLD_SCALE
        return int(x), int(y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for i in range(0, self.WORLD_WIDTH, 10):
            start = self._world_to_screen((i, 0))
            end = self._world_to_screen((i, self.WORLD_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(0, self.WORLD_HEIGHT, 10):
            start = self._world_to_screen((0, i))
            end = self._world_to_screen((self.WORLD_WIDTH, i))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw start and finish lines
        start_line_s = self._world_to_screen((5, 0))
        start_line_e = self._world_to_screen((5, self.WORLD_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_START, start_line_s, start_line_e, 3)

        finish_line_s = self._world_to_screen((self.FINISH_LINE_X, 0))
        finish_line_e = self._world_to_screen((self.FINISH_LINE_X, self.WORLD_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_FINISH, finish_line_s, finish_line_e, 3)

        # Draw particles
        for p in self.particles:
            sx, sy = self._world_to_screen(p['pos'])
            alpha = int(255 * (p['life'] / p['max_life']))
            radius = int(p['size'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, (*self.COLOR_PARTICLE, alpha))

        # Draw track segments
        for start, end in self.track_segments:
            s_start = self._world_to_screen(start)
            s_end = self._world_to_screen(end)
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, s_start, s_end, True)
            pygame.draw.line(self.screen, self.COLOR_TRACK, s_start, s_end, 2)

        # Draw sled
        sled_center_s = self._world_to_screen(self.sled_pos)
        w = self.SLED_WIDTH * self.WORLD_SCALE
        h = self.SLED_HEIGHT * self.WORLD_SCALE
        sled_rect = pygame.Rect(sled_center_s[0] - w/2, sled_center_s[1] - h/2, w, h)
        
        angle = -math.degrees(math.atan2(self.sled_vel[1], self.sled_vel[0])) if np.linalg.norm(self.sled_vel) > 0 else 0
        
        # Create a surface for the sled to rotate it
        sled_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(sled_surface, self.COLOR_SLED, (0, 0, w, h), border_radius=2)
        pygame.draw.rect(sled_surface, self.COLOR_SLED_ACCENT, (w * 0.1, h * 0.1, w * 0.8, h * 0.8), border_radius=2)
        rotated_sled = pygame.transform.rotate(sled_surface, angle)
        
        new_rect = rotated_sled.get_rect(center=sled_rect.center)
        self.screen.blit(rotated_sled, new_rect.topleft)

    def _render_ui(self):
        # Time display
        time_text = f"TIME: {self.steps / 30.0:.1f}s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (15, 10))

        # Speed display
        speed = np.linalg.norm(self.sled_vel) * 10
        speed_text = f"SPEED: {speed:.1f}"
        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(speed_surf, (self.SCREEN_WIDTH - speed_surf.get_width() - 15, 10))
        
        # Score display
        score_text = f"SCORE: {self.score:.2f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 35))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_surf = self.font_big.render(self.terminated_reason, True, self.COLOR_TEXT)
            text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_surf, text_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Sled Drawer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # Action mapping for human keyboard input
        keys = pygame.key.get_pressed()
        mov_action = 0 # No-op
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        clock.tick(30) # Limit to 30 FPS for playability
        
    print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
    env.close()