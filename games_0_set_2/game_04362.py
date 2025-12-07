
# Generated: 2025-08-28T02:10:47.937548
# Source Brief: brief_04362.md
# Brief Index: 4362

        
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

    user_guide = (
        "Controls: Arrow keys to draw short track segments. "
        "Space to draw a long segment in the rider's direction. "
        "Shift to draw an upward curve."
    )

    game_description = (
        "Draw a track in real-time for a physics-based rider to navigate. "
        "Reach the checkpoints and the finish line before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    WIDTH, HEIGHT = 640, 400
    # Physics
    GRAVITY = 0.3
    FRICTION = 0.99
    BOUNCE_DAMPENING = 0.7
    MIN_VELOCITY_FOR_REWARD = 0.5
    # Rider
    RIDER_RADIUS = 8
    TRAIL_LENGTH = 30
    # Track Drawing
    SHORT_SEGMENT_LENGTH = 30
    LONG_SEGMENT_LENGTH = 80
    CURVE_SEGMENTS = 5
    MAX_TRACK_SEGMENTS = 200
    # Time and Steps
    TIME_LIMIT_SECONDS = 180
    MAX_STEPS = 5000
    FPS = 30
    
    # --- Colors ---
    COLOR_BG = (224, 224, 224)
    COLOR_RIDER = (33, 33, 33)
    COLOR_TRACK = (33, 33, 33)
    COLOR_START = (76, 175, 80)
    COLOR_FINISH = (244, 67, 54)
    COLOR_CHECKPOINT = (33, 150, 243)
    COLOR_CHECKPOINT_REACHED = (13, 71, 161)
    COLOR_UI_TEXT = (33, 33, 33)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_ui = pygame.font.SysFont("Helvetica", 24)
            self.font_game_over = pygame.font.SysFont("Helvetica", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 48)

        self.game_over_text = ""
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_text = ""
        self.time_remaining = self.TIME_LIMIT_SECONDS

        self.rider_pos = np.array([60.0, 100.0])
        self.rider_vel = np.array([2.0, 0.0])
        self.last_rider_x = self.rider_pos[0]

        self.rider_trail = deque(maxlen=self.TRAIL_LENGTH)
        self.track_segments = deque(maxlen=self.MAX_TRACK_SEGMENTS)
        
        # Initial flat ground
        self.track_segments.append(((0, 150), (100, 150)))

        self.finish_line_x = self.WIDTH - 40
        
        self.checkpoints = [
            (self.WIDTH * 0.3, self.HEIGHT * 0.7),
            (self.WIDTH * 0.55, self.HEIGHT * 0.3),
            (self.WIDTH * 0.75, self.HEIGHT * 0.6),
        ]
        self.checkpoints_reached = [False] * len(self.checkpoints)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)
        
        dt = 1.0 / self.FPS
        reward = 0

        if self.game_over:
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info(),
            )

        self._handle_action(action)
        self._update_physics(dt)
        
        self.steps += 1
        self.time_remaining -= dt

        reward += self._calculate_reward()
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        self.score += reward
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        start_pos = tuple(self.rider_pos)

        if space_held:
            # Draw long segment in direction of velocity
            vel_norm = np.linalg.norm(self.rider_vel)
            if vel_norm > 0.1:
                direction = self.rider_vel / vel_norm
                end_pos = self.rider_pos + direction * self.LONG_SEGMENT_LENGTH
                self.track_segments.append((start_pos, tuple(end_pos)))
        
        elif shift_held:
            # Draw an upward curve
            p0 = self.rider_pos
            p1 = self.rider_pos + np.array([self.LONG_SEGMENT_LENGTH * 0.5, -self.LONG_SEGMENT_LENGTH * 0.8])
            p2 = self.rider_pos + np.array([self.LONG_SEGMENT_LENGTH, -self.LONG_SEGMENT_LENGTH * 0.3])
            
            last_pt = p0
            for i in range(1, self.CURVE_SEGMENTS + 1):
                t = i / self.CURVE_SEGMENTS
                # Quadratic Bezier curve formula
                pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
                self.track_segments.append((tuple(last_pt), tuple(pt)))
                last_pt = pt

        elif movement != 0:
            # Draw short segments with arrow keys
            end_pos = np.copy(self.rider_pos)
            if movement == 1: # Up
                end_pos[1] -= self.SHORT_SEGMENT_LENGTH
            elif movement == 2: # Down
                end_pos[1] += self.SHORT_SEGMENT_LENGTH
            elif movement == 3: # Left
                end_pos[0] -= self.SHORT_SEGMENT_LENGTH
            elif movement == 4: # Right
                end_pos[0] += self.SHORT_SEGMENT_LENGTH
            self.track_segments.append((start_pos, tuple(end_pos)))

    def _update_physics(self, dt):
        # Apply gravity
        self.rider_vel[1] += self.GRAVITY

        # Update position
        self.rider_pos += self.rider_vel

        # Collision detection and response
        collided = False
        for p1, p2 in self.track_segments:
            p1, p2 = np.array(p1), np.array(p2)
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0:
                continue

            t = np.dot(self.rider_pos - p1, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            closest_point = p1 + t * line_vec
            
            dist_vec = self.rider_pos - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)

            if dist_sq < self.RIDER_RADIUS**2:
                collided = True
                dist = math.sqrt(dist_sq)
                # Resolve penetration
                penetration_vec = (dist_vec / dist) * (self.RIDER_RADIUS - dist)
                self.rider_pos += penetration_vec

                # Calculate response
                normal = dist_vec / dist
                vel_component_normal = np.dot(self.rider_vel, normal)
                
                if vel_component_normal < 0:
                    # Reflect velocity
                    self.rider_vel -= (1 + self.BOUNCE_DAMPENING) * vel_component_normal * normal
        
        if collided:
            self.rider_vel *= self.FRICTION

        # Update trail
        self.rider_trail.append(tuple(self.rider_pos))
    
    def _calculate_reward(self):
        reward = 0
        
        # Forward progress reward
        progress = self.rider_pos[0] - self.last_rider_x
        if progress > 0:
            reward += 0.1 * progress
        self.last_rider_x = self.rider_pos[0]

        # Slow speed penalty
        if np.linalg.norm(self.rider_vel) < self.MIN_VELOCITY_FOR_REWARD:
            reward -= 0.01

        # Checkpoint reward
        for i, pos in enumerate(self.checkpoints):
            if not self.checkpoints_reached[i]:
                dist_sq = (self.rider_pos[0] - pos[0])**2 + (self.rider_pos[1] - pos[1])**2
                if dist_sq < (self.RIDER_RADIUS * 2)**2:
                    self.checkpoints_reached[i] = True
                    reward += 10.0 # Brief specified +1, but +10 makes it more significant
                    # sound: checkpoint_get.wav
        
        return reward

    def _check_termination(self):
        # Win condition
        if self.rider_pos[0] >= self.finish_line_x:
            self.game_over_text = "FINISH!"
            return True, 100.0

        # Loss conditions
        if not (0 < self.rider_pos[0] < self.WIDTH and 0 < self.rider_pos[1] < self.HEIGHT):
            self.game_over_text = "CRASHED!"
            return True, -100.0
        
        if self.time_remaining <= 0:
            self.game_over_text = "TIME UP!"
            return True, -100.0
            
        if self.steps >= self.MAX_STEPS:
            return True, -100.0 # Penalize for running out of steps

        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Start and Finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (40, 0), (40, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, 0), (self.finish_line_x, self.HEIGHT), 3)

        # Checkpoints
        for i, pos in enumerate(self.checkpoints):
            color = self.COLOR_CHECKPOINT_REACHED if self.checkpoints_reached[i] else self.COLOR_CHECKPOINT
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 12, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 12, color)

        # Track segments
        for p1, p2 in self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 3)

        # Rider trail
        for i, pos in enumerate(self.rider_trail):
            alpha = int(255 * (i / self.TRAIL_LENGTH))
            color = (*self.COLOR_RIDER, alpha)
            radius = int(self.RIDER_RADIUS * 0.5 * (i / self.TRAIL_LENGTH))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, color)

        # Rider
        rider_x, rider_y = int(self.rider_pos[0]), int(self.rider_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS // 2, self.COLOR_BG)

    def _render_ui(self):
        # Time remaining
        time_text = f"Time: {max(0, int(self.time_remaining))}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Game Over Text
        if self.game_over and self.game_over_text:
            text_surf = self.font_game_over.render(self.game_over_text, True, self.COLOR_FINISH)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "checkpoints_reached": sum(self.checkpoints_reached),
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


# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This requires a window, so we'll re-init pygame with a display
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    
    obs, info = env.reset()
    terminated = False
    
    print(GameEnv.user_guide)

    while not terminated:
        # Action defaults
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()