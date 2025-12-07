
# Generated: 2025-08-27T23:46:26.763784
# Source Brief: brief_03572.md
# Brief Index: 3572

        
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

    user_guide = (
        "Controls: Use arrows to set draw direction. Hold Space for a longer line, Shift for a shorter one. "
        "Draw a track to guide the sled to the finish line."
    )

    game_description = (
        "A physics-based puzzle game. Draw lines to create a track for a sled. "
        "Guide it safely to the finish line before time runs out to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH = 640
        self.HEIGHT = 400
        self.WORLD_WIDTH = self.WIDTH * 4 # Allow for a longer track

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_TRACK = (220, 220, 220)
        self.COLOR_SLED = (255, 80, 80)
        self.COLOR_SLED_OUTLINE = (255, 150, 150)
        self.COLOR_START = (80, 255, 80)
        self.COLOR_FINISH = (255, 80, 80)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        
        # Physics and game constants
        self.GRAVITY = 0.3
        self.FRICTION = 0.99
        self.BOUNCE = 0.4
        self.MAX_STEPS = 1500
        self.TIME_LIMIT = 45.0  # seconds
        self.SLED_SIZE = 10
        self.STUCK_LIMIT = 120 # steps before terminating if stuck
        
        self.START_LINE_X = 80
        self.FINISH_LINE_X = self.WORLD_WIDTH - 120

        # Initialize state variables
        self.sled_pos = [0.0, 0.0]
        self.sled_vel = [0.0, 0.0]
        self.track_segments = []
        self.last_track_endpoint = (0, 0)
        self.last_draw_direction = np.array([1.0, 0.0])
        self.particles = []
        self.camera_x = 0.0
        self.steps = 0
        self.score = 0.0
        self.timer = 0.0
        self.game_over = False
        self.game_over_message = ""
        self.max_progress_x = 0.0
        self.stuck_counter = 0
        self.last_stuck_check_pos = 0.0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.sled_pos = [float(self.START_LINE_X), float(self.HEIGHT / 2 - 50)]
        self.sled_vel = [2.0, 0.0]
        self.track_segments = []
        
        # Create initial platform
        start_platform_y = self.HEIGHT / 2
        p1 = (self.START_LINE_X - 50, start_platform_y)
        p2 = (self.START_LINE_X + 50, start_platform_y)
        self.track_segments.append((p1, p2))
        self.last_track_endpoint = p2

        self.particles = []
        self.camera_x = 0.0
        self.steps = 0
        self.score = 0.0
        self.timer = 0.0
        self.game_over = False
        self.game_over_message = ""
        self.max_progress_x = self.sled_pos[0]
        self.stuck_counter = 0
        self.last_stuck_check_pos = self.sled_pos[0]
        self.last_draw_direction = np.array([1.0, 0.0])

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- 1. Handle Action: Draw a new line segment ---
        self._handle_action(action)

        # --- 2. Update Physics ---
        self._update_physics()

        # --- 3. Update Game State ---
        self.steps += 1
        self.timer += 1.0 / 30.0  # Simulate 30 FPS
        
        # --- 4. Calculate Reward and Check Termination ---
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            
        # --- 5. Update Camera ---
        self._update_camera()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        direction_map = {
            0: np.array([1.0, 0.0]),  # No-op -> draw flat
            1: np.array([0.0, -1.0]), # Up
            2: np.array([0.0, 1.0]),  # Down
            3: np.array([-1.0, 0.0]), # Left
            4: np.array([1.0, 0.0]),  # Right
        }
        
        direction = direction_map[movement]
        if movement != 0:
            self.last_draw_direction = direction

        base_length = 40.0
        if space_held:
            length = base_length * 2.0
        elif shift_held:
            length = base_length * 0.5
        else:
            length = base_length

        start_point = self.last_track_endpoint
        end_point = (start_point[0] + direction[0] * length, 
                     start_point[1] + direction[1] * length)

        # Prevent drawing into the "past"
        if end_point[0] < start_point[0] - 5:
             end_point = (start_point[0] - 5, end_point[1])

        # Clamp to world bounds
        end_point = (
            max(0, min(self.WORLD_WIDTH, end_point[0])),
            max(0, min(self.HEIGHT, end_point[1]))
        )

        if np.linalg.norm(np.array(start_point) - np.array(end_point)) > 1:
            self.track_segments.append((start_point, end_point))
            self.last_track_endpoint = end_point
        
        # Prune old track segments for performance
        if len(self.track_segments) > 200:
            self.track_segments.pop(0)

    def _update_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY

        # Collision detection and response
        collided = False
        for p1, p2 in self.track_segments:
            # Broad phase check
            seg_min_x, seg_max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
            if self.sled_pos[0] + self.SLED_SIZE < seg_min_x or self.sled_pos[0] - self.SLED_SIZE > seg_max_x:
                continue

            # Closest point on segment to sled
            line_vec = np.array(p2) - np.array(p1)
            p1_to_sled = self.sled_pos - np.array(p1)
            line_len_sq = np.dot(line_vec, line_vec)

            if line_len_sq == 0: continue

            t = np.dot(p1_to_sled, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            
            closest_point = np.array(p1) + t * line_vec
            dist_vec = self.sled_pos - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)

            if dist_sq < self.SLED_SIZE**2:
                collided = True
                # SFX: // sled_scrape.wav
                dist = math.sqrt(dist_sq)
                penetration = self.SLED_SIZE - dist
                
                # Resolve penetration
                if dist > 0:
                    self.sled_pos += (dist_vec / dist) * penetration

                # Calculate response
                normal = dist_vec / dist if dist > 0 else np.array([0, -1])
                
                vel_dot_normal = np.dot(self.sled_vel, normal)
                
                if vel_dot_normal < 0:
                    # Separate velocity into normal and tangential components
                    normal_vel = vel_dot_normal * normal
                    tangent_vel = np.array(self.sled_vel) - normal_vel

                    # Apply bounce and friction
                    self.sled_vel = list((tangent_vel * self.FRICTION) - (normal_vel * self.BOUNCE))
                
                # Create collision particles
                self._create_particles(self.sled_pos, 5)
                break # Only handle one collision per frame

        # Update position
        self.sled_pos[0] += self.sled_vel[0]
        self.sled_pos[1] += self.sled_vel[1]
        
    def _calculate_reward(self):
        reward = -0.01  # Cost of living

        # Reward for forward progress
        progress = self.sled_pos[0] - self.max_progress_x
        if progress > 0:
            reward += progress * 0.1
            self.max_progress_x = self.sled_pos[0]
        
        # Penalize for moving backward
        elif progress < -1.0: # Small tolerance
            reward += progress * 0.05
        
        # Check termination conditions for large rewards/penalties
        if self.sled_pos[0] >= self.FINISH_LINE_X:
            reward += 10.0 # Reached finish line
            if self.timer <= self.TIME_LIMIT:
                reward += 50.0 # Bonus for finishing in time
        elif self._is_crashed():
            reward -= 10.0 # Crashed

        return reward

    def _is_crashed(self):
        return (self.sled_pos[1] > self.HEIGHT + self.SLED_SIZE * 2 or
                self.sled_pos[1] < -self.SLED_SIZE * 2 or
                self.sled_pos[0] < self.camera_x - self.SLED_SIZE * 2)
    
    def _check_termination(self):
        # Check for win
        if self.sled_pos[0] >= self.FINISH_LINE_X:
            self.game_over_message = "FINISH!"
            # SFX: // win_fanfare.wav
            return True

        # Check for crash
        if self._is_crashed():
            self.game_over_message = "CRASHED"
            # SFX: // crash_sound.wav
            return True

        # Check for time limit
        if self.timer >= self.TIME_LIMIT:
            self.game_over_message = "TIME UP"
            return True

        # Check max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over_message = "MAX STEPS"
            return True
        
        # Check if stuck
        if abs(self.sled_pos[0] - self.last_stuck_check_pos) < 0.5:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_stuck_check_pos = self.sled_pos[0]
        
        if self.stuck_counter >= self.STUCK_LIMIT:
            self.game_over_message = "STUCK"
            return True

        return False

    def _update_camera(self):
        # Camera follows sled with a dead zone in the middle of the screen
        dead_zone_start = self.camera_x + self.WIDTH * 0.4
        dead_zone_end = self.camera_x + self.WIDTH * 0.6
        
        if self.sled_pos[0] > dead_zone_end:
            self.camera_x += self.sled_pos[0] - dead_zone_end
        elif self.sled_pos[0] < dead_zone_start:
            self.camera_x += self.sled_pos[0] - dead_zone_start
        
        self.camera_x = max(0, self.camera_x)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WORLD_WIDTH, 40):
            x = int(i - self.camera_x)
            if 0 <= x < self.WIDTH:
                pygame.draw.aaline(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw start/finish lines
        start_x = int(self.START_LINE_X - self.camera_x)
        finish_x = int(self.FINISH_LINE_X - self.camera_x)
        pygame.draw.line(self.screen, self.COLOR_START, (start_x, 0), (start_x, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x, 0), (finish_x, self.HEIGHT), 2)

        # Draw track
        for p1, p2 in self.track_segments:
            cam_p1 = (int(p1[0] - self.camera_x), int(p1[1]))
            cam_p2 = (int(p2[0] - self.camera_x), int(p2[1]))
            pygame.draw.line(self.screen, self.COLOR_TRACK, cam_p1, cam_p2, 3)

        # Draw drawing cursor
        cursor_pos = (int(self.last_track_endpoint[0] - self.camera_x), int(self.last_track_endpoint[1]))
        pygame.gfxdraw.aacircle(self.screen, cursor_pos[0], cursor_pos[1], 5, self.COLOR_SLED_OUTLINE)
        
        # Draw particles
        self._update_and_draw_particles()

        # Draw sled
        sled_x = int(self.sled_pos[0] - self.camera_x)
        sled_y = int(self.sled_pos[1])
        sled_rect = pygame.Rect(sled_x - self.SLED_SIZE // 2, sled_y - self.SLED_SIZE // 2, self.SLED_SIZE, self.SLED_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect)
        pygame.draw.rect(self.screen, self.COLOR_SLED_OUTLINE, sled_rect, 1)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {max(0, self.TIME_LIMIT - self.timer):.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            msg_surf = self.font_large.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'max_life': lifetime})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                cam_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
                radius = int((p['life'] / p['max_life']) * 3)
                if radius > 0:
                    alpha = int((p['life'] / p['max_life']) * 200)
                    color = self.COLOR_PARTICLE + (alpha,)
                    pygame.gfxdraw.filled_circle(self.screen, cam_pos[0], cam_pos[1], radius, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "sled_pos_x": self.sled_pos[0],
            "sled_pos_y": self.sled_pos[1],
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action = env.action_space.sample() # Replace with your agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Step: {step_count}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")

    print(f"Episode finished after {step_count} steps. Final Score: {total_reward:.2f}")
    env.close()