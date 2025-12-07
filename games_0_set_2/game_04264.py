
# Generated: 2025-08-28T01:54:21.875161
# Source Brief: brief_04264.md
# Brief Index: 4264

        
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

    # Class attributes for state that persists across resets
    finish_line_distance_base = 200.0
    successful_episodes = 0

    user_guide = (
        "Use arrow keys to set line angle (None=horizontal). Hold Space for longer lines. Hold Shift for no line."
    )

    game_description = (
        "Draw a track for a sled to ride on. Guide it to the finish line by drawing different line segments."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 18, 26)
    COLOR_TRACK = (66, 165, 245)
    COLOR_SLED = (255, 255, 255)
    COLOR_START = (76, 175, 80)
    COLOR_FINISH = (244, 67, 54)
    COLOR_TEXT = (224, 224, 224)
    COLOR_PARTICLE = (180, 180, 180)
    
    # Physics
    GRAVITY = 0.15
    FRICTION = 0.995
    SLED_RADIUS = 4
    MIN_SPEED_FOR_STUCK = 0.1
    STUCK_FRAMES_LIMIT = 100
    MAX_STEPS = 2000

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

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
        
        self.font_ui = pygame.font.SysFont("Consolas", 18)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.sled_pos = np.zeros(2, dtype=float)
        self.sled_vel = np.zeros(2, dtype=float)
        self.sled_angle = 0.0
        self.track_points = []
        self.particles = []
        self.camera_x = 0.0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.stuck_counter = 0
        self.last_sled_x = 0.0
        self.finish_line_x = 0.0

        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game State
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        # Sled State
        start_pos = np.array([100.0, 200.0])
        self.sled_pos = start_pos.copy()
        self.sled_pos[1] -= self.SLED_RADIUS * 2 # Start slightly above the initial track
        self.sled_vel = np.array([1.0, 0.0])
        self.sled_angle = 0.0
        self.last_sled_x = self.sled_pos[0]
        self.stuck_counter = 0
        
        # Track State
        self.track_points = [start_pos]
        
        # World State
        self.finish_line_x = start_pos[0] + self.finish_line_distance_base
        self.camera_x = 0.0
        
        # Effects
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle action: Add a new line segment to the track
        self._handle_action(action)
        
        # 2. Update physics and game state
        self._update_physics()
        self._update_particles()
        self._update_camera()

        self.steps += 1
        
        # 3. Check for termination conditions
        terminated = self._check_termination()
        
        # 4. Calculate reward
        reward = self._calculate_reward(terminated)
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if shift_held:
            # Shift creates a gap, so we do nothing to the track
            return

        last_point = self.track_points[-1]
        length = 10.0 if space_held else 5.0
        
        angle_rad = 0
        if movement == 0: # None -> Horizontal Right
            angle_rad = 0
        elif movement == 1: # Up -> Up-Right
            angle_rad = -math.pi / 4
        elif movement == 2: # Down -> Down-Right
            angle_rad = math.pi / 4
        elif movement == 3: # Left -> Up-Left
            angle_rad = -3 * math.pi / 4
        elif movement == 4: # Right -> Down-Left
            angle_rad = 3 * math.pi / 4

        dx = length * math.cos(angle_rad)
        dy = length * math.sin(angle_rad)
        
        new_point = last_point + np.array([dx, dy])
        
        # Clamp new point to screen bounds to prevent impossible tracks
        new_point[0] = max(0, new_point[0])
        new_point[1] = np.clip(new_point[1], 0, self.SCREEN_HEIGHT)

        self.track_points.append(new_point)

    def _update_physics(self):
        # Store position for reward calculation
        self.last_sled_x = self.sled_pos[0]

        # Apply gravity (freefall)
        self.sled_vel[1] += self.GRAVITY

        # Update position
        self.sled_pos += self.sled_vel

        # Collision detection and response
        support_line_segment = None
        highest_support_y = -float('inf')

        # Find the highest line segment directly under the sled
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]

            # Bounding box check
            if min(p1[0], p2[0]) - self.SLED_RADIUS <= self.sled_pos[0] <= max(p1[0], p2[0]) + self.SLED_RADIUS:
                # Calculate line's y at sled's x
                line_dx = p2[0] - p1[0]
                if abs(line_dx) > 1e-6:
                    t = (self.sled_pos[0] - p1[0]) / line_dx
                    if 0 <= t <= 1:
                        line_y = p1[1] + t * (p2[1] - p1[1])
                        if self.sled_pos[1] + self.SLED_RADIUS >= line_y > highest_support_y:
                            highest_support_y = line_y
                            support_line_segment = (p1, p2)

        if support_line_segment is not None:
            # Collision response
            # Snap position to the line
            self.sled_pos[1] = highest_support_y - self.SLED_RADIUS
            
            p1, p2 = support_line_segment
            line_vec = p2 - p1
            line_angle = math.atan2(line_vec[1], line_vec[0])
            self.sled_angle = line_angle

            # Project velocity onto the line
            speed = np.linalg.norm(self.sled_vel)
            new_speed = speed * self.FRICTION
            
            self.sled_vel[0] = new_speed * math.cos(line_angle)
            self.sled_vel[1] = new_speed * math.sin(line_angle)
            
            # Add particles for sliding effect
            if new_speed > 1.0 and self.np_random.random() < 0.7:
                self._create_particles(1, self.sled_pos + np.array([0, self.SLED_RADIUS]), count_mult=0.5)
        else:
            # In air, angle follows velocity
            self.sled_angle = math.atan2(self.sled_vel[1], self.sled_vel[0])

    def _check_termination(self):
        # Win condition
        if self.sled_pos[0] >= self.finish_line_x:
            self.game_over = True
            self.win = True
            GameEnv.successful_episodes += 1
            if GameEnv.successful_episodes % 5 == 0 and GameEnv.successful_episodes > 0:
                GameEnv.finish_line_distance_base += 5.0
            return True

        # Loss conditions
        # Out of bounds
        if not (0 <= self.sled_pos[0] <= self.SCREEN_WIDTH + self.camera_x + 50 and 0 <= self.sled_pos[1] <= self.SCREEN_HEIGHT):
            self.game_over = True
            self._create_particles(50, self.sled_pos, color=self.COLOR_FINISH, count_mult=2.0) # Crash explosion
            return True
        
        # Stuck
        speed = np.linalg.norm(self.sled_vel)
        if speed < self.MIN_SPEED_FOR_STUCK:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        if self.stuck_counter >= self.STUCK_FRAMES_LIMIT:
            self.game_over = True
            self._create_particles(50, self.sled_pos, color=self.COLOR_FINISH, count_mult=2.0)
            return True

        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _calculate_reward(self, terminated):
        if terminated:
            if self.win:
                # Reward for winning, scaled by difficulty
                return 10.0 + (self.finish_line_distance_base / 50.0)
            else:
                # Large penalty for crashing or getting stuck/timeout
                return -10.0
        
        reward = 0.0
        # Reward for forward progress
        progress = self.sled_pos[0] - self.last_sled_x
        reward += progress * 0.5

        # Small penalty for being slow to encourage momentum
        if np.linalg.norm(self.sled_vel) < 1.0:
            reward -= 0.01

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw start and finish lines
        start_x = int(self.track_points[0][0] - self.camera_x)
        pygame.draw.line(self.screen, self.COLOR_START, (start_x, 0), (start_x, self.SCREEN_HEIGHT), 2)
        finish_x = int(self.finish_line_x - self.camera_x)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x, 0), (finish_x, self.SCREEN_HEIGHT), 2)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            alpha = p['life'] / p['max_life']
            color = (p['color'][0], p['color'][1], p['color'][2], int(255 * alpha))
            radius = int(p['size'] * alpha)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Draw track
        if len(self.track_points) > 1:
            points_on_screen = [(int(p[0] - self.camera_x), int(p[1])) for p in self.track_points]
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, points_on_screen, 2)
        
        # Draw sled
        if not (self.game_over and not self.win): # Don't draw sled if crashed
            sled_screen_pos = (self.sled_pos - np.array([self.camera_x, 0])).astype(int)
            
            # Create a rectangle for the sled and rotate it
            w, h = 12, 5
            points = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
            rotated_points = []
            for x, y in points:
                rx = x * math.cos(self.sled_angle) - y * math.sin(self.sled_angle)
                ry = x * math.sin(self.sled_angle) + y * math.cos(self.sled_angle)
                rotated_points.append((sled_screen_pos[0] + rx, sled_screen_pos[1] + ry))
            
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_SLED)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_SLED)

    def _render_ui(self):
        # Speed display
        speed = np.linalg.norm(self.sled_vel)
        speed_text = f"Speed: {speed:.1f}"
        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(speed_surf, (10, 10))
        
        # Score display
        score_text = f"Score: {self.score:.2f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Steps display
        steps_text = f"Step: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.font_ui.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (10, 30))

        # Game Over message
        if self.game_over:
            msg = "FINISH!" if self.win else "CRASHED"
            color = self.COLOR_START if self.win else self.COLOR_FINISH
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_surf, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": self.sled_pos.copy(),
            "sled_vel": self.sled_vel.copy(),
            "win": self.win
        }

    def _update_camera(self):
        # Camera follows the sled horizontally, keeping it in the center of the screen
        target_camera_x = self.sled_pos[0] - self.SCREEN_WIDTH / 2
        # Smooth camera movement
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

    def _update_particles(self):
        # Update and remove dead particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_particles(self, num, pos, color=COLOR_PARTICLE, count_mult=1.0):
        for _ in range(int(num * count_mult)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(1, 4),
                'color': color
            })

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

if __name__ == '__main__':
    # To play the game manually, you can use gymnasium.utils.play
    # Note: This requires installing additional dependencies like 'pygame' for the play utility
    # pip install gymnasium[classic-control]
    #
    # from gymnasium.utils.play import play
    #
    # env = GameEnv(render_mode="rgb_array")
    #
    # # Define keys for manual play
    # # Mapping:
    # # Arrow keys -> movement
    # # Space -> space_held
    # # Left Shift -> shift_held
    # # This doesn't map perfectly to MultiDiscrete, so play.py might need adjustment
    # # or a wrapper to convert key presses to MultiDiscrete actions.
    # # The default play utility might not support MultiDiscrete well.
    #
    # # A simple loop to test rendering and basic functionality
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Create a window to display the game
    pygame.display.set_caption("Line Rider Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    for _ in range(2000):
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Win: {info.get('win')}")
            obs, info = env.reset()
            total_reward = 0
            # Pause briefly on reset
            pygame.time.wait(1000)

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
        
        # Control frame rate
        env.clock.tick(30)

    env.close()