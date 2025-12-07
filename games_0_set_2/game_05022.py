
# Generated: 2025-08-28T03:43:36.998078
# Source Brief: brief_05022.md
# Brief Index: 5022

        
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
        "Controls: Use arrow keys to draw the track for the rider. "
        "↑ draws up, ↓ draws down, → draws forward. No action draws a straight line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track for a physics-based rider to navigate to the finish line. "
        "Collect green boosts for extra speed and reach the blue finish line as fast as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FINISH_LINE_X = 2500
    MAX_STEPS = 2000
    FPS = 30

    # Colors
    COLOR_BG = (230, 230, 240) # Light gray
    COLOR_GRID = (210, 210, 220)
    COLOR_TRACK = (255, 255, 255)
    COLOR_RIDER = (255, 80, 80)
    COLOR_RIDER_GLOW = (255, 150, 150)
    COLOR_BOOST = (80, 255, 80)
    COLOR_FINISH = (80, 80, 255)
    COLOR_TEXT = (50, 50, 70)
    
    # Physics
    GRAVITY = 0.3
    RIDER_RADIUS = 8
    FRICTION = 0.998
    BOUNCE_DAMPENING = 0.7
    
    # Gameplay
    TRACK_SEGMENT_LENGTH = 12
    DRAW_ANGLE = math.pi / 6  # 30 degrees
    STATIONARY_LIMIT = 100 # Steps
    STATIONARY_SPEED_THRESHOLD = 0.1
    
    # Rewards
    REWARD_FINISH = 100.0
    REWARD_CRASH = -50.0
    REWARD_BOOST = 5.0
    REWARD_FORWARD_MOVEMENT = 0.1
    REWARD_SLOW_PENALTY = -0.01

    # Camera
    CAMERA_LERP_FACTOR = 0.1
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        
        # Etc...
        self.render_mode = render_mode
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        
        # Rider state
        self.rider_pos = pygame.Vector2(50, self.SCREEN_HEIGHT / 2)
        self.rider_vel = pygame.Vector2(2, 0)
        self.last_rider_x = self.rider_pos.x
        
        # Track state
        self.track_points = []
        for i in range(10): # Create a starting platform
            self.track_points.append(pygame.Vector2(i * self.TRACK_SEGMENT_LENGTH, self.SCREEN_HEIGHT / 2 + 20))
            
        # Boosts
        self.boosts = []
        for i in range(3):
            self.boosts.append({
                "pos": pygame.Vector2(600 + i * 700 + self.np_random.uniform(-100, 100), self.SCREEN_HEIGHT / 2 + self.np_random.uniform(-100, 100)),
                "active": True,
                "radius": 15
            })
            
        # Effects and UI
        self.particles = []
        self.camera_pos = pygame.Vector2(0, self.rider_pos.y - self.SCREEN_HEIGHT / 2)
        self.stationary_steps = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        self.last_rider_x = self.rider_pos.x
        
        # Update game logic
        self._update_track(movement)
        self._update_rider()
        self._handle_collisions()
        self._update_particles()
        self._update_camera()
        
        self.steps += 1
        self.time_elapsed += 1 / self.FPS
        
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_track(self, movement):
        last_point = self.track_points[-1]
        
        angle = 0
        if movement == 1: # Up
            angle = -self.DRAW_ANGLE
        elif movement == 2: # Down
            angle = self.DRAW_ANGLE
        elif movement == 3: # Left (limited)
            new_x = last_point.x + self.TRACK_SEGMENT_LENGTH * 0.5
            new_y = last_point.y
            self.track_points.append(pygame.Vector2(new_x, new_y))
            return
        # 0 (None) and 4 (Right) draw straight ahead
        
        prev_point = self.track_points[-2] if len(self.track_points) > 1 else last_point
        base_angle = math.atan2(last_point.y - prev_point.y, last_point.x - prev_point.x)
        
        final_angle = base_angle * 0.5 + angle * 0.5 # Smooth the angle change
        
        new_x = last_point.x + math.cos(final_angle) * self.TRACK_SEGMENT_LENGTH
        new_y = last_point.y + math.sin(final_angle) * self.TRACK_SEGMENT_LENGTH
        
        new_x = max(new_x, last_point.x) # Clamp to prevent drawing backwards
        
        self.track_points.append(pygame.Vector2(new_x, new_y))

        if len(self.track_points) > 500: # Prune old track points for performance
            self.track_points.pop(0)

    def _update_rider(self):
        self.rider_vel.y += self.GRAVITY
        self.rider_vel *= self.FRICTION
        self.rider_pos += self.rider_vel

        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]

            if not (p1.x - self.RIDER_RADIUS < self.rider_pos.x < p2.x + self.RIDER_RADIUS or \
                    p2.x - self.RIDER_RADIUS < self.rider_pos.x < p1.x + self.RIDER_RADIUS):
                continue

            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            point_vec = self.rider_pos - p1
            t = max(0, min(1, point_vec.dot(line_vec) / line_vec.length_squared()))
            
            closest_point = p1 + t * line_vec
            dist_vec = self.rider_pos - closest_point
            
            if dist_vec.length() < self.RIDER_RADIUS:
                overlap = self.RIDER_RADIUS - dist_vec.length()
                normal = dist_vec.normalize() if dist_vec.length() > 0 else pygame.Vector2(0, -1)
                self.rider_pos += normal * overlap
                
                velocity_component = self.rider_vel.dot(normal)
                if velocity_component < 0:
                    self.rider_vel -= (1 + self.BOUNCE_DAMPENING) * velocity_component * normal
                # # Sound placeholder
                # if abs(velocity_component) > 1: # play_thud_sound()
                break
    
    def _handle_collisions(self):
        for boost in self.boosts:
            if boost["active"] and self.rider_pos.distance_to(boost["pos"]) < self.RIDER_RADIUS + boost["radius"]:
                boost["active"] = False
                self.rider_vel *= 1.5
                self.score += self.REWARD_BOOST
                # # Sound placeholder: play_boost_sound()
                for _ in range(30):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(2, 6)
                    vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                    pos = self.rider_pos.copy()
                    lifespan = self.np_random.integers(15, 30)
                    self.particles.append([pos, vel, lifespan, self.COLOR_BOOST])
                        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[1] *= 0.95 # friction
            p[2] -= 1 # lifespan--
            
    def _update_camera(self):
        target_cam_x = self.rider_pos.x - self.SCREEN_WIDTH / 4
        target_cam_y = self.rider_pos.y - self.SCREEN_HEIGHT / 2
        
        self.camera_pos.x += (target_cam_x - self.camera_pos.x) * self.CAMERA_LERP_FACTOR
        self.camera_pos.y += (target_cam_y - self.camera_pos.y) * self.CAMERA_LERP_FACTOR

    def _calculate_reward(self):
        reward = 0
        delta_x = self.rider_pos.x - self.last_rider_x
        if delta_x > 0:
            reward += delta_x * self.REWARD_FORWARD_MOVEMENT
        
        if self.rider_vel.length() < self.STATIONARY_SPEED_THRESHOLD * 5 and self.rider_pos.x < self.FINISH_LINE_X:
            reward += self.REWARD_SLOW_PENALTY
            
        return reward

    def _check_termination(self):
        if self.rider_pos.x >= self.FINISH_LINE_X:
            self.score += self.REWARD_FINISH
            return True
        
        if not (-200 < self.rider_pos.y < self.SCREEN_HEIGHT + 200):
            self.score += self.REWARD_CRASH
            return True
            
        if self.steps >= self.MAX_STEPS:
            return True
            
        if self.rider_vel.length() < self.STATIONARY_SPEED_THRESHOLD:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0
            
        if self.stationary_steps >= self.STATIONARY_LIMIT:
            self.score += self.REWARD_CRASH / 2
            return True
            
        return False

    def _world_to_screen(self, pos):
        return int(pos.x - self.camera_pos.x), int(pos.y - self.camera_pos.y)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        
        finish_pos = self._world_to_screen(pygame.Vector2(self.FINISH_LINE_X, 0))
        if finish_pos[0] < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_pos[0], 0), (finish_pos[0], self.SCREEN_HEIGHT), 5)
            
        for boost in self.boosts:
            if boost["active"]:
                pos = self._world_to_screen(boost["pos"])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], boost["radius"], self.COLOR_BOOST)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], boost["radius"], self.COLOR_BOOST)

        if len(self.track_points) > 1:
            screen_points = [self._world_to_screen(p) for p in self.track_points]
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, screen_points, 8)
            pygame.draw.lines(self.screen, (0,0,0,50), False, [(p[0], p[1]+4) for p in screen_points], 2)

        for p in self.particles:
            pos = self._world_to_screen(p[0])
            radius = int(p[2] / 5)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p[3])

        rider_screen_pos = self._world_to_screen(self.rider_pos)
        pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS + 2, self.COLOR_RIDER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)

    def _render_grid(self):
        grid_size = 50
        left = int(self.camera_pos.x // grid_size) * grid_size
        top = int(self.camera_pos.y // grid_size) * grid_size
        
        for i in range(left, int(left + self.SCREEN_WIDTH + grid_size), grid_size):
            x = int(i - self.camera_pos.x)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
            
        for i in range(top, int(top + self.SCREEN_HEIGHT + grid_size), grid_size):
            y = int(i - self.camera_pos.y)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
    def _render_ui(self):
        speed_text = f"Speed: {self.rider_vel.length():.1f}"
        time_text = f"Time: {self.time_elapsed:.1f}s"
        score_text = f"Score: {self.score:.0f}"
        
        speed_surf = self.font.render(speed_text, True, self.COLOR_TEXT)
        time_surf = self.font.render(time_text, True, self.COLOR_TEXT)
        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        
        self.screen.blit(speed_surf, (10, 10))
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        self.screen.blit(score_surf, (self.SCREEN_WIDTH / 2 - score_surf.get_width() / 2, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "distance_to_finish": max(0, self.FINISH_LINE_X - self.rider_pos.x),
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Line Rider Gym")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    action = np.array([0, 0, 0]) 
    
    print("--- Playing Game ---")
    print(env.user_guide)
    print("Press 'R' to reset.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(1000)
            
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()