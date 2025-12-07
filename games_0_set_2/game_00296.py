
# Generated: 2025-08-27T13:13:04.953953
# Source Brief: brief_00296.md
# Brief Index: 296

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: ↑ to accelerate, ←→ to steer. Hold space to boost."
    )

    # Short, user-facing description of the game
    game_description = (
        "Race against the clock in a procedurally generated side-view kart circuit. "
        "Complete 3 laps before time runs out, but watch out for obstacles!"
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_TRACK = (50, 55, 70)
    COLOR_TRACK_LINE = (200, 200, 220)
    COLOR_KART = (255, 80, 80)
    COLOR_KART_GLOW = (255, 80, 80, 60)
    COLOR_OBSTACLE = (220, 50, 50)
    COLOR_PARTICLE = (255, 220, 50)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_BOOST_BAR_BG = (80, 80, 80)
    COLOR_BOOST_BAR_FG = (255, 220, 50)

    # Game parameters
    FPS = 30
    MAX_TIME = 60.0
    MAX_LAPS = 3
    MAX_STEPS = 1000 * FPS // 30 # Scaled to FPS
    TRACK_LENGTH = SCREEN_WIDTH * 15
    TRACK_WIDTH = 180
    TRACK_Y_CENTER = SCREEN_HEIGHT // 2
    
    # Physics
    ACCELERATION = 0.2
    FRICTION = 0.98
    TURN_SPEED = 0.35
    MAX_SPEED_X = 6.0
    MAX_SPEED_Y = 3.0
    BOOST_FORCE = 0.6
    MAX_BOOST_SPEED_X = 12.0
    BOOST_CONSUMPTION = 2.5
    BOOST_RECHARGE = 0.5
    BOUNDARY_BOUNCE = -0.5

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
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)
        
        self.render_mode = render_mode
        self.np_random = None

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0.0
        self.laps = 0
        self.kart_pos = [0.0, 0.0]
        self.kart_vel = [0.0, 0.0]
        self.boost_level = 0.0
        self.particles = []
        self.obstacles = []
        self.track_top_y = []
        self.track_bottom_y = []
        self.camera_x = 0.0
        self.boost_just_used = False
        self.obstacles_passed_on_boost = set()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        self.laps = 0
        
        self.kart_pos = [100.0, self.TRACK_Y_CENTER]
        self.kart_vel = [0.0, 0.0]
        self.boost_level = 100.0
        
        self.particles.clear()
        self.camera_x = 0.0
        self.boost_just_used = False
        self.obstacles_passed_on_boost.clear()

        self._generate_track()
        self._generate_obstacles()
        
        return self._get_observation(), self._get_info()
    
    def _generate_track(self):
        self.track_top_y = np.zeros(self.TRACK_LENGTH, dtype=float)
        num_segments = 10
        amps = self.np_random.uniform(10, 50, num_segments)
        freqs = self.np_random.uniform(0.1, 0.5, num_segments)
        phases = self.np_random.uniform(0, 2 * math.pi, num_segments)
        
        for i in range(self.TRACK_LENGTH):
            y_offset = 0
            x = i / self.TRACK_LENGTH * 2 * math.pi
            for amp, freq, phase in zip(amps, freqs, phases):
                y_offset += amp * math.sin(freq * x * num_segments + phase)
            
            # Ensure it loops by fading out the wave towards the end
            loop_fade = (math.cos(x) + 1) / 2
            self.track_top_y[i] = self.TRACK_Y_CENTER - self.TRACK_WIDTH / 2 + y_offset * loop_fade
        
        self.track_bottom_y = self.track_top_y + self.TRACK_WIDTH

    def _generate_obstacles(self):
        self.obstacles.clear()
        num_obstacles = int(5 * (1 + 0.05 * self.laps))
        
        for i in range(num_obstacles):
            x = self.np_random.integers(500, self.TRACK_LENGTH - 500)
            track_top = self.track_top_y[x]
            track_bottom = self.track_bottom_y[x]
            y = self.np_random.uniform(track_top + 20, track_bottom - 20)
            size = self.np_random.integers(10, 15)
            obstacle_rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
            self.obstacles.append((i, obstacle_rect)) # Use ID for tracking

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        self.game_over = self.game_over or self.timer <= 0 or self.laps >= self.MAX_LAPS

        if not self.game_over:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # --- Update State ---
            self.timer -= 1.0 / self.FPS
            self.steps += 1
            
            # --- Handle Input & Physics ---
            is_boosting = space_held and self.boost_level > 0
            
            # Steering
            if movement == 3: # Left (Up screen)
                self.kart_vel[1] -= self.TURN_SPEED
            elif movement == 4: # Right (Down screen)
                self.kart_vel[1] += self.TURN_SPEED
            
            # Acceleration
            if movement == 1:
                self.kart_vel[0] += self.ACCELERATION
            
            # Boosting
            if is_boosting:
                # Sound: boost_loop.wav
                self.kart_vel[0] += self.BOOST_FORCE
                self.boost_level = max(0, self.boost_level - self.BOOST_CONSUMPTION)
                if not self.boost_just_used: # First frame of boost
                    self.boost_just_used = True
                    # Check for ineffective boost
                    nearby_obstacle = any(
                        obs.x > self.kart_pos[0] and obs.x < self.kart_pos[0] + 400
                        for _, obs in self.obstacles
                    )
                    if not nearby_obstacle:
                        reward -= 2.0
            else:
                self.boost_just_used = False
                self.boost_level = min(100, self.boost_level + self.BOOST_RECHARGE)

            # Apply friction and clamp velocity
            self.kart_vel[0] *= self.FRICTION
            self.kart_vel[1] *= self.FRICTION
            max_vx = self.MAX_BOOST_SPEED_X if is_boosting else self.MAX_SPEED_X
            self.kart_vel[0] = np.clip(self.kart_vel[0], 0, max_vx)
            self.kart_vel[1] = np.clip(self.kart_vel[1], -self.MAX_SPEED_Y, self.MAX_SPEED_Y)
            
            # Update position
            self.kart_pos[0] += self.kart_vel[0]
            self.kart_pos[1] += self.kart_vel[1]

            # Reward for forward movement
            if self.kart_vel[0] > 1.0:
                reward += 0.1

            # --- World Collisions & Laps ---
            kart_x_int = int(self.kart_pos[0]) % self.TRACK_LENGTH
            track_top = self.track_top_y[kart_x_int]
            track_bottom = self.track_bottom_y[kart_x_int]
            
            # Boundary collision
            if self.kart_pos[1] < track_top or self.kart_pos[1] > track_bottom:
                # Sound: boundary_hit.wav
                self.kart_pos[1] = np.clip(self.kart_pos[1], track_top, track_bottom)
                self.kart_vel[1] *= self.BOUNDARY_BOUNCE
                reward -= 0.1
            
            # Lap completion
            if self.kart_pos[0] >= self.TRACK_LENGTH:
                # Sound: lap_complete.wav
                self.laps += 1
                self.kart_pos[0] -= self.TRACK_LENGTH
                self.obstacles_passed_on_boost.clear()
                if self.laps < self.MAX_LAPS:
                    self._generate_obstacles()
                    reward += 1.0
            
            # Obstacle collision
            kart_rect = pygame.Rect(self.kart_pos[0]-5, self.kart_pos[1]-3, 10, 6)
            for obs_id, obs_rect in self.obstacles:
                if kart_rect.colliderect(obs_rect):
                    # Sound: crash.wav
                    self.game_over = True
                    reward -= 50.0
                    break
                # Check for effective boost reward
                if is_boosting and obs_id not in self.obstacles_passed_on_boost and kart_rect.left > obs_rect.right:
                    self.obstacles_passed_on_boost.add(obs_id)
                    reward += 5.0

            # --- Effects ---
            if is_boosting and self.steps % 2 == 0:
                p_vel_x = -self.kart_vel[0] * 0.5
                p_vel_y = (self.np_random.random() - 0.5) * 2.0
                self.particles.append({
                    'pos': [self.kart_pos[0] - 8, self.kart_pos[1]],
                    'vel': [p_vel_x, p_vel_y],
                    'life': 1.0
                })
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.05
        
        # Update camera
        self.camera_x = self.kart_pos[0] - self.SCREEN_WIDTH / 4

        # --- Termination and Final Reward ---
        terminated = self.steps >= self.MAX_STEPS or self.game_over or self.timer <= 0 or self.laps >= self.MAX_LAPS
        if terminated and not self.game_over:
            if self.laps >= self.MAX_LAPS:
                # Sound: win.wav
                reward += 50.0
            elif self.timer <= 0:
                # Sound: lose.wav
                reward -= 50.0
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # --- Track ---
        visible_start = max(0, int(self.camera_x))
        visible_end = min(self.TRACK_LENGTH, int(self.camera_x + self.SCREEN_WIDTH) + 2)
        
        # Track ground polygon
        track_poly = []
        for i in range(visible_start, visible_end):
            track_poly.append((i - self.camera_x, self.track_top_y[i]))
        for i in range(visible_end - 1, visible_start - 1, -1):
            track_poly.append((i - self.camera_x, self.track_bottom_y[i]))
        
        if len(track_poly) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, track_poly, self.COLOR_TRACK)
        
        # Track boundary lines with antialiasing
        for i in range(visible_start, visible_end - 1):
            x1, y1_top, y1_bot = i - self.camera_x, self.track_top_y[i], self.track_bottom_y[i]
            x2, y2_top, y2_bot = i + 1 - self.camera_x, self.track_top_y[i+1], self.track_bottom_y[i+1]
            pygame.draw.aaline(self.screen, self.COLOR_TRACK_LINE, (x1, y1_top), (x2, y2_top))
            pygame.draw.aaline(self.screen, self.COLOR_TRACK_LINE, (x1, y1_bot), (x2, y2_bot))
        
        # --- Particles ---
        for p in self.particles:
            screen_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            radius = int(p['life'] * 4)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_PARTICLE)

        # --- Obstacles ---
        for _, obs_rect in self.obstacles:
            screen_x = obs_rect.x - self.camera_x
            if -obs_rect.width < screen_x < self.SCREEN_WIDTH:
                screen_rect = obs_rect.move(-self.camera_x, 0)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
        
        # --- Kart ---
        kart_screen_pos = (
            int(self.kart_pos[0] - self.camera_x),
            int(self.kart_pos[1])
        )
        # Glow
        glow_rect = pygame.Rect(0, 0, 24, 16)
        glow_rect.center = kart_screen_pos
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(s, self.COLOR_KART_GLOW, s.get_rect())
        self.screen.blit(s, glow_rect.topleft)
        # Body
        kart_rect = pygame.Rect(0, 0, 18, 10)
        kart_rect.center = kart_screen_pos
        pygame.draw.rect(self.screen, self.COLOR_KART, kart_rect, border_radius=2)
        
    def _render_ui(self):
        # --- Lap Counter ---
        lap_text = self.font_ui.render(f"LAP: {min(self.laps + 1, self.MAX_LAPS)} / {self.MAX_LAPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_text, (10, 10))
        
        # --- Timer ---
        time_left = max(0, self.timer)
        time_text = self.font_ui.render(f"TIME: {time_left:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # --- Boost Meter ---
        bar_width, bar_height = 200, 20
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        
        boost_ratio = self.boost_level / 100.0
        
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR_FG, (bar_x, bar_y, int(bar_width * boost_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # --- Game Over Message ---
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill(self.COLOR_UI_BG)
            
            if self.laps >= self.MAX_LAPS:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
                
            text = self.font_big.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            s.blit(text, text_rect)
            self.screen.blit(s, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps,
            "timer": self.timer,
            "boost": self.boost_level
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Reset first to ensure everything is initialized for observation
        self.reset(seed=123)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=123)
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless

    env = GameEnv(render_mode="rgb_array")
    
    # --- Test Reset ---
    obs, info = env.reset(seed=42)
    print("Reset successful. Initial info:", info)
    
    # --- Test Step ---
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step successful. Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
    
    # --- Test a few more steps ---
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+2}: Reward: {reward:.2f}, Terminated: {terminated}")
        if terminated:
            print("Episode finished.")
            break

    # --- To visualize with Pygame (requires a display) ---
    try:
        del os.environ["SDL_VIDEODRIVER"]
        import sys
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Arcade Racer")
        clock = pygame.time.Clock()
        
        terminated = False
        total_reward = 0
        
        print("\n--- Starting Interactive Test ---")
        print(GameEnv.user_guide)

        while not terminated:
            # Map keyboard to MultiDiscrete action space
            keys = pygame.key.get_pressed()
            
            mov = 0 # no-op
            if keys[pygame.K_UP]: mov = 1
            elif keys[pygame.K_DOWN]: mov = 2 # Mapped to no-op in design
            elif keys[pygame.K_LEFT]: mov = 3
            elif keys[pygame.K_RIGHT]: mov = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [mov, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False

            clock.tick(GameEnv.FPS)
            
        print(f"Game Over! Final Score: {info['score']:.2f}")
        env.close()
        pygame.quit()
        sys.exit()

    except pygame.error as e:
        print("\nCould not start interactive test (is a display available?). Headless tests passed.")
        print(f"Pygame error: {e}")
    
    env.close()