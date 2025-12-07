import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to aim your track drawing. No-op continues straight. "
        "Hold Space to draw longer segments and Shift to draw shorter ones."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based arcade game where you draw the track for a sled in real-time. "
        "Navigate the procedurally generated terrain, maintain speed, and reach the finish line without crashing."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 10000

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_TERRAIN = (70, 80, 110)
        self.COLOR_SLED = (255, 255, 255)
        self.COLOR_TRACK = (255, 50, 100)
        self.COLOR_START = (255, 200, 0)
        self.COLOR_FINISH = (50, 255, 150)
        self.COLOR_PARTICLE = (255, 150, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)

        # Game constants
        self.GRAVITY = 0.2
        self.FRICTION = 0.995
        self.MAX_STEPS = 2000
        self.MAX_CRASHES = 5
        self.FINISH_X = self.WORLD_WIDTH - 500
        self.STATIONARY_CRASH_LIMIT = 50
        
        # Initialize state variables
        self.sled_pos = None
        self.sled_vel = None
        self.player_track_points = None
        self.last_draw_angle = None
        self.terrain_points = None
        self.terrain_roughness = None
        self.generated_terrain_x = None
        self.particles = None
        self.camera_x = None
        self.crashes_remaining = None
        self.stationary_steps = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        
        # This will be set properly in reset()
        self.just_crashed = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.crashes_remaining = self.MAX_CRASHES
        self.stationary_steps = 0
        self.just_crashed = False

        # Effects and camera
        self.particles = []
        self.camera_x = 0
        
        # Sled state
        start_pos = np.array([100.0, self.SCREEN_HEIGHT / 2.0])
        self.sled_pos = start_pos.copy()
        self.sled_vel = np.array([2.0, 0.0])
        
        # Track and terrain
        self.player_track_points = [start_pos - [20, 0], start_pos.copy()]
        self.last_draw_angle = 0
        self.terrain_points = []
        self.terrain_roughness = 0.5
        self.generated_terrain_x = 0
        self._generate_terrain_chunk(0, self.SCREEN_WIDTH * 2)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0
        if not self.game_over:
            # 1. Handle player input to draw track
            self._handle_input(movement, space_held, shift_held)
            
            # 2. Update sled physics
            prev_sled_x = self.sled_pos[0]
            self._update_physics()
            
            # 3. Update game state and check for events
            self._update_game_state()

            # 4. Calculate reward
            reward += (self.sled_pos[0] - prev_sled_x) * 0.1  # Reward for moving forward
            if np.linalg.norm(self.sled_vel) < 0.5:
                reward -= 0.01 # Penalty for being stationary
            
        # 5. Update camera and visual effects
        self._update_camera()
        self._update_particles()
        self._manage_terrain()
        
        terminated = self.game_over
        
        if self.just_crashed:
            # The large penalty is applied for the final crash that ends the game.
            # Non-terminal crashes have a score penalty but not a large negative reward.
            if terminated:
                reward = -100.0 
            self.score -= 5 # Score penalty for non-terminal crash
        if self.game_won:
            reward = 50.0

        self.score += reward
        self.steps += 1
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Determine segment length
        if space_held:
            length = 25.0
        elif shift_held:
            length = 8.0
        else:
            length = 15.0
        
        # Determine angle change
        angle_change = 0
        if movement == 1: angle_change = -0.2  # Up
        elif movement == 2: angle_change = 0.2   # Down
        elif movement == 3: angle_change = 0.4   # Left -> Sharp Down
        elif movement == 4: angle_change = -0.4  # Right -> Sharp Up
        
        self.last_draw_angle += angle_change
        self.last_draw_angle = np.clip(self.last_draw_angle, -math.pi/2, math.pi/2)

        last_point = self.player_track_points[-1]
        new_point = last_point + np.array([
            length * math.cos(self.last_draw_angle),
            length * math.sin(self.last_draw_angle)
        ])
        
        self.player_track_points.append(new_point)
        
        # Limit track length to avoid memory issues
        if len(self.player_track_points) > 300:
            self.player_track_points.pop(0)

    def _update_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY
        self.sled_pos += self.sled_vel
        
        # Apply friction
        self.sled_vel *= self.FRICTION

        # Find collision surface
        surface_info = self._get_collision_surface(self.sled_pos[0])
        if surface_info:
            surface_y, surface_angle, on_player_track = surface_info
            
            if self.sled_pos[1] > surface_y - 2: # Sled is on or below a surface
                self.sled_pos[1] = surface_y - 2
                
                # Project velocity onto the surface angle
                speed = np.linalg.norm(self.sled_vel)
                self.sled_vel[0] = speed * math.cos(surface_angle)
                self.sled_vel[1] = speed * math.sin(surface_angle)

                # Add a small bounce/push if on player track
                if on_player_track:
                    self.sled_vel *= 1.005

    def _get_collision_surface(self, x_pos):
        surfaces = []
        
        # Check player track
        for i in range(len(self.player_track_points) - 1):
            p1 = self.player_track_points[i]
            p2 = self.player_track_points[i+1]
            if p1[0] <= x_pos < p2[0] or p2[0] <= x_pos < p1[0]:
                if (p2[0] - p1[0]) != 0:
                    t = (x_pos - p1[0]) / (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    surfaces.append((y, angle, True))

        # Check terrain
        for i in range(len(self.terrain_points) - 1):
            p1 = self.terrain_points[i]
            p2 = self.terrain_points[i+1]
            if p1[0] <= x_pos < p2[0] or p2[0] <= x_pos < p1[0]:
                if (p2[0] - p1[0]) != 0:
                    t = (x_pos - p1[0]) / (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    surfaces.append((y, angle, False))
        
        # Return the highest surface (lowest y-value)
        if not surfaces:
            return None
        return min(surfaces, key=lambda s: s[0])

    def _update_game_state(self):
        self.just_crashed = False
        is_crash = False

        # Check boundaries
        if not (0 < self.sled_pos[1] < self.SCREEN_HEIGHT and self.sled_pos[0] > self.camera_x):
            is_crash = True
        
        # Check if stationary
        if np.linalg.norm(self.sled_vel) < 0.1:
            self.stationary_steps += 1
            if self.stationary_steps > self.STATIONARY_CRASH_LIMIT:
                is_crash = True
        else:
            self.stationary_steps = 0
            
        if is_crash:
            self.crashes_remaining -= 1
            self.just_crashed = True
            self._create_explosion(self.sled_pos)
            if self.crashes_remaining <= 0:
                self.game_over = True
            else:
                self._reset_sled_to_track()
        
        # Check win condition
        if self.sled_pos[0] >= self.FINISH_X:
            self.game_over = True
            self.game_won = True
            
        # Increase difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.terrain_roughness = min(3.0, self.terrain_roughness + 0.2)

    def _reset_sled_to_track(self):
        # Find a suitable point on the track to respawn
        for p in reversed(self.player_track_points):
            if self.camera_x < p[0] < self.camera_x + self.SCREEN_WIDTH:
                self.sled_pos = p.copy()
                self.sled_vel = np.array([1.0, 0.0])
                self.stationary_steps = 0
                return
        # If no suitable point, reset to start
        self.sled_pos = np.array([self.camera_x + 50.0, self.SCREEN_HEIGHT / 2.0])
        self.sled_vel = np.array([1.0, 0.0])

    def _create_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(20, 41)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'size': self.np_random.uniform(2, 6)})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_camera(self):
        target_cam_x = self.sled_pos[0] - self.SCREEN_WIDTH / 3
        self.camera_x = self.camera_x * 0.95 + target_cam_x * 0.05
    
    def _manage_terrain(self):
        if self.camera_x + self.SCREEN_WIDTH > self.generated_terrain_x - self.SCREEN_WIDTH:
            self._generate_terrain_chunk(self.generated_terrain_x, self.SCREEN_WIDTH * 2)

    def _generate_terrain_chunk(self, start_x, width):
        num_points = int(width / 20)
        if not self.terrain_points:
            last_y = self.SCREEN_HEIGHT * 0.8
            self.terrain_points.append(np.array([-100.0, last_y]))
        else:
            last_y = self.terrain_points[-1][1]
        
        for i in range(num_points):
            x = start_x + i * 20
            y_change = self.np_random.uniform(-15, 15) * self.terrain_roughness
            y = np.clip(last_y + y_change, self.SCREEN_HEIGHT * 0.6, self.SCREEN_HEIGHT - 20)
            self.terrain_points.append(np.array([float(x), float(y)]))
            last_y = y
        self.generated_terrain_x += width
        
        # Prune old terrain points
        self.terrain_points = [p for p in self.terrain_points if p[0] > self.camera_x - 200]

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
        cam_x, cam_y = int(self.camera_x), 0 # No vertical camera movement

        # Draw background grid
        for i in range(0, self.SCREEN_WIDTH, 50):
            x = i - (cam_x % 50)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw terrain
        terrain_screen_points = [(int(p[0] - cam_x), int(p[1] - cam_y)) for p in self.terrain_points if cam_x - 20 < p[0] < cam_x + self.SCREEN_WIDTH + 20]
        if len(terrain_screen_points) > 1:
            pygame.gfxdraw.aapolygon(self.screen, terrain_screen_points + [(self.SCREEN_WIDTH, self.SCREEN_HEIGHT), (0, self.SCREEN_HEIGHT)], self.COLOR_TERRAIN)
            pygame.gfxdraw.filled_polygon(self.screen, terrain_screen_points + [(self.SCREEN_WIDTH, self.SCREEN_HEIGHT), (0, self.SCREEN_HEIGHT)], self.COLOR_TERRAIN)

        # Draw player track
        track_screen_points = [(int(p[0] - cam_x), int(p[1] - cam_y)) for p in self.player_track_points]
        if len(track_screen_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, track_screen_points, 2)
        
        # Draw start/finish lines
        start_x_screen = int(100 - cam_x)
        if 0 < start_x_screen < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_START, (start_x_screen, 0), (start_x_screen, self.SCREEN_HEIGHT), 3)
        finish_x_screen = int(self.FINISH_X - cam_x)
        if 0 < finish_x_screen < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x_screen, 0), (finish_x_screen, self.SCREEN_HEIGHT), 3)

        # Draw sled
        sled_screen_pos = (int(self.sled_pos[0] - cam_x), int(self.sled_pos[1] - cam_y))
        pygame.gfxdraw.aacircle(self.screen, sled_screen_pos[0], sled_screen_pos[1], 5, self.COLOR_SLED)
        pygame.gfxdraw.filled_circle(self.screen, sled_screen_pos[0], sled_screen_pos[1], 5, self.COLOR_SLED)

        # Draw particles
        for p in self.particles:
            p_screen_pos = (int(p['pos'][0] - cam_x), int(p['pos'][1] - cam_y))
            # Create a temporary surface for alpha blending
            alpha_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            alpha = int(255 * (p['life'] / 40.0))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(alpha_surf, int(p['size']), int(p['size']), int(p['size']), color)
            self.screen.blit(alpha_surf, (p_screen_pos[0] - int(p['size']), p_screen_pos[1] - int(p['size'])))


    def _render_ui(self):
        # Score and time
        score_text = self.font_small.render(f"SCORE: {int(self.score):,}", True, self.COLOR_UI_TEXT)
        time_text = self.font_small.render(f"TIME: {self.steps / 30:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 30))
        
        # Crashes remaining
        crashes_text = self.font_small.render(f"LIVES: {self.crashes_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crashes_text, (self.SCREEN_WIDTH - crashes_text.get_width() - 10, 10))
        
        # Game over text
        if self.game_over:
            if self.game_won:
                end_text_str = "FINISH!"
                end_color = self.COLOR_FINISH
            else:
                end_text_str = "GAME OVER"
                end_color = self.COLOR_TRACK
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crashes_remaining": self.crashes_remaining,
            "sled_x": self.sled_pos[0],
            "sled_speed": np.linalg.norm(self.sled_vel),
        }

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Pygame setup for manual play
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Track Drawer")
    clock = pygame.time.Clock()

    action = env.action_space.sample() 
    action.fill(0) 

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space, shift])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) 

    env.close()