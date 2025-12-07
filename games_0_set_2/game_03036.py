
# Generated: 2025-08-27T22:10:14.078256
# Source Brief: brief_03036.md
# Brief Index: 3036

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to thrust horizontally. Hold SPACE for main vertical thrust. Hold SHIFT to engage braking thrusters."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Precisely pilot your ship to land on a moving platform. Master the physics to achieve a soft, accurate landing before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # World parameters
    WORLD_SCALE = 40
    GRAVITY = 0.002
    THRUST_HORIZONTAL = 0.015
    THRUST_VERTICAL = 0.005 # This is the continuous vertical thrust
    DRAG = 0.01
    BRAKE_DRAG = 0.05
    
    # Game parameters
    MAX_STEPS = 1000
    PLATFORM_SAFE_RADIUS = 0.5  # in world units
    PLATFORM_TOTAL_RADIUS = 1.2 # in world units
    
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_SHIP = (64, 224, 208)
    COLOR_SHIP_GLOW = (128, 255, 240, 50)
    COLOR_FLAME = (255, 200, 50)
    COLOR_FLAME_OUTER = (255, 100, 0)
    COLOR_PLATFORM_BASE = (40, 50, 80)
    COLOR_PLATFORM_DANGER = (200, 50, 80)
    COLOR_PLATFORM_SAFE = (50, 200, 80)
    COLOR_SHADOW = (0, 0, 0, 50)
    COLOR_TEXT = (230, 230, 240)
    COLOR_SUCCESS_PARTICLE = (100, 255, 120)
    COLOR_CRASH_PARTICLE = (255, 100, 100)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_hud = pygame.font.Font(None, 24)
        self.font_info = pygame.font.Font(None, 36)
        
        # Screen projection helpers
        self.screen_center_x = self.SCREEN_WIDTH // 2
        self.screen_center_y = self.SCREEN_HEIGHT // 2 + 50 # Push world center down

        # Initialize state variables
        self.ship_pos = None
        self.ship_vel = None
        self.platform_x = None
        self.platform_speed = None
        self.platform_amplitude = None
        self.platform_phase_offset = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_dist_to_platform = None
        self.info_message = None
        self.info_message_timer = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        start_x = self.np_random.uniform(-3, 3)
        start_z = self.np_random.uniform(6, 8)
        self.ship_pos = np.array([start_x, 0.0, start_z], dtype=np.float64) # x, y, z(height)
        self.ship_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        self.platform_x = 0.0
        self.platform_speed = 1.0
        self.platform_amplitude = 1.0
        self.platform_phase_offset = self.np_random.uniform(0, 2 * math.pi)
        
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        platform_pos_3d = np.array([self.platform_x, 0.0, 0.0])
        self.last_dist_to_platform = np.linalg.norm(self.ship_pos - platform_pos_3d)

        self.info_message = ""
        self.info_message_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing until reset
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean for vertical thrust
        shift_held = action[2] == 1  # Boolean for braking

        # --- Update game logic ---
        self.steps += 1
        
        # 1. Difficulty Scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.platform_speed += 0.05
        if self.steps > 0 and self.steps % 500 == 0:
            self.platform_amplitude += 0.1

        # 2. Platform Movement
        self.platform_x = self.platform_amplitude * math.sin(self.steps * self.platform_speed / 50.0 + self.platform_phase_offset)
        
        # 3. Apply Forces & Update Ship Physics
        # Horizontal thrust
        if movement == 3: # Left
            self.ship_vel[0] -= self.THRUST_HORIZONTAL
        elif movement == 4: # Right
            self.ship_vel[0] += self.THRUST_HORIZONTAL
        
        # Vertical thrust (counter-gravity)
        if space_held:
            self.ship_vel[2] += self.THRUST_VERTICAL
        
        # Gravity
        self.ship_vel[2] -= self.GRAVITY
        
        # Drag
        drag_factor = self.BRAKE_DRAG if shift_held else self.DRAG
        self.ship_vel[0] *= (1.0 - drag_factor)
        self.ship_vel[1] *= (1.0 - drag_factor) # y-axis drag
        self.ship_vel[2] *= (1.0 - self.DRAG * 0.1) # Air resistance on z-axis

        # Update position
        self.ship_pos += self.ship_vel
        
        # Prevent ship from going through the floor
        if self.ship_pos[2] < 0:
            self.ship_pos[2] = 0
            self.ship_vel[2] = 0

        # 4. Update Particles
        self._update_particles(movement, space_held)
        
        # 5. Check for Termination and Calculate Reward
        reward = 0
        terminated = False
        
        platform_pos_3d = np.array([self.platform_x, 0.0, 0.0])
        dist_vec = self.ship_pos - platform_pos_3d
        dist_to_platform_center = np.linalg.norm(dist_vec)
        
        # Check for landing
        if self.ship_pos[2] <= 0.05: # Landing height threshold
            dist_on_plane = np.linalg.norm(dist_vec[[0, 1]])
            landing_speed = np.linalg.norm(self.ship_vel)

            if dist_on_plane <= self.PLATFORM_SAFE_RADIUS:
                # SUCCESSFUL LANDING
                terminated = True
                self.info_message = "LANDING SUCCESSFUL!"
                self.info_message_timer = 90
                # Reward: base + bonus for accuracy and low speed
                accuracy_bonus = (self.PLATFORM_SAFE_RADIUS - dist_on_plane) / self.PLATFORM_SAFE_RADIUS
                speed_bonus = max(0, 1 - landing_speed)
                reward = 100 + (accuracy_bonus * 20) + (speed_bonus * 20)
                self._create_particles(self.ship_pos, self.COLOR_SUCCESS_PARTICLE, 50, 0.5)
            elif dist_on_plane <= self.PLATFORM_TOTAL_RADIUS:
                # CRASH ON PLATFORM EDGE
                terminated = True
                self.info_message = "CRASHED!"
                self.info_message_timer = 90
                reward = -10
                self._create_particles(self.ship_pos, self.COLOR_CRASH_PARTICLE, 30, 1.0)
        
        # Time limit termination
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            self.info_message = "TIME LIMIT REACHED"
            self.info_message_timer = 90
            reward = -10
        
        # Continuous rewards
        if not terminated:
            # Time penalty
            reward -= 0.01
            # Reward for getting closer
            if dist_to_platform_center < self.last_dist_to_platform:
                reward += 0.1
            self.last_dist_to_platform = dist_to_platform_center
        
        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _project(self, x, y, z):
        """Projects 3D world coordinates to 2D screen coordinates."""
        sx = self.screen_center_x + (x - y) * self.WORLD_SCALE
        sy = self.screen_center_y + (x + y) * 0.5 * self.WORLD_SCALE - z * self.WORLD_SCALE
        return int(sx), int(sy)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_platform_shadow()
        self._render_ship_shadow()
        self._render_particles()
        self._render_platform()
        self._render_ship()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for i in range(-15, 16):
            # Lines parallel to x-axis
            start_pos = self._project(-15, i, 0)
            end_pos = self._project(15, i, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_pos, end_pos)
            # Lines parallel to y-axis
            start_pos = self._project(i, -15, 0)
            end_pos = self._project(i, 15, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_pos, end_pos)
            
    def _render_platform(self):
        px, py = self._project(self.platform_x, 0, 0)
        
        # Draw base
        base_points = []
        for i in range(4):
            angle = math.pi / 2 * i + math.pi / 4
            x = self.platform_x + self.PLATFORM_TOTAL_RADIUS * 1.2 * math.cos(angle)
            y = self.PLATFORM_TOTAL_RADIUS * 1.2 * math.sin(angle)
            base_points.append(self._project(x, y, -0.2))
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_PLATFORM_BASE)
        
        # Draw danger zone
        danger_radius = int(self.PLATFORM_TOTAL_RADIUS * self.WORLD_SCALE)
        pygame.gfxdraw.filled_circle(self.screen, px, py, danger_radius, self.COLOR_PLATFORM_DANGER)
        pygame.gfxdraw.aacircle(self.screen, px, py, danger_radius, self.COLOR_PLATFORM_DANGER)
        
        # Draw safe zone
        safe_radius = int(self.PLATFORM_SAFE_RADIUS * self.WORLD_SCALE)
        pygame.gfxdraw.filled_circle(self.screen, px, py, safe_radius, self.COLOR_PLATFORM_SAFE)
        pygame.gfxdraw.aacircle(self.screen, px, py, safe_radius, self.COLOR_PLATFORM_SAFE)

    def _render_platform_shadow(self):
        shadow_surface = self.screen.copy()
        shadow_surface.set_colorkey(self.COLOR_BG)
        shadow_surface.set_alpha(80)
        
        shadow_points = []
        for i in range(8):
            angle = math.pi / 4 * i
            x = self.platform_x + self.PLATFORM_TOTAL_RADIUS * 1.2 * math.cos(angle)
            y = self.PLATFORM_TOTAL_RADIUS * 1.2 * math.sin(angle)
            shadow_points.append(self._project(x, y, 0))
        
        pygame.gfxdraw.filled_polygon(shadow_surface, shadow_points, (0,0,0))
        self.screen.blit(shadow_surface, (0, 0))

    def _render_ship_shadow(self):
        if self.ship_pos[2] > 0:
            sx, sy = self._project(self.ship_pos[0], self.ship_pos[1], 0)
            shadow_radius = int(max(1, (1 - self.ship_pos[2] / 10.0) * 10))
            
            shadow_surf = pygame.Surface((shadow_radius * 2, shadow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surf, self.COLOR_SHADOW, (shadow_radius, shadow_radius), shadow_radius)
            self.screen.blit(shadow_surf, (sx - shadow_radius, sy - shadow_radius))

    def _render_ship(self):
        sx, sy = self._project(self.ship_pos[0], self.ship_pos[1], self.ship_pos[2])
        
        # Ship body
        ship_points = [
            (sx, sy - 12), (sx + 8, sy), (sx, sy + 12), (sx - 8, sy)
        ]
        
        # Glow effect
        glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_SHIP_GLOW, (20, 20), 20)
        self.screen.blit(glow_surf, (sx - 20, sy - 20))

        pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_SHIP)
        
    def _update_particles(self, movement, space_held):
        # Create thruster particles
        if space_held: # Main engine
            p_pos = self.ship_pos + np.array([0, 0, -0.2])
            p_vel = np.array([self.np_random.uniform(-0.01, 0.01), self.np_random.uniform(-0.01, 0.01), -0.1])
            self.particles.append([p_pos, p_vel, 15, self.COLOR_FLAME])
        if movement == 3: # Left
            p_pos = self.ship_pos + np.array([0.2, 0, 0])
            p_vel = np.array([0.05, 0, 0])
            self.particles.append([p_pos, p_vel, 10, self.COLOR_FLAME_OUTER])
        if movement == 4: # Right
            p_pos = self.ship_pos + np.array([-0.2, 0, 0])
            p_vel = np.array([-0.05, 0, 0])
            self.particles.append([p_pos, p_vel, 10, self.COLOR_FLAME_OUTER])

        # Update and remove old particles
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1] # Update position
            p[2] -= 1 # Decrease lifetime
            
    def _create_particles(self, pos, color, count, speed_mult):
        for _ in range(count):
            angle_xy = self.np_random.uniform(0, 2 * math.pi)
            angle_z = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.01, 0.05) * speed_mult
            p_vel = np.array([
                math.cos(angle_xy) * math.sin(angle_z) * speed,
                math.sin(angle_xy) * math.sin(angle_z) * speed,
                math.cos(angle_z) * speed
            ])
            lifetime = self.np_random.integers(20, 40)
            self.particles.append([pos.copy(), p_vel, lifetime, color])

    def _render_particles(self):
        for pos, vel, life, color in self.particles:
            sx, sy = self._project(pos[0], pos[1], pos[2])
            radius = int(max(1, life / 5.0))
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, color)

    def _render_ui(self):
        # HUD
        speed = np.linalg.norm(self.ship_vel)
        dist = self.last_dist_to_platform
        time_left = self.MAX_STEPS - self.steps
        
        hud_texts = [
            f"Score: {self.score:.2f}",
            f"Time: {time_left}",
            f"Speed: {speed:.2f} m/s",
            f"Distance: {dist:.2f} m",
            f"Altitude: {self.ship_pos[2]:.2f} m",
        ]
        
        for i, text in enumerate(hud_texts):
            surf = self.font_hud.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surf, (10, 10 + i * 20))

        # Info message (landing/crash/timeout)
        if self.info_message_timer > 0:
            self.info_message_timer -= 1
            info_surf = self.font_info.render(self.info_message, True, self.COLOR_TEXT)
            pos = info_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(info_surf, pos)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ship_pos": self.ship_pos,
            "ship_vel": self.ship_vel,
            "platform_pos": np.array([self.platform_x, 0.0, 0.0]),
            "distance_to_platform": self.last_dist_to_platform,
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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'macOS' etc.

    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Player Controls ---
    # This setup allows holding keys for continuous actions
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0
            
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Run at 30 FPS for human play

    print(f"Game Over. Final Score: {info['score']:.2f} in {info['steps']} steps.")
    env.close()