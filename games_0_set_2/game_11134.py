import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:39:23.734104
# Source Brief: brief_01134.md
# Brief Index: 1134
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a ship through a hazardous 2D world to reach the core. "
        "Shift between dimensions with different physics to avoid obstacles and find a safe path."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to apply thrust to your ship. "
        "Press space to shift between dimensions, altering gravity and revealing different obstacles."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 2500
    WORLD_HEIGHT = 800
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_CORE = (255, 200, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    DIM_COLORS = {
        1: (255, 80, 80),   # Red: Normal Gravity
        2: (80, 255, 80),   # Green: Inverted Gravity
        3: (80, 80, 255),   # Blue: Zero Gravity
        4: (255, 255, 80)   # Yellow: High Gravity
    }
    
    # Physics
    PLAYER_FORCE = 0.3
    GRAVITY_NORMAL = 0.25
    GRAVITY_HIGH = 0.5
    DAMPING = 0.995 # Air resistance
    ZERO_G_DAMPING = 0.98
    
    # Game settings
    MAX_STEPS = 5000
    PLAYER_SIZE = 16
    CORE_SIZE = 40
    MAX_OBSTACLES_PER_DIM = 50
    CAMERA_LERP_RATE = 0.08
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.world_surface = pygame.Surface((self.WORLD_WIDTH, self.WORLD_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_dim = pygame.font.SysFont("monospace", 24, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.player_pos = [0.0, 0.0]
        self.player_vel = [0.0, 0.0]
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)

        self.core_pos = [0, 0]
        self.core_rect = pygame.Rect(0, 0, self.CORE_SIZE, self.CORE_SIZE)

        self.camera_pos = [0.0, 0.0]
        
        self.current_dimension = 1
        self.unlocked_dimensions = 1
        self.obstacles = {}
        self.obstacle_density_factor = 1.0

        self.particles = []
        
        self.prev_space_held = False
        self.prev_dist_to_core = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.player_pos = [200.0, float(self.WORLD_HEIGHT / 2)]
        self.player_vel = [2.0, 0.0] # Start with some forward momentum
        
        self.core_pos = [self.WORLD_WIDTH - 200, self.WORLD_HEIGHT / 2]
        self.core_rect = pygame.Rect(self.core_pos[0] - self.CORE_SIZE / 2, self.core_pos[1] - self.CORE_SIZE / 2, self.CORE_SIZE, self.CORE_SIZE)

        self.camera_pos = [self.player_pos[0] - self.SCREEN_WIDTH / 2, self.player_pos[1] - self.SCREEN_HEIGHT / 2]

        self.current_dimension = 1
        self.unlocked_dimensions = 1
        self.obstacle_density_factor = 1.0
        self._generate_obstacles()

        self.particles = []
        self.prev_space_held = False
        self.prev_dist_to_core = self._get_distance_to_core()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # 1. Process Actions
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Apply movement force
        if movement == 1: self.player_vel[1] -= self.PLAYER_FORCE # Up
        elif movement == 2: self.player_vel[1] += self.PLAYER_FORCE # Down
        elif movement == 3: self.player_vel[0] -= self.PLAYER_FORCE # Left
        elif movement == 4: self.player_vel[0] += self.PLAYER_FORCE # Right
        
        # Handle dimension shift on button press (not hold)
        reward = 0
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            # SFX: Dimension Shift sound
            self.current_dimension = (self.current_dimension % self.unlocked_dimensions) + 1
            reward += 1.0 # Reward for using the mechanic
            self._create_dimension_shift_particles()
        self.prev_space_held = space_held

        # 2. Update Game Logic
        self._update_physics()
        self._update_particles()
        self._update_game_progression()

        # 3. Check for termination and calculate rewards
        terminated = False
        
        # Distance-based reward
        current_dist = self._get_distance_to_core()
        dist_change = self.prev_dist_to_core - current_dist
        reward += dist_change * 0.1 # Continuous reward for getting closer
        self.prev_dist_to_core = current_dist

        # Check for collisions
        if self._check_collision():
            # SFX: Player Explosion sound
            reward = -100.0
            terminated = True
            self.game_over = True

        # Check for victory
        if self.player_rect.colliderect(self.core_rect):
            # SFX: Victory sound
            reward = 100.0
            terminated = True
            self.game_over = True
        
        # Check for max steps
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            np.clip(reward, -100, 100), # Clip reward to specified range
            terminated,
            truncated,
            self._get_info()
        )

    def _update_physics(self):
        # Apply gravity based on dimension
        if self.current_dimension == 1: # Normal Gravity
            self.player_vel[1] += self.GRAVITY_NORMAL
        elif self.current_dimension == 2: # Inverted Gravity
            self.player_vel[1] -= self.GRAVITY_NORMAL
        elif self.current_dimension == 3: # Zero Gravity
            self.player_vel[0] *= self.ZERO_G_DAMPING
            self.player_vel[1] *= self.ZERO_G_DAMPING
        elif self.current_dimension == 4: # High Gravity
            self.player_vel[1] += self.GRAVITY_HIGH
        
        # Apply universal damping
        self.player_vel[0] *= self.DAMPING
        self.player_vel[1] *= self.DAMPING

        # Update position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # World boundary collision
        if self.player_pos[0] < self.PLAYER_SIZE / 2:
            self.player_pos[0] = self.PLAYER_SIZE / 2
            self.player_vel[0] *= -0.5
        if self.player_pos[0] > self.WORLD_WIDTH - self.PLAYER_SIZE / 2:
            self.player_pos[0] = self.WORLD_WIDTH - self.PLAYER_SIZE / 2
            self.player_vel[0] *= -0.5
        if self.player_pos[1] < self.PLAYER_SIZE / 2:
            self.player_pos[1] = self.PLAYER_SIZE / 2
            self.player_vel[1] *= -0.5
        if self.player_pos[1] > self.WORLD_HEIGHT - self.PLAYER_SIZE / 2:
            self.player_pos[1] = self.WORLD_HEIGHT - self.PLAYER_SIZE / 2
            self.player_vel[1] *= -0.5

        self.player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))

    def _check_collision(self):
        if self.current_dimension in self.obstacles:
            for obstacle in self.obstacles[self.current_dimension]:
                if self.player_rect.colliderect(obstacle['rect']):
                    return True
        return False

    def _update_game_progression(self):
        # Unlock new dimensions
        if self.steps > 0 and self.steps % 1000 == 0:
            if self.unlocked_dimensions < 4:
                self.unlocked_dimensions += 1
                # SFX: New Dimension Unlocked sound
        
        # Increase obstacle density
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_density_factor = min(2.0, self.obstacle_density_factor + 0.05)
            self._generate_obstacles(regenerate=True)

    def _get_observation(self):
        # --- Update Camera ---
        target_cam_x = self.player_pos[0] - self.SCREEN_WIDTH / 2
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT / 2
        self.camera_pos[0] += (target_cam_x - self.camera_pos[0]) * self.CAMERA_LERP_RATE
        self.camera_pos[1] += (target_cam_y - self.camera_pos[1]) * self.CAMERA_LERP_RATE
        
        # Clamp camera to world bounds
        self.camera_pos[0] = max(0, min(self.camera_pos[0], self.WORLD_WIDTH - self.SCREEN_WIDTH))
        self.camera_pos[1] = max(0, min(self.camera_pos[1], self.WORLD_HEIGHT - self.SCREEN_HEIGHT))
        
        # --- Render World ---
        self.world_surface.fill(self.COLOR_BG)
        self._render_world_elements()
        
        # --- Blit World to Screen ---
        self.screen.blit(self.world_surface, (-int(self.camera_pos[0]), -int(self.camera_pos[1])))
        
        # --- Render UI (on top of everything) ---
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_world_elements(self):
        # Render Core
        self._render_pulsating_core()
        
        # Render Obstacles for the current dimension
        dim_color = self.DIM_COLORS[self.current_dimension]
        if self.current_dimension in self.obstacles:
            for obs in self.obstacles[self.current_dimension]:
                pygame.draw.rect(self.world_surface, dim_color, obs['rect'])

        # Render Particles
        self._render_particles()

        # Render Player
        self._render_player_glow()
        pygame.draw.rect(self.world_surface, self.COLOR_PLAYER, self.player_rect)

    def _render_pulsating_core(self):
        pulse = (math.sin(self.steps * 0.05) + 1) / 2 # Varies between 0 and 1
        current_size = self.CORE_SIZE * (1 + pulse * 0.2)
        
        # Glow effect
        glow_radius = current_size * 1.5
        glow_alpha = 100 + pulse * 50
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_CORE, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.world_surface.blit(glow_surf, (self.core_pos[0] - glow_radius, self.core_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Main core circle
        pygame.gfxdraw.filled_circle(self.world_surface, int(self.core_pos[0]), int(self.core_pos[1]), int(current_size/2), self.COLOR_CORE)
        pygame.gfxdraw.aacircle(self.world_surface, int(self.core_pos[0]), int(self.core_pos[1]), int(current_size/2), self.COLOR_CORE)

    def _render_player_glow(self):
        glow_color = self.DIM_COLORS[self.current_dimension]
        for i in range(4, 0, -1):
            glow_rect = self.player_rect.inflate(i * 4, i * 4)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            glow_surf.fill((*glow_color, 20))
            self.world_surface.blit(glow_surf, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Dimension Indicator
        dim_color = self.DIM_COLORS[self.current_dimension]
        dim_text = f"DIMENSION {self.current_dimension}"
        text_surf = self.font_dim.render(dim_text, True, dim_color)
        self.screen.blit(text_surf, (10, 10))
        pygame.draw.rect(self.screen, dim_color, (10, 40, text_surf.get_width(), 4))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Distance to Core
        dist_text = f"DISTANCE: {int(self.prev_dist_to_core)}"
        dist_surf = self.font_ui.render(dist_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_surf, (self.SCREEN_WIDTH - dist_surf.get_width() - 10, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "player_vel": self.player_vel,
            "distance_to_core": self.prev_dist_to_core,
            "current_dimension": self.current_dimension
        }
    
    def _generate_obstacles(self, regenerate=False):
        if not regenerate:
            self.obstacles = {i: [] for i in range(1, 5)}
        
        for dim in range(1, 5):
            if regenerate:
                self.obstacles[dim].clear()

            num_obstacles = int(self.MAX_OBSTACLES_PER_DIM * self.obstacle_density_factor * (dim / 4.0))
            
            for _ in range(num_obstacles):
                while True:
                    width = self.np_random.integers(20, 100)
                    height = self.np_random.integers(20, 100)
                    x = self.np_random.integers(300, self.WORLD_WIDTH - 300)
                    y = self.np_random.integers(0, self.WORLD_HEIGHT - height)
                    rect = pygame.Rect(x, y, width, height)
                    
                    # Ensure obstacles don't block start or end
                    if not rect.colliderect(self.player_rect.inflate(200, 200)) and \
                       not rect.colliderect(self.core_rect.inflate(200, 200)):
                        self.obstacles[dim].append({'rect': rect})
                        break
    
    def _get_distance_to_core(self):
        return math.hypot(self.player_pos[0] - self.core_pos[0], self.player_pos[1] - self.core_pos[1])

    def _create_dimension_shift_particles(self):
        player_center = self.player_rect.center
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            color = self.DIM_COLORS[self.current_dimension]
            self.particles.append({'pos': list(player_center), 'vel': vel, 'life': lifespan, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40.0))
            color = (*p['color'], alpha)
            size = int(p['life'] / 10) + 1
            
            # Use a surface to handle alpha correctly
            part_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(part_surf, color, (size, size), size)
            self.world_surface.blit(part_surf, (p['pos'][0] - size, p['pos'][1] - size), special_flags=pygame.BLEND_RGBA_ADD)

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override the Pygame screen with a display one for human play
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dimension Shift")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # We need to manually blit the observation to the display screen
        # This is because the env renders to a surface, not the display
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(frame_surface, (0, 0))

        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            # A small pause on reset
            pygame.time.wait(1000)

        env.clock.tick(60) # Run at 60 FPS for smooth human gameplay
        
    env.close()