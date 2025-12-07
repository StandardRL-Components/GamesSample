import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:00:36.419994
# Source Brief: brief_02041.md
# Brief Index: 2041
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Bounce Down
    Control a bouncing ball using tilt controls, chaining bounces off accelerating
    platforms to reach the bottom within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball, chaining bounces off moving platforms to reach the bottom before time runs out."
    )
    user_guide = (
        "Use ← and → arrow keys to tilt the ball. Use ↑ to reduce gravity and ↓ to increase it."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000 # As per brief's termination condition

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (40, 80, 120)
    COLOR_BALL = (255, 255, 255)
    COLOR_PLATFORM = (200, 200, 220)
    COLOR_GLOW = (255, 255, 0)
    COLOR_PARTICLE = (220, 220, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Physics
    GRAVITY = 0.3
    TILT_ACCEL = 0.4
    VERTICAL_ACCEL = 0.2
    BALL_RADIUS = 12
    BALL_FRICTION = 0.995 # Horizontal velocity decay
    BOUNCE_VELOCITY = -8.0
    CHAIN_BOUNCE_VELOCITY = -12.0
    CHAIN_TIME_THRESHOLD = 15 # 0.5 seconds at 30 FPS

    # Game Objects
    PLATFORM_COUNT = 10
    PLATFORM_HEIGHT = 10
    PLATFORM_WIDTH = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_chain = pygame.font.SysFont("Consolas", 36, bold=True)
        self._precompute_gradient()

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.ball_pos = None
        self.ball_vel = None
        self.platforms = []
        self.particles = []
        self.chain_count = 0
        self.last_bounce_time = -1
        self.terminated = False

    def _precompute_gradient(self):
        """Creates a surface with the background gradient for faster blitting."""
        self.gradient_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.gradient_surface, color, (0, y), (self.WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset core state
        self.steps = 0
        self.score = 0
        self.terminated = False

        # Reset ball
        self.ball_pos = [self.WIDTH / 2, self.BALL_RADIUS * 3]
        self.ball_vel = [0, 0]

        # Reset platforms
        self.platforms = []
        v_spacing = (self.HEIGHT - 80) / self.PLATFORM_COUNT
        for i in range(self.PLATFORM_COUNT):
            self.platforms.append({
                "rect": pygame.Rect(
                    self.np_random.uniform(0, self.WIDTH - self.PLATFORM_WIDTH),
                    80 + i * v_spacing,
                    self.PLATFORM_WIDTH,
                    self.PLATFORM_HEIGHT
                ),
                "speed": (i + 1) * self.np_random.choice([-0.5, 0.5]),
            })

        # Reset effects and counters
        self.particles = []
        self.chain_count = 0
        self.last_bounce_time = -self.CHAIN_TIME_THRESHOLD - 1

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # 1. Apply actions and physics
        self._apply_actions(action)
        self._update_ball_physics()
        
        # 2. Update platforms
        self._update_platforms()
        
        # 3. Handle collisions and get bounce rewards
        bounce_reward, is_chain = self._handle_collisions()
        reward += bounce_reward
        if is_chain:
            reward += 1.0 # Event-based reward for chain reaction

        # 4. Update visual effects
        self._update_particles()
        
        # 5. Check for termination
        self.terminated, term_reward = self._check_termination()
        reward += term_reward
        
        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.terminated = True

        return (
            self._get_observation(),
            reward,
            self.terminated,
            truncated,
            self._get_info(),
        )

    def _apply_actions(self, action):
        movement = action[0]

        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 3: # Left
            self.ball_vel[0] -= self.TILT_ACCEL
        elif movement == 4: # Right
            self.ball_vel[0] += self.TILT_ACCEL
        elif movement == 1: # Up (anti-gravity)
            self.ball_vel[1] -= self.VERTICAL_ACCEL
        elif movement == 2: # Down (extra gravity)
            self.ball_vel[1] += self.VERTICAL_ACCEL

    def _update_ball_physics(self):
        # Apply gravity
        self.ball_vel[1] += self.GRAVITY
        
        # Apply horizontal friction
        self.ball_vel[0] *= self.BALL_FRICTION
        
        # Update position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        # Wall bounces
        if self.ball_pos[0] < self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -0.8 # Lose some energy on wall hit
        if self.ball_pos[0] > self.WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -0.8

        # Prevent going above screen
        if self.ball_pos[1] < self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] = 0

    def _update_platforms(self):
        for p in self.platforms:
            p["rect"].x += p["speed"]
            if p["rect"].left < 0 or p["rect"].right > self.WIDTH:
                p["speed"] *= -1
                p["rect"].x += p["speed"] # Prevent getting stuck

    def _handle_collisions(self):
        reward = 0
        is_chain = False
        
        # Only check for collision if ball is falling
        if self.ball_vel[1] > 0:
            for p in self.platforms:
                # Simple AABB collision check
                is_colliding = (
                    p["rect"].colliderect(
                        self.ball_pos[0] - self.BALL_RADIUS,
                        self.ball_pos[1] - self.BALL_RADIUS,
                        self.BALL_RADIUS * 2,
                        self.BALL_RADIUS * 2,
                    )
                    and self.ball_pos[1] < p["rect"].centery
                )

                if is_colliding:
                    # sfx: bounce.wav
                    # Correct position
                    self.ball_pos[1] = p["rect"].top - self.BALL_RADIUS
                    
                    # Check for chain reaction
                    if self.steps - self.last_bounce_time < self.CHAIN_TIME_THRESHOLD:
                        self.chain_count += 1
                        self.ball_vel[1] = self.CHAIN_BOUNCE_VELOCITY
                        is_chain = True
                        # sfx: chain_bounce.wav
                    else:
                        self.chain_count = 1
                        self.ball_vel[1] = self.BOUNCE_VELOCITY
                    
                    self.last_bounce_time = self.steps
                    reward += 0.1 # Continuous feedback reward
                    
                    # Create particles
                    self._create_particles(self.ball_pos[0], p["rect"].top)
                    
                    # Only one bounce per frame
                    break
        return reward, is_chain

    def _check_termination(self):
        # Win condition: reached bottom
        if self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS:
            # sfx: win.wav
            return True, 100.0

        # Lose condition: timeout (this is handled by truncation now)
        if self.steps >= self.MAX_STEPS:
            # sfx: lose.wav
            return True, -100.0
        
        return False, 0.0

    def _create_particles(self, x, y):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30, endpoint=True), # lifespan in frames
                "radius": self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # particle gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        # 1. Draw background
        self.screen.blit(self.gradient_surface, (0, 0))

        # 2. Draw game elements
        self._render_game()

        # 3. Draw UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color
            )

        # Draw platforms
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p["rect"], border_radius=3)
        
        # Draw ball glow if in a chain
        if self.chain_count > 1:
            glow_size = self.chain_count * 4
            for i in range(glow_size, 0, -2):
                alpha = 100 - int((i / glow_size) * 100)
                color = (*self.COLOR_GLOW, alpha)
                radius = self.BALL_RADIUS + i
                pygame.gfxdraw.filled_circle(
                    self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), radius, color
                )

        # Draw ball
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL
        )

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, x, y, color=self.COLOR_TEXT):
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (x + 2, y + 2))
            self.screen.blit(text_surf, (x, y))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {time_left:.2f}"
        draw_text(time_text, self.font_ui, self.WIDTH - 180, 10)

        # Chain counter
        chain_text = f"CHAIN: {self.chain_count}"
        draw_text(chain_text, self.font_ui, self.WIDTH - 180, 40)

        # Show big chain text on screen
        if self.chain_count > 2:
            chain_pop_text = f"x{self.chain_count} CHAIN!"
            color = self.COLOR_GLOW
            text_surf = self.font_chain.render(chain_pop_text, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/4))
            
            shadow_surf = self.font_chain.render(chain_pop_text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(text_surf, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "chain_count": self.chain_count,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Use arrow keys for tilt, up/down for vertical thrust
    # Un-comment the line below to run with a display window
    # os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.set_caption("Bounce Down - Manual Play")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    running = True
    total_reward = 0
    
    # Set a higher FPS for smoother manual play
    clock = pygame.time.Clock()
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Total Reward: {total_reward}")
            print(f"Final Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(60) # Run manual play at a smooth 60 FPS
        
    env.close()