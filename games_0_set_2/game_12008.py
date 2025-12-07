import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:52:31.722011
# Source Brief: brief_02008.md
# Brief Index: 2008
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a bouncing ball.
    The goal is to build momentum by bouncing off the floor and reach the top
    of the screen while dodging horizontally moving obstacles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball and build momentum to reach the top of the screen while dodging moving obstacles."
    )
    user_guide = "Controls: Use ← and → arrow keys to move the ball left and right."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_DURATION_SECONDS = 60
    FPS = 60 # Using a fixed step rate for physics
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_PLAYER = (0, 255, 255) # Cyan
    COLOR_OBSTACLE = (255, 50, 50) # Red
    COLOR_UI_TEXT = (255, 255, 0) # Yellow
    COLOR_TIMER_GOOD = (50, 255, 50) # Green
    COLOR_TIMER_WARN = (255, 50, 50) # Red
    COLOR_BOUNDARY = (255, 255, 255)

    # Player Physics
    PLAYER_RADIUS = 12
    PLAYER_GRAVITY = 0.25
    PLAYER_TILT_FORCE = 0.4
    PLAYER_INITIAL_BOUNCE_VEL = -8.0
    PLAYER_FLOOR_ELASTICITY = 1.0 # Perfect bounce, momentum handles height

    # Obstacle Config
    NUM_OBSTACLES = 5
    OBSTACLE_WIDTH = 80
    OBSTACLE_HEIGHT = 15
    OBSTACLE_SPEED_INCREASE_INTERVAL = 10 * FPS # Every 10 seconds
    OBSTACLE_SPEED_INCREASE_AMOUNT = 0.1

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        self.render_mode = render_mode
        self.bg_surface = self._create_gradient_background()

        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_over_message = ""
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_momentum = 1.0
        self.obstacles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Player state
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - self.PLAYER_RADIUS - 6)
        self.player_vel = pygame.math.Vector2(0, self.PLAYER_INITIAL_BOUNCE_VEL)
        self.player_momentum = 1.0

        # Obstacle state
        self.obstacles = []
        for i in range(self.NUM_OBSTACLES):
            y_pos = self.SCREEN_HEIGHT * (0.2 + 0.6 * (i / (self.NUM_OBSTACLES - 1)))
            self.obstacles.append({
                "rect": pygame.Rect(0, 0, self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT),
                "base_y": y_pos,
                "speed": self.np_random.uniform(0.5, 1.5),
                "amplitude": self.np_random.uniform(self.OBSTACLE_WIDTH, self.SCREEN_WIDTH / 2 - self.OBSTACLE_WIDTH),
                "offset": self.np_random.uniform(0, 2 * math.pi)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = 0.0
        terminated = False

        # --- 1. Handle Action ---
        movement = action[0]
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_TILT_FORCE
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_TILT_FORCE
        
        # Actions 1, 2 (up/down) and space/shift have no effect per brief

        # --- 2. Update Game State ---
        self._update_player()
        self._update_obstacles()

        # --- 3. Handle Collisions and Boundaries ---
        reward += self._handle_collisions_and_boundaries()

        # --- 4. Update Score and Steps ---
        self.score += reward
        self.steps += 1

        # --- 5. Check Termination Conditions ---
        # Win: Reached the top
        if self.player_pos.y - self.PLAYER_RADIUS <= 5: # 5px top boundary
            win_reward = 100.0
            reward += win_reward
            self.score += win_reward
            terminated = True
            self.game_over_message = "VICTORY!"

        # Loss: Out of time
        if self.steps >= self.MAX_STEPS:
            time_penalty = -10.0
            reward += time_penalty
            self.score += time_penalty
            terminated = True
            self.game_over_message = "TIME UP"

        # Loss: Momentum dead
        if self.player_momentum < 0.1:
            momentum_penalty = -5.0
            reward += momentum_penalty
            self.score += momentum_penalty
            terminated = True
            self.game_over_message = "NO MOMENTUM"
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.PLAYER_GRAVITY
        
        # Apply air friction/drag
        self.player_vel.x *= 0.98

        # Update position
        self.player_pos += self.player_vel

    def _update_obstacles(self):
        # Increase obstacle speed over time
        if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
            for obs in self.obstacles:
                obs["speed"] += self.OBSTACLE_SPEED_INCREASE_AMOUNT

        # Move obstacles
        for obs in self.obstacles:
            time_factor = self.steps / self.FPS
            x_pos = self.SCREEN_WIDTH / 2 + math.sin(time_factor * obs["speed"] + obs["offset"]) * obs["amplitude"]
            obs["rect"].center = (x_pos, obs["base_y"])

    def _handle_collisions_and_boundaries(self):
        reward = 0.0
        
        # --- Wall boundaries (left/right) ---
        if self.player_pos.x - self.PLAYER_RADIUS < 0:
            self.player_pos.x = self.PLAYER_RADIUS
            self.player_vel.x *= -0.8 # Bounce off walls
        elif self.player_pos.x + self.PLAYER_RADIUS > self.SCREEN_WIDTH:
            self.player_pos.x = self.SCREEN_WIDTH - self.PLAYER_RADIUS
            self.player_vel.x *= -0.8

        # --- Floor boundary ---
        floor_y = self.SCREEN_HEIGHT - 5 # 5px bottom boundary
        if self.player_pos.y + self.PLAYER_RADIUS > floor_y and self.player_vel.y > 0:
            self.player_pos.y = floor_y - self.PLAYER_RADIUS
            # Apply bounce with momentum
            self.player_vel.y = -abs(self.player_vel.y * self.PLAYER_FLOOR_ELASTICITY) * self.player_momentum
            
            # Increase momentum for next bounce
            self.player_momentum *= 1.10
            self.player_momentum = min(self.player_momentum, 5.0) # Cap momentum

            # Reward for bouncing up
            reward += 0.1
            # Sound: Boing!

        # --- Obstacle collisions ---
        player_circle_center = (int(self.player_pos.x), int(self.player_pos.y))
        
        for obs in self.obstacles:
            # More accurate circle-rectangle collision check
            closest_x = max(obs["rect"].left, min(player_circle_center[0], obs["rect"].right))
            closest_y = max(obs["rect"].top, min(player_circle_center[1], obs["rect"].bottom))
            distance_sq = (player_circle_center[0] - closest_x)**2 + (player_circle_center[1] - closest_y)**2

            if distance_sq < self.PLAYER_RADIUS**2:
                # Collision confirmed
                reward -= 1.0
                self.player_momentum *= 0.75 # Lose 25% momentum
                # Sound: Clank!

                # Simple bounce logic
                self.player_vel.y *= -0.5 # Bounce off vertically
                # Nudge player out of obstacle to prevent sticking
                if self.player_pos.y < obs["rect"].centery:
                    self.player_pos.y = obs["rect"].top - self.PLAYER_RADIUS - 1
                else:
                    self.player_pos.y = obs["rect"].bottom + self.PLAYER_RADIUS + 1
                
                break # Only one collision per frame

        return reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.bg_surface, (0, 0))
        
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "momentum": self.player_momentum,
        }
        
    def _render_game(self):
        # Draw boundaries
        pygame.draw.line(self.screen, self.COLOR_BOUNDARY, (0, 5), (self.SCREEN_WIDTH, 5), 3)
        pygame.draw.line(self.screen, self.COLOR_BOUNDARY, (0, self.SCREEN_HEIGHT - 5), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 5), 3)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs["rect"], border_radius=3)

        # Draw player with glow
        self._draw_glow_circle(
            self.screen,
            (int(self.player_pos.x), int(self.player_pos.y)),
            self.PLAYER_RADIUS,
            self.COLOR_PLAYER
        )

    def _render_ui(self):
        # Render momentum
        momentum_text = f"Momentum: {self.player_momentum:.2f}x"
        text_surf = self.font_ui.render(momentum_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, self.SCREEN_HEIGHT - 35))

        # Render timer
        time_left = self.GAME_DURATION_SECONDS - (self.steps / self.FPS)
        time_left = max(0, time_left)
        timer_color = self.COLOR_TIMER_GOOD if time_left > 10 else self.COLOR_TIMER_WARN
        timer_text = f"Time: {time_left:.1f}"
        text_surf = self.font_ui.render(timer_text, True, timer_color)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        text_surf = self.font_big.render(self.game_over_message, True, self.COLOR_BOUNDARY)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _draw_glow_circle(self, surface, pos, radius, color, glow_strength=5):
        for i in range(glow_strength, 0, -1):
            alpha = int(255 * (1 - (i / glow_strength))**2 * 0.3)
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(
                surface,
                pos[0],
                pos[1],
                int(radius + i * 2),
                glow_color
            )
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Momentum Bounce")
    clock = pygame.time.Clock()
    
    running = True
    
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and env.game_over:
                obs, info = env.reset()

        if not env.game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                action[0] = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if env.game_over:
            # Game is paused, waiting for reset
            pass
        
        clock.tick(env.FPS)
        
    env.close()