
# Generated: 2025-08-27T17:14:58.535165
# Source Brief: brief_01472.md
# Brief Index: 1472

        
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
        "Controls: ←→ to move the paddle. Break all the bricks or reach the top to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Bounce a ball off a paddle to break bricks in this fast-paced arcade action game. Don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 7
        self.BRICK_WIDTH = 58
        self.BRICK_HEIGHT = 20
        self.TOP_BOUNDARY_Y = 40
        self.MAX_STEPS = 2000 # Increased for more gameplay potential

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BOUNDARY = (100, 100, 120)
        self.COLOR_TEXT = (200, 200, 220)
        self.COLOR_HEART = (255, 80, 80)
        self.BRICK_COLORS = {
            1: (0, 200, 100),  # Green
            3: (0, 150, 255),  # Blue
            5: (255, 100, 0),  # Red
        }

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 16)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = None
        self.bricks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.bricks_destroyed_count = None

        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.bricks_destroyed_count = 0

        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        
        # Start ball with a random upward angle
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = [math.cos(angle), math.sin(angle)]
        self.ball_speed = 4.0

        self._create_brick_layout()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _create_brick_layout(self):
        self.bricks = []
        brick_values = [5, 5, 3, 3, 1, 1]
        num_cols = self.SCREEN_WIDTH // (self.BRICK_WIDTH + 2)
        x_offset = (self.SCREEN_WIDTH - num_cols * (self.BRICK_WIDTH + 2)) / 2

        for row in range(len(brick_values)):
            value = brick_values[row]
            color = self.BRICK_COLORS[value]
            for col in range(num_cols):
                brick_rect = pygame.Rect(
                    x_offset + col * (self.BRICK_WIDTH + 2),
                    self.TOP_BOUNDARY_Y + 20 + row * (self.BRICK_HEIGHT + 2),
                    self.BRICK_WIDTH,
                    self.BRICK_HEIGHT,
                )
                self.bricks.append({"rect": brick_rect, "color": color, "value": value})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.01 # Small penalty per step to encourage speed
        
        # --- Player Input ---
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # --- Game Logic ---
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over: # Win conditions
            if not self.bricks:
                reward += 50 # Cleared all bricks
            else: # Must have reached the top
                reward += 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0] * self.ball_speed
        self.ball_pos[1] += self.ball_vel[1] * self.ball_speed
    
    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(1, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH - 1, ball_rect.right)
            # sfx: ball_bounce_wall
        if ball_rect.top <= self.TOP_BOUNDARY_Y:
            self.ball_vel[1] *= -1
            ball_rect.top = self.TOP_BOUNDARY_Y + 1
            # sfx: ball_bounce_wall

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            
            # Add "spin" based on where it hits the paddle
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = max(-0.95, min(0.95, self.ball_vel[0] + offset * 0.7))
            
            # Normalize velocity vector
            norm = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel = [v / norm for v in self.ball_vel]
            
            # Prevent ball from getting stuck in paddle
            ball_rect.bottom = self.paddle.top - 1
            self.ball_pos[1] = ball_rect.centery
            # sfx: ball_bounce_paddle
        
        # Brick collisions
        for brick in self.bricks[:]:
            if ball_rect.colliderect(brick["rect"]):
                reward += brick["value"]
                self.score += brick["value"]
                
                # Determine bounce direction
                prev_ball_rect = pygame.Rect(
                    (self.ball_pos[0] - self.ball_vel[0] * self.ball_speed) - self.BALL_RADIUS,
                    (self.ball_pos[1] - self.ball_vel[1] * self.ball_speed) - self.BALL_RADIUS,
                    self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
                )
                if prev_ball_rect.centery < brick["rect"].top or prev_ball_rect.centery > brick["rect"].bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1

                self._create_particles(brick["rect"].center, brick["color"])
                self.bricks.remove(brick)
                self.bricks_destroyed_count += 1
                
                # Increase ball speed every 25 bricks
                if self.bricks_destroyed_count > 0 and self.bricks_destroyed_count % 25 == 0:
                    self.ball_speed = min(8.0, self.ball_speed + 0.5)
                
                # sfx: brick_break
                break # Only break one brick per frame

        # Bottom boundary (lose life)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 5
            # sfx: lose_life
            if self.lives > 0:
                self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = [math.cos(angle), math.sin(angle)]
            else:
                self.game_over = True
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "radius": radius, "color": color, "lifetime": lifetime})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            p["radius"] -= 0.1
            if p["lifetime"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return (
            self.lives <= 0 or 
            not self.bricks or 
            self.ball_pos[1] - self.BALL_RADIUS <= self.TOP_BOUNDARY_Y or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw top boundary
        pygame.draw.line(self.screen, self.COLOR_BOUNDARY, (0, self.TOP_BOUNDARY_Y), (self.SCREEN_WIDTH, self.TOP_BOUNDARY_Y), 2)
        
        # Draw particles
        for p in self.particles:
            alpha_color = (*p["color"], max(0, min(255, int(p["lifetime"] * 10))))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), max(0, int(p["radius"])), alpha_color)

        # Draw bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in brick["color"]), brick["rect"], 2) # Border

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with a glow effect
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_color = (*self.COLOR_BALL, 50)
        pygame.gfxdraw.filled_circle(self.screen, *ball_pos_int, self.BALL_RADIUS + 4, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, *ball_pos_int, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, *ball_pos_int, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Draw score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 5))
        
        # Draw lives as hearts
        for i in range(self.lives):
            self._draw_heart(15 + i * 30, 20)
    
    def _draw_heart(self, x, y):
        points = [
            (x, y - 8), (x + 8, y - 16), (x + 16, y - 8),
            (x + 16, y), (x, y + 16), (x - 16, y),
            (x - 16, y - 8), (x - 8, y - 16), (x, y-8)
        ]
        scaled_points = [(p[0], p[1]) for p in points]
        pygame.gfxdraw.filled_polygon(self.screen, scaled_points, self.COLOR_HEART)
        pygame.gfxdraw.aapolygon(self.screen, scaled_points, self.COLOR_HEART)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks),
            "ball_speed": self.ball_speed
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
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This block allows a human to play the game.
    # It sets up a pygame window to display the frames.
    
    # Set render_mode to "human" to display the game
    # For this example, we manually render the "rgb_array"
    pygame.display.set_caption("Breakout")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Main game loop for human play
    while not terminated:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        # Collect keyboard inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3 # Left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4 # Right
            
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(60) # Run at 60 FPS for smooth human play

    env.close()
    print("Game Over. Final Info:", info)