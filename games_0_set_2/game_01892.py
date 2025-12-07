
# Generated: 2025-08-27T18:36:38.302388
# Source Brief: brief_01892.md
# Brief Index: 1892

        
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
    """
    A procedurally generated brick breaker where risky plays are rewarded and
    safe plays are penalized, challenging RL agents to find the optimal balance
    between aggression and safety.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ←→ to move the paddle. Hold Shift for faster movement. Press Space to launch the ball."
    )

    # User-facing description of the game
    game_description = (
        "A fast-paced, neon-drenched arcade brick breaker. Break all the bricks to win, but lose all your balls and it's game over. Risky side-wall bounces are rewarded, but cautious top-wall bounces are penalized."
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game constants and colors
        self._define_colors()
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED, self.PADDLE_SPEED_FAST = 8, 16
        self.BALL_RADIUS = 8
        self.MAX_BALL_SPEED_Y = 8
        self.MAX_BALL_DEFLECTION_X = 6
        self.MAX_STEPS = 2500 # Extended to allow more time for completion
        self.BRICK_ROWS, self.BRICK_COLS = 5, 20
        self.TOTAL_BRICKS = self.BRICK_ROWS * self.BRICK_COLS

        # Initialize state variables
        self.paddle_rect = None
        self.ball_rect = None
        self.ball_velocity = None
        self.ball_on_paddle = None
        self.bricks = None
        self.brick_color_map = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.balls_remaining = None

        self.reset()
        self.validate_implementation()

    def _define_colors(self):
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (0, 255, 255) # Bright Cyan
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_TEXT = (255, 255, 255)
        self.BRICK_COLORS = {
            10: (0, 255, 100),   # Green
            20: (0, 150, 255),   # Blue
            30: (255, 255, 0),   # Yellow
            40: (255, 50, 50),    # Red
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_remaining = 3
        self.particles = []

        # Paddle state
        self.paddle_rect = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Ball state
        self._reset_ball()

        # Brick layout
        self.bricks = []
        self.brick_color_map = {}
        brick_width = self.WIDTH // self.BRICK_COLS
        brick_height = 20
        points = list(self.BRICK_COLORS.keys())
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                brick = pygame.Rect(
                    c * brick_width,
                    r * brick_height + 40,
                    brick_width - 1,
                    brick_height - 1,
                )
                self.bricks.append(brick)
                # Use numpy RNG for point selection
                point_value = self.np_random.choice(points)
                self.brick_color_map[brick.topleft] = point_value

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_rect = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball_rect.centerx = self.paddle_rect.centerx
        self.ball_rect.bottom = self.paddle_rect.top
        self.ball_velocity = [0, 0]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02  # Per-step penalty to encourage speed

        # Handle player input
        self._handle_input(action)

        # Update game logic
        if not self.ball_on_paddle:
            reward += self._update_physics()

        self.steps += 1
        terminated = self._check_termination()

        # Add terminal rewards
        if terminated:
            if not self.bricks: # Win condition
                reward += 100
            elif self.balls_remaining <= 0: # Lose condition
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        speed = self.PADDLE_SPEED_FAST if shift_held else self.PADDLE_SPEED

        if movement == 3:  # Left
            self.paddle_rect.x -= speed
        elif movement == 4:  # Right
            self.paddle_rect.x += speed

        # Clamp paddle to screen
        self.paddle_rect.left = max(0, self.paddle_rect.left)
        self.paddle_rect.right = min(self.WIDTH, self.paddle_rect.right)

        if self.ball_on_paddle:
            self.ball_rect.centerx = self.paddle_rect.centerx
            if space_held:
                self.ball_on_paddle = False
                # Launch with a slight random x component for variety
                vx = self.np_random.uniform(-1, 1)
                self.ball_velocity = [vx, -self.MAX_BALL_SPEED_Y]
                # sfx: launch_ball.wav

    def _update_physics(self):
        reward = 0
        self.ball_rect.x += self.ball_velocity[0]
        self.ball_rect.y += self.ball_velocity[1]

        # Wall collisions
        if self.ball_rect.left <= 0 or self.ball_rect.right >= self.WIDTH:
            self.ball_velocity[0] *= -1
            self.ball_rect.left = max(0, self.ball_rect.left)
            self.ball_rect.right = min(self.WIDTH, self.ball_rect.right)
            reward += 0.1  # Risky play reward
            self._create_particles(self.ball_rect.center, self.COLOR_WALL, 5)
            # sfx: wall_bounce.wav
        if self.ball_rect.top <= 0:
            self.ball_velocity[1] *= -1
            reward -= 2.0  # Safe play penalty
            self._create_particles(self.ball_rect.center, self.COLOR_WALL, 5)
            # sfx: wall_bounce.wav

        # Ball lost
        if self.ball_rect.top >= self.HEIGHT:
            self.balls_remaining -= 1
            # sfx: lose_life.wav
            if self.balls_remaining > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return reward # Return early, no other collisions matter

        # Paddle collision
        if self.ball_rect.colliderect(self.paddle_rect) and self.ball_velocity[1] > 0:
            self.ball_velocity[1] *= -1
            # Add deflection based on hit location
            offset = (self.ball_rect.centerx - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_velocity[0] = self.MAX_BALL_DEFLECTION_X * offset
            self.ball_rect.bottom = self.paddle_rect.top # Prevent sticking
            self._create_particles(self.ball_rect.center, self.COLOR_PADDLE, 10)
            # sfx: paddle_hit.wav

        # Brick collisions
        hit_brick_index = self.ball_rect.collidelist(self.bricks)
        if hit_brick_index != -1:
            brick = self.bricks.pop(hit_brick_index)
            
            # Determine bounce direction (simple approach)
            self.ball_velocity[1] *= -1

            # Add score and reward
            point_value = self.brick_color_map.pop(brick.topleft)
            self.score += point_value
            reward += point_value # Use point value directly as reward
            
            # Visual and audio feedback
            color = self.BRICK_COLORS[point_value]
            self._create_particles(brick.center, color, 20)
            # sfx: brick_break.wav

        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if not self.bricks: # Win
            self.game_over = True
            return True
        if self.balls_remaining <= 0: # Lose
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS: # Timeout
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(p['life'] * (255 / p['start_life']))))
                color = p['color'] + (alpha,)
                size = int(self.BALL_RADIUS * 0.3 * (p['life'] / p['start_life']))
                if size > 0:
                    # Create a temporary surface for transparency
                    temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (size, size), size)
                    self.screen.blit(temp_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))


        # Draw bricks
        for brick in self.bricks:
            color = self.BRICK_COLORS[self.brick_color_map[brick.topleft]]
            pygame.draw.rect(self.screen, color, brick)
            # Add a subtle inner glow/bevel effect
            highlight = tuple(min(255, c + 50) for c in color)
            pygame.draw.rect(self.screen, highlight, brick.inflate(-6, -6), 2)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=5)

        # Draw ball with glow
        ball_center = (int(self.ball_rect.centerx), int(self.ball_rect.centery))
        glow_color = self.COLOR_BALL + (50,) # Add alpha for glow
        glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (self.BALL_RADIUS*2, self.BALL_RADIUS*2), self.BALL_RADIUS * 1.5)
        self.screen.blit(glow_surf, (ball_center[0] - self.BALL_RADIUS*2, ball_center[1] - self.BALL_RADIUS*2))
        
        # Draw main ball using anti-aliasing for smoothness
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Remaining balls
        for i in range(self.balls_remaining - 1): # Don't show the one in play
            pos_x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pos_y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Game Over / Win Message
        if self.game_over:
            msg_text = "YOU WIN!" if not self.bricks else "GAME OVER"
            color = self.BRICK_COLORS[10] if not self.bricks else self.BRICK_COLORS[40]
            rendered_msg = self.font_msg.render(msg_text, True, color)
            msg_rect = rendered_msg.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(rendered_msg, msg_rect)


    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': life,
                'start_life': life,
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_remaining": self.balls_remaining,
            "bricks_remaining": len(self.bricks),
        }
        
    def close(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To visualize the game, we can use a simple pygame loop
    # Note: This is for human play/viewing and not part of the Gym env
    try:
        import os
        # Set a non-dummy driver if you want to see the window
        if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
             del os.environ["SDL_VIDEODRIVER"]
             
        pygame.display.set_caption("Brick Breaker")
        real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        
        obs, info = env.reset()
        terminated = False
        
        # Game loop
        running = True
        while running:
            # Map pygame keys to gym actions
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False

            if not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the real screen
            # Pygame uses (width, height), numpy uses (height, width)
            # We need to transpose back for pygame display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Control frame rate
            env.clock.tick(60) # Run at 60 FPS for smooth human play

    finally:
        env.close()