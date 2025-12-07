
# Generated: 2025-08-27T14:54:44.809161
# Source Brief: brief_00830.md
# Brief Index: 830

        
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
    """
    An arcade-style brick-breaking game implemented as a Gymnasium environment.

    The player controls a paddle at the bottom of the screen, moving it left and
    right to bounce a ball upwards. The objective is to destroy all the bricks
    at the top of the screen by hitting them with the ball. The player has a
    limited number of lives, losing one each time the ball passes the paddle
    and goes off the bottom of the screen. The game ends when all bricks are
    destroyed (win) or all lives are lost (loss). The ball's speed increases
    as more bricks are broken, adding to the challenge.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control and game descriptions
    user_guide = (
        "Controls: ← to move the paddle left, → to move right. Bounce the ball to destroy all bricks."
    )
    game_description = (
        "Bounce a pixelated ball off your paddle to destroy all the bricks in a top-down arcade environment."
    )

    # Frame advance setting
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_LIVES = 3
        self.MAX_STEPS = 10000
        self.INITIAL_BALL_SPEED = 3.0
        self.PADDLE_SPEED = 6.0
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.BALL_RADIUS = 7
        self.BRICK_ROWS = 5
        self.BRICK_COLS = 5

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 60)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_UI = (220, 220, 220)
        self.BRICK_COLORS = [
            (217, 87, 99), (217, 143, 87), (189, 217, 87),
            (87, 217, 143), (87, 189, 217)
        ]

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = None
        self.particles = None
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.bricks_destroyed_count = 0
        self.current_ball_speed = self.INITIAL_BALL_SPEED

        # Initialize state
        self.reset()
        
        # Self-validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.SCREEN_HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.reset_ball()
        self._create_bricks()

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.bricks_destroyed_count = 0
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack action
            movement = action[0]

            # 1. Update paddle position
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

            # 2. Update ball position
            self.ball_pos[0] += self.ball_vel[0] * self.current_ball_speed
            self.ball_pos[1] += self.ball_vel[1] * self.current_ball_speed
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # 3. Handle collisions
            # Wall collisions
            if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
                self.ball_vel[0] *= -1
                # sound: wall_bounce
            if self.ball_pos[1] <= self.BALL_RADIUS:
                self.ball_vel[1] *= -1
                # sound: wall_bounce

            # Paddle collision
            if ball_rect.colliderect(self.paddle):
                # Ensure ball is above paddle to prevent it getting stuck
                self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
                
                # Calculate bounce angle based on hit location
                offset = (self.paddle.centerx - self.ball_pos[0]) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = -np.clip(offset, -0.9, 0.9)
                self.ball_vel[1] *= -1
                
                # Normalize velocity vector
                norm = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
                self.ball_vel = [self.ball_vel[0]/norm, self.ball_vel[1]/norm]
                
                reward += 0.1 # Reward for successful paddle hit
                # sound: paddle_hit

            # Brick collisions
            hit_brick_index = ball_rect.collidelist(self.bricks)
            if hit_brick_index != -1:
                brick = self.bricks.pop(hit_brick_index)
                self.ball_vel[1] *= -1 # Simple vertical bounce
                reward += 1.0 # Reward for breaking a brick
                self.score += 10
                self.bricks_destroyed_count += 1
                
                # Create particle explosion
                self._create_particles(brick.center, self.BRICK_COLORS[brick.y // 20 % len(self.BRICK_COLORS)])
                # sound: brick_break
                
                # Increase ball speed
                speed_increase_tiers = self.bricks_destroyed_count // 5
                self.current_ball_speed = self.INITIAL_BALL_SPEED + speed_increase_tiers * 0.4


            # Missed ball (bottom wall)
            if self.ball_pos[1] >= self.SCREEN_HEIGHT + self.BALL_RADIUS:
                self.lives -= 1
                # sound: life_lost
                if self.lives <= 0:
                    self.game_over = True
                    terminated = True
                    reward = -100.0 # Large negative reward for losing
                else:
                    self.reset_ball()

            # 4. Check for win/loss conditions
            if not self.bricks:
                self.game_over = True
                terminated = True
                reward = 100.0 # Large positive reward for winning
            
            if self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
        
        self._update_particles()
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render game elements
        self._render_game()

        # Render UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "lives": self.lives,
            "steps": self.steps,
            "bricks_left": len(self.bricks)
        }

    def _render_game(self):
        # Draw bricks
        for i, brick in enumerate(self.bricks):
            color = self.BRICK_COLORS[brick.y // 20 % len(self.BRICK_COLORS)]
            pygame.draw.rect(self.screen, color, brick)
            pygame.draw.rect(self.screen, tuple(min(255, c+30) for c in color), brick.inflate(-6, -6))


        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw ball with antialiasing for smoothness
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0], p['pos'][1], 2, 2))

    def _render_ui(self):
        # Display score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Display lives as hearts
        for i in range(self.lives):
            self._draw_heart(self.SCREEN_WIDTH - 30 - i * 25, 18, self.COLOR_PADDLE)
        
        # Display game over message
        if self.game_over:
            msg = "YOU WIN!" if not self.bricks else "GAME OVER"
            color = (100, 255, 100) if not self.bricks else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_bricks(self):
        self.bricks = []
        brick_width = 60
        brick_height = 15
        padding = 5
        top_offset = 60
        side_offset = (self.SCREEN_WIDTH - (self.BRICK_COLS * (brick_width + padding))) // 2

        for row in range(self.BRICK_ROWS):
            for col in range(self.BRICK_COLS):
                x = side_offset + col * (brick_width + padding)
                y = top_offset + row * (brick_height + padding)
                self.bricks.append(pygame.Rect(x, y, brick_width, brick_height))

    def reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 5]
        angle = self.np_random.uniform(np.pi * 1.25, np.pi * 1.75)
        self.ball_vel = [math.cos(angle), math.sin(angle)]

    def _create_particles(self, pos, color):
        for _ in range(20):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
                'lifespan': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
    def _draw_heart(self, x, y, color):
        points = [
            (x, y-5), (x+5, y-10), (x+10, y-5),
            (x, y+5), (x-10, y-5), (x-5, y-10)
        ]
        pygame.draw.polygon(self.screen, color, points)

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
    
    # --- To play manually ---
    # This requires setting render_mode to "human" in __init__ and
    # handling pygame events. The current implementation is for rgb_array.
    # For demonstration, we'll run a simple loop and show one frame.
    
    # Reset the environment
    obs, info = env.reset()
    print("Initial state:")
    print(f"Score: {info['score']}, Lives: {info['lives']}")

    # Take a sample action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("\nAfter one step:")
    print(f"Action taken: {action}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Info: {info}")

    # To visualize a frame, you'd need a library like matplotlib or opencv
    # For example, using matplotlib:
    try:
        import matplotlib.pyplot as plt
        plt.imshow(obs)
        plt.title("Game Frame")
        plt.axis('off')
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Cannot display the game frame.")
        print("To visualize, run: pip install matplotlib")

    env.close()