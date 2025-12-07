import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A retro arcade block breaker. Use the paddle to bounce the ball and destroy all the blocks. Don't let the ball fall!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 8
        self.BASE_BALL_SPEED = 4.0
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 3
        self.BLOCK_ROWS = 5
        self.BLOCK_COLS = 10
        self.TOTAL_BLOCKS = self.BLOCK_ROWS * self.BLOCK_COLS

        # EXACT spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (200, 200, 220)
        self.BLOCK_COLORS = [
            (255, 70, 70), (255, 165, 0), (0, 200, 200), (70, 255, 70), (180, 70, 255)
        ]

        # Fonts
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 32)
        
        # State variables (initialized in reset)
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.blocks_destroyed_count = None
        self.score = None
        self.lives = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.particles = None
        self.ball_stuck_counter = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset paddle
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Reset ball
        self.ball_launched = False
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=np.float64)
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        # Reset blocks
        self.blocks = []
        block_width = self.WIDTH // self.BLOCK_COLS
        block_height = 20
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                block_rect = pygame.Rect(
                    j * block_width,
                    i * block_height + 50,
                    block_width - 2,
                    block_height - 2,
                )
                self.blocks.append(block_rect)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False
        self.blocks_destroyed_count = 0
        self.particles = []
        self.ball_stuck_counter = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty per step to encourage efficiency
        self.steps += 1

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Actions ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        if not self.ball_launched:
            # Ball follows paddle
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            if space_held:
                # Launch ball
                self.ball_launched = True
                # FIX: low must be less than or equal to high for np.random.uniform
                angle = self.np_random.uniform(-2 * math.pi / 3, -math.pi / 3) # Upwards
                self.ball_vel = np.array([math.cos(angle), math.sin(angle)])
                # sfx: launch_ball.wav

        # --- Update Game Logic ---
        block_reward = 0
        if self.ball_launched:
            block_reward = self._update_ball()
        
        reward += block_reward
        
        # Update particles
        self._update_particles()
        
        # --- Check Post-Update Game State ---
        
        # Check for losing a life
        if self.ball_launched and self.ball_pos[1] > self.HEIGHT + self.BALL_RADIUS:
            self.lives -= 1
            reward -= 5
            self.ball_launched = False
            self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=np.float64)
            self.ball_vel = np.array([0.0, 0.0])
            # sfx: lose_life.wav
            if self.lives <= 0:
                self.game_over = True
                self.win = False

        # Check for win condition
        if self.blocks_destroyed_count == self.TOTAL_BLOCKS:
            if not self.game_over: # Give win reward only once
                reward += 100
                # sfx: game_win.wav
            self.game_over = True
            self.win = True
        
        # Check for max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        terminated = self.game_over
        truncated = False # Game ends on max_steps, considered termination

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_ball(self):
        # Calculate current speed
        speed_increase = (self.blocks_destroyed_count // 10) * 0.5
        current_speed = self.BASE_BALL_SPEED + speed_increase
        
        # Normalize velocity and apply speed
        norm = np.linalg.norm(self.ball_vel)
        if norm > 0:
            self.ball_vel = self.ball_vel / norm * current_speed
        
        self.ball_pos += self.ball_vel
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce.wav
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce.wav

        # Paddle collision
        if ball_rect.colliderect(self.paddle):
            if self.ball_vel[1] > 0: # Only reflect if moving downwards
                self.ball_vel[1] *= -1
                # Add "spin" based on hit location
                offset = (self.paddle.centerx - self.ball_pos[0]) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] -= offset * 2.0
                self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
                self.ball_stuck_counter = 0 # Reset stuck counter
                # sfx: paddle_bounce.wav
        
        # Anti-softlock mechanism
        if abs(self.ball_vel[1]) < 0.2:
            self.ball_stuck_counter += 1
            if self.ball_stuck_counter > 200:
                self.ball_vel[1] += self.np_random.choice([-0.5, 0.5])
                self.ball_stuck_counter = 0
        else:
            self.ball_stuck_counter = 0

        # Block collisions
        reward = 0
        collided_index = ball_rect.collidelist(self.blocks)
        if collided_index != -1:
            collided_block = self.blocks.pop(collided_index)
            # sfx: block_break.wav

            # Create particle explosion
            self._create_particles(collided_block.center, self.BLOCK_COLORS[self.blocks_destroyed_count % len(self.BLOCK_COLORS)])

            # Determine bounce direction
            prev_ball_pos = self.ball_pos - self.ball_vel
            if (prev_ball_pos[0] < collided_block.left or prev_ball_pos[0] > collided_block.right):
                 self.ball_vel[0] *= -1
            else:
                 self.ball_vel[1] *= -1
            
            self.score += 10
            reward += 10
            self.blocks_destroyed_count += 1
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw blocks
        for i, block in enumerate(self.blocks):
            # The color should be based on the original position, not the current index
            # but this is a minor visual detail. Using current index for now.
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
            pygame.draw.rect(self.screen, color, block, border_radius=3)
        
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = max(0, p['lifetime'] * 8)
            color = (*p['color'], alpha)
            size = max(1, int(p['lifetime'] / 5))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        self.reset()
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
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    
    # For this example, we'll run a random agent headless.
    obs, info = env.reset(seed=42)
    terminated = False
    total_reward = 0
    frame_count = 0
    
    print("Running random agent...")
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frame_count += 1
        
        if terminated or truncated:
            print(f"Episode finished after {frame_count} frames.")
            print(f"Final Info: {info}")
            print(f"Total Reward: {total_reward:.2f}")
            break

    env.close()