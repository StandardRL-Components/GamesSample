import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Controls: Use ← and → to move the paddle."

    # User-facing description of the game
    game_description = "Classic arcade block-breaker. Use the paddle to keep the ball in play and destroy all the blocks."

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.BALL_SPEED_INITIAL = 4.0
        self.BALL_SPEED_MIN = 3.0
        self.BALL_SPEED_MAX = 8.0
        
        self.BLOCK_COLS, self.BLOCK_ROWS = 10, 10
        self.BLOCK_TOTAL_WIDTH = 580
        self.BLOCK_SPACING = 4
        self.BLOCK_WIDTH = (self.BLOCK_TOTAL_WIDTH - (self.BLOCK_COLS - 1) * self.BLOCK_SPACING) // self.BLOCK_COLS
        self.BLOCK_HEIGHT = 12
        
        self.MAX_STEPS = 5000
        self.INITIAL_LIVES = 3

        # --- Colors ---
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 60, 60)
        self.COLOR_BALL_GLOW = (255, 60, 60)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = [
            (50, 220, 220), (220, 50, 220), (220, 220, 50),
            (50, 220, 50), (220, 140, 50)
        ]
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        # Setups pygame to run headlessly
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_msg = pygame.font.Font(None, 50)
        
        # --- Game State Variables ---
        self.paddle = None
        self.ball = None
        self.ball_velocity = None
        self.blocks = None
        self.block_colors = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False

        # self.reset() is called here to set up the initial state,
        # which is needed for validate_implementation to run correctly.
        # However, the user is still expected to call reset() once before starting.
        self.reset(seed=42)
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False
        
        self._setup_paddle()
        self._setup_ball()
        self._setup_blocks()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement = action[0]  # 3=left, 4=right

        reward = -0.02  # Small penalty for each step to encourage efficiency

        self._handle_input(movement)
        event_reward = self._update_ball_and_collisions()
        reward += event_reward
        self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100.0
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated is true, terminated should also be true

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _setup_paddle(self):
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

    def _setup_ball(self, new_life=False):
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        if not new_life:
            angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards
        else: # After losing a life, make it less random
            angle = -math.pi/2
            
        self.ball_velocity = [
            math.cos(angle) * self.BALL_SPEED_INITIAL,
            math.sin(angle) * self.BALL_SPEED_INITIAL
        ]

    def _setup_blocks(self):
        self.blocks = []
        self.block_colors = []
        start_x = (self.SCREEN_WIDTH - self.BLOCK_TOTAL_WIDTH) / 2
        start_y = 50
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                block_rect = pygame.Rect(
                    start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING),
                    start_y + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING),
                    self.BLOCK_WIDTH,
                    self.BLOCK_HEIGHT
                )
                self.blocks.append(block_rect)
                self.block_colors.append(self.BLOCK_COLORS[(i // 2) % len(self.BLOCK_COLORS)])

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))

    def _update_ball_and_collisions(self):
        reward = 0.0
        
        self.ball.x += self.ball_velocity[0]
        self.ball.y += self.ball_velocity[1]

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
            self.ball_velocity[0] *= -1
            self.ball.x = max(0, min(self.SCREEN_WIDTH - self.ball.width, self.ball.x))
        if self.ball.top <= 0:
            self.ball_velocity[1] *= -1
            self.ball.y = max(0, self.ball.y)

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_velocity[1] > 0:
            self.ball.bottom = self.paddle.top
            self.ball_velocity[1] *= -1
            
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_velocity[0] += offset * 2.5

        # Block collision
        collided_idx = self.ball.collidelist(self.blocks)
        if collided_idx != -1:
            block = self.blocks.pop(collided_idx)
            color = self.block_colors.pop(collided_idx)
            
            self.score += 10
            reward += 1.0

            self._create_particles(block.center, color)
            
            self.ball_velocity[1] *= -1

        # Bottom wall (lose life)
        if self.ball.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward = -10.0
            if self.lives > 0:
                self._setup_ball(new_life=True)
            else:
                self.game_over = True

        # Speed control
        speed = math.hypot(*self.ball_velocity)
        if speed < self.BALL_SPEED_MIN and speed > 0:
            self.ball_velocity[0] = (self.ball_velocity[0] / speed) * self.BALL_SPEED_MIN
            self.ball_velocity[1] = (self.ball_velocity[1] / speed) * self.BALL_SPEED_MIN
        elif speed > self.BALL_SPEED_MAX:
            self.ball_velocity[0] = (self.ball_velocity[0] / speed) * self.BALL_SPEED_MAX
        
        # Prevent horizontal trapping
        if abs(self.ball_velocity[1]) < 0.5:
            self.ball_velocity[1] = math.copysign(0.5, self.ball_velocity[1]) if self.ball_velocity[1] != 0 else 0.5

        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(1, 4),
                'color': color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        if self.lives <= 0:
            self.game_over = True
        if not self.blocks:
            self.game_over = True
            self.win = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw blocks
        for i, block in enumerate(self.blocks):
            pygame.draw.rect(self.screen, self.block_colors[i], block)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.block_colors[i]), block, 1)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_BALL_GLOW, 60), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (self.ball.centerx - glow_radius, self.ball.centery - glow_radius))
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 120, 10))
        for i in range(self.lives):
            life_rect = pygame.Rect(self.SCREEN_WIDTH - 60 + i * 20, 12, 15, 5)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=2)

        # Game Over / Win Message
        if self.game_over and not (self.steps >= self.MAX_STEPS):
            msg = "GAME OVER" if not self.win else "YOU WIN!"
            msg_text = self.font_msg.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

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
        assert self.lives == self.INITIAL_LIVES
        assert len(self.blocks) == self.BLOCK_COLS * self.BLOCK_ROWS
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        # Test termination by losing a life
        self.reset()
        self.lives = 1
        self.ball.y = self.SCREEN_HEIGHT - 1  # Position ball just before the bottom
        self.ball_velocity = [0, 10]          # Ensure it moves downwards to trigger loss
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert term is True, "Termination flag should be True after losing the last life"
        assert self.lives == 0, "Lives should be 0 after losing the last life"
        
        # Test termination by winning
        self.reset()
        self.blocks = []
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert term is True, "Termination flag should be True after clearing all blocks"
        assert self.win is True, "Win flag should be True after clearing all blocks"

        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # The environment is validated upon initialization
    env = GameEnv(render_mode="rgb_array")
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful. Initial info:", info)
    
    # Test a few random steps
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if (i+1) % 20 == 0:
            print(f"Step {i+1}: Reward={reward:.2f}, Info={info}, Terminated={terminated}, Truncated={truncated}")
        if terminated or truncated:
            print("Episode finished.")
            break
            
    env.close()

    # To visualize the game, you would need to change the environment setup
    # and run a loop that renders to the screen.
    print("\nTo visualize, run the following code block:")
    print("""
# --- Visualization Code ---
# if __name__ == '__main__':
#     import pygame
#     # Note: remove os.environ["SDL_VIDEODRIVER"] = "dummy" from __init__ to visualize
#     env = GameEnv(render_mode='rgb_array')
#     obs, info = env.reset()
#     
#     screen = pygame.display.set_mode((640, 400))
#     pygame.display.set_caption("Block Breaker")
#     clock = pygame.time.Clock()
#     
#     terminated, truncated = False, False
#     while not (terminated or truncated):
#         # Simple agent: follow the ball
#         ball_x = env.ball.centerx
#         paddle_x = env.paddle.centerx
#         
#         movement = 0 # no-op
#         if ball_x < paddle_x - 10:
#             movement = 3 # left
#         elif ball_x > paddle_x + 10:
#             movement = 4 # right
#
#         action = [movement, 0, 0] # Move, no other actions
#         
#         obs, reward, terminated, truncated, info = env.step(action)
#         
#         # Render to screen
#         surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
#         screen.blit(surf, (0, 0))
#         pygame.display.flip()
#         
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 terminated = True
#                 
#         clock.tick(60) # Limit frame rate
#         
#     env.close()
#     pygame.quit()
    """)