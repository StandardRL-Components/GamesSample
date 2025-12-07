# Generated: 2025-08-27T15:26:13.115314
# Source Brief: brief_00990.md
# Brief Index: 990

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw  # Import the missing module
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Use the paddle to bounce the ball and "
        "destroy all blocks before time runs out."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 6.0
        self.MAX_STAGES = 3
        self.TIME_PER_STAGE = 60 # in seconds

        # Colors
        self._define_colors()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_main = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_velocity = None
        self.ball_speed_magnitude = None
        self.ball_attached = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.stage = 0
        self.balls_left = 0
        self.time_left = 0
        self.game_over = False
        self.game_won = False
        
        # This will call reset, which needs np_random to be initialized first.
        # super().reset() initializes the RNG.
        super().reset(seed=None)
        self.reset()
        self.validate_implementation()

    def _define_colors(self):
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_WALL = (150, 150, 180)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = {
            1: (217, 87, 99),   # Red
            2: (99, 217, 87),   # Green
            3: (87, 99, 217),   # Blue
            5: (217, 187, 87),  # Yellow
        }
        self.BLOCK_POINTS = list(self.BLOCK_COLORS.keys())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.stage = 1
        self.balls_left = 3
        self.game_over = False
        self.game_won = False
        self.particles = []
        self.ball_speed_magnitude = self.INITIAL_BALL_SPEED

        self._reset_level()
        
        return self._get_observation(), self._get_info()

    def _reset_level(self):
        self.time_left = self.TIME_PER_STAGE * self.FPS
        self._reset_ball_and_paddle()
        self._generate_blocks()
    
    def _reset_ball_and_paddle(self):
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball_attached = True
        self.ball_velocity = [0, 0]

    def _generate_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = self.np_random.integers(3, 6)
        cols = self.np_random.integers(8, 12)
        
        start_y = 50
        start_x = (self.SCREEN_WIDTH - cols * (block_width + 5)) // 2

        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() < 0.8: # Chance to spawn a block
                    points = self.np_random.choice(self.BLOCK_POINTS)
                    block_rect = pygame.Rect(
                        start_x + c * (block_width + 5),
                        start_y + r * (block_height + 5),
                        block_width,
                        block_height,
                    )
                    self.blocks.append({
                        "rect": block_rect,
                        "color": self.BLOCK_COLORS[points],
                        "points": points
                    })
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02  # Time penalty to encourage efficiency

        self._handle_input(action)
        ball_reward = self._update_ball()
        reward += ball_reward
        self._update_particles()
        
        condition_reward, terminated = self._check_game_conditions()
        reward += condition_reward
        
        self.game_over = terminated
        if self.game_over and self.balls_left <= 0:
            reward -= 100 # Final penalty for losing
        elif self.game_over and self.game_won:
            reward += 100 # Final bonus for winning

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # Launch ball
        if space_held and self.ball_attached:
            # sfx: ball_launch
            self.ball_attached = False
            angle = self.np_random.uniform(-0.3, 0.3) # Slight random angle
            self.ball_velocity = [
                self.ball_speed_magnitude * math.sin(angle),
                -self.ball_speed_magnitude * math.cos(angle)
            ]

    def _update_ball(self):
        reward = 0
        self.time_left = max(0, self.time_left - 1)

        if self.ball_attached:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
            return reward

        self.ball.x += self.ball_velocity[0]
        self.ball.y += self.ball_velocity[1]

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
            self.ball_velocity[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.SCREEN_WIDTH, self.ball.right)
            # sfx: wall_bounce
        if self.ball.top <= 0:
            self.ball_velocity[1] *= -1
            self.ball.top = max(0, self.ball.top)
            # sfx: wall_bounce

        # Bottom wall (lose ball)
        if self.ball.top >= self.SCREEN_HEIGHT:
            self.balls_left -= 1
            reward -= 5
            # sfx: lose_ball
            if self.balls_left > 0:
                self._reset_ball_and_paddle()
            else:
                self.game_over = True
            return reward

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_velocity[1] > 0:
            self.ball.bottom = self.paddle.top
            offset = (self.ball.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            offset = np.clip(offset, -0.9, 0.9) # Limit extreme angles
            
            new_vx = self.ball_speed_magnitude * offset
            new_vy_sq = self.ball_speed_magnitude**2 - new_vx**2
            new_vy = -math.sqrt(max(0.1, new_vy_sq)) # Ensure some vertical speed
            
            self.ball_velocity = [new_vx, new_vy]
            reward += 0.1 # Reward for keeping ball in play
            # sfx: paddle_bounce

        # Block collisions
        collided_block = None
        for block in self.blocks:
            if self.ball.colliderect(block['rect']):
                collided_block = block
                break
        
        if collided_block:
            # sfx: block_break
            reward += collided_block['points']
            self.score += collided_block['points']
            self._create_particles(collided_block['rect'].center, collided_block['color'])
            self.blocks.remove(collided_block)

            # Collision response
            prev_ball_center = (self.ball.centerx - self.ball_velocity[0], self.ball.centery - self.ball_velocity[1])
            
            # Check horizontal penetration
            if prev_ball_center[0] <= collided_block['rect'].left or prev_ball_center[0] >= collided_block['rect'].right:
                self.ball_velocity[0] *= -1
            # Check vertical penetration
            if prev_ball_center[1] <= collided_block['rect'].top or prev_ball_center[1] >= collided_block['rect'].bottom:
                self.ball_velocity[1] *= -1
            
            # Simple fallback to prevent sticking
            if self.ball.colliderect(collided_block['rect']):
                self.ball_velocity[1] *= -1

        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.2
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _check_game_conditions(self):
        reward = 0
        terminated = False

        if not self.blocks: # Stage clear
            reward += 50
            # sfx: stage_clear
            if self.stage < self.MAX_STAGES:
                self.stage += 1
                self.ball_speed_magnitude += 0.2
                self._reset_level()
            else:
                self.game_won = True
                terminated = True
        
        if self.time_left <= 0:
            terminated = True
        
        if self.balls_left <= 0:
            terminated = True
            
        return reward, terminated

    def _get_observation(self):
        # --- Render Background ---
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # --- Render Game Elements ---
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), block['rect'], 2)

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, p['radius']))
        
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        pygame.draw.circle(self.screen, self.COLOR_BALL, self.ball.center, self.BALL_RADIUS)
        # Add a glow effect to the ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS + 2, (*self.COLOR_BALL, 100))
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS + 4, (*self.COLOR_BALL, 50))

        # --- Render UI ---
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_text = self.font_ui.render(f"TIME: {int(self.time_left / self.FPS)}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH / 2 - time_text.get_width() / 2, 10))
        
        balls_text = self.font_ui.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.SCREEN_WIDTH - balls_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.game_won:
                end_text = self.font_main.render("YOU WIN!", True, self.COLOR_BALL)
            else:
                end_text = self.font_main.render("GAME OVER", True, self.BLOCK_COLORS[1])
                
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "stage": self.stage,
            "balls_left": self.balls_left,
            "time_left_seconds": int(self.time_left / self.FPS),
        }

    def close(self):
        pygame.font.quit()
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Un-comment the next line to run with a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()