
# Generated: 2025-08-27T23:57:22.897073
# Source Brief: brief_03638.md
# Brief Index: 3638

        
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

    # Short, user-facing control string
    user_guide = "Controls: ←→ to move the paddle. Press space to launch the ball."

    # Short, user-facing description of the game
    game_description = "A fast-paced, top-down block breaker where strategic paddle positioning and risky plays are rewarded."

    # Frames auto-advance at 30fps
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            30: (255, 70, 70),   # Red
            20: (70, 150, 255),  # Blue
            10: (70, 255, 150),  # Green
        }

        # Paddle settings
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12

        # Ball settings
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 4.0
        self.MAX_BOUNCE_ANGLE = math.pi / 2.4 # ~75 degrees

        # Block settings
        self.BLOCK_ROWS = 5
        self.BLOCK_COLS = 10
        self.TOTAL_BLOCKS = self.BLOCK_ROWS * self.BLOCK_COLS
        self.BLOCK_WIDTH = 58
        self.BLOCK_HEIGHT = 18
        self.BLOCK_SPACING = 6

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = None
        self.ball_in_play = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.blocks_destroyed_count = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.blocks_destroyed_count = 0
        
        # Paddle state
        paddle_y = self.HEIGHT - 40
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Particle system
        self.particles = []

        # Create blocks
        self.blocks = []
        block_y_offset = 40
        point_values = [30, 30, 20, 20, 10]
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                points = point_values[i]
                color = self.BLOCK_COLORS[points]
                bx = j * (self.BLOCK_WIDTH + self.BLOCK_SPACING) + (self.WIDTH - self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) + self.BLOCK_SPACING) / 2
                by = i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING) + block_y_offset
                block_rect = pygame.Rect(bx, by, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({'rect': block_rect, 'color': color, 'points': points})
        
        self._reset_ball()

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_in_play = False
        self.ball_speed = self.INITIAL_BALL_SPEED + 0.05 * (self.blocks_destroyed_count // 20)
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        
        # --- Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if not self.ball_in_play:
            self.ball_pos[0] = self.paddle.centerx
            if space_held:
                # Sound: Ball Launch
                self.ball_in_play = True
                angle = (self.np_random.random() - 0.5) * (math.pi / 4) # Random launch angle up to 45 deg
                self.ball_vel = [self.ball_speed * math.sin(angle), -self.ball_speed * math.cos(angle)]

        # --- Update Game State ---
        if self.ball_in_play:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]

        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # --- Collision Detection ---
        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # Sound: Wall Bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery
            # Sound: Wall Bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            offset = ball_rect.centerx - self.paddle.centerx
            normalized_offset = offset / (self.PADDLE_WIDTH / 2)
            bounce_angle = normalized_offset * self.MAX_BOUNCE_ANGLE
            
            self.ball_vel[0] = self.ball_speed * math.sin(bounce_angle)
            self.ball_vel[1] = -self.ball_speed * math.cos(bounce_angle)
            
            # Ensure ball velocity is not purely horizontal
            if abs(self.ball_vel[1]) < 0.1:
                self.ball_vel[1] = -0.1 if self.ball_vel[1] <= 0 else 0.1
            
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            # Sound: Paddle Hit
            
        # Block collisions
        collided_block = None
        for block in self.blocks:
            if ball_rect.colliderect(block['rect']):
                collided_block = block
                break
        
        if collided_block:
            # Sound: Block Break
            self.blocks.remove(collided_block)
            reward += collided_block['points']
            self.score += collided_block['points']
            self.blocks_destroyed_count += 1
            
            # Create particles
            for _ in range(15):
                self._create_particle(ball_rect.center, collided_block['color'])

            # Increase speed every 20 blocks
            if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 20 == 0:
                self.ball_speed += 0.25
            
            # Collision response
            prev_ball_rect = pygame.Rect(ball_rect.x - self.ball_vel[0], ball_rect.y - self.ball_vel[1], ball_rect.width, ball_rect.height)
            
            if prev_ball_rect.bottom <= collided_block['rect'].top or prev_ball_rect.top >= collided_block['rect'].bottom:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1

        # Ball lost
        if ball_rect.top >= self.HEIGHT:
            # Sound: Ball Lost
            self.balls_left -= 1
            reward -= 10
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
                reward -= 100

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # --- Termination ---
        self.steps += 1
        terminated = self.game_over
        
        if len(self.blocks) == 0:
            terminated = True
            reward += 100
            self.score += 100 # Bonus for clearing
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _create_particle(self, pos, color):
        angle = self.np_random.random() * 2 * math.pi
        speed = self.np_random.random() * 2 + 1
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        lifespan = self.np_random.integers(15, 30)
        self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            # Add a subtle 3D effect
            highlight_color = tuple(min(255, c + 30) for c in block['color'])
            shadow_color = tuple(max(0, c - 30) for c in block['color'])
            pygame.draw.line(self.screen, highlight_color, block['rect'].topleft, block['rect'].topright, 2)
            pygame.draw.line(self.screen, highlight_color, block['rect'].topleft, block['rect'].bottomleft, 2)
            pygame.draw.line(self.screen, shadow_color, block['rect'].bottomleft, block['rect'].bottomright, 2)
            pygame.draw.line(self.screen, shadow_color, block['rect'].topright, block['rect'].bottomright, 2)

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Render ball
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30))))
            color_with_alpha = p['color'] + (alpha,)
            size = max(1, int(self.BALL_RADIUS / 3 * (p['lifespan'] / 30)))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

        # Render UI
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        for i in range(self.balls_left - 1):
            ball_icon_pos = (self.WIDTH - 20 - i * (self.BALL_RADIUS * 2 + 5), 15)
            pygame.gfxdraw.filled_circle(self.screen, ball_icon_pos[0], ball_icon_pos[1], self.BALL_RADIUS - 2, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, ball_icon_pos[0], ball_icon_pos[1], self.BALL_RADIUS - 2, self.COLOR_PADDLE)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
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

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
        
        clock.tick(30) # Match the auto_advance rate
        
    env.close()