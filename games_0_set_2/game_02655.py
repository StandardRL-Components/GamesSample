
# Generated: 2025-08-28T05:30:59.540356
# Source Brief: brief_02655.md
# Brief Index: 2655

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where risk-taking is rewarded and cautious play is penalized."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Used for physics scaling, not frame-advance rate
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (220, 220, 250)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BORDER = (80, 80, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = [
            (227, 99, 99), (99, 227, 99), (99, 99, 227), (227, 227, 99), (227, 150, 99)
        ]

        # Paddle settings
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12

        # Ball settings
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 8

        # Block settings
        self.NUM_BLOCKS_X = 10
        self.NUM_BLOCKS_Y = 5
        self.BLOCK_WIDTH = 58
        self.BLOCK_HEIGHT = 20
        self.BLOCK_SPACING_X = 6
        self.BLOCK_SPACING_Y = 5
        self.BLOCK_AREA_TOP = 50

        # Max episode steps
        self.MAX_STEPS = 1000

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize paddle
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        paddle_x = self.WIDTH / 2 - self.PADDLE_WIDTH / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Initialize ball
        self.ball_attached = True
        self._attach_ball_to_paddle()
        self.ball_vel = [0, 0]

        # Initialize blocks
        self.blocks = []
        total_block_width = self.NUM_BLOCKS_X * (self.BLOCK_WIDTH + self.BLOCK_SPACING_X) - self.BLOCK_SPACING_X
        start_x = (self.WIDTH - total_block_width) / 2
        for i in range(self.NUM_BLOCKS_Y):
            for j in range(self.NUM_BLOCKS_X):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING_X)
                y = self.BLOCK_AREA_TOP + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING_Y)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "color": color})

        # Initialize other state
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        
        reward = -0.02  # Per-step penalty

        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        if movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(1, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH - 1))

        if self.ball_attached:
            self._attach_ball_to_paddle()
            if space_pressed:
                self._launch_ball()
                # sound: ball_launch.wav
        
        # 2. Update game logic
        event_reward = self._update_ball()
        self._update_particles()
        reward += event_reward

        # 3. Check termination conditions
        self.steps += 1
        terminated = False
        if self.balls_left <= 0:
            reward -= 10  # Penalty for losing all balls
            terminated = True
            self.game_over = True
        elif not self.blocks:
            reward += 100  # Bonus for winning
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += event_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _attach_ball_to_paddle(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]

    def _launch_ball(self):
        self.ball_attached = False
        angle_rad = self.np_random.uniform(-math.pi * 0.2, math.pi * 0.2) # Launch within a 72-degree cone upwards
        self.ball_vel = [self.BALL_SPEED * math.sin(angle_rad), -self.BALL_SPEED * math.cos(angle_rad)]

    def _update_ball(self):
        if self.ball_attached:
            return 0
        
        reward = 0
        
        # Move ball
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.ball_pos[0], self.WIDTH - self.BALL_RADIUS))
            # sound: wall_bounce.wav
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sound: wall_bounce.wav

        # Bottom wall (lose ball)
        if ball_rect.top >= self.HEIGHT:
            self.balls_left -= 1
            self.ball_attached = True
            self._attach_ball_to_paddle()
            reward -= 10
            # sound: lose_ball.wav
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # prevent sticking

            # Risk/reward calculation
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            if abs(offset) > 0.75:
                reward += 2 # Risky play
            elif abs(offset) < 0.25:
                reward -= 0.2 # Safe play

            # Change horizontal velocity based on paddle hit location
            self.ball_vel[0] += offset * 4
            # Normalize to maintain constant speed
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED
            # sound: paddle_bounce.wav

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                self._create_particles(ball_rect.center, block["color"])
                self.blocks.remove(block)
                reward += 1 # Standard block break reward
                self.ball_vel[1] *= -1 # Simple vertical bounce
                # sound: block_break.wav
                break # Only break one block per frame
        
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

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
        # Draw border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 1) # Dark outline

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 3, 3))
            self.screen.blit(temp_surf, (int(p['pos'][0]-1), int(p['pos'][1]-1)))

    def _render_ui(self):
        # Render score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render balls left
        for i in range(self.balls_left):
            x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            msg = "GAME OVER"
            if not self.blocks:
                msg = "YOU WIN!"
            
            end_text = self.font_main.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
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
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement = 0 # No-op
        space = 0 # Released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
            
        action = [movement, space, 0] # shift is unused
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS) # Control the speed of manual play

    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    pygame.quit()