
# Generated: 2025-08-27T19:00:37.189731
# Source Brief: brief_02022.md
# Brief Index: 2022

        
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
        "A minimalist, neon-drenched block breaker. Clear all the blocks to win, but lose all your balls and it's game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_PADDLE = (57, 255, 20) # Neon Green
    COLOR_PADDLE_GLOW = (57, 255, 20, 50)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (255, 255, 255, 100)
    BLOCK_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0)] # Cyan, Magenta, Yellow
    COLOR_TEXT = (240, 240, 240)

    # Game element properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 7
    BALL_SPEED_INITIAL = 6.0
    BALL_SPEED_MAX = 10.0
    RISKY_SHOT_THRESHOLD = 0.7 # Normalized horizontal velocity

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.Font(None, 36)
        self.font_big = pygame.font.Font(None, 72)

        # This will be initialized in reset()
        self.np_random = None

        # Game state variables are initialized in reset()
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.prev_space_held = None
        self.is_risky_shot = None
        
        self.reset()
        # self.validate_implementation() # Validation can be run during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self._reset_ball()
        self._create_blocks()

        self.particles = []
        self.balls_left = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.is_risky_shot = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1

        # Handle paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.02 # Small penalty for moving
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.02 # Small penalty for moving
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # Handle ball launch
        is_space_press = space_held and not self.prev_space_held
        if is_space_press and not self.ball_launched:
            self.ball_launched = True
            # Launch angle depends on where ball is relative to paddle center
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            angle = math.pi / 2 - offset * (math.pi / 3) # Launch between 30 and 150 degrees
            self.ball_vel = [self.BALL_SPEED_INITIAL * math.cos(angle), -self.BALL_SPEED_INITIAL * math.sin(angle)]
            # SFX: Ball launch

        self.prev_space_held = space_held

        # Update game state
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        # Check termination conditions
        terminated = False
        if not self.blocks:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 50  # Win bonus
        elif self.balls_left <= 0:
            self.win = False
            self.game_over = True
            terminated = True
            reward -= 50 # Lose penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _reset_ball(self):
        self.ball_launched = False
        self.is_risky_shot = False
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.center = self.paddle.centerx, self.paddle.top - self.BALL_RADIUS
        self.ball_vel = [0, 0]

    def _create_blocks(self):
        self.blocks = []
        block_width, block_height = 40, 20
        rows, cols = 5, 14
        start_x = (self.SCREEN_WIDTH - cols * (block_width + 5)) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() > 0.2:  # 80% chance of a block
                    x = start_x + c * (block_width + 5)
                    y = start_y + r * (block_height + 5)
                    color = self.np_random.choice(self.BLOCK_COLORS)
                    self.blocks.append({"rect": pygame.Rect(x, y, block_width, block_height), "color": color})

    def _update_ball(self):
        if self.ball_launched:
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]
        else:
            # Ball follows paddle before launch
            self.ball.center = self.paddle.centerx, self.paddle.top - self.BALL_RADIUS

    def _handle_collisions(self):
        if not self.ball_launched:
            return 0
        
        reward = 0

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.SCREEN_WIDTH, self.ball.right)
            # SFX: Wall bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = max(0, self.ball.top)
            # SFX: Wall bounce

        # Lost ball
        if self.ball.top >= self.SCREEN_HEIGHT:
            self.balls_left -= 1
            reward -= 5
            if self.balls_left > 0:
                self._reset_ball()
            # SFX: Lose ball
            return reward

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # SFX: Paddle bounce
            reward += 0.1
            
            # Reverse vertical velocity
            self.ball_vel[1] *= -1
            self.ball.bottom = self.paddle.top

            # Add "spin" based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 3.0
            
            # Normalize and cap speed
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > self.BALL_SPEED_MAX:
                self.ball_vel = [(v / speed) * self.BALL_SPEED_MAX for v in self.ball_vel]
            
            # Check for risky shot
            norm_vx = abs(self.ball_vel[0]) / (math.hypot(self.ball_vel[0], self.ball_vel[1]) + 1e-6)
            if norm_vx > self.RISKY_SHOT_THRESHOLD:
                self.is_risky_shot = True
            else:
                self.is_risky_shot = False

        # Block collisions
        hit_block_idx = self.ball.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            # SFX: Block break
            block_data = self.blocks.pop(hit_block_idx)
            block_rect = block_data['rect']
            self.score += 1
            reward += 1
            
            if self.is_risky_shot:
                reward += 1 # Risky shot bonus (total reward for risky block break is +2)
                self.is_risky_shot = False # Consume bonus

            self._create_particles(block_rect.center, block_data['color'])

            # Simple bounce logic: prioritize vertical bounce
            self.ball_vel[1] *= -1

        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            size = int(self.BALL_RADIUS * 0.5 * (p['life'] / 40))
            if size > 0:
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

        # Render paddle with glow
        glow_surf = pygame.Surface((self.PADDLE_WIDTH + 20, self.PADDLE_HEIGHT + 20), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PADDLE_GLOW, (10, 10, self.PADDLE_WIDTH, self.PADDLE_HEIGHT), border_radius=8)
        self.screen.blit(glow_surf, (self.paddle.x - 10, self.paddle.y - 10))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=8)

        # Render ball with glow
        ball_center = (int(self.ball.centerx), int(self.ball.centery))
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Render score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render balls left
        for i in range(self.balls_left - 1): # Don't draw the ball in play
            pos_x = self.SCREEN_WIDTH - 20 - i * (self.BALL_RADIUS * 2 + 5)
            pos_y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            text_surf = self.font_big.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

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
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Get key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(30)

    env.close()