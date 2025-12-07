
# Generated: 2025-08-27T13:47:52.150469
# Source Brief: brief_00490.md
# Brief Index: 490

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist, procedurally generated block breaker where risk-taking is rewarded. Clear all blocks to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 7
    BALL_SPEED = 7
    MAX_STEPS = 2000
    INITIAL_LIVES = 3

    # --- Colors ---
    COLOR_BG_TOP = (15, 20, 45)
    COLOR_BG_BOTTOM = (30, 40, 70)
    COLOR_PADDLE = (240, 240, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (200, 200, 0, 64)
    COLOR_BORDER = (80, 90, 120)
    COLOR_UI_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 87, 34), (255, 193, 7), (139, 195, 74), (0, 188, 212),
        (33, 150, 243), (156, 39, 176), (233, 30, 99)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # Game state variables are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.on_paddle = None
        self.blocks = None
        self.block_colors = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        # Initialize game state
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.on_paddle = True
        self._reset_ball()

        self.blocks = []
        self.block_colors = []
        block_w, block_h = 58, 20
        margin_x, margin_y = (self.WIDTH - 10 * block_w) / 2, 50
        for i in range(10):
            for j in range(10):
                self.blocks.append(pygame.Rect(
                    margin_x + i * block_w,
                    margin_y + j * block_h,
                    block_w - 2,
                    block_h - 2
                ))
                self.block_colors.append(self.np_random.choice(self.BLOCK_COLORS))

        self.particles = []
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.on_paddle = True
        self.ball_pos = np.array([
            self.paddle.centerx,
            self.paddle.top - self.BALL_RADIUS
        ], dtype=np.float64)
        self.ball_vel = np.array([0.0, 0.0])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1

        reward = -0.02  # Small penalty for each step to encourage efficiency

        # 1. Handle player actions
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.clamp_ip(self.screen.get_rect())

        if self.on_paddle and space_held:
            # Launch ball
            self.on_paddle = False
            launch_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = np.array([math.cos(launch_angle), math.sin(launch_angle)]) * self.BALL_SPEED
            # Sound: Ball launch

        # 2. Update game state
        if self.on_paddle:
            self.ball_pos[0] = self.paddle.centerx
        else:
            reward += self._update_ball()

        self._update_particles()
        self.steps += 1

        # 3. Check for termination
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if not self.blocks: # Win condition
            reward += 100
            self.game_over = True
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball(self):
        reward = 0
        self.ball_pos += self.ball_vel

        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.clamp_ip(self.screen.get_rect())
            self.ball_pos[0] = ball_rect.centerx
            # Sound: Wall bounce

        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.clamp_ip(self.screen.get_rect())
            self.ball_pos[1] = ball_rect.centery
            # Sound: Wall bounce

        # Bottom wall (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            reward -= 10
            # Sound: Lose life
            if self.lives <= 0:
                self.game_over = True
            else:
                self._reset_ball()
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound: Paddle bounce
            self.ball_vel[1] *= -1
            # Add horizontal velocity based on where it hit the paddle
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.0
            # Normalize to maintain constant speed
            self.ball_vel = self.ball_vel / np.linalg.norm(self.ball_vel) * self.BALL_SPEED
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking

        # Block collisions
        hit_index = ball_rect.collidelist(self.blocks)
        if hit_index != -1:
            # Sound: Block break
            block = self.blocks.pop(hit_index)
            block_color = self.block_colors.pop(hit_index)
            self._create_particles(block.center, block_color)
            self.score += 1
            reward += 1

            # Determine bounce direction
            # A simple but effective way is to check the relative position of the ball and block centers
            dx = self.ball_pos[0] - block.centerx
            dy = self.ball_pos[1] - block.centery
            w, h = block.width / 2, block.height / 2

            if abs(dx / w) > abs(dy / h): # Horizontal collision
                self.ball_vel[0] *= -1
            else: # Vertical collision
                self.ball_vel[1] *= -1

        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            # Interpolate between top and bottom colors
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.HEIGHT
            pygame.draw.line(self.screen, (int(r), int(g), int(b)), (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw blocks
        for i, block in enumerate(self.blocks):
            pygame.draw.rect(self.screen, self.block_colors[i], block, border_radius=3)
            # Add a subtle highlight
            highlight_color = tuple(min(255, c + 40) for c in self.block_colors[i])
            pygame.draw.rect(self.screen, highlight_color, block.inflate(-block.width*0.8, -block.height*0.8).move(0, -2), border_radius=2)

        # Draw particles
        for p in self.particles:
            size = int(self.BALL_RADIUS * (p['life'] / 30.0))
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (p['pos'][0]-size/2, p['pos'][1]-size/2, size, size))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        pygame.draw.rect(self.screen, (255,255,255), self.paddle.inflate(-4,-4), border_radius=3)

        # Draw ball with glow
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        for i in range(self.BALL_RADIUS, 0, -2):
             alpha = 100 - (i / self.BALL_RADIUS) * 100
             glow_color = (*self.COLOR_BALL, int(alpha))
             pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + i, glow_color)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            pos = (self.WIDTH - 70 + i * 20, 18)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_PADDLE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_PADDLE)

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            color = (100, 255, 100) if not self.blocks else (255, 100, 100)
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
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
        assert "score" in info

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        assert "steps" in info

        # Test game logic assertions
        self.reset()
        assert self.on_paddle
        self.step([0, 1, 0]) # Launch ball
        assert not self.on_paddle
        assert np.any(self.ball_vel != 0)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0]  # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            total_reward = 0

        clock.tick(60) # Control the frame rate for human play
        
    env.close()