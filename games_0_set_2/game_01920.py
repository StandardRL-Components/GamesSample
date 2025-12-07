
# Generated: 2025-08-27T18:42:01.017304
# Source Brief: brief_01920.md
# Brief Index: 1920

        
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


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = np_random.integers(20, 40)
        self.radius = np_random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.radius -= 0.1
        return self.life > 0 and self.radius > 0

    def draw(self, surface):
        if self.life > 0 and self.radius > 1:
            alpha = max(0, min(255, int(255 * (self.life / 20))))
            r, g, b = self.color
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), (r, g, b, alpha))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A minimalist, neon-themed block-breaking game. Destroy all 100 blocks to win, but don't lose all your balls!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    BALL_SPEED = 6
    MAX_STEPS = 3000
    STARTING_BALLS = 3
    TOP_MARGIN = 40  # Space for UI

    # --- Colors ---
    COLOR_BG_TOP = (10, 10, 20)
    COLOR_BG_BOTTOM = (30, 10, 40)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = {
        10: (50, 200, 50),   # Green
        20: (50, 100, 255),  # Blue
        30: (180, 50, 255),  # Purple
    }

    # --- Rewards ---
    REWARD_BREAK_BLOCK = 1.0
    REWARD_RISKY_CATCH = 0.1
    REWARD_CHAIN_BONUS = 5.0
    REWARD_WIN = 100.0
    PENALTY_LOSE_BALL = -50.0
    PENALTY_STEP = -0.01


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.block_data = None
        self.particles = None
        self.score = None
        self.balls_left = None
        self.steps = None
        self.blocks_broken_this_trajectory = None
        self.chain_bonus_awarded = None
        self.game_over = None

        self.reset()
        
        # Self-validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_attached = True
        self._reset_ball()

        self.blocks = []
        self.block_data = []
        block_rows = 10
        block_cols = 10
        block_width = (self.WIDTH - (block_cols - 1) * 2) / block_cols
        block_height = 15
        for i in range(block_rows):
            for j in range(block_cols):
                points = random.choice(list(self.BLOCK_COLORS.keys()))
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(
                    j * (block_width + 2),
                    self.TOP_MARGIN + 20 + i * (block_height + 2),
                    block_width,
                    block_height,
                )
                self.blocks.append(rect)
                self.block_data.append({"color": color, "points": points})

        self.particles = []
        self.score = 0
        self.balls_left = self.STARTING_BALLS
        self.steps = 0
        self.game_over = False
        self.blocks_broken_this_trajectory = 0
        self.chain_bonus_awarded = False

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.blocks_broken_this_trajectory = 0
        self.chain_bonus_awarded = False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = self.PENALTY_STEP
        self.steps += 1

        # 1. Unpack and handle actions
        movement = action[0]
        space_pressed = action[1] == 1

        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if self.ball_attached and space_pressed:
            # Sound: Ball Launch
            self.ball_attached = False
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)).normalize() * self.BALL_SPEED

        # 2. Update game state
        self._update_particles()

        if self.ball_attached:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel
            reward += self._handle_collisions()

        # 3. Check for termination
        terminated = False
        if len(self.blocks) == 0:
            reward += self.REWARD_WIN
            terminated = True
            self.game_over = True
        elif self.balls_left <= 0:
            terminated = True
            self.game_over = True
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

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel.x *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            # Sound: Wall Bounce
        if ball_rect.top <= self.TOP_MARGIN:
            self.ball_vel.y *= -1
            ball_rect.top = self.TOP_MARGIN
            # Sound: Wall Bounce

        # Bottom wall (lose ball)
        if ball_rect.top >= self.HEIGHT:
            self.balls_left -= 1
            reward += self.PENALTY_LOSE_BALL
            if self.balls_left > 0:
                self._reset_ball()
            # Sound: Lose Ball
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            # Sound: Paddle Hit
            self.ball_vel.y *= -1
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel.x = self.BALL_SPEED * offset * 1.2
            self.ball_vel = self.ball_vel.normalize() * self.BALL_SPEED
            
            # Risky catch reward
            if abs(offset) > 0.75:
                reward += self.REWARD_RISKY_CATCH
            
            # Reset chain counter
            self.blocks_broken_this_trajectory = 0
            self.chain_bonus_awarded = False
            ball_rect.bottom = self.paddle.top

        # Block collisions
        collided_idx = ball_rect.collidelist(self.blocks)
        if collided_idx != -1:
            # Sound: Block Break
            block_rect = self.blocks[collided_idx]
            
            # Determine bounce direction
            prev_ball_pos = self.ball_pos - self.ball_vel
            if (prev_ball_pos.y < block_rect.top) or (prev_ball_pos.y > block_rect.bottom):
                 self.ball_vel.y *= -1
            else:
                 self.ball_vel.x *= -1

            data = self.block_data[collided_idx]
            self._create_particles(block_rect.centerx, block_rect.centery, data["color"])
            
            self.score += data["points"]
            reward += self.REWARD_BREAK_BLOCK

            self.blocks.pop(collided_idx)
            self.block_data.pop(collided_idx)
            
            self.blocks_broken_this_trajectory += 1
            if self.blocks_broken_this_trajectory >= 3 and not self.chain_bonus_awarded:
                reward += self.REWARD_CHAIN_BONUS
                self.chain_bonus_awarded = True

        self.ball_pos.x = ball_rect.centerx
        self.ball_pos.y = ball_rect.centery
        return reward

    def _create_particles(self, x, y, color):
        for _ in range(self.np_random.integers(15, 25)):
            self.particles.append(Particle(x, y, color, self.np_random))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # Draw play area top border
        pygame.draw.line(self.screen, self.COLOR_TEXT, (0, self.TOP_MARGIN), (self.WIDTH, self.TOP_MARGIN))

    def _render_game(self):
        # Draw blocks
        for i, block in enumerate(self.blocks):
            color = self.block_data[i]["color"]
            pygame.draw.rect(self.screen, color, block)
            # Add a slight inner bevel for depth
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, highlight, block.topleft, block.topright)
            pygame.draw.line(self.screen, highlight, block.topleft, block.bottomleft)
            pygame.draw.line(self.screen, shadow, block.bottomleft, block.bottomright)
            pygame.draw.line(self.screen, shadow, block.topright, block.bottomright)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls
        balls_text = self.font_small.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 10))

        # Blocks remaining
        blocks_text = self.font_large.render(f"{len(self.blocks)} BLOCKS", True, self.COLOR_TEXT)
        text_rect = blocks_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 25))
        self.screen.blit(blocks_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks)
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
    # This block allows you to play the game with keyboard controls
    # Requires `pip install gymnasium[classic-control]`
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    
    while running:
        # Action defaults
        movement = 0 # no-op
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
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            # A small delay before restarting
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS
        
    env.close()