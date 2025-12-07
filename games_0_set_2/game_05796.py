
# Generated: 2025-08-28T06:06:58.274107
# Source Brief: brief_05796.md
# Brief Index: 5796

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set Pygame to run in a headless mode for server environments
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Break all the blocks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated block-breaking game where risk-taking is rewarded. "
        "Clear all the blocks with 3 balls to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_BORDER = (200, 200, 220)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = {
        1: (0, 200, 100),  # Green
        2: (100, 150, 255), # Blue
        3: (255, 100, 100), # Red
    }
    
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 7
    MAX_BALL_SPEED = 12
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        self.np_random = None
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.balls_left = 3
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self._create_blocks()
        self._spawn_ball()

        self.particles = []
        self.last_hit_step = -10
        self.combo_bonus = 5

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        self._current_reward = -0.01  # Small penalty for time
        
        if not self.game_over:
            self._handle_input(movement)
            self._update_ball()
            self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                self._current_reward += 100
            elif self.balls_left <= 0:
                self._current_reward -= 100
        
        return (
            self._get_observation(),
            self._current_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

    def _update_ball(self):
        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            # sfx: wall_bounce

        # Paddle collision
        if self.paddle.colliderect(self._get_ball_rect()):
            if self.ball_vel.y > 0:
                self.ball_vel.y *= -1
                
                # Influence angle based on hit position
                offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel.x += offset * 2
                
                # Clamp ball speed
                speed = self.ball_vel.length()
                if speed > self.MAX_BALL_SPEED:
                    self.ball_vel = self.ball_vel.normalize() * self.MAX_BALL_SPEED
                elif speed < self.INITIAL_BALL_SPEED:
                    self.ball_vel = self.ball_vel.normalize() * self.INITIAL_BALL_SPEED
                
                self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
                # sfx: paddle_hit

        # Block collisions
        ball_rect = self._get_ball_rect()
        for i, block_data in reversed(list(enumerate(self.blocks))):
            block_rect, color, points = block_data
            if block_rect.colliderect(ball_rect):
                self._handle_block_collision(i, block_rect, color, points)
                break # Handle one block per frame to avoid weird physics

        # Ball lost
        if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
            self.balls_left -= 1
            # sfx: lose_ball
            if self.balls_left > 0:
                self._spawn_ball()
            else:
                self.game_over = True

    def _handle_block_collision(self, block_index, block_rect, color, points):
        # Determine collision side to correctly reflect the ball
        ball_rect = self._get_ball_rect()
        dx = self.ball_pos.x - block_rect.centerx
        dy = self.ball_pos.y - block_rect.centery
        w = (ball_rect.width + block_rect.width) / 2
        h = (ball_rect.height + block_rect.height) / 2
        wy, hx = w * dy, h * dx
        
        if wy > hx:
            if wy > -hx: # Top
                self.ball_vel.y = -abs(self.ball_vel.y)
            else: # Left
                self.ball_vel.x = -abs(self.ball_vel.x)
        else:
            if wy > -hx: # Right
                self.ball_vel.x = abs(self.ball_vel.x)
            else: # Bottom
                self.ball_vel.y = abs(self.ball_vel.y)

        # Calculate angle-based reward
        angle = math.atan2(abs(self.ball_vel.y), abs(self.ball_vel.x))
        if angle > math.pi / 3: # Steeper than 60 degrees
            self._current_reward += 0.1
        else: # Shallower
            self._current_reward -= 0.2

        # Combo reward
        if self.steps - self.last_hit_step < 5:
            self._current_reward += self.combo_bonus
            self.score += self.combo_bonus
            # sfx: combo_bonus

        self.last_hit_step = self.steps

        self.score += points
        self._current_reward += points
        self._create_particles(block_rect.center, color)
        self.blocks.pop(block_index)
        # sfx: block_break

        if not self.blocks:
            self.game_over = True
            self.game_won = True
            # sfx: win_game

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 2)

    def _render_game(self):
        # Blocks
        for block, color, _ in self.blocks:
            pygame.draw.rect(self.screen, color, block)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), block, 2)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.rect(s, (*p['color'], alpha), s.get_rect())
                self.screen.blit(s, (int(p['pos'].x - size), int(p['pos'].y - size)))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball (with glow)
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        for i in range(self.BALL_RADIUS, 0, -2):
            alpha = 80 - (i * 10)
            pygame.gfxdraw.filled_circle(self.screen, *ball_pos_int, i + 3, (*self.COLOR_BALL, alpha))
        pygame.gfxdraw.aacircle(self.screen, *ball_pos_int, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, *ball_pos_int, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        for i in range(self.balls_left -1):
            pos = (self.WIDTH - 30 - i * (self.BALL_RADIUS * 2 + 5), 10 + self.BALL_RADIUS)
            pygame.gfxdraw.aacircle(self.screen, *pos, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, *pos, self.BALL_RADIUS, self.COLOR_BALL)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _create_blocks(self):
        self.blocks = []
        rows, cols = 5, 10
        block_width = self.WIDTH // cols
        block_height = 20
        y_offset = 40
        
        for r in range(rows):
            for c in range(cols):
                points = 1 if r >= 3 else (2 if r >= 1 else 3)
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(
                    c * block_width,
                    y_offset + r * block_height,
                    block_width - 2,
                    block_height - 2
                )
                self.blocks.append((rect, color, points))

    def _spawn_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = pygame.Vector2(math.sin(angle), math.cos(angle)) * self.INITIAL_BALL_SPEED
    
    def _get_ball_rect(self):
        return pygame.Rect(
            self.ball_pos.x - self.BALL_RADIUS,
            self.ball_pos.y - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        import sys
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Block Breaker")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement = 0 # No-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            action = [movement, 0, 0] # Space/Shift not used
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Convert observation back to a surface for display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
                # Wait for a moment before auto-restarting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

            clock.tick(GameEnv.FPS)
            
        env.close()
        pygame.quit()
        sys.exit()

    except pygame.error as e:
        print("\nCould not create display. This is expected in a headless environment.")
        print("The environment is still valid for training.")