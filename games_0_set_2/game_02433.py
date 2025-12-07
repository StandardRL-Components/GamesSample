
# Generated: 2025-08-28T04:51:21.641948
# Source Brief: brief_02433.md
# Brief Index: 2433

        
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
        "Controls: Use ↑ and ↓ to move the paddle vertically. Try to break all the bricks!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A vibrant, side-view take on the classic brick-breaker. "
        "Control your paddle to bounce the ball and shatter the wall of bricks for points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.PADDLE_WIDTH = 15
        self.PADDLE_HEIGHT = 80
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_MIN_SPEED_X = 4
        self.BALL_MAX_SPEED_X = 8
        self.BALL_MAX_SPEED_Y = 8
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 1000
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG_TOP = (10, 0, 30)
        self.COLOR_BG_BOTTOM = (40, 0, 70)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 60)
        self.BRICK_COLORS = [
            (255, 50, 50), (255, 150, 50), (50, 255, 50), 
            (50, 150, 255), (150, 50, 255), (255, 50, 150)
        ]
        self.COLOR_TEXT = (255, 255, 255)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('Consolas', 32, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 24)
        
        # Etc...
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = None
        self.particles = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.paddle = pygame.Rect(
            self.PADDLE_WIDTH * 2, 
            (self.SCREEN_HEIGHT - self.PADDLE_HEIGHT) // 2, 
            self.PADDLE_WIDTH, 
            self.PADDLE_HEIGHT
        )
        
        self._reset_ball()

        self.bricks = []
        brick_rows = 8
        brick_cols = 10
        brick_width = 30
        brick_height = 15
        brick_padding = 5
        start_x = self.SCREEN_WIDTH - (brick_cols * (brick_width + brick_padding)) - 20
        start_y = 50
        for i in range(brick_rows):
            for j in range(brick_cols):
                color_index = self.np_random.integers(0, len(self.BRICK_COLORS))
                color = self.BRICK_COLORS[color_index]
                brick_rect = pygame.Rect(
                    start_x + j * (brick_width + brick_padding),
                    start_y + i * (brick_height + brick_padding),
                    brick_width,
                    brick_height
                )
                self.bricks.append({'rect': brick_rect, 'color': color})

        self.particles = []
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.steps = 0
        self.steps_since_brick_hit = 0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle.centerx + 20, self.paddle.centery)
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        speed = self.np_random.uniform(self.BALL_MIN_SPEED_X, self.BALL_MIN_SPEED_X + 2)
        self.ball_vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Boolean (unused)
        # shift_held = action[2] == 1  # Boolean (unused)
        
        # Update game logic
        if movement == 1:  # Up
            self.paddle.y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle.y += self.PADDLE_SPEED
        
        self.paddle.y = max(0, min(self.paddle.y, self.SCREEN_HEIGHT - self.PADDLE_HEIGHT))

        bricks_hit = self._update_game_state()
        
        reward = self._calculate_reward(bricks_hit)
        terminated = self._check_termination(reward)
        if terminated:
            if self.lives <= 0 or self.steps >= self.MAX_STEPS:
                reward -= 100.0
            elif self.score >= self.WIN_SCORE:
                reward += 100.0
        
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_game_state(self):
        self.ball_pos += self.ball_vel

        bricks_hit_this_step = 0
        
        if self.ball_pos.y - self.BALL_RADIUS <= 0 or self.ball_pos.y + self.BALL_RADIUS >= self.SCREEN_HEIGHT:
            self.ball_vel.y *= -1 # sfx: bounce_wall.wav
        if self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1 # sfx: bounce_wall.wav

        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.x < 0:
            self.ball_vel.x *= -1
            offset = (self.ball_pos.y - self.paddle.centery) / (self.PADDLE_HEIGHT / 2)
            self.ball_vel.y = offset * self.BALL_MAX_SPEED_Y
            self.ball_vel.x = max(self.BALL_MIN_SPEED_X, abs(self.ball_vel.x))
            self.ball_vel.x = min(self.BALL_MAX_SPEED_X, self.ball_vel.x)
            # sfx: bounce_paddle.wav

        if self.ball_pos.x - self.BALL_RADIUS < 0:
            self.lives -= 1 # sfx: life_lost.wav
            if self.lives > 0:
                self._reset_ball()

        hit_brick_indices = []
        for i, brick_data in enumerate(self.bricks):
            if brick_data['rect'].colliderect(ball_rect):
                hit_brick_indices.append(i)
                dx = self.ball_pos.x - brick_data['rect'].centerx
                dy = self.ball_pos.y - brick_data['rect'].centery
                w = (self.BALL_RADIUS + brick_data['rect'].width) / 2
                h = (self.BALL_RADIUS + brick_data['rect'].height) / 2
                if abs(dx) / w > abs(dy) / h: self.ball_vel.x *= -1
                else: self.ball_vel.y *= -1
                self._create_particles(brick_data['rect'].center, brick_data['color'])
                # sfx: brick_break.wav
                break 

        if hit_brick_indices:
            bricks_hit_this_step = len(hit_brick_indices)
            self.score += bricks_hit_this_step
            for i in sorted(hit_brick_indices, reverse=True): del self.bricks[i]
        
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0: self.particles.remove(p)

        return bricks_hit_this_step

    def _calculate_reward(self, bricks_hit):
        reward = 0.0
        self.steps_since_brick_hit += 1
        if bricks_hit > 0:
            reward += 1.0 * bricks_hit  # Event reward
            reward += 0.1 # Continuous feedback
            self.steps_since_brick_hit = 0
        elif self.steps_since_brick_hit > 5:
            reward -= 0.02
        return reward
    
    def _check_termination(self, reward):
        if self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(
                self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp for i in range(3)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick['color'], brick['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in brick['color']), brick['rect'], 1)

        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 30))
            size = int(self.BALL_RADIUS / 2 * (p['lifespan'] / 30))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p['color'], alpha), (size, size), size)
                self.screen.blit(s, (int(p['pos'].x - size), int(p['pos'].y - size)))

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        
        if self.lives > 0:
            ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
            s = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_BALL_GLOW, (self.BALL_RADIUS * 2, self.BALL_RADIUS * 2), self.BALL_RADIUS * 1.5)
            self.screen.blit(s, (ball_pos_int[0] - self.BALL_RADIUS * 2, ball_pos_int[1] - self.BALL_RADIUS * 2))
            pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        life_radius = 8
        life_padding = 5
        for i in range(self.lives):
            pos_x = self.SCREEN_WIDTH - (i * (life_radius * 2 + life_padding)) - 20
            pygame.gfxdraw.aacircle(self.screen, pos_x, 20, life_radius, self.COLOR_PADDLE)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 20, life_radius, self.COLOR_PADDLE)
        
        if self.game_over:
            message = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install gymnasium[classic-control]
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    key_to_action = { pygame.K_UP: 1, pygame.K_DOWN: 2, }
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Brick Breaker")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op for movement
    
    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action[0] = key_to_action[event.key]
                if event.key == pygame.K_r and terminated:
                    obs, info = env.reset()
                    terminated = False
            if event.type == pygame.KEYUP:
                if event.key in key_to_action and action[0] == key_to_action[event.key]:
                    action[0] = 0 # No movement

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()