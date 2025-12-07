
# Generated: 2025-08-28T03:59:53.169597
# Source Brief: brief_02175.md
# Brief Index: 2175

        
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
        "Controls: ← to move left, → to move right. Deflect the ball to break bricks and score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade brick-breaker. Control a paddle to deflect a ball, break all the bricks, and aim for a high score. Catch falling power-ups for an advantage!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (220, 220, 240)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (200, 200, 220)
    BRICK_COLORS = [
        (255, 50, 50), (255, 150, 50), (255, 255, 50),
        (50, 255, 50), (50, 150, 255), (150, 50, 255)
    ]
    COLOR_POWERUP = (50, 255, 255)

    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 5.0
    MAX_BALL_SPEED_X_FACTOR = 1.2
    
    BRICK_ROWS = 6
    BRICK_COLS = 12
    BRICK_WIDTH = 50
    BRICK_HEIGHT = 20
    BRICK_MARGIN_TOP = 50
    BRICK_MARGIN_X = (SCREEN_WIDTH - BRICK_COLS * BRICK_WIDTH) / 2
    
    MAX_STEPS = 3000
    WIN_SCORE = 5000
    INITIAL_LIVES = 3
    
    MULTIPLIER_TIMEOUT = 90 # frames (3 seconds at 30fps)
    POWERUP_DROP_CHANCE = 0.20
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)

        # Etc...
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball = None
        self.ball_vel = [0, 0]
        self.current_ball_speed = 0
        self.bricks = []
        self.particles = []
        self.powerups = []
        self.multiplier = 1
        self.last_brick_hit_time = 0
        self.score_milestone = 0
        self.step_reward = 0

        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.step_reward = 0
        self.multiplier = 1
        self.last_brick_hit_time = 0
        self.score_milestone = 500
        
        # Paddle
        paddle_y = self.SCREEN_HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2, paddle_y,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )

        # Ball
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
        )
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards, not too steep
        self.ball_vel = [math.cos(angle) * self.current_ball_speed, math.sin(angle) * self.current_ball_speed]

        # Bricks
        self._setup_bricks()

        # Clear lists
        self.particles.clear()
        self.powerups.clear()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _setup_bricks(self):
        self.bricks.clear()
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                brick_x = self.BRICK_MARGIN_X + j * self.BRICK_WIDTH
                brick_y = self.BRICK_MARGIN_TOP + i * self.BRICK_HEIGHT
                brick_rect = pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH - 2, self.BRICK_HEIGHT - 2)
                color_index = i % len(self.BRICK_COLORS)
                self.bricks.append({"rect": brick_rect, "color": self.BRICK_COLORS[color_index]})

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        self.step_reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            # space_held = action[1] == 1  # Boolean - not used
            # shift_held = action[2] == 1  # Boolean - not used
            
            # 1. Handle player input
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            # Clamp paddle to screen
            self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.paddle.width))

            # 2. Update game logic
            self._update_ball()
            self._update_powerups()
            self._update_multiplier()
            self._update_difficulty()
        
        self._update_particles()
        
        # 3. Update counters and check for termination
        self.steps += 1
        terminated = self._check_termination()
        
        reward = self.step_reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_ball(self):
        # Move ball
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
            self.ball.x = max(0, min(self.ball.x, self.SCREEN_WIDTH - self.ball.width)) # Clamp
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            # sfx: wall_bounce

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            
            offset = (self.ball.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel[0] = self.MAX_BALL_SPEED_X_FACTOR * self.current_ball_speed * offset
            self.ball_vel[1] *= -1
            
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel[0] = (self.ball_vel[0] / speed) * self.current_ball_speed
            self.ball_vel[1] = (self.ball_vel[1] / speed) * self.current_ball_speed
            
            self.multiplier = 1 # Reset multiplier on paddle hit
            # sfx: paddle_hit

        # Brick collisions
        hit_brick_idx = self.ball.collidelist([b['rect'] for b in self.bricks])
        if hit_brick_idx != -1:
            brick_data = self.bricks.pop(hit_brick_idx)
            brick = brick_data['rect']
            
            overlap_x = (self.ball.width / 2 + brick.width / 2) - abs(self.ball.centerx - brick.centerx)
            overlap_y = (self.ball.height / 2 + brick.height / 2) - abs(self.ball.centery - brick.centery)

            if overlap_x < overlap_y:
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1
            
            points = 10 * self.multiplier
            self.score += points
            self.step_reward += 0.1
            self.multiplier += 1
            self.last_brick_hit_time = self.steps
            
            self._create_particles(brick.center, brick_data['color'], 20)
            if self.np_random.random() < self.POWERUP_DROP_CHANCE:
                self._spawn_powerup(brick.center)
            # sfx: brick_break

        # Bottom wall (miss)
        if self.ball.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            self.multiplier = 1
            if self.lives > 0:
                self.ball.centerx = self.paddle.centerx
                self.ball.bottom = self.paddle.top - 5
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = [math.cos(angle) * self.current_ball_speed, math.sin(angle) * self.current_ball_speed]
                # sfx: life_lost
            else:
                self.game_over = True
                self.step_reward -= 100
                # sfx: game_over_lose

    def _update_powerups(self):
        for powerup in self.powerups[:]:
            powerup['rect'].y += 3
            if powerup['rect'].colliderect(self.paddle):
                self._apply_powerup(powerup['type'])
                self.powerups.remove(powerup)
                self.step_reward += 1.0
                # sfx: powerup_collect
            elif powerup['rect'].top > self.SCREEN_HEIGHT:
                self.powerups.remove(powerup)

    def _spawn_powerup(self, pos):
        powerup_rect = pygame.Rect(pos[0] - 10, pos[1] - 10, 20, 20)
        self.powerups.append({'rect': powerup_rect, 'type': 'wide_paddle', 'color': self.COLOR_POWERUP})

    def _apply_powerup(self, type):
        if type == 'wide_paddle':
            self.paddle.width = min(self.PADDLE_WIDTH * 1.5, self.SCREEN_WIDTH)
            self.paddle.left = max(0, self.paddle.left - (self.PADDLE_WIDTH * 0.25))

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_multiplier(self):
        if self.steps - self.last_brick_hit_time > self.MULTIPLIER_TIMEOUT and self.multiplier > 1:
            self.multiplier = 1

    def _update_difficulty(self):
        if self.score >= self.score_milestone:
            self.current_ball_speed += 0.5
            self.score_milestone += 500
            
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.current_ball_speed
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.current_ball_speed

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.step_reward += 100
            # sfx: game_win
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if not self.bricks:
            self.game_over = True
            self.step_reward += 50
            return True
        return False
        
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "multiplier": self.multiplier,
        }

    def _render_game(self):
        # Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick['color'], brick['rect'], border_radius=3)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p['color'], alpha), (size, size), size)
                self.screen.blit(s, (p['pos'][0] - size, p['pos'][1] - size), special_flags=pygame.BLEND_RGBA_ADD)

        # Powerups
        for powerup in self.powerups:
            if (self.steps // 5) % 2 == 0:
                pygame.draw.rect(self.screen, powerup['color'], powerup['rect'], border_radius=5)
                pygame.draw.rect(self.screen, (255,255,255), powerup['rect'], 2, border_radius=5)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Ball with glow
        ball_center = (int(self.ball.centerx), int(self.ball.centery))
        glow_radius = int(self.BALL_RADIUS * 2.5)
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_BALL, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (ball_center[0] - glow_radius, ball_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, 10))
        
        # Multiplier
        if self.multiplier > 1 and not self.game_over:
            alpha = max(0, 255 * (1 - (self.steps - self.last_brick_hit_time) / self.MULTIPLIER_TIMEOUT))
            if alpha > 0:
                mult_text = self.font_large.render(f"x{self.multiplier}", True, self.COLOR_BALL)
                mult_text.set_alpha(int(alpha))
                self.screen.blit(mult_text, (self.paddle.centerx - mult_text.get_width()/2, self.paddle.y - 60))

        # Game Over message
        if self.game_over:
            if self.score >= self.WIN_SCORE or not self.bricks:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
        
        # Test game-specific logic
        self.reset()
        initial_speed = self.current_ball_speed
        self.score = 499
        self._update_difficulty()
        assert abs(self.current_ball_speed - initial_speed) < 1e-6
        self.score = 500
        self._update_difficulty()
        assert self.current_ball_speed > initial_speed

        self.reset()
        initial_lives = self.lives
        self.ball.y = self.SCREEN_HEIGHT + 10 # Simulate a miss
        self._update_ball()
        assert self.lives == initial_lives - 1

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Brick Breaker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            action = [movement, 0, 0]

            obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()