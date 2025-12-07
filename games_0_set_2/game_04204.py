
# Generated: 2025-08-28T01:42:38.973020
# Source Brief: brief_04204.md
# Brief Index: 4204

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic brick breaker. Destroy all bricks to win. Don't lose your balls!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1500  # Increased for longer play
    
    # Colors (Neon Aesthetic)
    COLOR_BG_START = (10, 0, 30)
    COLOR_BG_END = (30, 0, 50)
    COLOR_PADDLE = (0, 255, 255)  # Cyan
    COLOR_BALL = (0, 255, 255)    # Cyan
    COLOR_WALL = (100, 100, 200, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (255, 255, 100)

    BRICK_COLORS = {
        "green": {"color": (0, 255, 100), "points": 1, "reward": 1},
        "blue": {"color": (0, 150, 255), "points": 2, "reward": 2},
        "red": {"color": (255, 50, 100), "points": 3, "reward": 3},
        "gold": {"color": (255, 223, 0), "points": 5, "reward": 5},
    }

    # Game Parameters
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    BALL_SPEED_MIN = 4
    BALL_SPEED_MAX = 8
    
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
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Initialize state variables
        self.paddle = None
        self.balls = None
        self.bricks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.balls_left = 0
        self.ball_on_paddle = False
        self.combo_hits = 0
        self.combo_timer = 0
        self.steps_since_last_hit = 0
        self.game_over_message = ""
        
        # Initialize state
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.balls_left = 3
        self.combo_hits = 0
        self.combo_timer = 0
        self.steps_since_last_hit = 0
        self.game_over_message = ""

        # Paddle
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect((self.WIDTH - self.PADDLE_WIDTH) // 2, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Bricks
        self._generate_bricks()

        # Ball
        self.balls = []
        self._create_new_ball(on_paddle=True)

        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        # --- Action Handling ---
        movement = action[0]
        space_pressed = action[1] == 1

        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.01 # Small penalty for movement
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.01 # Small penalty for movement

        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        if self.ball_on_paddle and space_pressed:
            # --- SFX: Ball Launch ---
            self.ball_on_paddle = False
            ball = self.balls[0]
            ball['vel'] = pygame.Vector2(self.np_random.uniform(-1, 1), -1).normalize() * self.BALL_SPEED_MIN

        # --- Game Logic Update ---
        self._update_balls()
        self._update_particles()

        # Combo timer
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo_hits = 0

        # --- Collision Handling & Reward Calculation ---
        brick_hit_this_step = False
        for ball in self.balls:
            if 'vel' not in ball: continue

            # Wall collisions
            if ball['pos'].x - self.BALL_RADIUS <= 0 or ball['pos'].x + self.BALL_RADIUS >= self.WIDTH:
                ball['vel'].x *= -1
                ball['pos'].x = np.clip(ball['pos'].x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # --- SFX: Wall Bounce ---
            if ball['pos'].y - self.BALL_RADIUS <= 0:
                ball['vel'].y *= -1
                ball['pos'].y = self.BALL_RADIUS
                # --- SFX: Wall Bounce ---

            # Paddle collision
            ball_rect = pygame.Rect(ball['pos'].x - self.BALL_RADIUS, ball['pos'].y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if self.paddle.colliderect(ball_rect) and ball['vel'].y > 0:
                # --- SFX: Paddle Hit ---
                offset = (ball['pos'].x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                angle = offset * (math.pi / 2.5) # Max angle ~72 degrees
                
                ball['vel'].x = math.sin(angle) * self.BALL_SPEED_MAX
                ball['vel'].y = -math.cos(angle) * self.BALL_SPEED_MAX
                
                # Ensure minimum vertical speed to prevent flat bounces
                ball['vel'].y = min(ball['vel'].y, -self.BALL_SPEED_MIN / 2)
                ball['pos'].y = self.paddle.top - self.BALL_RADIUS

                # Speed normalization
                speed = ball['vel'].length()
                if speed > self.BALL_SPEED_MAX:
                    ball['vel'] = ball['vel'].normalize() * self.BALL_SPEED_MAX
                if speed < self.BALL_SPEED_MIN:
                    ball['vel'] = ball['vel'].normalize() * self.BALL_SPEED_MIN

            # Brick collisions
            for brick in self.bricks[:]:
                if brick['rect'].colliderect(ball_rect):
                    # --- SFX: Brick Break ---
                    self.bricks.remove(brick)
                    brick_hit_this_step = True
                    
                    # Calculate reward
                    brick_info = self.BRICK_COLORS[brick['type']]
                    base_reward = brick_info['reward']
                    multiplier = 1.0 + self.combo_hits * 0.2
                    reward += base_reward * multiplier
                    
                    self.score += int(brick_info['points'] * multiplier)

                    # Bounce logic
                    ball['vel'].y *= -1
                    
                    # Add particles
                    self._create_particles(brick['rect'].center, brick_info['color'])
                    break # Only one brick per ball per frame

        # Update combo and anti-softlock counters
        if brick_hit_this_step:
            reward += 0.1 # General reward for hitting any brick
            self.combo_hits += 1
            self.combo_timer = 90 # 3 seconds at 30fps
            self.steps_since_last_hit = 0
        else:
            self.steps_since_last_hit += 1

        # Ball loss
        for ball in self.balls[:]:
            if ball['pos'].y > self.HEIGHT:
                # --- SFX: Ball Lost ---
                self.balls.remove(ball)
                reward -= 5.0 # Harsher penalty for losing a ball

        if not self.balls:
            self.balls_left -= 1
            if self.balls_left > 0:
                self._create_new_ball(on_paddle=True)
            else:
                terminated = True
                self.game_over_message = "GAME OVER"
                reward -= 100

        # --- Termination Checks ---
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over_message = "TIME UP"

        if not self.bricks:
            terminated = True
            self.game_over_message = "YOU WIN!"
            reward += 100

        # Anti-softlock
        if self.steps_since_last_hit > 200 and not self.ball_on_paddle and len(self.balls) == 1:
            self._create_new_ball(on_paddle=False) # Launch a bonus ball
            self.steps_since_last_hit = 0
            # --- SFX: Powerup ---

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_new_ball(self, on_paddle=True):
        self.ball_on_paddle = on_paddle
        ball = {'pos': pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)}
        if not on_paddle:
            ball['vel'] = pygame.Vector2(self.np_random.uniform(-1, 1), -1).normalize() * self.BALL_SPEED_MIN
            ball['pos'] = pygame.Vector2(self.WIDTH/2, self.HEIGHT/2)
        self.balls.append(ball)

    def _generate_bricks(self):
        self.bricks = []
        brick_types = list(self.BRICK_COLORS.keys())
        brick_w, brick_h = 40, 15
        gap = 5
        rows = self.np_random.integers(5, 9)
        cols = 12
        
        start_x = (self.WIDTH - (cols * (brick_w + gap))) // 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                # Make pattern more interesting
                if self.np_random.random() < 0.2:
                    continue
                
                brick_type = self.np_random.choice(brick_types, p=[0.4, 0.3, 0.2, 0.1])
                x = start_x + c * (brick_w + gap)
                y = start_y + r * (brick_h + gap)
                rect = pygame.Rect(x, y, brick_w, brick_h)
                self.bricks.append({'rect': rect, 'type': brick_type})

    def _update_balls(self):
        if self.ball_on_paddle:
            if self.balls:
                self.balls[0]['pos'].x = self.paddle.centerx
        else:
            for ball in self.balls:
                if 'vel' in ball:
                    ball['pos'] += ball['vel']

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Drag
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "bricks_left": len(self.bricks),
            "combo": self.combo_hits
        }

    def _render_background(self):
        # Optimized gradient background
        rect = pygame.Rect(0, 0, self.WIDTH, 1)
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp),
                int(self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp),
                int(self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp)
            )
            rect.top = y
            pygame.draw.rect(self.screen, color, rect)

    def _render_game_elements(self):
        # Bricks with glow
        for brick in self.bricks:
            r = brick['rect']
            color = self.BRICK_COLORS[brick['type']]['color']
            glow_color = (*color, 30)
            pygame.draw.rect(self.screen, glow_color, r.inflate(6, 6))
            pygame.draw.rect(self.screen, color, r)
            pygame.draw.rect(self.screen, (255,255,255), r, 1) # White outline

        # Paddle with glow
        glow_color = (*self.COLOR_PADDLE, 40)
        pygame.draw.rect(self.screen, glow_color, self.paddle.inflate(8, 8), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Balls with glow
        for ball in self.balls:
            self._draw_glow_circle(self.screen, self.COLOR_BALL, ball['pos'], self.BALL_RADIUS, 15)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (*p['color'], alpha)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 2, color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Balls
        ball_text = self.font_large.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        text_rect = ball_text.get_rect(topright=(self.WIDTH - 10, 5))
        self.screen.blit(ball_text, text_rect)

        # Combo
        if self.combo_hits > 1:
            combo_text = self.font_large.render(f"x{1.0 + self.combo_hits * 0.2:.1f}", True, self.BRICK_COLORS['gold']['color'])
            text_rect = combo_text.get_rect(midtop=(self.WIDTH // 2, 5))
            self.screen.blit(combo_text, text_rect)
        
        # Game Over Message
        if self.game_over_message:
            msg_surf = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _draw_glow_circle(self, surface, color, center, radius, glow_strength):
        center_x, center_y = int(center.x), int(center.y)
        for i in range(glow_strength, 0, -2):
            alpha = 150 * (1 - (i / glow_strength))**2
            glow_color = (*color, int(alpha))
            pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius + i, glow_color)
        
        pygame.gfxdraw.aacircle(surface, center_x, center_y, radius, color)
        pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius, color)

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

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to run headlessly if you want, but for testing, a window is fine.
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # For interactive testing with a display
    pygame.display.set_caption("Brick Breaker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0

    while not terminated:
        # Map keyboard keys to actions for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # unused

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Control the frame rate for human play

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()