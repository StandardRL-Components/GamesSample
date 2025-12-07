import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
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
        "A retro arcade Breakout variant. Position your paddle to control the ball, destroy all the bricks, and aim for a high score by making risky edge-of-paddle hits."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 8
        self.INITIAL_LIVES = 3
        self.BRICK_ROWS = 5
        self.BRICK_COLS = 15
        self.TOTAL_BRICKS = self.BRICK_ROWS * self.BRICK_COLS
        self.BRICK_WIDTH = self.WIDTH // self.BRICK_COLS
        self.BRICK_HEIGHT = 15
        
        # Colors
        self.COLOR_BG = (10, 10, 40)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEART = (255, 50, 50)
        self.BRICK_COLORS = [
            (0, 255, 255), (255, 0, 255), (0, 255, 0),
            (255, 128, 0), (255, 255, 0)
        ]

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
        
        # Game state variables
        self.paddle = None
        self.ball = None
        self.bricks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        self.bricks_destroyed = 0
        self.bricks_destroyed_since_speed_increase = 0
        
        self.np_random = None
        
        # Initialize state
        # self.reset() is called by the wrapper/runner
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False
        self.bricks_destroyed = 0
        self.bricks_destroyed_since_speed_increase = 0

        self.paddle = self._Paddle(
            x=self.WIDTH / 2 - self.PADDLE_WIDTH / 2,
            y=self.HEIGHT - 40,
            width=self.PADDLE_WIDTH,
            height=self.PADDLE_HEIGHT,
            speed=self.PADDLE_SPEED,
            color=self.COLOR_PADDLE,
            bounds=(0, self.WIDTH)
        )
        
        self.ball = self._Ball(
            radius=self.BALL_RADIUS,
            color=self.COLOR_BALL
        )
        self.ball.reset(self.paddle)

        self._create_bricks()
        
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Handle input
        if not self.game_over:
            if movement == 3:  # Left
                self.paddle.move(-1)
            elif movement == 4:  # Right
                self.paddle.move(1)
            
            if space_held and self.ball.is_on_paddle:
                self.ball.launch(self.np_random)
                # Sound: launch

        # Update game logic
        self.paddle.update()
        self.ball.update(self.paddle)

        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)

        # --- Collision Detection & Game Logic ---
        if not self.ball.is_on_paddle and not self.game_over:
            # Ball vs Walls
            if self.ball.x - self.ball.radius < 0 or self.ball.x + self.ball.radius > self.WIDTH:
                self.ball.vx *= -1
                self.ball.x = max(self.ball.radius, min(self.ball.x, self.WIDTH - self.ball.radius))
                # Sound: wall_hit
            if self.ball.y - self.ball.radius < 0:
                self.ball.vy *= -1
                self.ball.y = self.ball.radius
                # Sound: wall_hit

            # Ball vs Paddle
            if self.ball.vy > 0 and self.paddle.rect.colliderect(self.ball.rect):
                hit_pos = (self.ball.x - self.paddle.rect.centerx) / (self.paddle.width / 2)
                hit_pos = max(-1, min(1, hit_pos))
                
                self.ball.vy *= -1
                self.ball.vx = hit_pos * (self.ball.base_speed * 1.5)
                self.ball.y = self.paddle.y - self.ball.radius
                # Sound: paddle_hit

                # Risk/reward for paddle hit position
                if abs(hit_pos) > 0.6:
                    reward += 0.1  # Risky edge hit
                else:
                    reward -= 0.02 # Safe center hit

            # Ball vs Bricks
            for brick in self.bricks[:]:
                if brick.rect.colliderect(self.ball.rect):
                    self._handle_brick_collision(brick)
                    reward += 1.0
                    self.score += 10
                    # Sound: brick_break
                    break # Only one brick per frame

            # Ball miss
            if self.ball.y - self.ball.radius > self.HEIGHT:
                self.lives -= 1
                # Sound: miss
                if self.lives > 0:
                    self.ball.reset(self.paddle)
                else:
                    self.game_over = True
                    reward -= 10.0 # Loss penalty

        self.steps += 1
        terminated = self._check_termination()
        
        if self.win and not self.game_over:
             reward += 100.0 # Win bonus
             self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            self.steps >= self.MAX_STEPS,
            self._get_info()
        )

    def _handle_brick_collision(self, brick):
        # Determine collision side to correctly reflect the ball
        prev_ball_rect = self.ball.rect.copy()
        prev_ball_rect.move_ip(-self.ball.vx, -self.ball.vy)
        
        if prev_ball_rect.bottom <= brick.rect.top or prev_ball_rect.top >= brick.rect.bottom:
             self.ball.vy *= -1
        if prev_ball_rect.right <= brick.rect.left or prev_ball_rect.left >= brick.rect.right:
             self.ball.vx *= -1

        # Anti-softlock: nudge if velocity is too horizontal/vertical
        if abs(self.ball.vx) < 0.2:
            self.ball.vx += self.np_random.choice([-0.5, 0.5])
        if abs(self.ball.vy) < 0.2:
            self.ball.vy += self.np_random.choice([-0.5, 0.5])

        self.bricks.remove(brick)
        self._create_particles(brick.rect.center, brick.color)
        self.bricks_destroyed += 1
        self.bricks_destroyed_since_speed_increase += 1

        # Difficulty scaling
        if self.bricks_destroyed_since_speed_increase >= 15:
            self.ball.base_speed += 0.5
            self.bricks_destroyed_since_speed_increase = 0
            
        if not self.bricks:
            self.win = True

    def _create_bricks(self):
        self.bricks.clear()
        y_offset = 40
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                brick_x = j * self.BRICK_WIDTH
                brick_y = y_offset + i * self.BRICK_HEIGHT
                color = self.BRICK_COLORS[i % len(self.BRICK_COLORS)]
                self.bricks.append(self._Brick(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT, color))

    def _create_particles(self, pos, color, count=10):
        for _ in range(count):
            self.particles.append(self._Particle(pos[0], pos[1], color, self.np_random))

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if not self.bricks: # Win condition
            self.win = True
            self.game_over = True
            return True
        if self.lives <= 0: # Loss condition
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks)
        }

    def _render_game(self):
        # Draw bricks
        for brick in self.bricks:
            brick.draw(self.screen)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
        
        # Draw paddle
        self.paddle.draw(self.screen)
        
        # Draw ball
        self.ball.draw(self.screen)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Lives
        for i in range(self.lives):
            self._draw_heart(self.WIDTH - 30 - (i * 25), 22, self.COLOR_HEART)

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(over_surf, over_rect)

    def _draw_heart(self, x, y, color):
        points = [
            (x, y - 5), (x + 5, y - 10), (x + 10, y - 5),
            (x, y + 5),
            (x - 10, y - 5), (x - 5, y - 10)
        ]
        pygame.draw.polygon(self.screen, color, points)

    def close(self):
        pygame.quit()

    class _Paddle:
        def __init__(self, x, y, width, height, speed, color, bounds):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.speed = speed
            self.color = color
            self.bounds = bounds
            self.dx = 0
            self.rect = pygame.Rect(x, y, width, height)
        
        def move(self, direction):
            self.dx = direction * self.speed

        def update(self):
            self.x += self.dx
            self.x = max(self.bounds[0], min(self.x, self.bounds[1] - self.width))
            self.rect.x = self.x
            self.dx = 0 # Reset movement intention each frame

        def draw(self, surface):
            pygame.draw.rect(surface, self.color, self.rect, border_radius=3)

    class _Ball:
        def __init__(self, radius, color):
            self.radius = radius
            self.color = color
            self.x, self.y = 0, 0
            self.vx, self.vy = 0, 0
            self.base_speed = 5.0 # Initial speed
            self.is_on_paddle = True
            self.rect = pygame.Rect(0, 0, radius*2, radius*2)

        def reset(self, paddle):
            self.is_on_paddle = True
            self.vx, self.vy = 0, 0
            self.base_speed = 5.0
            self.update(paddle)
        
        def launch(self, np_random):
            if self.is_on_paddle:
                self.is_on_paddle = False
                self.vy = -self.base_speed
                self.vx = np_random.uniform(-0.5, 0.5) * self.base_speed

        def update(self, paddle):
            if self.is_on_paddle:
                self.x = paddle.rect.centerx
                self.y = paddle.y - self.radius
            else:
                speed_magnitude = math.sqrt(self.vx**2 + self.vy**2)
                if speed_magnitude > 0:
                    scale = self.base_speed / speed_magnitude
                    self.vx *= scale
                    self.vy *= scale
                
                self.x += self.vx
                self.y += self.vy
            
            self.rect.center = (int(self.x), int(self.y))
        
        def draw(self, surface):
            pos = (int(self.x), int(self.y))
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], self.radius, self.color)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], self.radius, self.color)
            # Add a subtle glow
            glow_color = (*self.color, 50)
            temp_surf = pygame.Surface((self.radius * 4, self.radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (self.radius * 2, self.radius * 2), self.radius + 3)
            surface.blit(temp_surf, (pos[0] - self.radius * 2, pos[1] - self.radius * 2))


    class _Brick:
        def __init__(self, x, y, width, height, color):
            self.rect = pygame.Rect(x, y, width - 1, height - 1) # -1 for gap
            self.color = color

        def draw(self, surface):
            pygame.draw.rect(surface, self.color, self.rect)
            # Add a slight 3D effect
            highlight = tuple(min(255, c + 30) for c in self.color)
            shadow = tuple(max(0, c - 30) for c in self.color)
            pygame.draw.line(surface, highlight, self.rect.topleft, self.rect.topright)
            pygame.draw.line(surface, highlight, self.rect.topleft, self.rect.bottomleft)
            pygame.draw.line(surface, shadow, self.rect.bottomleft, self.rect.bottomright)
            pygame.draw.line(surface, shadow, self.rect.topright, self.rect.bottomright)


    class _Particle:
        def __init__(self, x, y, color, np_random):
            self.x = x
            self.y = y
            self.color = color
            angle = np_random.uniform(0, 2 * math.pi)
            speed = np_random.uniform(1, 4)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            self.lifespan = np_random.integers(15, 30) # frames
            self.radius = np_random.integers(2, 5)

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.lifespan -= 1
            self.vx *= 0.95 # friction
            self.vy *= 0.95

        def is_dead(self):
            return self.lifespan <= 0

        def draw(self, surface):
            if not self.is_dead():
                # FIX: The particle lifespan can be initialized to a value greater than 20,
                # which caused the original alpha calculation `255 * (lifespan / 20)` to
                # exceed 255, resulting in an "invalid color argument" error.
                # The corrected logic clamps the alpha ratio to [0, 1] before scaling.
                alpha_ratio = self.lifespan / 20.0
                alpha = int(max(0.0, min(1.0, alpha_ratio)) * 255)
                color_with_alpha = (*self.color, alpha)
                
                temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color_with_alpha, (self.radius, self.radius), self.radius)
                surface.blit(temp_surf, (int(self.x - self.radius), int(self.y - self.radius)))