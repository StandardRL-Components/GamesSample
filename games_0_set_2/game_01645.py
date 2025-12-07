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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Clear all bricks on the screen by deflecting the ball with your paddle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    WIDTH, HEIGHT = 640, 400
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (20, 40, 80)
    COLOR_PADDLE = (220, 220, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 150, 60)
    COLOR_UI_TEXT = (200, 200, 220)
    BRICK_COLORS = [
        (214, 81, 74), (228, 149, 74), (228, 220, 74),
        (130, 214, 74), (74, 185, 214), (133, 74, 214)
    ]
    # Game parameters
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 5.0
    MAX_EPISODE_STEPS = 1000
    INITIAL_BALLS = 3
    TOTAL_STAGES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Set headless mode for pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Create a static background gradient surface for performance
        self.background = self._create_gradient_background()

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.balls_left = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.ball_attached = True
        self.bricks = []
        self.particles = []
        self.last_space_held = False
        self.np_random = None

        # Initialize state
        # self.reset() # reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Create a default generator if one doesn't exist
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.balls_left = self.INITIAL_BALLS
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.last_space_held = False

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self._reset_ball()
        self._generate_bricks(self.stage)
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage efficiency
        terminated = False

        # --- Action Handling ---
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean

        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if space_held and self.ball_attached:
            self.ball_attached = False
            # Launch with a slight random horizontal component
            self.ball_vel = [self.np_random.uniform(-0.5, 0.5), -1]
            # Normalize velocity
            norm = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel = [
                self.ball_vel[0] / norm * self.ball_speed,
                self.ball_vel[1] / norm * self.ball_speed
            ]

        self.last_space_held = space_held

        # --- Game Logic Update ---
        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            
            # --- Collision Detection ---
            # Wall collisions
            if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            if self.ball_pos[1] <= self.BALL_RADIUS:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT)

            # Ball lost
            if self.ball_pos[1] >= self.HEIGHT:
                self.balls_left -= 1
                reward = -10.0
                if self.balls_left > 0:
                    self._reset_ball()
                else:
                    self.game_over = True
                    terminated = True

            # Paddle collision
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
                self.ball_vel[1] *= -1
                # Influence horizontal direction based on hit location
                offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = self.ball_speed * offset * 1.2
                # Clamp y position to prevent tunneling
                self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
                reward += 0.1 # Reward for keeping ball in play

            # Brick collisions
            hit_brick_index = -1
            for i, brick_info in enumerate(self.bricks):
                if brick_info['rect'].colliderect(ball_rect):
                    hit_brick_index = i
                    break
            
            if hit_brick_index != -1:
                brick_info = self.bricks.pop(hit_brick_index)
                brick_rect = brick_info['rect']
                self.score += 10
                reward += 1.0
                
                # Create particles
                self._create_particles(brick_rect.center, self.BRICK_COLORS[brick_info['color_index'] % len(self.BRICK_COLORS)])
                
                # Determine bounce direction
                cy = self.ball_pos[1]
                if cy < brick_rect.top or cy > brick_rect.bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1
                
                # Add a tiny random perturbation to prevent loops
                self.ball_vel[0] += self.np_random.uniform(-0.1, 0.1)
        
        # Anti-softlock: ensure ball has minimum vertical speed
        if not self.ball_attached and abs(self.ball_vel[1]) < 0.2 * self.ball_speed:
            self.ball_vel[1] = 0.2 * self.ball_speed * np.sign(self.ball_vel[1] if self.ball_vel[1] != 0 else 1)

        # Update particles
        self._update_particles()
        
        # Check for stage clear
        if not self.bricks and not self.game_over:
            reward += 5.0
            self.stage += 1
            if self.stage > self.TOTAL_STAGES:
                self.game_over = True
                terminated = True
                reward += 100.0 # Big reward for winning
            else:
                self.ball_speed += 0.2
                self._reset_ball()
                self._generate_bricks(self.stage)

        self.steps += 1
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            self.game_over = True
        
        if self.game_over:
            terminated = True
            if self.balls_left == 0 and self.stage <= self.TOTAL_STAGES:
                 reward = -10.0 # Ensure final step reward for loss is consistent

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _generate_bricks(self, stage):
        self.bricks = []
        brick_width, brick_height = 50, 20
        gap = 4
        rows, cols = 0, 0
        layout = []

        if stage == 1:
            rows, cols = 5, 11
            layout = [[1]*cols for _ in range(rows)]
        elif stage == 2:
            rows, cols = 6, 11
            layout = [[(c % 2) for c in range(cols)] for r in range(rows)]
        elif stage == 3:
            rows, cols = 7, 11
            layout = [[1 if (r+c) % 3 != 0 else 0 for c in range(cols)] for r in range(rows)]
        
        total_brick_width = cols * (brick_width + gap) - gap
        start_x = (self.WIDTH - total_brick_width) // 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                if layout[r][c] == 1:
                    brick_rect = pygame.Rect(
                        start_x + c * (brick_width + gap),
                        start_y + r * (brick_height + gap),
                        brick_width, brick_height
                    )
                    color_index = (r + c) % len(self.BRICK_COLORS)
                    self.bricks.append({'rect': brick_rect, 'color_index': color_index})
    
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        # Blit the pre-rendered background
        self.screen.blit(self.background, (0, 0))

        # Render game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw bricks
        for brick_info in self.bricks:
            color = self.BRICK_COLORS[brick_info['color_index'] % len(self.BRICK_COLORS)]
            pygame.draw.rect(self.screen, color, brick_info['rect'])
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / 30))
            size = max(1, int(3 * (p['lifetime'] / 30)))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (size, size), size)
            self.screen.blit(s, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw ball with glow
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        self.screen.blit(glow_surf, (ball_center[0] - self.BALL_RADIUS * 2, ball_center[1] - self.BALL_RADIUS * 2))
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
    
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_large.render(f"STAGE {self.stage}", True, self.COLOR_UI_TEXT)
        stage_rect = stage_text.get_rect(center=(self.WIDTH // 2, 20))
        self.screen.blit(stage_text, stage_rect)

        # Balls left
        ball_icon = pygame.Surface((self.BALL_RADIUS*2, self.BALL_RADIUS*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(ball_icon, self.BALL_RADIUS, self.BALL_RADIUS, self.BALL_RADIUS, self.COLOR_BALL)
        for i in range(self.balls_left):
            self.screen.blit(ball_icon, (self.WIDTH - 30 - i * 20, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
        }
        
    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            # Linear interpolation between top and bottom colors
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.HEIGHT
            pygame.draw.line(bg, (int(r), int(g), int(b)), (0, y), (self.WIDTH, y))
        return bg

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run with a display, comment out the `os.environ` line in `__init__`
    # and re-enable the display creation here.
    
    # For headed mode, pygame display must be initialized before GameEnv
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Block Breaker")

    # Now, create the environment. It will use the existing display.
    env = GameEnv(render_mode="rgb_array")
    # We need to unset the dummy driver if we want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    obs, info = env.reset()

    clock = pygame.time.Clock()
    running = True

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used in this game

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already the rendered frame
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose it back for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Game Loop ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset() # Automatically reset for a new game

        clock.tick(60)

    env.close()