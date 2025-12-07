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
        "Bounce a ball off a paddle to break bricks and achieve the highest score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 4.0
    MAX_STEPS = 10000

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (0, 150, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    BRICK_COLORS = [
        (255, 87, 34),   # Deep Orange
        (255, 193, 7),   # Amber
        (76, 175, 80),   # Green
        (33, 150, 243),  # Blue
        (156, 39, 176),  # Purple
    ]

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

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.ball_launched = False
        self.current_ball_speed = 0
        self.bricks_broken_count = 0
        self.consecutive_breaks = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_pos = np.array(
            [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=float
        )
        self.ball_vel = np.array([0.0, 0.0], dtype=float)

        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.ball_launched = False
        self.bricks = self._create_bricks()
        self.particles = []
        
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        self.bricks_broken_count = 0
        self.consecutive_breaks = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time encourages faster completion
        
        if self.game_over:
            # On subsequent steps after termination, return a dummy response
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        if movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if space_held and not self.ball_launched:
            self.ball_launched = True
            # FIX: The low argument must be less than the high argument for np.random.uniform.
            # -2*pi/3 is smaller than -pi/3.
            angle = self.np_random.uniform(-2 * math.pi / 3, -math.pi / 3)  # Upwards
            self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.current_ball_speed
            # sfx: launch_ball.wav

        # --- Update Game Logic ---
        self.steps += 1
        
        if self.ball_launched:
            reward += 0.1 # Reward for keeping ball in play
            self.ball_pos += self.ball_vel
            
            # Update ball speed based on bricks broken
            self.current_ball_speed = self.INITIAL_BALL_SPEED + (self.bricks_broken_count // 10) * 0.5

            # Ball-wall collision
            if self.ball_pos[0] - self.BALL_RADIUS <= 0 or self.ball_pos[0] + self.BALL_RADIUS >= self.WIDTH:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # sfx: wall_bounce.wav
            if self.ball_pos[1] - self.BALL_RADIUS <= 0:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
                # sfx: wall_bounce.wav

            # Ball-paddle collision
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
                self.ball_vel[1] *= -1
                offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = offset * self.current_ball_speed * 0.8 # Add horizontal influence
                
                # Normalize velocity to maintain constant speed
                norm = np.linalg.norm(self.ball_vel)
                if norm > 0:
                    self.ball_vel = (self.ball_vel / norm) * self.current_ball_speed
                
                self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
                self.consecutive_breaks = 0 # Reset combo on paddle hit
                # sfx: paddle_hit.wav

            # Ball-brick collision
            hit_brick = None
            for brick in self.bricks:
                if brick['rect'].colliderect(ball_rect):
                    hit_brick = brick
                    break
            
            if hit_brick:
                self.bricks.remove(hit_brick)
                self.score += hit_brick['points']
                reward += 1.0 + self.consecutive_breaks * 0.5 # Brick break + combo bonus
                self.consecutive_breaks += 1
                self.bricks_broken_count += 1
                self._create_particles(hit_brick['rect'].center, hit_brick['color'])
                # sfx: brick_break.wav

                # Simple bounce logic
                self.ball_vel[1] *= -1

            # Ball out of bounds (lose life)
            if self.ball_pos[1] - self.BALL_RADIUS > self.HEIGHT:
                self.lives -= 1
                self.ball_launched = False
                self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=float)
                self.ball_vel = np.array([0.0, 0.0], dtype=float)
                self.consecutive_breaks = 0
                reward -= 5 # Penalty for losing a life
                # sfx: lose_life.wav
        else:
            # Ball follows paddle
            self.ball_pos[0] = self.paddle.centerx

        # Update particles
        self._update_particles()

        # Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.lives <= 0:
                reward = -100 # Lose
            elif not self.bricks:
                reward = 100 # Win
        
        # Truncation is not used in this environment
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated or truncated,
            truncated,
            self._get_info()
        )

    def _create_bricks(self):
        bricks = []
        brick_rows = 5
        brick_cols = 10
        brick_width = (self.WIDTH - (brick_cols + 1) * 4) / brick_cols
        brick_height = 20
        
        for i in range(brick_rows):
            for j in range(brick_cols):
                brick_x = j * (brick_width + 4) + 4
                brick_y = i * (brick_height + 4) + 40
                rect = pygame.Rect(brick_x, brick_y, brick_width, brick_height)
                color = self.BRICK_COLORS[i % len(self.BRICK_COLORS)]
                points = (brick_rows - i) * 10
                bricks.append({'rect': rect, 'color': color, 'points': points})
        return bricks
    
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.lives <= 0 or not self.bricks:
            self.game_over = True
            return True
        return False

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
        # Draw bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick['color'], brick['rect'])

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        
        # Draw ball with a glow effect
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_color = (100, 100, 100)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 2, glow_color)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 2, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = p['color'] + (alpha,)
            size = int(p['life'] / 4)
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (p['pos'][0] - size, p['pos'][1] - size))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Game Over message
        if self.game_over:
            win = not self.bricks and self.lives > 0
            message = "YOU WIN!" if win else "GAME OVER"
            color = (0, 255, 100) if win else (255, 50, 50)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            game_over_text = self.font_game_over.render(message, True, color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks)
        }
        
    def close(self):
        pygame.quit()
        
if __name__ == "__main__":
    # To run the game with manual controls
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    # This part is not needed for the headless environment but is useful for testing
    pygame.display.init()
    pygame.display.set_caption("Brick Breaker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Transpose observation back to pygame's (width, height, channels) format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds before auto-reset
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()