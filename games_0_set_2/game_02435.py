
# Generated: 2025-08-28T04:47:54.696845
# Source Brief: brief_02435.md
# Brief Index: 2435

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import random
import os
import os
import pygame


# Set a dummy video driver to run pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Break all the bricks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-styled brick breaker game. Control a paddle to bounce a ball and destroy bricks for points. Don't let the ball hit the bottom!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WALL_THICKNESS = 10

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_WALL = (180, 180, 180)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.BRICK_COLORS = {
            1: (200, 70, 70),   # Red
            2: (70, 200, 70),   # Green
            3: (70, 70, 200),   # Blue
        }
        self.COLOR_TEXT = (220, 220, 220)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts - use a common monospace font
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_game_over = pygame.font.SysFont("Consolas", 64)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 30)
            self.font_game_over = pygame.font.SysFont(None, 80)

        # Game parameters
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.INITIAL_BALL_SPEED = 5.0
        self.MAX_STEPS = 10000
        self.WIN_SCORE = 500

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = None
        self.bricks = None
        self.particles = None
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_score_milestone = 0
        self.np_random = None

        # Call reset to set up the initial state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            random.seed(seed)
        else:
            self.np_random = np.random.default_rng()


        # Reset game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.win = False
        self.last_score_milestone = 0

        # Player paddle
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Ball
        self._reset_ball()

        # Bricks
        self._create_bricks()

        # Particles
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        self.ball_speed = self.INITIAL_BALL_SPEED
        angle = random.uniform(math.pi * 1.25, math.pi * 1.75)  # Upwards angle
        self.ball_vel = [self.ball_speed * math.cos(angle), self.ball_speed * math.sin(angle)]

    def _create_bricks(self):
        self.bricks = []
        brick_rows = 5
        brick_cols = 10
        brick_width = (self.WIDTH - self.WALL_THICKNESS * 2) / brick_cols
        brick_height = 20
        y_offset = 50

        for r in range(brick_rows):
            for c in range(brick_cols):
                points = (brick_rows - r) // 2 + 1  # More points for higher rows
                color = self.BRICK_COLORS[points]
                brick_rect = pygame.Rect(
                    self.WALL_THICKNESS + c * brick_width,
                    y_offset + r * brick_height,
                    brick_width - 2,  # Gaps
                    brick_height - 2,
                )
                self.bricks.append({"rect": brick_rect, "color": color, "points": points})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small time penalty to encourage speed

        # --- Action Handling ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(self.WALL_THICKNESS, min(self.paddle.x, self.WIDTH - self.WALL_THICKNESS - self.PADDLE_WIDTH))

        # --- Game Logic ---
        event_reward, life_lost = self._update_ball()
        reward += event_reward
        self._update_particles()
        
        # Difficulty scaling
        if self.score // 50 > self.last_score_milestone:
            self.last_score_milestone = self.score // 50
            new_speed = self.INITIAL_BALL_SPEED + (self.last_score_milestone * 0.2)
            current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if current_speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / current_speed) * new_speed
                self.ball_vel[1] = (self.ball_vel[1] / current_speed) * new_speed
                self.ball_speed = new_speed

        # --- Termination Check ---
        terminated = False
        if self.lives <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward = -100.0
        elif self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            terminated = True
            reward = 100.0
        elif not self.bricks: # All bricks cleared
            self.game_over = True
            self.win = True
            terminated = True
            reward = 100.0
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        # Ensure clock ticks for auto-advance
        if self.auto_advance:
            self.clock.tick(30)
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        reward = 0
        life_lost = False

        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= self.WALL_THICKNESS:
            ball_rect.left = self.WALL_THICKNESS
            self.ball_vel[0] *= -1
        if ball_rect.right >= self.WIDTH - self.WALL_THICKNESS:
            ball_rect.right = self.WIDTH - self.WALL_THICKNESS
            self.ball_vel[0] *= -1
        if ball_rect.top <= self.WALL_THICKNESS:
            ball_rect.top = self.WALL_THICKNESS
            self.ball_vel[1] *= -1
        
        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound effect placeholder: # pygame.mixer.Sound('paddle_hit.wav').play()
            ball_rect.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.0
            
            # Normalize to maintain speed
            current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if current_speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / current_speed) * self.ball_speed
                self.ball_vel[1] = (self.ball_vel[1] / current_speed) * self.ball_speed

        # Brick collisions
        collided_brick_index = ball_rect.collidelist([b['rect'] for b in self.bricks])
        if collided_brick_index != -1:
            brick = self.bricks[collided_brick_index]
            # Sound effect placeholder: # pygame.mixer.Sound('brick_break.wav').play()
            
            # Determine collision side to correctly reflect
            # A simple approximation: check overlap
            dx = ball_rect.centerx - brick['rect'].centerx
            dy = ball_rect.centery - brick['rect'].centery
            w = (ball_rect.width + brick['rect'].width) / 2
            h = (ball_rect.height + brick['rect'].height) / 2
            wy = w * dy
            hx = h * dx
            
            if wy > hx:
                if wy > -hx: # Top
                    self.ball_vel[1] *= -1
                    ball_rect.bottom = brick['rect'].top
                else: # Left
                    self.ball_vel[0] *= -1
                    ball_rect.right = brick['rect'].left
            else:
                if wy > -hx: # Right
                    self.ball_vel[0] *= -1
                    ball_rect.left = brick['rect'].right
                else: # Bottom
                    self.ball_vel[1] *= -1
                    ball_rect.top = brick['rect'].bottom

            # Rewards and effects
            reward += brick['points'] + 0.1
            self.score += brick['points']
            self._create_particles(brick['rect'].center, brick['color'])
            self.bricks.pop(collided_brick_index)
        
        # Bottom wall collision (lose life)
        if ball_rect.top > self.HEIGHT:
            # Sound effect placeholder: # pygame.mixer.Sound('lose_life.wav').play()
            self.lives -= 1
            life_lost = True
            reward = -10.0
            if self.lives > 0:
                self._reset_ball()

        self.ball_pos = [ball_rect.centerx, ball_rect.centery]
        return reward, life_lost

    def _create_particles(self, pos, color):
        for _ in range(15):
            particle_vel = [random.uniform(-2, 2), random.uniform(-2, 2)]
            particle_pos = list(pos)
            lifespan = random.randint(10, 20)
            self.particles.append({"pos": particle_pos, "vel": particle_vel, "lifespan": lifespan, "color": color})
            
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))
        
        # Render bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick['color'], brick['rect'])

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Render ball
        pygame.draw.circle(self.screen, self.COLOR_BALL, (int(self.ball_pos[0]), int(self.ball_pos[1])), self.BALL_RADIUS)

        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 20.0))
            size = max(1, int(3 * (p['lifespan'] / 20.0)))
            p_color = (p['color'][0], p['color'][1], p['color'][2])
            
            # Using a surface for alpha blending
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            s.fill((p_color[0], p_color[1], p_color[2], alpha))
            self.screen.blit(s, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 10))
        
        # Lives
        lives_text = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - self.WALL_THICKNESS - 10, self.WALL_THICKNESS + 10))
        
        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks),
        }

    def close(self):
        pygame.font.quit()
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
    # This block allows you to play the game manually for testing
    # Requires pygame to be installed with display support
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Brick Breaker")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()
            terminated = False
            total_reward = 0
            
    env.close()