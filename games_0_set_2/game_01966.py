
# Generated: 2025-08-27T18:50:00.934654
# Source Brief: brief_01966.md
# Brief Index: 1966

        
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
        "Controls: Use ← and → to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade game where the player controls a paddle to bounce a ball and destroy all the bricks. Build combos for a high score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (100, 200, 255)
        self.COLOR_BALL_GLOW = (50, 100, 155)
        self.COLOR_TEXT = (220, 220, 220)
        self.BRICK_COLORS = [
            (217, 87, 99), (217, 144, 87), (195, 217, 87),
            (87, 217, 134), (87, 166, 217)
        ]
        self.FONT_UI = pygame.font.SysFont("Consolas", 24, bold=True)
        self.FONT_COMBO = pygame.font.SysFont("Impact", 36)
        self.FONT_MSG = pygame.font.SysFont("Impact", 60)

        # --- Game Constants ---
        self.MAX_STEPS = 10000
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 6.0
        self.MAX_BALL_SPEED = 12.0
        self.BALL_SPIN_FACTOR = 2.5
        self.INITIAL_LIVES = 3
        self.COMBO_TIMEOUT = 0.5 * 30  # 0.5 seconds at 30fps

        self.BRICK_ROWS = 5
        self.BRICK_COLS = 15
        self.BRICK_WIDTH = 40
        self.BRICK_HEIGHT = 18
        self.BRICK_GAP = 2
        self.BRICK_OFFSET_X = (self.screen_width - (self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_GAP))) / 2
        self.BRICK_OFFSET_Y = 50

        # --- State Variables ---
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = None
        self.ball_on_paddle = None
        self.bricks = None
        self.total_bricks = 0
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.combo = 0
        self.last_brick_hit_step = 0
        self.particles = []
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle_rect = pygame.Rect(
            (self.screen_width - self.PADDLE_WIDTH) / 2,
            self.screen_height - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball_on_paddle = True
        self._reset_ball()

        self.bricks = []
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                brick_x = self.BRICK_OFFSET_X + j * (self.BRICK_WIDTH + self.BRICK_GAP)
                brick_y = self.BRICK_OFFSET_Y + i * (self.BRICK_HEIGHT + self.BRICK_GAP)
                brick_rect = pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                self.bricks.append({"rect": brick_rect, "color": self.BRICK_COLORS[i % len(self.BRICK_COLORS)]})
        self.total_bricks = len(self.bricks)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.combo = 0
        self.last_brick_hit_step = -self.COMBO_TIMEOUT
        self.particles = []

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.ball_on_paddle = True

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        
        # --- Game Logic ---
        self.steps += 1
        reward = -0.001 # Small penalty for existing

        # Paddle Movement
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        self.paddle_rect.x = np.clip(self.paddle_rect.x, 0, self.screen_width - self.PADDLE_WIDTH)

        # Ball Launch
        if self.ball_on_paddle:
            self.ball_pos[0] = self.paddle_rect.centerx
            if movement in [3, 4]: # Launch on first move
                self.ball_on_paddle = False
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                self.ball_vel = [math.cos(angle) * self.ball_speed, math.sin(angle) * self.ball_speed]
                # sfx: ball_launch
        else:
            # Ball Movement
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]

        # Ball Collision
        reward += self._handle_ball_collisions()
        
        # Update particles
        self._update_particles()

        # --- Termination Check ---
        win = len(self.bricks) == 0
        lose = self.lives <= 0
        timeout = self.steps >= self.MAX_STEPS
        terminated = win or lose or timeout

        if win:
            reward += 100
            self.game_over_message = "YOU WIN!"
        if lose:
            reward -= 100
            self.game_over_message = "GAME OVER"
        if timeout and not (win or lose):
            self.game_over_message = "TIME UP"
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_ball_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.screen_width - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.screen_width - self.BALL_RADIUS)
            # sfx: wall_bounce

        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce

        # Bottom wall (lose life)
        if self.ball_pos[1] >= self.screen_height:
            self.lives -= 1
            self.combo = 0
            # sfx: lose_life
            if self.lives > 0:
                self._reset_ball()
            return -10 # Penalty for losing a life

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS

            # Add spin
            offset = self.ball_pos[0] - self.paddle_rect.centerx
            normalized_offset = offset / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += normalized_offset * self.BALL_SPIN_FACTOR
            
            # Normalize velocity to maintain constant speed
            current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if current_speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / current_speed) * self.ball_speed
                self.ball_vel[1] = (self.ball_vel[1] / current_speed) * self.ball_speed
            
            self.combo = 0 # Reset combo on paddle hit
            # sfx: paddle_bounce
            
        # Brick collision
        hit_brick_idx = ball_rect.collidelist([b['rect'] for b in self.bricks])
        if hit_brick_idx != -1:
            brick_hit = self.bricks.pop(hit_brick_idx)
            # sfx: brick_hit
            
            # Handle combo
            if self.steps - self.last_brick_hit_step < self.COMBO_TIMEOUT:
                self.combo += 1
                reward += 1.0 + self.combo * 0.5 # Combo bonus
            else:
                self.combo = 1
            self.last_brick_hit_step = self.steps
            
            reward += 1.0 # Base reward for hitting a brick
            self.score += 10 * self.combo
            
            # Create explosion particles
            self._create_explosion(brick_hit['rect'].center, brick_hit['color'])
            
            # Ball reflection logic
            # Determine which side of the brick was hit
            dx = self.ball_pos[0] - brick_hit['rect'].centerx
            dy = self.ball_pos[1] - brick_hit['rect'].centery
            w, h = brick_hit['rect'].width / 2, brick_hit['rect'].height / 2
            
            if abs(dx / w) > abs(dy / h): # Horizontal collision
                self.ball_vel[0] *= -1
            else: # Vertical collision
                self.ball_vel[1] *= -1

            # Difficulty scaling
            bricks_destroyed = self.total_bricks - len(self.bricks)
            if bricks_destroyed > 0 and bricks_destroyed % 10 == 0:
                self.ball_speed = min(self.MAX_BALL_SPEED, self.ball_speed * 1.02)

        return reward

    def _create_explosion(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 
                'max_life': lifespan, 'color': color, 'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick['color'], brick['rect'])

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            size = int(p['size'] * (p['lifespan'] / p['max_life']))
            if size > 0:
                # Use a surface for alpha blending
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, (*p['color'], alpha), (size, size), size)
                self.screen.blit(particle_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        
        # Ball
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.FONT_UI.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.screen_width - lives_text.get_width() - 10, 10))
        
        # Combo
        if self.combo > 1:
            combo_text = self.FONT_COMBO.render(f"{self.combo}x COMBO!", True, self.BRICK_COLORS[self.combo % len(self.BRICK_COLORS)])
            text_rect = combo_text.get_rect(center=(self.screen_width / 2, 30))
            self.screen.blit(combo_text, text_rect)

        # Game Over Message
        if self.game_over:
            msg_text = self.FONT_MSG.render(self.game_over_message, True, (255, 255, 255))
            text_rect = msg_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks),
            "combo": self.combo,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        assert len(self.bricks) == self.BRICK_ROWS * self.BRICK_COLS
        assert self.lives == self.INITIAL_LIVES
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test paddle bounds
        self.paddle_rect.x = -100
        self.step(self.action_space.sample())
        assert self.paddle_rect.x >= 0
        self.paddle_rect.x = self.screen_width + 100
        self.step(self.action_space.sample())
        assert self.paddle_rect.right <= self.screen_width

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "dummy", "windows", "quartz" etc. depending on your OS

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Breakout")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Key state handling for continuous movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Limit to 30 FPS
        
    env.close()