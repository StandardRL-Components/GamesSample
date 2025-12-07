
# Generated: 2025-08-28T03:51:32.622136
# Source Brief: brief_05064.md
# Brief Index: 5064

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-inspired brick breaker game where risk-taking is rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (10, 10, 40)
    COLOR_GRID = (20, 20, 60)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (50, 205, 50)
    BRICK_COLORS = [
        (200, 50, 50), (50, 200, 50), (50, 50, 200),
        (200, 200, 50), (50, 200, 200), (200, 50, 200)
    ]

    PADDLE_WIDTH, PADDLE_HEIGHT = 80, 12
    PADDLE_SPEED = 8
    BALL_RADIUS = 6
    BALL_INITIAL_SPEED = 4.0
    BALL_SPEED_INCREMENT = 0.5
    
    MAX_STEPS = 10000
    INITIAL_LIVES = 3

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.stage = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = []
        self.particles = []
        self.last_paddle_hit_offset = 0.0
        self.last_brick_hit_info = {"pos": None, "count": 0}
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.particles = []
        self.last_paddle_hit_offset = 0.0
        self.last_brick_hit_info = {"pos": None, "count": 0}

        self._create_bricks(self.stage)
        self._reset_ball()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]
        
        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # 2. Update game logic
        step_rewards = self._update_ball()
        reward += step_rewards
        self._update_particles()
        
        # 3. Check for stage clear
        if not any(any(row) for row in self.bricks):
            # sfx: stage_clear
            reward += 10.0
            self.stage += 1
            if self.stage > 3:
                self.game_over = True
                reward += 100.0 # Win game bonus
            else:
                self._create_bricks(self.stage)
                self._reset_ball()

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        reward = 0.0
        
        # Store previous position
        prev_ball_pos = self.ball_pos[:]
        
        # Move ball
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left < 0 or ball_rect.right > self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = np.clip(ball_rect.left, 0, self.WIDTH - ball_rect.width)
            self.ball_pos[0] = ball_rect.centerx
            # sfx: bounce_wall
        if ball_rect.top < 0:
            self.ball_vel[1] *= -1
            ball_rect.top = 0
            self.ball_pos[1] = ball_rect.centery
            # sfx: bounce_wall

        # Bottom wall (lose life)
        if ball_rect.top > self.HEIGHT:
            self.lives -= 1
            reward -= 1.0
            # sfx: lose_life
            if self.lives <= 0:
                self.game_over = True
            else:
                self._reset_ball()
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: bounce_paddle
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2.0)
            offset = np.clip(offset, -1.0, 1.0)
            self.last_paddle_hit_offset = offset

            self.ball_vel[1] *= -1
            self.ball_vel[0] = offset * 5.0
            
            # Normalize to maintain speed
            speed = self.BALL_INITIAL_SPEED + (self.stage - 1) * self.BALL_SPEED_INCREMENT
            current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if current_speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / current_speed) * speed
                self.ball_vel[1] = (self.ball_vel[1] / current_speed) * speed
            
            ball_rect.bottom = self.paddle.top
            self.ball_pos[1] = ball_rect.centery

        # Brick collisions
        brick_hit = False
        for r, row in enumerate(self.bricks):
            for c, brick in enumerate(row):
                if brick and ball_rect.colliderect(brick['rect']):
                    brick_hit = True
                    # sfx: brick_hit
                    self._create_particles(brick['rect'].center, brick['color'])
                    
                    # Reward logic
                    reward += 1.0
                    if abs(self.last_paddle_hit_offset) > 0.7:
                        reward += 0.1
                    elif abs(self.last_paddle_hit_offset) < 0.3:
                        reward -= 0.02
                    self.score += 10

                    # Anti-softlock
                    brick_pos_id = (r, c)
                    if self.last_brick_hit_info["pos"] == brick_pos_id:
                        self.last_brick_hit_info["count"] += 1
                    else:
                        self.last_brick_hit_info["pos"] = brick_pos_id
                        self.last_brick_hit_info["count"] = 1

                    if self.last_brick_hit_info["count"] >= 100:
                        self.bricks[r][c] = None # Force destroy
                        break

                    # Collision response
                    prev_ball_rect = pygame.Rect(prev_ball_pos[0] - self.BALL_RADIUS, prev_ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
                    
                    # Determine if collision was primarily horizontal or vertical
                    if prev_ball_rect.bottom <= brick['rect'].top or prev_ball_rect.top >= brick['rect'].bottom:
                         self.ball_vel[1] *= -1
                    if prev_ball_rect.right <= brick['rect'].left or prev_ball_rect.left >= brick['rect'].right:
                         self.ball_vel[0] *= -1
                    
                    self.bricks[r][c] = None
                    break
            if brick_hit:
                break
                
        return reward

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        speed = self.BALL_INITIAL_SPEED + (self.stage - 1) * self.BALL_SPEED_INCREMENT
        self.ball_vel = [speed * math.sin(angle), -speed * math.cos(angle)]

    def _create_bricks(self, stage):
        self.bricks = []
        brick_width = 40
        brick_height = 15
        layout = self._get_brick_layout(stage)
        
        for r, row_pattern in enumerate(layout):
            brick_row = []
            for c, cell in enumerate(row_pattern):
                if cell == 'X':
                    x = 50 + c * (brick_width + 5)
                    y = 50 + r * (brick_height + 5)
                    color_index = (r + c) % len(self.BRICK_COLORS)
                    brick_row.append({
                        'rect': pygame.Rect(x, y, brick_width, brick_height),
                        'color': self.BRICK_COLORS[color_index]
                    })
                else:
                    brick_row.append(None)
            self.bricks.append(brick_row)

    def _get_brick_layout(self, stage):
        if stage == 1:
            return [
                "XXXXXXXXXXXX",
                "XXXXXXXXXXXX",
                "XXXXXXXXXXXX",
                "XXXXXXXXXXXX",
            ]
        elif stage == 2:
            return [
                "     XX     ",
                "    XXXX    ",
                "   XXXXXX   ",
                "  XXXXXXXX  ",
                " XXXXXXXXXX ",
                "XXXXXXXXXXXX",
            ]
        elif stage == 3:
            return [
                "X X X X X X ",
                " X X X X X X",
                "X X X X X X ",
                " X X X X X X",
                "X X X X X X ",
                " X X X X X X",
            ]
        return []
    
    def _create_particles(self, pos, color):
        for _ in range(10):
            vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 1)]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': 20, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

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
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Bricks
        for row in self.bricks:
            for brick in row:
                if brick:
                    pygame.draw.rect(self.screen, brick['color'], brick['rect'])
                    pygame.draw.rect(self.screen, tuple(c*0.7 for c in brick['color']), brick['rect'], 1)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        pos = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            alpha = max(0, p['life'] * 12)
            color = (*p['color'], alpha)
            particle_surface = pygame.Surface((3, 3), pygame.SRCALPHA)
            particle_surface.fill(color)
            self.screen.blit(particle_surface, (int(p['pos'][0]), int(p['pos'][1])))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(centerx=self.WIDTH // 2, top=10)
        self.screen.blit(stage_text, stage_rect)

        # Lives
        lives_text = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            life_icon_rect = pygame.Rect(self.WIDTH - 80 + i * 20, 12, 15, 8)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_icon_rect, border_radius=2)
            
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.lives <= 0:
                msg = "GAME OVER"
            else:
                msg = "YOU WIN!"
            
            end_text = self.font_large.render(msg, True, self.COLOR_BALL)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
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
    # To run and play the game manually
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11" or "windows" or "mac" etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Brick Breaker")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        action = [0, 0, 0]  # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_q]:
            running = False
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS
        
    env.close()