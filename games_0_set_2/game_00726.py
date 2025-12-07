
# Generated: 2025-08-27T14:34:23.964148
# Source Brief: brief_00726.md
# Brief Index: 726

        
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
        "Controls: Arrow keys change gravity direction. Hold space to move your paddle left, shift to move right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control gravity to bounce a ball and score points against an AI paddle in a top-down arcade game."
    )

    # Frames auto-advance for smooth real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (0, 0, 10) # Dark navy
        self.COLOR_WALL = (20, 20, 80)
        self.COLOR_PLAYER = (0, 255, 255) # Cyan
        self.COLOR_AI = (255, 0, 255) # Magenta
        self.COLOR_BALL = (255, 255, 255) # White
        self.COLOR_GRAVITY_ARROW = (0, 255, 0, 50) # Transparent Green
        self.COLOR_UI = (255, 255, 0) # Yellow
        self.PARTICLE_COLORS = [(255, 69, 0), (255, 165, 0), (255, 215, 0)]

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.BALL_RADIUS = 8
        self.GRAVITY_STRENGTH = 0.15
        self.PADDLE_SPEED = 8
        self.MAX_BALL_SPEED = 12
        self.AI_REACTION_SPEED = 0.08
        self.MAX_STEPS = 1500 # Extended for longer rallies
        self.WIN_SCORE = 10
        self.MAX_LIVES = 3

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.player_paddle = None
        self.ai_paddle = None
        self.ball = None
        self.ball_vel = None
        self.gravity_dir = None
        self.particles = []

        # Initialize state variables
        self.reset()
        
        # Self-validation
        # self.validate_implementation() # Commented out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        self.player_paddle = pygame.Rect(
            self.WIDTH / 2 - self.PADDLE_WIDTH / 2, 
            self.HEIGHT - self.PADDLE_HEIGHT - 10, 
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        self.ai_paddle = pygame.Rect(
            self.WIDTH / 2 - self.PADDLE_WIDTH / 2, 
            10, 
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        
        self.particles = []
        self._reset_ball()
        
        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball = pygame.Rect(self.WIDTH / 2 - self.BALL_RADIUS, self.HEIGHT / 2 - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        # Start ball moving towards the player
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        initial_speed = 5
        self.ball_vel = [math.cos(angle) * initial_speed, math.sin(angle) * initial_speed]
        self.gravity_dir = (0, 1) # Gravity starts downwards
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Small penalty per step to encourage action

        self._handle_input(action)
        self._update_ai()
        reward += self._update_ball()
        self._update_particles()
        
        self.steps += 1
        
        terminated = self.score >= self.WIN_SCORE or self.lives <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 10 # Win bonus
            elif self.lives <= 0:
                reward -= 10 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        gravity_action = action[0]
        move_left = action[1] == 1
        move_right = action[2] == 1

        # Update gravity
        if gravity_action == 1: self.gravity_dir = (0, -1) # Up
        elif gravity_action == 2: self.gravity_dir = (0, 1) # Down
        elif gravity_action == 3: self.gravity_dir = (-1, 0) # Left
        elif gravity_action == 4: self.gravity_dir = (1, 0) # Right
        
        # Update player paddle
        if move_left:
            self.player_paddle.x -= self.PADDLE_SPEED
        if move_right:
            self.player_paddle.x += self.PADDLE_SPEED
        
        self.player_paddle.left = max(0, self.player_paddle.left)
        self.player_paddle.right = min(self.WIDTH, self.player_paddle.right)

    def _update_ai(self):
        target_x = self.ball.centerx
        self.ai_paddle.centerx += (target_x - self.ai_paddle.centerx) * self.AI_REACTION_SPEED
        self.ai_paddle.left = max(0, self.ai_paddle.left)
        self.ai_paddle.right = min(self.WIDTH, self.ai_paddle.right)

    def _update_ball(self):
        reward = 0
        # Apply gravity
        self.ball_vel[0] += self.gravity_dir[0] * self.GRAVITY_STRENGTH
        self.ball_vel[1] += self.gravity_dir[1] * self.GRAVITY_STRENGTH

        # Clamp speed
        speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
        if speed > self.MAX_BALL_SPEED:
            self.ball_vel[0] = (self.ball_vel[0] / speed) * self.MAX_BALL_SPEED
            self.ball_vel[1] = (self.ball_vel[1] / speed) * self.MAX_BALL_SPEED

        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall collisions
        if self.ball.left < 0 or self.ball.right > self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.WIDTH, self.ball.right)
            self._create_particles(self.ball.center, 10)
            # sfx: wall_bounce.wav
        
        # Paddle collisions
        if self.ball.colliderect(self.player_paddle) and self.ball_vel[1] > 0:
            reward += self._handle_paddle_collision(self.player_paddle)
            # sfx: player_hit.wav
        
        if self.ball.colliderect(self.ai_paddle) and self.ball_vel[1] < 0:
            self._handle_paddle_collision(self.ai_paddle)
            # sfx: ai_hit.wav

        # Scoring
        if self.ball.top > self.HEIGHT:
            self.lives -= 1
            self._reset_ball()
            # sfx: lose_life.wav
        elif self.ball.bottom < 0:
            self.score += 1
            reward += 1.0 # Reward for scoring a point
            self._reset_ball()
            # sfx: score_point.wav
            
        return reward

    def _handle_paddle_collision(self, paddle):
        self.ball_vel[1] *= -1.05 # Add a bit of speed on hit
        
        # Adjust horizontal velocity based on where it hit the paddle
        hit_pos = (self.ball.centerx - paddle.centerx) / (paddle.width / 2)
        self.ball_vel[0] += hit_pos * 2.0
        
        # Ensure ball is outside the paddle to prevent sticking
        if self.ball_vel[1] < 0: # Hit player paddle, moving up
            self.ball.bottom = paddle.top
        else: # Hit AI paddle, moving down
            self.ball.top = paddle.bottom
        
        self._create_particles(self.ball.center, 20)

        # Calculate reward for player hits
        if paddle == self.player_paddle:
            hit_abs = abs(hit_pos)
            if hit_abs > 0.7: # Risky hit near the edge
                return 1.0
            elif hit_abs < 0.3: # Safe hit near the center
                return -0.2
            else: # Normal hit
                return 0.1
        return 0

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifespan = self.np_random.integers(15, 30)
            color = random.choice(self.PARTICLE_COLORS)
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': radius, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] -= 0.1
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw gravity indicator
        self._draw_gravity_arrow()
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color_with_alpha = p['color'] + (alpha,)
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color_with_alpha)

        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, 10))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.HEIGHT - 10, self.WIDTH, 10))
        
        # Draw paddles
        pygame.draw.rect(self.screen, self.COLOR_AI, self.ai_paddle, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_paddle, border_radius=5)
        
        # Draw ball with glow
        ball_pos_int = (int(self.ball.centerx), int(self.ball.centery))
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_color = self.COLOR_BALL + (50,) # White with low alpha
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], glow_radius, glow_color)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _draw_gravity_arrow(self):
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        arrow_len = 30
        p1 = (center_x, center_y)
        
        dx, dy = self.gravity_dir
        p2 = (center_x + dx * arrow_len, center_y + dy * arrow_len)
        
        # Orthogonal vector for arrow wings
        ox, oy = -dy, dx
        wing_len = 15
        
        p3 = (p2[0] - dx * wing_len + ox * wing_len, p2[1] - dy * wing_len + oy * wing_len)
        p4 = (p2[0] - dx * wing_len - ox * wing_len, p2[1] - dy * wing_len - oy * wing_len)
        
        points = [p2, p3, p4]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRAVITY_ARROW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRAVITY_ARROW)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (20, self.HEIGHT - 40))
        
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 20, self.HEIGHT - 40))
        
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_PLAYER)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_AI)
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
if __name__ == "__main__":
    import time

    # To run with display, you need to set up a Pygame window
    # This is for testing and visualization purposes only
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame window setup for human play ---
    pygame.display.set_caption("Gravity Pong")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    # ---

    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Map keyboard keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        # --- Human input handling ---
        action = [0, 0, 0] # [no-op, space_released, shift_released]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        for key, val in key_map.items():
            if keys[key]:
                action[0] = val
                break # Only one gravity direction at a time
        
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Space held
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1 # Shift held
        # ---

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            time.sleep(2)
            obs, info = env.reset()
            total_reward = 0

        # --- Render to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        # ---
        
        clock.tick(30) # Run at 30 FPS

    env.close()