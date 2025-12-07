
# Generated: 2025-08-28T04:14:20.761259
# Source Brief: brief_02243.md
# Brief Index: 2243

        
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

    user_guide = (
        "Controls: ←→ to move the paddle. Break all the bricks to win."
    )

    game_description = (
        "A minimalist Breakout clone. Control the paddle to reflect the ball and destroy all the bricks. You have two lives."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.COLOR_BG = (0, 0, 0)
        self.COLOR_FG = (255, 255, 255)
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 10
        self.PADDLE_Y = self.HEIGHT - 30
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.BALL_SPEED = 5 # Changed from 3 to 5 for better game feel
        self.BRICK_ROWS, self.BRICK_COLS = 4, 5
        self.BRICK_WIDTH, self.BRICK_HEIGHT = 100, 20
        self.BRICK_GAP = 10
        self.BRICK_AREA_TOP = 50
        self.MAX_STEPS = 1500 # Increased from 1000 for longer potential games
        self.INITIAL_LIVES = 2
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_game_over = pygame.font.SysFont("Consolas", 50)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 28)
            self.font_game_over = pygame.font.SysFont(None, 60)

        # --- Game State ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle_x = 0
        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]
        self.bricks = []
        self.particles = []
        
        self.reset()
        
        # This will fail if the implementation is incorrect
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        
        # Paddle
        self.paddle_x = self.WIDTH / 2 - self.PADDLE_WIDTH / 2
        
        # Ball
        self.ball_pos = [self.WIDTH / 2, self.PADDLE_Y - self.BALL_RADIUS - 5]
        angle = self.np_random.uniform(low=-math.pi * 0.75, high=-math.pi * 0.25)
        self.ball_vel = [self.BALL_SPEED * math.cos(angle), self.BALL_SPEED * math.sin(angle)]

        # Bricks
        self.bricks = []
        total_brick_width = self.BRICK_COLS * self.BRICK_WIDTH + (self.BRICK_COLS - 1) * self.BRICK_GAP
        start_x = (self.WIDTH - total_brick_width) / 2
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                x = start_x + j * (self.BRICK_WIDTH + self.BRICK_GAP)
                y = self.BRICK_AREA_TOP + i * (self.BRICK_HEIGHT + self.BRICK_GAP)
                self.bricks.append(pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT))

        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # --- 1. Handle Input ---
        movement = action[0]
        if movement == 3: # Left
            self.paddle_x -= self.PADDLE_SPEED
            reward -= 0.01
        elif movement == 4: # Right
            self.paddle_x += self.PADDLE_SPEED
            reward -= 0.01
        
        self.paddle_x = np.clip(self.paddle_x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        # --- 2. Update Game State ---
        self._update_particles()
        self._update_ball()
        
        # --- 3. Handle Collisions & Rewards ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- 4. Check Termination ---
        self.steps += 1
        terminated = False
        
        if self.lives <= 0:
            self.game_over = True
            terminated = True
            reward -= 100 # Lose penalty
        
        if not self.bricks:
            self.game_over = True
            terminated = True
            reward += 100 # Win bonus
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # Sound: Wall Hit
            
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            # Sound: Wall Hit

        # Bottom wall (miss)
        if self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS:
            self.lives -= 1
            # Sound: Miss
            if self.lives > 0:
                # Reset ball
                self.ball_pos = [self.paddle_x + self.PADDLE_WIDTH / 2, self.PADDLE_Y - self.BALL_RADIUS - 5]
                angle = self.np_random.uniform(low=-math.pi * 0.75, high=-math.pi * 0.25)
                self.ball_vel = [self.BALL_SPEED * math.cos(angle), self.BALL_SPEED * math.sin(angle)]

    def _handle_collisions(self):
        reward = 0.0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_x, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if ball_rect.colliderect(paddle_rect) and self.ball_vel[1] > 0:
            reward += 0.1
            # Sound: Paddle Hit
            
            # Change ball angle based on hit location
            offset = (ball_rect.centerx - paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            angle = math.pi * (0.5 - offset * 0.4) # Map offset to angle range
            
            self.ball_vel[0] = self.BALL_SPEED * math.cos(angle)
            self.ball_vel[1] = -abs(self.BALL_SPEED * math.sin(angle)) # Ensure it goes up
            
            # Prevent sticking
            self.ball_pos[1] = self.PADDLE_Y - self.BALL_RADIUS

        # Brick collision
        for i, brick in enumerate(self.bricks):
            if ball_rect.colliderect(brick):
                reward += 1.0
                self.score += 1
                # Sound: Brick Break
                
                self._create_particles(brick.center, 20)
                
                # Determine bounce direction
                prev_ball_pos_x = self.ball_pos[0] - self.ball_vel[0]
                prev_ball_pos_y = self.ball_pos[1] - self.ball_vel[1]

                if prev_ball_pos_x < brick.left or prev_ball_pos_x > brick.right:
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1
                
                self.bricks.pop(i)
                break # Only one brick per frame
        
        return reward

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, self.COLOR_FG, brick)
            
        # Draw Paddle
        paddle_rect = pygame.Rect(self.paddle_x, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_FG, paddle_rect)
        
        # Draw Ball
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_FG)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_FG)
        
        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (alpha, alpha, alpha)
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
                pygame.draw.rect(self.screen, color, rect)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_FG)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_FG)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if not self.bricks else "GAME OVER"
            game_over_surf = self.font_game_over.render(msg, True, self.COLOR_FG)
            pos = (self.WIDTH / 2 - game_over_surf.get_width() / 2, self.HEIGHT / 2 - game_over_surf.get_height() / 2)
            self.screen.blit(game_over_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game in a window ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Breakout Gym Environment")
    
    obs, info = env.reset()
    done = False
    
    # Use a simple agent to play the game
    def simple_agent(ball_pos, paddle_x, paddle_width):
        action = [0, 0, 0] # Default: no-op
        paddle_center = paddle_x + paddle_width / 2
        if ball_pos[0] < paddle_center - 10:
            action[0] = 3 # Move left
        elif ball_pos[0] > paddle_center + 10:
            action[0] = 4 # Move right
        return action
        
    while not done:
        # Agent decides action
        action = simple_agent(env.ball_pos, env.paddle_x, env.PADDLE_WIDTH)
        # action = env.action_space.sample() # Random agent

        # Environment steps
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(60) # Limit to 60 FPS for human play
        
    print(f"Game Over! Final Info: {info}")
    env.close()