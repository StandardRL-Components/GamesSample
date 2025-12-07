
# Generated: 2025-08-28T04:41:18.911175
# Source Brief: brief_05330.md
# Brief Index: 5330

        
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
        "Controls: Use ← and → arrow keys to move the paddle left and right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 60 seconds in a Multiball Pong arena. Keep the balls in play with your paddle to score points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    COLOR_BG = (20, 25, 40)
    COLOR_PADDLE = (200, 255, 255)
    COLOR_BALLS = [(255, 80, 80), (80, 255, 80), (80, 150, 255)]
    COLOR_UI = (220, 220, 220)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)

    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 8
    BALL_SPEED = 4
    NUM_BALLS = 3
    PARTICLE_LIFESPAN = 20

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 50, bold=True)
        
        # Etc...        
        self.paddle = None
        self.balls = []
        self.particles = []
        self.win = False
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Initialize paddle
        paddle_y = self.HEIGHT - 40
        self.paddle = pygame.Rect(
            self.WIDTH / 2 - self.PADDLE_WIDTH / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Initialize balls
        self.balls = []
        for i in range(self.NUM_BALLS):
            angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
            vel_x = math.cos(angle) * self.BALL_SPEED
            vel_y = abs(math.sin(angle) * self.BALL_SPEED) # Always start downwards
            
            ball_rect = pygame.Rect(
                self.np_random.integers(self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS),
                self.np_random.integers(self.BALL_RADIUS, self.HEIGHT // 3),
                self.BALL_RADIUS * 2,
                self.BALL_RADIUS * 2
            )

            self.balls.append({
                'rect': ball_rect,
                'vel': [vel_x, vel_y],
                'color': self.COLOR_BALLS[i % len(self.COLOR_BALLS)]
            })

        # Clear particles
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Action Handling ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen bounds
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)
        
        reward = 0
        
        # --- Ball Physics and Logic ---
        active_balls = []
        for ball in self.balls:
            ball['rect'].x += ball['vel'][0]
            ball['rect'].y += ball['vel'][1]

            # Wall collisions
            if ball['rect'].left <= 0 or ball['rect'].right >= self.WIDTH:
                ball['vel'][0] *= -1
                ball['rect'].left = max(0, ball['rect'].left)
                ball['rect'].right = min(self.WIDTH, ball['rect'].right)
            if ball['rect'].top <= 0:
                ball['vel'][1] *= -1
                ball['rect'].top = max(0, ball['rect'].top)

            # Paddle collision
            if ball['rect'].colliderect(self.paddle) and ball['vel'][1] > 0:
                ball['vel'][1] *= -1.02 # Slight speed up on hit
                # Add "spin" based on where it hits the paddle
                hit_pos_norm = (ball['rect'].centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                ball['vel'][0] += hit_pos_norm * 2.0
                ball['vel'][0] = np.clip(ball['vel'][0], -self.BALL_SPEED*1.5, self.BALL_SPEED*1.5)
                
                ball['rect'].bottom = self.paddle.top

                reward += 1.0  # Event-based reward for hitting a ball
                self._create_particles(ball['rect'].midbottom)
                # Sound effect placeholder: // Paddle hit sound

            # Check if ball is still in play
            if ball['rect'].top < self.HEIGHT:
                active_balls.append(ball)

        self.balls = active_balls
        
        self._update_particles()
        
        # --- Reward Calculation ---
        reward += 0.1 * len(self.balls)
        self.score += reward

        # --- Termination Check ---
        self.steps += 1
        terminated = False

        if len(self.balls) == 0:
            terminated = True
            self.game_over = True
            self.win = False
            # Sound effect placeholder: // Game over sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win = True
            win_bonus = 100.0
            reward += win_bonus
            self.score += win_bonus
            # Sound effect placeholder: // Win sound
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
            "balls_left": len(self.balls),
            "win": self.win
        }

    def _render_game(self):
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw balls with anti-aliasing
        for ball in self.balls:
            pygame.gfxdraw.filled_circle(
                self.screen, int(ball['rect'].centerx), int(ball['rect'].centery), self.BALL_RADIUS, ball['color']
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(ball['rect'].centerx), int(ball['rect'].centery), self.BALL_RADIUS, ball['color']
            )

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / self.PARTICLE_LIFESPAN))
            color = (*self.COLOR_PARTICLE, alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Render timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Render game over/win message
        if self.game_over:
            if self.win:
                msg_text = "YOU WIN!"
                msg_color = self.COLOR_WIN
            else:
                msg_text = "GAME OVER"
                msg_color = self.COLOR_LOSE
            
            text_surf = self.font_msg.render(msg_text, True, msg_color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
            
    def _create_particles(self, pos):
        # Sound effect placeholder: // Particle burst sound
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.integers(2, 5),
                'life': self.PARTICLE_LIFESPAN
            })
            
    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] > 0 and p['radius'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()
        
# Example of how to run the environment for visualization
if __name__ == '__main__':
    # Set this to run pygame in a window
    import os
    # os.environ['SDL_VIDEODRIVER'] = 'dummy' # For headless execution
    
    env = GameEnv(render_mode="rgb_array")
    env.reset()

    # Create a display window
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Multiball Pong")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    while running:
        # --- Event Handling ---
        action = env.action_space.sample()
        action[0] = 0 # Default to no-op for movement
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # The game will now just show the final screen until reset (R key)
            
        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(GameEnv.FPS)

    env.close()