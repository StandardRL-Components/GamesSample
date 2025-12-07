
# Generated: 2025-08-27T20:21:30.959116
# Source Brief: brief_02432.md
# Brief Index: 2432

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper function to draw glowing circles for a neon effect
def draw_glowing_circle(surface, color, center, radius, glow_factor=2.0):
    """Draws a circle with a glow effect."""
    glow_radius = int(radius * glow_factor)
    for i in range(glow_radius, int(radius), -1):
        alpha = int(50 * (1 - (i - radius) / (glow_radius - radius)))
        if alpha > 0:
            glow_color = (*color, alpha)
            pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), i, glow_color)
    pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), color)
    pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: ↑ to move the paddle up, ↓ to move down."
    game_description = "Survive an accelerating barrage of bouncing balls by deflecting them with your paddle."

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 100  # Game logic runs at 100 steps per second as per brief
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_BALL = (255, 0, 100)
        self.COLOR_PARTICLE = (255, 255, 0)
        self.COLOR_WALL = (200, 200, 255)
        self.COLOR_UI = (50, 255, 50)

        # Paddle
        self.PADDLE_WIDTH = 12
        self.PADDLE_HEIGHT = 80
        self.PADDLE_SPEED = 6
        self.PADDLE_X = 30

        # Ball
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 3.0
        self.BALL_SPAWN_INTERVAL = 5 * self.FPS
        self.SPEED_INCREASE_INTERVAL = 10 * self.FPS
        self.BALL_SPEED_INCREMENT = 0.2

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.paddle_y = 0
        self.balls = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_ball_speed = 0
        self.ball_spawn_timer = 0
        self.speed_increase_timer = 0
        
        # This will be properly initialized in reset()
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_y = self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2
        
        self.balls = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_ball_speed = self.INITIAL_BALL_SPEED
        self.ball_spawn_timer = 0
        self.speed_increase_timer = 0

        self._spawn_ball()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # --- Update Game State ---
            self.steps += 1
            
            self._handle_input(action)
            self._update_difficulty()
            self._update_balls()
            self._update_particles()
            
            # --- Collision and Rewards ---
            reward += self._handle_collisions()

            # --- Check Termination ---
            if self.game_over:
                terminated = True
                reward = -100.0 # Penalty for losing
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True
                reward += 100.0 # Bonus for survival
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _spawn_ball(self):
        # Spawn ball in the center-right area with a random velocity towards the paddle
        x = self.WIDTH * 0.75
        y = self.np_random.uniform(self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
        
        angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25) # Aim towards left
        vel_x = self.base_ball_speed * math.cos(angle)
        vel_y = self.base_ball_speed * math.sin(angle)
        
        ball = {
            "pos": pygame.Vector2(x, y),
            "vel": pygame.Vector2(vel_x, vel_y),
            "trail": collections.deque(maxlen=10)
        }
        self.balls.append(ball)

    def _handle_input(self, action):
        movement = action[0]
        if movement == 1:  # Up
            self.paddle_y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle_y += self.PADDLE_SPEED
        
        self.paddle_y = np.clip(self.paddle_y, 0, self.HEIGHT - self.PADDLE_HEIGHT)

    def _update_difficulty(self):
        # Increase ball speed over time
        self.speed_increase_timer += 1
        if self.speed_increase_timer >= self.SPEED_INCREASE_INTERVAL:
            self.speed_increase_timer = 0
            self.base_ball_speed += self.BALL_SPEED_INCREMENT
            # Update existing balls' speed
            for ball in self.balls:
                if ball["vel"].length() > 0:
                    ball["vel"].scale_to_length(self.base_ball_speed)

        # Spawn new balls over time
        self.ball_spawn_timer += 1
        if self.ball_spawn_timer >= self.BALL_SPAWN_INTERVAL:
            self.ball_spawn_timer = 0
            self._spawn_ball()

    def _update_balls(self):
        for ball in self.balls:
            ball["trail"].append(ball["pos"].copy())
            ball["pos"] += ball["vel"]

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            p["radius"] = max(0, p["radius"] * 0.95)
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _handle_collisions(self):
        reward = 0
        paddle_rect = pygame.Rect(self.PADDLE_X, self.paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        for ball in self.balls:
            # Wall collisions
            if ball["pos"].y - self.BALL_RADIUS <= 0 or ball["pos"].y + self.BALL_RADIUS >= self.HEIGHT:
                ball["vel"].y *= -1
                ball["pos"].y = np.clip(ball["pos"].y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
                # sfx: wall_bounce
            if ball["pos"].x + self.BALL_RADIUS >= self.WIDTH:
                ball["vel"].x *= -1
                ball["pos"].x = self.WIDTH - self.BALL_RADIUS
                # sfx: wall_bounce

            # Game Over condition
            if ball["pos"].x - self.BALL_RADIUS <= 0:
                self.game_over = True
                # sfx: game_over_sound
                break

            # Paddle collision
            ball_rect = pygame.Rect(ball["pos"].x - self.BALL_RADIUS, ball["pos"].y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if paddle_rect.colliderect(ball_rect) and ball["vel"].x < 0:
                ball["vel"].x *= -1
                
                # Add "spin" based on impact position
                offset = (ball["pos"].y - paddle_rect.centery) / (self.PADDLE_HEIGHT / 2)
                ball["vel"].y += offset * 2.0
                
                # Normalize to maintain speed
                ball["vel"].scale_to_length(self.base_ball_speed)
                
                # Ensure ball is pushed out of paddle
                ball["pos"].x = paddle_rect.right + self.BALL_RADIUS

                reward += 0.1
                self.score += 0.1
                self._create_particles(ball["pos"])
                # sfx: paddle_hit
        
        return reward

    def _create_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            particle = {
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "lifetime": self.np_random.integers(15, 30)
            }
            self.particles.append(particle)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (self.WIDTH, 0), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (self.WIDTH - 1, 0), (self.WIDTH - 1, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, self.HEIGHT - 1), (self.WIDTH, self.HEIGHT - 1), 2)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30.0))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)

        # Draw balls and trails
        for ball in self.balls:
            # Trail
            for i, pos in enumerate(ball["trail"]):
                alpha = int(80 * (i / len(ball["trail"])))
                radius = self.BALL_RADIUS * (i / len(ball["trail"])) * 0.5
                if radius > 1:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), (*self.COLOR_BALL, alpha))
            # Ball
            draw_glowing_circle(self.screen, self.COLOR_BALL, ball["pos"], self.BALL_RADIUS)

        # Draw paddle
        paddle_rect = pygame.Rect(self.PADDLE_X, self.paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        glow_color = (*self.COLOR_PADDLE, 50)
        pygame.draw.rect(self.screen, glow_color, paddle_rect.inflate(8, 8), border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=5)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (15, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 15, 10))

        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                end_text = "YOU SURVIVED!"
            else:
                end_text = "GAME OVER"
            
            end_surf = self.font.render(end_text, True, self.COLOR_UI)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_surf, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_elapsed_seconds": self.steps / self.FPS,
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires `pip install pygame`
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set up window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Paddle Panic")
    clock = pygame.time.Clock()

    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2

        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Total Reward: {total_reward:.1f}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(60) # Run at 60 FPS for human play

    env.close()