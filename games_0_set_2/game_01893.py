import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to move the paddle. Survive for 60 seconds to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game. Reflect the neon ball with your paddle, avoid moving "
        "obstacles, and survive for 60 seconds to achieve the highest score."
    )

    # Frames auto-advance at a rate of 30fps.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (25, 10, 40)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_BALL = (255, 0, 150)
        self.COLOR_OBSTACLE = (255, 120, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE_BALL = (255, 100, 200)
        self.COLOR_PARTICLE_PADDLE = (100, 200, 255)
        
        # Paddle
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_ACCEL = 1.0
        self.PADDLE_FRICTION = 0.90
        self.PADDLE_MAX_SPEED = 12.0
        
        # Ball
        self.BALL_RADIUS = 8
        self.BALL_INITIAL_SPEED = 5.0
        self.BALL_MAX_X_VEL = 7.0
        
        # Obstacles
        self.NUM_OBSTACLES = 5
        self.OBSTACLE_BASE_SPEED = 2.0
        self.OBSTACLE_SPEED_INCREASE_INTERVAL = 10 * self.FPS # every 10 seconds
        self.OBSTACLE_SPEED_INCREASE_AMOUNT = 0.5

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paddle = None
        self.paddle_vel = 0.0
        self.ball = None
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.obstacles = []
        self.obstacle_current_speed = self.OBSTACLE_BASE_SPEED
        self.particles = []
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Paddle
        paddle_x = self.WIDTH / 2 - self.PADDLE_WIDTH / 2
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT * 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.paddle_vel = 0.0

        # Ball
        self.ball = pygame.Rect(self.WIDTH / 2 - self.BALL_RADIUS, self.HEIGHT / 2 - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        angle = self.np_random.uniform(math.pi * 0.4, math.pi * 0.6) # Downward angle
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * self.BALL_INITIAL_SPEED
        
        # Obstacles
        self.obstacles = []
        self.obstacle_current_speed = self.OBSTACLE_BASE_SPEED
        for i in range(self.NUM_OBSTACLES):
            size = self.np_random.integers(30, 80)
            obstacle = {
                "rect": pygame.Rect(0, 0, size if i % 2 == 0 else 20, 20 if i % 2 == 0 else size),
                "base_pos": np.array([
                    self.np_random.uniform(size, self.WIDTH - size),
                    self.np_random.uniform(50, self.HEIGHT - 150)
                ], dtype=np.float32),
                "amplitude": self.np_random.uniform(50, 150),
                "frequency": self.np_random.uniform(0.01, 0.03),
                "phase": self.np_random.uniform(0, 2 * math.pi),
                "type": "horizontal" if i % 2 == 0 else "vertical"
            }
            self.obstacles.append(obstacle)

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 3=left, 4=right

        # --- Update Game Logic ---
        
        # 1. Update Paddle
        if movement == 3:  # Left
            self.paddle_vel -= self.PADDLE_ACCEL
        elif movement == 4:  # Right
            self.paddle_vel += self.PADDLE_ACCEL
        else: # no-op or other actions
            self.paddle_vel *= self.PADDLE_FRICTION
        
        self.paddle_vel = np.clip(self.paddle_vel, -self.PADDLE_MAX_SPEED, self.PADDLE_MAX_SPEED)
        self.paddle.x += self.paddle_vel
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # 2. Update Ball
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # 3. Collision Detection
        # Walls
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.x = np.clip(self.ball.x, 0, self.WIDTH - self.ball.width)
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.y = np.clip(self.ball.y, 0, self.HEIGHT - self.ball.height)

        # Paddle
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.0
            self.ball_vel[0] = np.clip(self.ball_vel[0], -self.BALL_MAX_X_VEL, self.BALL_MAX_X_VEL)
            
            reward += 1.0
            self.score += 10
            self._create_particles(self.ball.center, self.COLOR_PARTICLE_PADDLE, 15)

            for obs in self.obstacles:
                dist = math.hypot(self.ball.centerx - obs['rect'].centerx, self.ball.centery - obs['rect'].centery)
                if dist < (self.BALL_RADIUS + max(obs['rect'].width, obs['rect'].height) + 10):
                    reward += 5.0
                    self.score += 50
                    break

        # Obstacles
        for obs in self.obstacles:
            if self.ball.colliderect(obs["rect"]):
                dx = self.ball.centerx - obs["rect"].centerx
                dy = self.ball.centery - obs["rect"].centery
                
                if abs(dx / obs["rect"].width) > abs(dy / obs["rect"].height):
                    if dx > 0: self.ball.left = obs["rect"].right
                    else: self.ball.right = obs["rect"].left
                    self.ball_vel[0] *= -1
                else:
                    if dy > 0: self.ball.top = obs["rect"].bottom
                    else: self.ball.bottom = obs["rect"].top
                    self.ball_vel[1] *= -1
                
                self._create_particles(self.ball.center, self.COLOR_OBSTACLE, 10)
                break

        # 4. Update Obstacles
        if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
            self.obstacle_current_speed += self.OBSTACLE_SPEED_INCREASE_AMOUNT

        for obs in self.obstacles:
            time_val = self.steps * obs["frequency"] + obs["phase"]
            if obs["type"] == "horizontal":
                obs["rect"].centerx = obs["base_pos"][0] + math.sin(time_val) * obs["amplitude"]
                obs["rect"].centery = obs["base_pos"][1]
            else: # vertical
                obs["rect"].centerx = obs["base_pos"][0]
                obs["rect"].centery = obs["base_pos"][1] + math.sin(time_val) * obs["amplitude"]
        
        # 5. Update Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            p['radius'] -= 0.2

        # 6. Update State & Rewards
        self.steps += 1
        reward += 0.01  # Survival reward

        if self.ball_vel[1] < 0: # Ball moving away from paddle
            reward -= 0.02

        # 7. Check Termination
        if self.ball.top >= self.HEIGHT:
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True # Game won, but episode ends
            reward += 100.0 # Survival bonus
            self.score += 10000

        return self._get_observation(), reward, terminated, truncated, self._get_info()

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
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }
    
    def _render_game(self):
        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs["rect"], border_radius=5)

        # Particles
        for p in self.particles:
            if p['radius'] > 0:
                # FIX: Clamp the alpha value to the valid [0, 255] range for colors.
                # The original `int(p['life'] * 10)` could exceed 255, causing an error.
                alpha = min(255, int(p['life'] * 10))
                color_with_alpha = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]),
                    int(p['radius']), color_with_alpha
                )

        # Paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PADDLE, 50), glow_surface.get_rect(), border_radius=10)
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=8)

        # Ball with glow
        pygame.gfxdraw.filled_circle(self.screen, self.ball.centerx, self.ball.centery, self.BALL_RADIUS + 4, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.filled_circle(self.screen, self.ball.centerx, self.ball.centery, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, self.ball.centerx, self.ball.centery, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': np.array(pos, dtype=np.float32),
                'vel': np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * speed,
                'radius': self.np_random.uniform(3, 6),
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    # It is not used by the automated tests but is helpful for debugging.
    # To run, execute `python your_file_name.py`
    
    # --- Re-initialize Pygame without the dummy driver ---
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    pygame.quit() # Quit the dummy instance
    pygame.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Neon Paddle Survival")
    clock = pygame.time.Clock()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    print("--- Human Controls ---")
    print(env.user_guide)
    print("----------------------")

    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # The other actions (space, shift) are not used in this example
        # but are part of the action space.
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()