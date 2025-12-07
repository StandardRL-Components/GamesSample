import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- Fixes Start ---
    game_description = "Bounce a ball on moving platforms to climb as high as possible. Control the ball's horizontal impulse at each bounce to aim for the next platform."
    user_guide = "Controls: Use ←→ arrow keys to set horizontal direction for the next bounce. ↑ and ↓ keys adjust the strength of the bounce."
    auto_advance = True
    # --- Fixes End ---

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 3000  # 60 seconds at 50 steps/sec
    TARGET_SCORE = 1000

    # --- Colors ---
    COLOR_BG_TOP = (20, 30, 50)
    COLOR_BG_BOTTOM = (40, 60, 100)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (200, 200, 255)
    COLOR_PLATFORM = (120, 120, 140)
    COLOR_PLATFORM_TOP = (150, 150, 170)
    COLOR_UI_SCORE = (100, 255, 100)
    COLOR_PARTICLE = (200, 220, 255)

    # --- Physics & Game Parameters ---
    GRAVITY = 0.2
    BASE_BOUNCE_VELOCITY = 7.5
    BALL_RADIUS = 12
    HORIZONTAL_IMPULSE_INCREMENT = 0.4
    HORIZONTAL_IMPULSE_DECAY = 0.99
    MAX_HORIZONTAL_IMPULSE = 6.0
    PLATFORM_COUNT = 5
    PLATFORM_HEIGHT = 10
    PLATFORM_WIDTH = 100
    PLATFORM_OSCILLATION_PERIOD_STEPS = 250  # 5 seconds at 50 steps/sec

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)

        self.background = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            r = int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio)
            g = int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio)
            b = int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            pygame.draw.line(self.background, (r, g, b), (0, y), (self.WIDTH, y))

        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.platforms = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.last_score_milestone = 0
        self.game_over = False
        self.horizontal_impulse = 0.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.last_score_milestone = 0
        self.game_over = False

        self.ball_pos = np.array([self.WIDTH / 2, 80.0], dtype=np.float32)
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.horizontal_impulse = 0.0

        self.platforms = []
        for i in range(self.PLATFORM_COUNT):
            y_pos = self.HEIGHT * (i + 1.5) / (self.PLATFORM_COUNT + 1.5)
            
            # First platform is static and centered for a stable start
            if i == 0:
                x_pos = self.ball_pos[0] - self.PLATFORM_WIDTH / 2
                amplitude = 0
                phase_offset = 0
            else:
                x_pos = self.np_random.uniform(0, self.WIDTH - self.PLATFORM_WIDTH)
                amplitude = self.np_random.uniform(20, 50)
                phase_offset = self.np_random.uniform(0, self.PLATFORM_OSCILLATION_PERIOD_STEPS)

            self.platforms.append({
                "rect": pygame.Rect(
                    x_pos,
                    y_pos,
                    self.PLATFORM_WIDTH,
                    self.PLATFORM_HEIGHT
                ),
                "base_y": y_pos,
                "amplitude": amplitude,
                "phase_offset": phase_offset
            })
            
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        if not self.game_over:
            self._handle_input(action)
            self._update_platforms()
            bounce_reward, bounce_score = self._update_ball()
            
            reward += bounce_reward
            self.score += bounce_score

            if self.score // 100 > self.last_score_milestone:
                self.last_score_milestone = self.score // 100
                reward += 1.0

            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated: # Game over by falling or winning
            if self.score >= self.TARGET_SCORE:
                reward = 100.0
            else:
                reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]

        if movement == 1: # Up: Decrease magnitude
            self.horizontal_impulse *= 0.95
        elif movement == 2: # Down: Increase magnitude
            self.horizontal_impulse *= 1.05
        elif movement == 3: # Left
            self.horizontal_impulse -= self.HORIZONTAL_IMPULSE_INCREMENT
        elif movement == 4: # Right
            self.horizontal_impulse += self.HORIZONTAL_IMPULSE_INCREMENT
        
        self.horizontal_impulse = np.clip(
            self.horizontal_impulse, -self.MAX_HORIZONTAL_IMPULSE, self.MAX_HORIZONTAL_IMPULSE
        )

    def _update_platforms(self):
        for p in self.platforms:
            time_arg = 2 * math.pi * (self.steps + p["phase_offset"]) / self.PLATFORM_OSCILLATION_PERIOD_STEPS
            p["rect"].y = p["base_y"] + p["amplitude"] * math.sin(time_arg)

    def _update_ball(self):
        bounce_reward = 0.0
        bounce_score = 0.0

        self.ball_vel[1] += self.GRAVITY
        self.ball_vel[0] *= self.HORIZONTAL_IMPULSE_DECAY
        self.ball_pos += self.ball_vel

        if self.ball_vel[1] > 0:
            for p in self.platforms:
                platform_rect = p["rect"]
                ball_bottom = self.ball_pos[1] + self.BALL_RADIUS
                
                if platform_rect.collidepoint(self.ball_pos[0], ball_bottom) and \
                   platform_rect.top < ball_bottom < platform_rect.top + self.ball_vel[1]:
                    self.ball_pos[1] = platform_rect.top - self.BALL_RADIUS
                    self.ball_vel[1] = -self.BASE_BOUNCE_VELOCITY
                    self.ball_vel[0] = self.horizontal_impulse

                    bounce_reward = 0.1
                    bounce_score = max(0, (self.HEIGHT - platform_rect.y) / 10.0)
                    self._create_particles(self.ball_pos[0], platform_rect.top)
                    break

        if self.ball_pos[0] < self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -0.8
        elif self.ball_pos[0] > self.WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -0.8
        
        if self.ball_pos[1] < self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -0.5

        return bounce_reward, bounce_score

    def _check_termination(self):
        if self.ball_pos[1] > self.HEIGHT + self.BALL_RADIUS:
            self.game_over = True
        if self.score >= self.TARGET_SCORE:
            self.game_over = True
        
        return self.game_over

    def _create_particles(self, x, y):
        for _ in range(15):
            angle = self.np_random.uniform(0, math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, -math.sin(angle) * speed]
            life = self.np_random.integers(20, 40, endpoint=True)
            radius = self.np_random.uniform(1, 4)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': life, 'radius': radius})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += self.GRAVITY * 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 40.0))
            color = (*self.COLOR_PARTICLE, int(alpha))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color
            )

        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p["rect"])
            top_rect = pygame.Rect(p["rect"].x, p["rect"].y, p["rect"].width, 3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, top_rect)

        for i in range(4, 0, -1):
            alpha = 80 - i * 15
            radius = self.BALL_RADIUS + i * 2
            color = (*self.COLOR_BALL_GLOW, alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), radius, color
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), radius, color
            )

        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL
        )

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_SCORE)
        self.screen.blit(score_text, (10, 5))

        timer_ratio = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        timer_width = int(self.WIDTH * timer_ratio)
        timer_color = (
            int(255 * (1 - timer_ratio)),
            int(255 * timer_ratio),
            50
        )
        pygame.draw.rect(self.screen, timer_color, (0, 0, timer_width, 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The main loop requires a display, which is not available in the test environment.
    # To run this locally, you may need to comment out the os.environ line at the top.
    try:
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Bouncing Ball")
        clock = pygame.time.Clock()
        
        done = False
        total_reward = 0
        
        while not done:
            action = np.array([0, 0, 0])
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            elif keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(50)
            
        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    except pygame.error as e:
        print(f"Pygame display could not be initialized: {e}")
        print("This is expected in a headless environment. The GameEnv class is still functional.")

    env.close()