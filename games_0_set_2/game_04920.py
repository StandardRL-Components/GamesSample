import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Helper classes for game objects to keep the main class cleaner
class Ball:
    def __init__(self, pos, vel, color, radius):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.radius = radius

    def update(self):
        self.pos += self.vel

    def draw(self, surface):
        # Draw a soft glow effect
        glow_radius = int(self.radius * 1.5)
        glow_alpha = 60
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.color + (glow_alpha,), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw the main ball with anti-aliasing
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)

class Particle:
    def __init__(self, pos, vel, color, start_radius, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.radius = start_radius
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius = max(0, self.radius * (self.lifespan / self.initial_lifespan))

    def draw(self, surface):
        if self.radius > 1:
            pygame.draw.circle(surface, self.color, self.pos, int(self.radius))

class FloatingText:
    def __init__(self, pos, text, color, font, lifespan=45):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, -0.5)
        self.text = text
        self.color = color
        self.font = font
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1

    def draw(self, surface):
        alpha = int(255 * (self.lifespan / self.initial_lifespan))
        if alpha > 0:
            text_surf = self.font.render(self.text, True, self.color)
            text_surf.set_alpha(alpha)
            surface.blit(text_surf, self.pos)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑ and ↓ to move the paddle. Hit the colored balls to score points."
    )

    game_description = (
        "A retro-inspired arcade game. Score 5 points for both red and blue by hitting the balls with your paddle. Missing 10 balls ends the game."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 5
        self.MAX_MISSES = 10
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 10, 60
        self.PADDLE_SPEED = 6
        self.BALL_RADIUS = 8
        self.NUM_BALLS_ON_SCREEN = 2

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_RED = (255, 80, 80)
        self.COLOR_BLUE = (80, 120, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SCORE_RED = (255, 120, 120)
        self.COLOR_SCORE_BLUE = (120, 160, 255)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_float = pygame.font.SysFont("Verdana", 16, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.balls = []
        self.particles = []
        self.floating_texts = []
        self.steps = 0
        self.score_red = 0
        self.score_blue = 0
        self.missed_balls = 0
        self.ball_speed = 0.0
        
        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle = pygame.Rect(30, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.balls = []
        self.particles = []
        self.floating_texts = []
        
        self.steps = 0
        self.score_red = 0
        self.score_blue = 0
        self.missed_balls = 0
        
        self.ball_speed = 3.0
        
        for _ in range(self.NUM_BALLS_ON_SCREEN):
            self._spawn_ball()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        movement = action[0]
        
        self._update_paddle(movement)
        
        hit_events = self._update_balls()
        self._update_particles()
        self._update_floating_texts()
        
        self._spawn_ball_if_needed()
        self._update_difficulty()

        reward = self._calculate_reward(hit_events)
        
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated:
            if self.score_red >= self.WIN_SCORE and self.score_blue >= self.WIN_SCORE:
                reward += 100  # Win reward
            else:
                reward -= 100  # Loss reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_paddle(self, movement):
        if movement == 1:  # Up
            self.paddle.y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle.y += self.PADDLE_SPEED
        
        self.paddle.y = np.clip(self.paddle.y, 0, self.HEIGHT - self.PADDLE_HEIGHT)

    def _update_balls(self):
        hit_events = []
        balls_to_remove = []

        for ball in self.balls:
            ball.update()

            # Wall bounces
            if ball.pos.y - ball.radius < 0 or ball.pos.y + ball.radius > self.HEIGHT:
                ball.vel.y *= -1
                ball.pos.y = np.clip(ball.pos.y, ball.radius, self.HEIGHT - ball.radius)
            if ball.pos.x + ball.radius > self.WIDTH:
                ball.vel.x *= -1
                ball.pos.x = self.WIDTH - ball.radius

            # Paddle miss
            if ball.pos.x - ball.radius < 0:
                self.missed_balls += 1
                balls_to_remove.append(ball)
                self._create_floating_text(f"Miss!", (self.paddle.centerx + 20, self.paddle.centery), self.COLOR_RED)
                continue

            # Paddle hit
            if self.paddle.collidepoint(ball.pos.x - ball.radius, ball.pos.y) and ball.vel.x < 0:
                ball.vel.x *= -1
                
                hit_offset = (ball.pos.y - self.paddle.centery) / (self.PADDLE_HEIGHT / 2)
                ball.vel.y += hit_offset * 2.0
                ball.vel.normalize_ip()
                ball.vel *= self.ball_speed

                ball.pos.x = self.paddle.right + ball.radius
                
                hit_color_str = 'red' if ball.color == self.COLOR_RED else 'blue'
                hit_events.append({'color': hit_color_str, 'pos': ball.pos.copy()})
                
                if hit_color_str == 'red':
                    self.score_red = min(self.WIN_SCORE, self.score_red + 1)
                else:
                    self.score_blue = min(self.WIN_SCORE, self.score_blue + 1)
                
                self._create_hit_particles(ball.pos, ball.color)

        self.balls = [b for b in self.balls if b not in balls_to_remove]
        return hit_events

    def _calculate_reward(self, hit_events):
        reward = 0
        if not hit_events:
            reward -= 0.01  # Small penalty for not hitting a ball
        
        for hit in hit_events:
            reward += 0.1  # Base reward for any hit
            
            is_red_hit = hit['color'] == 'red'
            bonus_reward = 0
            if is_red_hit and self.score_red <= self.score_blue:
                bonus_reward = 1.0
            elif not is_red_hit and self.score_blue <= self.score_red:
                bonus_reward = 1.0
            
            if bonus_reward > 0:
                reward += bonus_reward
                self._create_floating_text(f"+{bonus_reward:.1f}", hit['pos'], (255, 255, 0))

        return reward

    def _check_termination(self):
        win = self.score_red >= self.WIN_SCORE and self.score_blue >= self.WIN_SCORE
        loss = self.missed_balls >= self.MAX_MISSES
        return win or loss

    def _spawn_ball_if_needed(self):
        while len(self.balls) < self.NUM_BALLS_ON_SCREEN:
            self._spawn_ball()
            
    def _spawn_ball(self):
        pos = (self.WIDTH / 2, self.np_random.uniform(self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS))
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        if self.np_random.random() > 0.5:
             angle += math.pi
        
        vel = (math.cos(angle) * self.ball_speed, math.sin(angle) * self.ball_speed)
        color = self.COLOR_RED if self.np_random.random() > 0.5 else self.COLOR_BLUE
        
        self.balls.append(Ball(pos, vel, color, self.BALL_RADIUS))
        
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.ball_speed = min(5.0, self.ball_speed + 0.1)
            for ball in self.balls:
                ball.vel.normalize_ip()
                ball.vel *= self.ball_speed

    def _create_hit_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            start_radius = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos, vel, color, start_radius, lifespan))

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _create_floating_text(self, text, pos, color):
        self.floating_texts.append(FloatingText(pos, text, color, self.font_float))

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft.update()
        self.floating_texts = [ft for ft in self.floating_texts if ft.lifespan > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score_red": self.score_red,
            "score_blue": self.score_blue,
            "score": self.score_red + self.score_blue,
            "steps": self.steps,
            "missed_balls": self.missed_balls,
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw paddle with rounded corners and glow
        paddle_color_glow = self.COLOR_PADDLE + (50,)
        glow_rect = self.paddle.inflate(6, 6)
        
        # Create a temporary surface for the glow effect
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, paddle_color_glow, glow_surf.get_rect(), border_radius=5)
        
        # Blit the glow surface onto the main screen using additive blending
        self.screen.blit(glow_surf, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw the main paddle on top
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw balls
        for ball in self.balls:
            ball.draw(self.screen)

    def _render_ui(self):
        # Draw scores
        red_text = f"Red: {self.score_red}/{self.WIN_SCORE}"
        blue_text = f"Blue: {self.score_blue}/{self.WIN_SCORE}"
        
        red_surf = self.font_ui.render(red_text, True, self.COLOR_SCORE_RED)
        blue_surf = self.font_ui.render(blue_text, True, self.COLOR_SCORE_BLUE)
        
        self.screen.blit(red_surf, (10, 5))
        self.screen.blit(blue_surf, (self.WIDTH - blue_surf.get_width() - 10, 5))
        
        # Draw misses
        miss_text = f"Misses: {self.missed_balls}/{self.MAX_MISSES}"
        miss_surf = self.font_ui.render(miss_text, True, self.COLOR_TEXT)
        self.screen.blit(miss_surf, (self.WIDTH // 2 - miss_surf.get_width() // 2, 5))

        # Draw floating texts
        for ft in self.floating_texts:
            ft.draw(self.screen)

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful. Initial info:", info)
    
    # Run a few random steps
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if (i + 1) % 20 == 0:
            print(f"Step {i+1}: Reward={reward:.2f}, Info={info}, Terminated={terminated}, Truncated={truncated}")
        if terminated or truncated:
            print("Episode finished. Resetting.")
            env.reset()

    env.close()
    print("Environment closed.")