import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:02:00.369101
# Source Brief: brief_01472.md
# Brief Index: 1472
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

# Helper class for a Ball
class Ball:
    """Represents a single bouncing ball with position, velocity, and appearance."""
    def __init__(self, x, y, radius, color, np_random: np.random.Generator):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(np_random.uniform(-1, 1), 0)
        self.radius = radius
        self.color = color

# Helper class for a Particle
class Particle:
    """Represents a short-lived particle for visual effects."""
    def __init__(self, x, y, color, np_random: np.random.Generator):
        self.pos = pygame.Vector2(x, y)
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        self.lifespan = np_random.integers(20, 41)
        self.radius = self.lifespan / 8
        self.color = color

    def update(self):
        """Updates particle position and lifespan."""
        self.pos += self.vel
        self.lifespan -= 1
        self.radius = max(0, self.lifespan / 8)

    def is_alive(self):
        """Checks if the particle's lifespan has expired."""
        return self.lifespan > 0

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls 20 bouncing balls
    to hit ascending targets within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a swarm of bouncing balls and make them hit an ascending target line before time runs out."
    )
    user_guide = (
        "Press Space for a low bounce and Shift for a high bounce to get all balls above the target line."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BALL_COUNT = 20
    BALL_RADIUS = 7
    NUM_LEVELS = 5
    MAX_STEPS = 3600  # 60 seconds at 60 FPS

    # Physics
    GRAVITY = 0.25
    FLOOR_Y = 385
    WALL_DAMPING = -0.9
    LOW_BOUNCE_VEL = 9
    HIGH_BOUNCE_VEL = 16

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_BALL = (0, 255, 255)  # Cyan
    COLOR_TARGET = (255, 0, 255) # Magenta
    COLOR_TEXT = (50, 255, 50)  # Bright Green
    COLOR_GRID = (20, 40, 80)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- GYM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('monospace', 20, bold=True)
        self.font_score = pygame.font.SysFont('monospace', 30, bold=True)
        
        # --- GAME STATE ---
        self.balls = []
        self.particles = []
        self.hit_balls_this_level = set()
        self.target_y = 0
        self.initial_target_y = self.SCREEN_HEIGHT * 0.7
        self.steps = 0
        self.score = 0
        self.level = 0
        self.game_over = False
        self.last_bounce_effect_time = -100 # Initialize to a long-past time

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.target_y = self.initial_target_y
        
        self.balls.clear()
        self.particles.clear()
        self.hit_balls_this_level.clear()

        for i in range(self.BALL_COUNT):
            x = self.np_random.uniform(self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            self.balls.append(Ball(x, self.FLOOR_Y, self.BALL_RADIUS, self.COLOR_BALL, self.np_random))
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        reward = self._update_game_state(action)
        
        terminated = self._check_termination()
        truncated = False # This environment does not truncate based on steps, only terminates
        
        if terminated:
            self.game_over = True
            if self.level > self.NUM_LEVELS:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Timeout penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
        
    def _update_game_state(self, action):
        # Unpack factorized action
        space_held = action[1] == 1
        shift_held = action[2] == 1

        self._handle_actions(space_held, shift_held)
        self._update_physics()
        
        new_hits = self._check_collisions()
        reward = self._calculate_reward(new_hits)
        
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update()
        
        if len(self.hit_balls_this_level) == self.BALL_COUNT:
            # sfx: level_complete.wav
            self.level += 1
            self.score += self.level * 10
            if self.level <= self.NUM_LEVELS:
                self.target_y *= 0.9 # Target moves up
                self.hit_balls_this_level.clear()
                for _ in range(50):
                    px = self.np_random.uniform(0, self.SCREEN_WIDTH)
                    self.particles.append(Particle(px, self.target_y, self.COLOR_TARGET, self.np_random))
        return reward

    def _handle_actions(self, space_held, shift_held):
        bounce_vel = 0
        if shift_held:
            bounce_vel = self.HIGH_BOUNCE_VEL
        elif space_held:
            bounce_vel = self.LOW_BOUNCE_VEL

        if bounce_vel > 0:
            bounced_this_frame = False
            for ball in self.balls:
                if ball.pos.y >= self.FLOOR_Y - 1:
                    ball.vel.y = -bounce_vel
                    ball.vel.x += self.np_random.uniform(-0.3, 0.3)
                    ball.pos.y = self.FLOOR_Y - 1
                    bounced_this_frame = True
            
            if bounced_this_frame:
                # sfx: bounce.wav
                self.last_bounce_effect_time = self.steps

    def _update_physics(self):
        for ball in self.balls:
            ball.vel.y += self.GRAVITY
            ball.pos += ball.vel

            if ball.pos.y >= self.FLOOR_Y:
                ball.pos.y = self.FLOOR_Y
                ball.vel.y = 0
            
            if ball.pos.y <= ball.radius:
                ball.pos.y = ball.radius
                ball.vel.y *= self.WALL_DAMPING

            if ball.pos.x <= ball.radius or ball.pos.x >= self.SCREEN_WIDTH - ball.radius:
                ball.pos.x = np.clip(ball.pos.x, ball.radius, self.SCREEN_WIDTH - ball.radius)
                ball.vel.x *= self.WALL_DAMPING
            
            ball.vel.x *= 0.998

    def _check_collisions(self):
        new_hits = 0
        for i, ball in enumerate(self.balls):
            if i not in self.hit_balls_this_level:
                is_above = ball.pos.y - ball.radius < self.target_y
                was_below = (ball.pos - ball.vel).y - ball.radius >= self.target_y
                moving_up = ball.vel.y < 0
                
                if is_above and was_below and moving_up:
                    self.hit_balls_this_level.add(i)
                    self.score += 1
                    new_hits += 1
                    # sfx: target_hit.wav
                    for _ in range(15):
                        self.particles.append(Particle(ball.pos.x, self.target_y, self.COLOR_TARGET, self.np_random))
        return new_hits

    def _calculate_reward(self, new_hits):
        reward = 0
        reward += new_hits * 1.0
        
        proximity_reward_zone = 50
        for ball in self.balls:
            if abs(ball.pos.y - self.target_y) < proximity_reward_zone:
                reward += 0.01

        return reward

    def _check_termination(self):
        if self.level > self.NUM_LEVELS:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_left": (self.MAX_STEPS - self.steps) / 60.0
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_background_grid()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_background_grid(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_game(self):
        if self.steps - self.last_bounce_effect_time < 10:
            alpha = 100 * (1 - (self.steps - self.last_bounce_effect_time) / 10)
            s = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
            s.fill((255, 255, 255, alpha))
            self.screen.blit(s, (0, self.FLOOR_Y - 40))

        target_y_int = int(self.target_y)
        for i in range(5, 0, -1):
            alpha = int(150 - i * 25)
            color = (*self.COLOR_TARGET, alpha)
            pygame.gfxdraw.hline(self.screen, 0, self.SCREEN_WIDTH, target_y_int - i, color)
            pygame.gfxdraw.hline(self.screen, 0, self.SCREEN_WIDTH, target_y_int + i, color)
        pygame.gfxdraw.hline(self.screen, 0, self.SCREEN_WIDTH, target_y_int, self.COLOR_TARGET)

        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), p.color)

        for ball in self.balls:
            x, y = int(ball.pos.x), int(ball.pos.y)
            for i in range(int(ball.radius * 1.5), int(ball.radius), -1):
                alpha = int(50 * (1 - (i - ball.radius) / (ball.radius * 0.5)))
                pygame.gfxdraw.filled_circle(self.screen, x, y, i, (*self.COLOR_BALL, alpha))
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(ball.radius), self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, y, int(ball.radius), self.COLOR_BALL)

    def _render_ui(self):
        level_text = self.font_ui.render(f"LEVEL: {self.level}/{self.NUM_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / 60)
        time_text = self.font_ui.render(f"TIME: {time_left:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        score_text = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.FLOOR_Y + (self.SCREEN_HEIGHT - self.FLOOR_Y)/2 + 5))
        self.screen.blit(score_text, score_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not used by the tests
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bounce Mania")
    
    total_reward = 0
    running = True
    while running:
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        # The action space is [5, 2, 2], but this human-play script only uses the last two parts.
        # action[0] is for directional input, which is not used in this game's logic.
        action = [0, 1 if space_held else 0, 1 if shift_held else 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(frame_surface, (0, 0))
        
        if done:
            font_game_over = pygame.font.SysFont('monospace', 50, bold=True)
            win_status = "YOU WIN!" if info['level'] > GameEnv.NUM_LEVELS else "TIME'S UP!"
            msg_text = font_game_over.render(win_status, True, (255, 255, 0))
            msg_rect = msg_text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 20))
            display_screen.blit(msg_text, msg_rect)
            
            font_restart = pygame.font.SysFont('monospace', 20, bold=True)
            restart_text = font_restart.render("Press 'R' to Restart", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 30))
            display_screen.blit(restart_text, restart_rect)

        pygame.display.flip()
        env.clock.tick(60)

    env.close()