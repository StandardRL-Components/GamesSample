import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:58:39.426787
# Source Brief: brief_02025.md
# Brief Index: 2025
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gravity Juggler Environment.

    The player controls the gravity affecting five bouncing balls. The goal is to
    synchronize their bounces on the floor for points while preventing any ball
    from falling off the bottom of the screen.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Selects which ball to affect (0, 1, 2, 3, or 4).
    - action[1]: If 1, sets the selected ball's gravity to "UP".
    - action[2]: If 1, sets the selected ball's gravity to "DOWN".
      (If both are 1, "DOWN" takes precedence).

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.1 per frame for survival.
    - +5 for a synchronized bounce (two balls bounce within 3 frames).
    - -2 for an asynchronous bounce.
    - +100 for surviving the full 45 seconds.
    - -100 if any ball falls off the screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control the gravity affecting five bouncing balls. Synchronize their bounces for points "
        "while preventing any ball from falling off the screen."
    )
    user_guide = (
        "Controls: Use keys 1-5 to select a ball. Hold space to apply upward gravity "
        "or shift to apply downward gravity."
    )
    auto_advance = True

    # --- Ball helper class ---
    class Ball:
        def __init__(self, pos, vel, color, radius):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.color = color
            self.radius = radius
            self.gravity_dir = -1  # -1 for down (default), 1 for up
            self.is_targeted = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 45 * self.FPS  # 2700 steps for a 45-second game

        # --- Visuals & Style ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_LINE = (220, 220, 220)
        self.BALL_COLORS = [
            (255, 80, 80),   # Bright Red
            (80, 255, 80),   # Bright Green
            (80, 120, 255),  # Bright Blue
            (255, 255, 80),  # Bright Yellow
            (200, 80, 255)   # Bright Purple
        ]

        # --- Physics & Mechanics ---
        self.BALL_RADIUS = 15
        self.GRAVITY_DOWN = 0.2
        self.GRAVITY_UP = -0.1  # Counteracts some of the downward gravity
        self.BOUNCE_LINE_Y = self.HEIGHT - 10
        self.ELASTICITY = -0.95  # Energy retained on bounce
        self.SYNC_WINDOW = 3  # Frames within which bounces are "synchronized"

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
        try:
            self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font = pygame.font.SysFont(None, 30) # Fallback font

        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.balls = []
        self.last_bounce_frames = {}
        self.targeted_ball_idx = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.balls = []
        self.last_bounce_frames = {}
        self.targeted_ball_idx = -1

        for i in range(5):
            # Spawn in upper half to avoid immediate loss
            x = self.np_random.uniform(self.BALL_RADIUS + 20, self.WIDTH - self.BALL_RADIUS - 20)
            y = self.np_random.uniform(self.BALL_RADIUS + 20, self.HEIGHT / 2)
            vx = self.np_random.uniform(-2, 2)
            vy = self.np_random.uniform(-1, 1)

            ball = self.Ball(
                pos=(x, y),
                vel=(vx, vy),
                color=self.BALL_COLORS[i],
                radius=self.BALL_RADIUS
            )
            self.balls.append(ball)
            # Initialize bounce times to a long time ago
            self.last_bounce_frames[i] = -self.SYNC_WINDOW * 2

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self.reset()

        self.steps += 1
        reward = 0.0

        # 1. Unpack and process actions
        ball_selection_idx = action[0]
        set_gravity_up = action[1] == 1
        set_gravity_down = action[2] == 1

        self.targeted_ball_idx = ball_selection_idx

        # Set gravity. Shift (down) overrides Space (up).
        if set_gravity_down:
            if self.balls[ball_selection_idx].gravity_dir != -1:
                # Sfx: Gravity shift down sound
                self.balls[ball_selection_idx].gravity_dir = -1
        elif set_gravity_up:
            if self.balls[ball_selection_idx].gravity_dir != 1:
                # Sfx: Gravity shift up sound
                self.balls[ball_selection_idx].gravity_dir = 1

        # 2. Update physics and check for bounces
        bounce_event_reward = 0
        bounced_this_frame = []

        for i, ball in enumerate(self.balls):
            gravity_force = self.GRAVITY_DOWN if ball.gravity_dir == -1 else self.GRAVITY_UP
            ball.vel.y += gravity_force
            ball.pos += ball.vel

            # Wall collisions
            if ball.pos.x - ball.radius < 0 or ball.pos.x + ball.radius > self.WIDTH:
                ball.pos.x = np.clip(ball.pos.x, ball.radius, self.WIDTH - ball.radius)
                ball.vel.x *= self.ELASTICITY
            if ball.pos.y - ball.radius < 0:
                ball.pos.y = ball.radius
                ball.vel.y *= self.ELASTICITY

            # Floor bounce
            if ball.pos.y + ball.radius >= self.BOUNCE_LINE_Y and ball.vel.y > 0:
                ball.pos.y = self.BOUNCE_LINE_Y - ball.radius
                ball.vel.y *= self.ELASTICITY
                # Sfx: Ball bounce sound
                bounced_this_frame.append(i)

        # 3. Calculate rewards from bounce events
        for ball_idx in bounced_this_frame:
            is_synced = False
            for other_idx, last_bounce in self.last_bounce_frames.items():
                if ball_idx != other_idx and abs(self.steps - last_bounce) <= self.SYNC_WINDOW:
                    is_synced = True
                    break
            
            if is_synced:
                bounce_event_reward += 5.0
                # Sfx: Positive sync chime
            else:
                bounce_event_reward -= 2.0
                # Sfx: Negative async buzz
            
            self.last_bounce_frames[ball_idx] = self.steps

        self.score += bounce_event_reward
        reward += bounce_event_reward

        # 4. Check termination conditions
        terminated = False
        truncated = False
        for ball in self.balls:
            if ball.pos.y + ball.radius > self.HEIGHT:
                terminated = True
                self.game_over = True
                reward = -100.0  # Loss penalty
                # Sfx: Game over failure sound
                break
        
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True # In Gymnasium, time limit is a termination, not truncation for this setup
            self.game_over = True
            reward = 100.0  # Win bonus
            self.score += 100.0 # Add to score for display
            # Sfx: Game win fanfare

        if not terminated:
            reward += 0.1  # Survival reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_LINE, (0, self.BOUNCE_LINE_Y, self.WIDTH, self.HEIGHT - self.BOUNCE_LINE_Y))

        for i, ball in enumerate(self.balls):
            # Glow effect for visual appeal
            glow_radius = int(ball.radius * 1.8)
            alpha = 100 if i == self.targeted_ball_idx else 50
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*ball.color, alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (int(ball.pos.x - glow_radius), int(ball.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

            # Main ball body (anti-aliased)
            pygame.gfxdraw.aacircle(self.screen, int(ball.pos.x), int(ball.pos.y), ball.radius, ball.color)
            pygame.gfxdraw.filled_circle(self.screen, int(ball.pos.x), int(ball.pos.y), ball.radius, ball.color)

            # Gravity indicator arrow
            arrow_y = ball.pos.y - ball.radius - 10
            arrow_color = (200, 200, 200) if ball.gravity_dir == 1 else (100, 100, 100)
            if ball.gravity_dir == 1:  # Up arrow
                points = [(ball.pos.x, arrow_y - 5), (ball.pos.x - 5, arrow_y + 5), (ball.pos.x + 5, arrow_y + 5)]
            else:  # Down arrow
                points = [(ball.pos.x, arrow_y + 5), (ball.pos.x - 5, arrow_y - 5), (ball.pos.x + 5, arrow_y - 5)]
            pygame.draw.polygon(self.screen, arrow_color, points)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_LINE)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font.render(f"TIME: {time_left:.2f}", True, self.COLOR_LINE)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy videodriver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gravity Juggler")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # --- Manual Control Mapping ---
    # Keys 1-5 to select a ball
    # Space to set gravity UP
    # Shift to set gravity DOWN
    
    selected_ball = 0
    
    while not terminated:
        action = [selected_ball, 0, 0] # Default action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: selected_ball = 0
                if event.key == pygame.K_2: selected_ball = 1
                if event.key == pygame.K_3: selected_ball = 2
                if event.key == pygame.K_4: selected_ball = 3
                if event.key == pygame.K_5: selected_ball = 4
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        action[0] = selected_ball
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()