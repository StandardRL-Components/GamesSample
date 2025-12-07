import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame


# Set up headless Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls three bouncing balls.
    The goal is to synchronize their bounces to maximize a score multiplier
    and reach a target score before time runs out.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=boost ball 1, 2=boost ball 2, 3=boost ball 3, 4=unused)
    - action[1]: Space button (unused)
    - action[2]: Shift button (unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - an RGB image of the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control three bouncing balls and synchronize their bounces to maximize your score before time runs out."
    user_guide = "Controls: Use the ↑, ↓, and ← arrow keys to boost the first, second, and third balls respectively. Synchronize their bounces for a score multiplier."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds * 30 FPS
    WIN_SCORE = 1000

    # Colors
    COLOR_BG = (10, 10, 10)
    COLOR_GROUND = (51, 51, 51)
    COLOR_TEXT = (238, 238, 238)
    BALL_COLORS = [(255, 68, 68), (68, 255, 68), (68, 68, 255)] # Red, Green, Blue
    MULTIPLIER_COLORS = {
        1: (128, 128, 128),
        2: (100, 200, 100),
        3: (255, 255, 100),
        4: (255, 165, 0),
        5: (255, 0, 0)
    }

    # Physics
    GRAVITY = 0.4
    BOUNCE_DAMPENING = 0.98
    BOOST_STRENGTH = -10
    GROUND_Y = HEIGHT - 40
    BASE_RADIUS = 15

    # Multiplier thresholds (vertical pixel spread)
    MULTIPLIER_THRESHOLDS = {
        5: 15,   # x5
        4: 30,   # x4
        3: 50,   # x3
        2: 80,   # x2
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls = []
        self.particles = []
        self.multiplier = 1
        self.prev_multiplier = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.multiplier = 1
        self.prev_multiplier = 1
        self.particles.clear()

        self.balls = []
        for i in range(3):
            self.balls.append({
                'pos': pygame.Vector2(self.WIDTH * (i + 1) / 4, self.GROUND_Y - self.BASE_RADIUS - self.np_random.uniform(50, 150)),
                'vel_y': self.np_random.uniform(-2, 2),
                'color': self.BALL_COLORS[i],
                'trail': deque(maxlen=20),
                'radius_scale': 1.0,
                'scale_vel': 0.0,
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- ACTION HANDLING ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        if 1 <= movement <= 3:
            ball_idx = movement - 1
            self.balls[ball_idx]['vel_y'] += self.BOOST_STRENGTH
            self._spawn_boost_particles(self.balls[ball_idx]['pos'], self.balls[ball_idx]['color'])

        # --- GAME LOGIC & PHYSICS ---
        for ball in self.balls:
            # Apply gravity
            ball['vel_y'] += self.GRAVITY
            
            # Update position
            ball['pos'].y += ball['vel_y']

            # Bounce animation physics
            ball['scale_vel'] += (1.0 - ball['radius_scale']) * 0.1  # Spring back to 1.0
            ball['scale_vel'] *= 0.9  # Dampen
            ball['radius_scale'] += ball['scale_vel']

            # Ground collision
            radius = self.BASE_RADIUS * ball['radius_scale']
            if ball['pos'].y > self.GROUND_Y - radius:
                ball['pos'].y = self.GROUND_Y - radius
                ball['vel_y'] *= -self.BOUNCE_DAMPENING
                
                # Trigger bounce animation
                ball['radius_scale'] = 0.8
                ball['scale_vel'] = 0.5
                
                # Reward penalty for bouncing
                reward -= 2.0

            # Update trail
            ball['trail'].append(pygame.Vector2(ball['pos']))

        # Update particles
        self._update_particles()

        # --- SCORING & MULTIPLIER ---
        self.prev_multiplier = self.multiplier
        self._update_multiplier()
        
        # Add score based on current multiplier
        self.score += self.multiplier

        # --- REWARD CALCULATION ---
        reward += self._calculate_reward()

        # --- TERMINATION CHECK ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward -= 10

        truncated = False

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _calculate_reward(self):
        reward = 0
        
        # Continuous reward for synchronization
        ball_ys = [b['pos'].y for b in self.balls]
        spread = max(ball_ys) - min(ball_ys)
        if spread < self.MULTIPLIER_THRESHOLDS[5]:
             reward += 0.1

        # Event-based reward for increasing multiplier
        if self.multiplier > self.prev_multiplier:
            if self.multiplier == 2: reward += 1
            elif self.multiplier == 3: reward += 2
            elif self.multiplier == 4: reward += 3
            elif self.multiplier == 5: reward += 5
        
        return reward

    def _update_multiplier(self):
        ball_ys = [b['pos'].y for b in self.balls]
        spread = max(ball_ys) - min(ball_ys)

        self.multiplier = 1
        for mult, threshold in sorted(self.MULTIPLIER_THRESHOLDS.items(), key=lambda item: item[1]):
            if spread <= threshold:
                self.multiplier = mult
                break

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
            "multiplier": self.multiplier,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }
        
    def _render_game(self):
        # Draw ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 5)

        # Draw particles
        for p in self.particles:
            p_alpha = max(0, 255 * (p['life'] / p['max_life']))
            p_color = (*p['color'], p_alpha)
            p_surface = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(p_surface, p_color, (p['size'], p['size']), p['size'])
            self.screen.blit(p_surface, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # Draw balls and trails
        for ball in self.balls:
            # Draw trail
            for i, pos in enumerate(ball['trail']):
                alpha = int(100 * (i / len(ball['trail'])))
                radius = int(self.BASE_RADIUS * 0.3 * (i / len(ball['trail'])))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, (*ball['color'], alpha))
            
            # Draw ball with glow
            radius = int(self.BASE_RADIUS * ball['radius_scale'])
            glow_radius = int(radius * 1.8)
            glow_color = (*ball['color'], 60)
            
            # Using a surface for alpha blending the glow
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surface, glow_radius, glow_radius, glow_radius, glow_color)
            pygame.gfxdraw.aacircle(glow_surface, glow_radius, glow_radius, glow_radius, glow_color)
            self.screen.blit(glow_surface, (int(ball['pos'].x - glow_radius), int(ball['pos'].y - glow_radius)))
            
            # Draw main ball
            pygame.gfxdraw.filled_circle(self.screen, int(ball['pos'].x), int(ball['pos'].y), radius, ball['color'])
            pygame.gfxdraw.aacircle(self.screen, int(ball['pos'].x), int(ball['pos'].y), radius, ball['color'])

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_main.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 15, 10))

        # Multiplier
        mult_color = self.MULTIPLIER_COLORS[self.multiplier]
        mult_text_main = self.font_main.render("x", True, self.COLOR_TEXT)
        mult_text_val = self.font_main.render(f"{self.multiplier}", True, mult_color)
        
        mult_x_pos = 15
        self.screen.blit(mult_text_main, (mult_x_pos, 45))
        self.screen.blit(mult_text_val, (mult_x_pos + mult_text_main.get_width() + 2, 45))

    def _spawn_boost_particles(self, pos, color):
        for _ in range(15):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(1, 4)),
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Override the screen for direct display
    # Set the SDL video driver back to a real one
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bounce Sync")

    action = [0, 0, 0] # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Manual controls
        keys = pygame.key.get_pressed()
        action[0] = 0 # No-op
        if keys[pygame.K_1] or keys[pygame.K_UP]:
            action[0] = 1 # Boost ball 1
        elif keys[pygame.K_2] or keys[pygame.K_DOWN]:
            action[0] = 2 # Boost ball 2
        elif keys[pygame.K_3] or keys[pygame.K_LEFT]:
            action[0] = 3 # Boost ball 3

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to the display window
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

    env.close()