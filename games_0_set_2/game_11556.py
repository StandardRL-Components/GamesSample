import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
from gymnasium.spaces import MultiDiscrete
import os
import pygame


# Set SDL to dummy to run headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade-style Gymnasium environment where the player controls a recursively
    accelerating bouncing ball. The goal is to guide the ball into a target
    zone within a time limit. The ball's speed and size change every 10 bounces,
    creating an escalating challenge.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a recursively accelerating bouncing ball into a target zone. The ball's speed and size change with every 10 bounces, creating an escalating challenge."
    )
    user_guide = (
        "Controls: Use the ← and → arrow keys to adjust the ball's trajectory when it is on the ground."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30.0
    DT = 1.0 / TARGET_FPS

    # Colors
    COLOR_BG = (16, 16, 48)  # Dark Blue
    COLOR_BALL = (255, 51, 51)  # Bright Red
    COLOR_BALL_GLOW = (255, 100, 100)
    COLOR_TARGET = (51, 255, 51)  # Bright Green
    COLOR_TRAJECTORY = (255, 255, 153)  # Faint Yellow
    COLOR_TEXT = (255, 255, 255)  # White
    COLOR_SHOCKWAVE = (200, 200, 255)

    # Physics
    GRAVITY = 980.0
    BOUNCE_DAMPENING = 0.95
    HORIZONTAL_ADJUST_FORCE = 800.0
    MIN_BALL_RADIUS = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.time_remaining = None
        self.max_episode_steps = 1000
        self.game_over = None
        
        self.ball_pos = None
        self.ball_vel = None
        self.ball_radius = None
        self.bounce_count = None
        self.last_bounce_dist_to_target = None

        self.target_zone = pygame.Rect(
            self.SCREEN_WIDTH / 2 - 50, 20, 100, 40
        )

        self.particles = []
        self.shockwaves = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.time_remaining = 90.0
        self.game_over = False

        self.ball_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30)
        initial_vx = self.np_random.uniform(-150, 150)
        self.ball_vel = pygame.Vector2(initial_vx, -400)
        
        self.ball_radius = 20
        self.bounce_count = 0
        
        self.last_bounce_dist_to_target = self._get_dist_to_target()

        self.particles.clear()
        self.shockwaves.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        movement = action[0]

        self._handle_input(movement)
        
        bounce_reward = self._update_physics()
        reward += bounce_reward
        
        self._update_effects()

        self.steps += 1
        self.time_remaining -= self.DT

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward

        truncated = self.time_remaining <= 0 or self.steps >= self.max_episode_steps
        if truncated and not terminated:
            if self.time_remaining <= 0:
                reward -= 100.0 # Penalty for running out of time
        
        self.score += reward
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        # Player can only adjust trajectory when the ball is on/near the ground
        is_on_ground = self.ball_pos.y >= self.SCREEN_HEIGHT - self.ball_radius - 1
        
        if is_on_ground:
            if movement == 3:  # Left
                self.ball_vel.x -= self.HORIZONTAL_ADJUST_FORCE * self.DT
            elif movement == 4:  # Right
                self.ball_vel.x += self.HORIZONTAL_ADJUST_FORCE * self.DT

    def _update_physics(self):
        bounce_reward = 0.0

        # Apply gravity
        self.ball_vel.y += self.GRAVITY * self.DT
        
        # Update position
        self.ball_pos += self.ball_vel * self.DT

        # Wall bounces (left/right)
        if self.ball_pos.x - self.ball_radius < 0:
            self.ball_pos.x = self.ball_radius
            self.ball_vel.x *= -self.BOUNCE_DAMPENING
        elif self.ball_pos.x + self.ball_radius > self.SCREEN_WIDTH:
            self.ball_pos.x = self.SCREEN_WIDTH - self.ball_radius
            self.ball_vel.x *= -self.BOUNCE_DAMPENING

        # Floor bounce
        if self.ball_pos.y + self.ball_radius > self.SCREEN_HEIGHT:
            self.ball_pos.y = self.SCREEN_HEIGHT - self.ball_radius
            self.ball_vel.y *= -self.BOUNCE_DAMPENING
            
            self.bounce_count += 1
            self._create_shockwave(pygame.Vector2(self.ball_pos.x, self.SCREEN_HEIGHT))
            
            current_dist = self._get_dist_to_target()
            if current_dist < self.last_bounce_dist_to_target:
                bounce_reward += 0.1
            else:
                bounce_reward -= 0.1
            self.last_bounce_dist_to_target = current_dist

            if self.bounce_count > 0 and self.bounce_count % 10 == 0:
                self.ball_vel *= 2.0
                self.ball_radius = max(self.MIN_BALL_RADIUS, self.ball_radius / 2.0)
                bounce_reward += 1.0

        # Ceiling bounce
        if self.ball_pos.y - self.ball_radius < 0:
            self.ball_pos.y = self.ball_radius
            self.ball_vel.y *= -self.BOUNCE_DAMPENING
            
        return bounce_reward

    def _update_effects(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel'] * self.DT
            p['life'] -= self.DT * 2.0
            p['radius'] = max(0, p['life'] * p['start_radius'])
            if p['life'] <= 0:
                self.particles.remove(p)

        # Create new particles
        if self.np_random.random() < 0.8: # Create particles frequently
            p_vel = -self.ball_vel.normalize() * self.np_random.uniform(20, 40) + pygame.Vector2(self.np_random.uniform(-10, 10), self.np_random.uniform(-10, 10))
            start_rad = self.ball_radius * 0.5
            self.particles.append({
                'pos': self.ball_pos.copy(),
                'vel': p_vel,
                'life': 1.0,
                'start_radius': start_rad,
                'radius': start_rad,  # Initialize radius to prevent KeyError
                'color': self.COLOR_BALL_GLOW
            })
            
        # Update shockwaves
        for sw in self.shockwaves[:]:
            sw['life'] -= self.DT * 3.0
            sw['radius'] += 300 * self.DT
            if sw['life'] <= 0:
                self.shockwaves.remove(sw)

    def _check_termination(self):
        # Victory condition
        if self.target_zone.collidepoint(self.ball_pos):
            return True, 100.0
            
        return False, 0.0

    def _get_observation(self):
        # Background
        self.screen.fill(self.COLOR_BG)

        # Game elements
        self._render_game()

        # UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw target zone
        pygame.draw.rect(self.screen, self.COLOR_TARGET, self.target_zone, border_radius=5)
        
        # Draw shockwaves
        for sw in self.shockwaves:
            alpha = max(0, min(255, int(sw['life'] * 255)))
            if alpha > 0:
                pygame.gfxdraw.aacircle(
                    self.screen, int(sw['pos'].x), int(sw['pos'].y), int(sw['radius']),
                    (self.COLOR_SHOCKWAVE[0], self.COLOR_SHOCKWAVE[1], self.COLOR_SHOCKWAVE[2], alpha)
                )

        # Draw particles
        for p in self.particles:
            if p['radius'] > 1:
                self._draw_aa_circle(
                    self.screen, p['pos'], p['radius'], p['color']
                )

        # Draw trajectory prediction line (if on ground)
        if self.ball_pos.y >= self.SCREEN_HEIGHT - self.ball_radius - 1:
            self._draw_trajectory_prediction()

        # Draw ball with glow
        glow_radius = int(self.ball_radius * 2.5)
        for i in range(glow_radius, int(self.ball_radius), -2):
            alpha = 30 * (1 - (i - self.ball_radius) / (glow_radius - self.ball_radius))
            color = (*self.COLOR_BALL_GLOW, int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), i, color)
        
        self._draw_aa_circle(self.screen, self.ball_pos, self.ball_radius, self.COLOR_BALL)

    def _render_ui(self):
        # Timer
        timer_text = f"{max(0, self.time_remaining):.1f}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 10))

        # Score
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Bounce Count
        bounce_text = f"Bounces: {self.bounce_count}"
        bounce_surf = self.font_small.render(bounce_text, True, self.COLOR_TEXT)
        self.screen.blit(bounce_surf, (20, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "bounce_count": self.bounce_count,
            "ball_speed": self.ball_vel.magnitude()
        }

    def _get_dist_to_target(self):
        return self.ball_pos.distance_to(self.target_zone.center)

    def _create_shockwave(self, pos):
        self.shockwaves.append({
            'pos': pos,
            'radius': 0,
            'life': 1.0
        })

    def _draw_aa_circle(self, surface, pos, radius, color):
        x, y = int(pos.x), int(pos.y)
        rad = int(radius)
        if rad > 0:
            pygame.gfxdraw.filled_circle(surface, x, y, rad, color)
            pygame.gfxdraw.aacircle(surface, x, y, rad, color)

    def _draw_trajectory_prediction(self):
        temp_pos = self.ball_pos.copy()
        temp_vel = self.ball_vel.copy()
        points = []
        for _ in range(30): # Predict 1 second into the future
            temp_vel.y += self.GRAVITY * self.DT
            temp_pos += temp_vel * self.DT
            if temp_pos.y > self.SCREEN_HEIGHT: break
            points.append((int(temp_pos.x), int(temp_pos.y)))
        
        if len(points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TRAJECTORY, False, points)

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows for human play testing of the environment.
    # Use Left/Right arrow keys to control the ball on bounce.
    
    # Un-set the dummy driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Recursive Bouncing Ball")
    
    terminated = False
    truncated = False
    running = True
    
    while running:
        action = [0, 0, 0]  # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back to a Surface for display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(GameEnv.TARGET_FPS)
        
    env.close()