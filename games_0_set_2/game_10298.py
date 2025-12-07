import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball,
    dodging oscillating neon platforms to reach the bottom of the screen.

    **Visuals:**
    - Minimalist, neon-inspired aesthetic.
    - Glowing ball and platforms against a dark background.
    - Particle effects for bounces and collisions.

    **Gameplay:**
    - The player controls the horizontal acceleration of a ball.
    - The ball is subject to gravity.
    - The goal is to navigate past a series of horizontally oscillating platforms
      that fade in and out of existence.
    - Reaching the bottom of the screen constitutes a win for the level.
    - Colliding with a platform or running out of time is a loss.
    - The number of platforms increases with each successful level.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Movement (3=left, 4=right, others=no-op)
    - `actions[1]`: Unused
    - `actions[2]`: Unused

    **Reward Structure:**
    - +0.01 per step for survival.
    - +100 for each platform successfully avoided (when it fades out).
    - +1000 for reaching the bottom of the screen.
    - -100 for colliding with a platform or running out of time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control a bouncing ball to dodge oscillating neon platforms and reach the bottom of the screen."
    user_guide = "Use the ← and → arrow keys to apply horizontal force to the ball."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        self.np_random = None

        # --- Visuals & Style ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_BALL = (57, 255, 20)
        self.COLOR_PLATFORM = (255, 20, 57)
        self.COLOR_WALL = (180, 180, 190)
        self.COLOR_TEXT = (240, 240, 240)
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)

        # --- Game Constants ---
        self.FPS = 60
        self.MAX_TIME_SECONDS = 60
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        self.GRAVITY = 0.08
        self.BALL_RADIUS = 12
        self.BALL_ACCEL = 0.25
        self.FRICTION = 0.985
        self.WALL_PADDING = 10
        self.PLATFORM_HEIGHT = 12
        self.PLATFORM_CYCLE_SECONDS = 3

        # --- State Variables ---
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.platforms = []
        self.particles = []
        self.level = 1
        self.score = 0
        self.steps = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             # Seed the random number generator for reproducibility
             random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Reset ball state
        self.ball_pos = pygame.Vector2(self.WIDTH / 2, self.BALL_RADIUS * 3)
        self.ball_vel = pygame.Vector2(random.uniform(-1, 1), 0)

        # Clear dynamic elements
        self.particles.clear()
        self._generate_platforms()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Survival reward

        self._handle_input(action)
        self._update_ball()
        reward += self._update_platforms()
        self._update_particles()

        self.steps += 1
        terminated = False
        truncated = False

        # Check for collision
        if self._check_collisions():
            self._create_explosion(self.ball_pos, self.COLOR_BALL)
            reward = -100
            terminated = True
            self.level = 1 # Reset level on failure

        # Check for win condition
        if self.ball_pos.y > self.HEIGHT + self.BALL_RADIUS:
            reward += 1000
            terminated = True
            self.level += 1 # Progress to next level

        # Check for timeout
        if self.steps >= self.MAX_STEPS:
            reward = -100
            truncated = True # Use truncated for time limit
            self.level = 1 # Reset level on failure
        
        if terminated or truncated:
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _generate_platforms(self):
        self.platforms.clear()
        num_platforms = 2 + (self.level - 1) * 2
        
        # Start platforms lower to avoid immediate termination in stability tests
        platform_y_start = 200
        platform_y_end = self.HEIGHT - 40
        available_space = platform_y_end - platform_y_start
        vertical_spacing = available_space / max(1, num_platforms)

        for i in range(num_platforms):
            y_pos = platform_y_start + i * vertical_spacing
            width = random.randint(80, 150)
            center_x = random.randint(self.WALL_PADDING + width, self.WIDTH - self.WALL_PADDING - width)
            oscillation_range = random.randint(50, 150)
            
            self.platforms.append({
                'rect': pygame.Rect(center_x - width / 2, y_pos, width, self.PLATFORM_HEIGHT),
                'center_x': center_x,
                'oscillation_range': oscillation_range,
                'phase': random.uniform(0, 2 * math.pi),
                'phase_speed': random.uniform(0.01, 0.02),
                'timer': random.randint(0, self.PLATFORM_CYCLE_SECONDS * self.FPS),
                'cycle_duration': self.PLATFORM_CYCLE_SECONDS * self.FPS,
                'is_visible': True,
                'alpha': 0,
                'reward_claimed': False
            })

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.ball_vel.x -= self.BALL_ACCEL
        elif movement == 4:  # Right
            self.ball_vel.x += self.BALL_ACCEL

    def _update_ball(self):
        self.ball_vel.y += self.GRAVITY
        self.ball_vel.x *= self.FRICTION
        self.ball_pos += self.ball_vel

        # Wall bounces
        if self.ball_pos.x - self.BALL_RADIUS < self.WALL_PADDING:
            self.ball_pos.x = self.WALL_PADDING + self.BALL_RADIUS
            self.ball_vel.x *= -0.8
            self._create_bounce_particles(pygame.Vector2(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y))
        elif self.ball_pos.x + self.BALL_RADIUS > self.WIDTH - self.WALL_PADDING:
            self.ball_pos.x = self.WIDTH - self.WALL_PADDING - self.BALL_RADIUS
            self.ball_vel.x *= -0.8
            self._create_bounce_particles(pygame.Vector2(self.ball_pos.x + self.BALL_RADIUS, self.ball_pos.y))

    def _update_platforms(self):
        avoid_reward = 0
        fade_duration = self.FPS * 0.75  # 0.75 second fade

        for p in self.platforms:
            # Oscillation
            p['phase'] += p['phase_speed']
            p['rect'].centerx = p['center_x'] + math.sin(p['phase']) * p['oscillation_range']

            # Visibility cycle
            p['timer'] += 1
            if p['timer'] >= p['cycle_duration']:
                p['timer'] = 0
                p['is_visible'] = not p['is_visible']
                if not p['is_visible'] and not p['reward_claimed']:
                    avoid_reward += 100
                    p['reward_claimed'] = True

            # Alpha for fade effect
            time_in_phase = p['timer']
            if p['is_visible']:
                p['alpha'] = min(255, (time_in_phase / fade_duration) * 255)
            else:
                p['alpha'] = max(0, 255 - (time_in_phase / fade_duration) * 255)

        return avoid_reward

    def _check_collisions(self):
        ball_rect = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        ball_rect.center = self.ball_pos
        for p in self.platforms:
            if p['alpha'] > 100 and ball_rect.colliderect(p['rect']):
                return True
        return False

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": int(self.score), "steps": self.steps, "level": self.level}

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_PADDING, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_PADDING, 0, self.WALL_PADDING, self.HEIGHT))

        self._render_particles()
        self._render_platforms()
        self._render_glow_circle(self.ball_pos, self.BALL_RADIUS, self.COLOR_BALL, 5, 25)

    def _render_ui(self):
        score_surf = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        time_left = self.MAX_TIME_SECONDS - (self.steps / self.FPS)
        timer_surf = self.font_small.render(f"Time: {max(0, time_left):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 20, 10))

        level_surf = self.font_large.render(f"Level {self.level}", True, self.COLOR_TEXT)
        level_rect = level_surf.get_rect(center=(self.WIDTH / 2, 40))
        self.screen.blit(level_surf, level_rect)

    def _render_glow_circle(self, pos, radius, color, layers, max_alpha):
        center_x, center_y = int(pos.x), int(pos.y)
        for i in range(layers, 0, -1):
            alpha = max_alpha * (1 - (i / layers))**2
            glow_color = (*color, int(alpha))
            glow_radius = radius + i * 2
            
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (center_x - glow_radius, center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(radius), color)

    def _render_glow_rect(self, rect, color, base_alpha, layers, max_glow_alpha):
        if base_alpha <= 0: return
        
        # Draw glow
        glow_alpha = max_glow_alpha * (base_alpha / 255.0)
        for i in range(layers, 0, -1):
            alpha = glow_alpha * (1 - (i / layers))**2
            glow_color = (*color, int(alpha))
            glow_rect = rect.inflate(i * 4, i * 4)
            
            temp_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, glow_color, temp_surf.get_rect(), border_radius=8)
            self.screen.blit(temp_surf, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
            
        # Draw main rect
        main_color = (*color, int(base_alpha))
        temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, main_color, temp_surf.get_rect(), border_radius=3)
        self.screen.blit(temp_surf, rect.topleft)

    def _render_platforms(self):
        for p in self.platforms:
            self._render_glow_rect(p['rect'], self.COLOR_PLATFORM, p['alpha'], 4, 20)

    def _render_particles(self):
        for p in self.particles:
            if p['radius'] > 0:
                alpha = 255 * (p['lifespan'] / p['max_lifespan'])
                color = (*p['color'], int(alpha))
                pos = (int(p['pos'].x), int(p['pos'].y))
                radius = int(p['radius'])
                
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0]-radius, pos[1]-radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _create_explosion(self, pos, color):
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(20, 50)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'radius': random.uniform(2, 6),
                'color': color, 'lifespan': lifespan, 'max_lifespan': lifespan
            })

    def _create_bounce_particles(self, pos):
        for _ in range(5):
            vel_x = (pos.x - self.ball_pos.x) * 0.2 + random.uniform(-0.5, 0.5)
            vel_y = random.uniform(-1, 1)
            lifespan = random.randint(15, 30)
            self.particles.append({
                'pos': pos.copy(), 'vel': pygame.Vector2(vel_x, vel_y),
                'radius': random.uniform(1, 4), 'color': self.COLOR_WALL,
                'lifespan': lifespan, 'max_lifespan': lifespan
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window even with the dummy video driver,
    # but it might be blank or non-responsive depending on the OS.
    # The environment itself is headless and works correctly.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Try to force a display driver
        import pygame.display
        pygame.display.init()
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Bouncer Dodge")
        has_display = True
    except pygame.error:
        print("No display available, running headlessly.")
        has_display = False

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Left/Right Arrow Keys: Move")
    print("R: Reset Environment")
    print("Q: Quit")
    
    while not (terminated or truncated):
        action = [0, 0, 0] # Default: no-op
        
        # This event loop is for manual play and requires a display
        if has_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        terminated = True
                    if event.key == pygame.K_r:
                        obs, info = env.reset(seed=42)
                        total_reward = 0
                        print(f"--- Env Reset --- Level: {info['level']}")

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            if keys[pygame.K_RIGHT]:
                action[0] = 4
        else: # In headless mode, just run with no-ops to see it doesn't crash
            if env.steps > 200: # Run for a few hundred steps
                break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if has_display:
            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            if has_display:
                pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset(seed=42)
            total_reward = 0
            terminated, truncated = False, False
            print(f"--- New Game Started --- Level: {info['level']}")
            
        clock.tick(env.FPS)
        
    env.close()