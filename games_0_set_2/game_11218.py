import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:47:43.716487
# Source Brief: brief_01218.md
# Brief Index: 1218
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a ball, launching it to bounce
    off sinusoidally moving platforms. The goal is to achieve a high score by
    chaining bounces together.

    Visual Style:
    - Neon-inspired with a dark gradient background.
    - Bright, high-contrast interactive elements (ball, platforms).
    - Particle effects for ball trails and bounce impacts.
    - Glow effect for active bounce chains.

    Gameplay:
    - Before launch, the agent adjusts the launch angle.
    - The 'space' action launches the ball.
    - The ball bounces off walls and platforms.
    - Bouncing off platforms increases the score.
    - Rapid, successive bounces build a chain multiplier for bonus points.
    - The episode ends if the score reaches 150 or 5000 steps are taken.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch a neon ball and bounce it off moving platforms to score points. "
        "Chain bounces together for a higher score multiplier."
    )
    user_guide = "Use arrow keys (↑↓←→) to aim the launcher. Press space to launch the ball."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    SCORE_GOAL = 150

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (40, 0, 60)
    COLOR_BALL = (255, 255, 0)
    COLOR_PLATFORM = (0, 200, 255)
    COLOR_TRAIL = (255, 255, 0)
    COLOR_IMPACT = (0, 255, 255)
    COLOR_CHAIN_GLOW = (0, 255, 100)
    COLOR_UI = (255, 255, 255)
    COLOR_LAUNCH_INDICATOR = (255, 255, 255, 150)

    # Game Parameters
    BALL_RADIUS = 10
    BALL_SPEED = 8.0
    PLATFORM_HEIGHT = 12
    PLATFORM_WIDTH = 80
    NUM_PLATFORMS = 15
    ANGLE_ADJUST_RATE = math.radians(3) # Radians per step
    CHAIN_TIMEOUT_STEPS = 120 # 4 seconds at 30 FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self._background_surf = self._create_background()

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.terminated = False
        self.ball_launched = False
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.launch_angle = 0.0
        self.platforms = []
        self.particles = []
        self.chain_count = 0
        self.last_bounce_step = -self.CHAIN_TIMEOUT_STEPS
        self.prev_space_held = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.terminated = False
        self.ball_launched = False
        self.ball_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.launch_angle = self.np_random.uniform(0, 2 * math.pi)
        self.particles.clear()
        self.chain_count = 0
        self.last_bounce_step = -self.CHAIN_TIMEOUT_STEPS
        self.prev_space_held = False
        
        self._initialize_platforms()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        reward = 0
        self.terminated = False

        self._handle_input(action)
        self._update_game_state()

        reward += self._check_collisions()

        self.steps += 1

        truncated = self.steps >= self.MAX_STEPS

        if self.score >= self.SCORE_GOAL:
            self.terminated = True
            reward += 100.0
        elif truncated:
            self.terminated = True # In this game, truncation is also termination

        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), self.terminated, truncated, info

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if not self.ball_launched:
            # Adjust launch angle
            if movement == 1 or movement == 3: # Up or Left -> Counter-clockwise
                self.launch_angle += self.ANGLE_ADJUST_RATE
            elif movement == 2 or movement == 4: # Down or Right -> Clockwise
                self.launch_angle -= self.ANGLE_ADJUST_RATE
            self.launch_angle %= (2 * math.pi)

            # Launch ball on space press (rising edge)
            if space_held and not self.prev_space_held:
                self.ball_launched = True
                self.ball_vel = np.array([math.cos(self.launch_angle), math.sin(self.launch_angle)]) * self.BALL_SPEED
                # SFX: Launch sound

        self.prev_space_held = space_held

    def _update_game_state(self):
        if self.ball_launched:
            self.ball_pos += self.ball_vel
            self._add_particle(self.ball_pos.copy(), count=1, p_type='trail')

        self._update_platforms()
        self._update_particles()

    def _update_platforms(self):
        for p in self.platforms:
            time_factor = self.steps * p['freq'] + p['phase']
            p['rect'].centerx = p['center_y_anchor'] + p['amp'] * math.sin(time_factor)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1

    def _check_collisions(self):
        bounce_reward = 0
        
        if not self.ball_launched:
            return 0

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # SFX: Wall bounce
        if self.ball_pos[1] <= self.BALL_RADIUS or self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            # SFX: Wall bounce

        # Platform collisions
        for p in self.platforms:
            if p['rect'].clipline(self.ball_pos - self.ball_vel, self.ball_pos):
                # Simple and effective arcade bounce physics
                # Check if ball is moving towards the platform to avoid sticking
                relative_y_vel = self.ball_vel[1] - 0 # Platform has 0 y-velocity
                is_colliding_top = self.ball_pos[1] < p['rect'].centery and relative_y_vel > 0
                is_colliding_bottom = self.ball_pos[1] > p['rect'].centery and relative_y_vel < 0

                if is_colliding_top or is_colliding_bottom:
                    self.ball_vel[1] *= -1
                    # Nudge ball out of platform
                    self.ball_pos[1] += self.ball_vel[1]
                    # Add a bit of the platform's horizontal velocity
                    platform_vel_x = (p['rect'].centerx - p['last_x'])
                    self.ball_vel[0] = (self.ball_vel[0] * 0.6) + (platform_vel_x * 0.2)
                    # Renormalize speed
                    current_speed = np.linalg.norm(self.ball_vel)
                    if current_speed > 0:
                        self.ball_vel = (self.ball_vel / current_speed) * self.BALL_SPEED
                    
                    # SFX: Platform bounce
                    self._add_particle(self.ball_pos.copy(), count=15, p_type='impact')
                    bounce_reward += self._handle_bounce_scoring()
                    break # Only one platform bounce per frame
            p['last_x'] = p['rect'].centerx
            
        return bounce_reward
    
    def _handle_bounce_scoring(self):
        reward = 5.0
        
        if self.steps - self.last_bounce_step < self.CHAIN_TIMEOUT_STEPS:
            self.chain_count += 1
        else:
            self.chain_count = 1
        
        self.last_bounce_step = self.steps
        self.score += 1

        if self.chain_count >= 3:
            chain_bonus = self.chain_count * 5.0
            reward += chain_bonus
            # SFX: Chain bonus sound
        
        return reward

    def _get_observation(self):
        self.screen.blit(self._background_surf, (0, 0))
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "chain": self.chain_count,
            "ball_launched": self.ball_launched
        }

    def _render_game_elements(self):
        # Render particles
        for p in self.particles:
            if p['radius'] > 0:
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                color = p['color']
                if len(color) == 3: color = (*color, alpha)
                else: color = (color[0], color[1], color[2], alpha)
                
                if p['radius'] > 1:
                    pygame.gfxdraw.filled_circle(
                        self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color
                    )

        # Render platforms
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p['rect'], border_radius=4)
            # Add a subtle inner highlight for depth
            highlight_rect = p['rect'].inflate(-4, -4)
            highlight_color = tuple(min(255, c + 50) for c in self.COLOR_PLATFORM)
            pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=3)

        # Render ball and chain glow
        if self.chain_count > 0 and self.ball_launched:
            glow_radius = self.BALL_RADIUS + self.chain_count * 2 + 5
            glow_alpha = min(100, 20 + self.chain_count * 15)
            # Draw multiple layers for a soft glow
            for i in range(4):
                r = glow_radius * (1 - i * 0.15)
                alpha = glow_alpha * (1 - i * 0.2)
                if r > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]),
                                                 int(r), (*self.COLOR_CHAIN_GLOW, int(alpha)))
        
        # Draw the main ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]),
                                     self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]),
                                self.BALL_RADIUS, self.COLOR_BALL)

        # Render launch indicator
        if not self.ball_launched:
            end_pos_x = self.ball_pos[0] + 40 * math.cos(self.launch_angle)
            end_pos_y = self.ball_pos[1] + 40 * math.sin(self.launch_angle)
            pygame.draw.line(self.screen, self.COLOR_LAUNCH_INDICATOR, 
                             (int(self.ball_pos[0]), int(self.ball_pos[1])),
                             (int(end_pos_x), int(end_pos_y)), 3)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        if self.chain_count > 1:
            chain_text = self.font.render(f"CHAIN x{self.chain_count}", True, self.COLOR_CHAIN_GLOW)
            self.screen.blit(chain_text, (10, 35))

    def _initialize_platforms(self):
        self.platforms.clear()
        y_positions = np.linspace(50, self.HEIGHT - 50, self.NUM_PLATFORMS)
        self.np_random.shuffle(y_positions)
        
        for i in range(self.NUM_PLATFORMS):
            y = y_positions[i]
            center_x = self.WIDTH / 2
            max_amp = (self.WIDTH - self.PLATFORM_WIDTH) / 2
            
            p = {
                'rect': pygame.Rect(0, 0, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT),
                'center_y_anchor': center_x,
                'amp': self.np_random.uniform(max_amp * 0.2, max_amp * 0.95),
                'freq': self.np_random.uniform(0.01, 0.05),
                'phase': self.np_random.uniform(0, 2 * math.pi),
                'last_x': center_x
            }
            p['rect'].centery = y
            self.platforms.append(p)

    def _add_particle(self, pos, count, p_type):
        for _ in range(count):
            if p_type == 'trail':
                vel = (self.np_random.random(2) - 0.5) * 0.5
                lifespan = 20
                radius = self.BALL_RADIUS * 0.5
                color = self.COLOR_TRAIL
            elif p_type == 'impact':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                lifespan = self.np_random.integers(25, 40)
                radius = self.np_random.uniform(2, 5)
                color = self.COLOR_IMPACT
            else:
                continue

            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'radius': radius,
                'color': color
            })

    def _create_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neon Bounce")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        mov = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: mov = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: mov = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: mov = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        if terminated or truncated:
            break

        clock.tick(GameEnv.FPS)
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()