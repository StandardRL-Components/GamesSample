import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:53:33.055561
# Source Brief: brief_00086.md
# Brief Index: 86
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple, deque

# Helper data structures for clarity
Particle = namedtuple("Particle", ["pos", "vel", "radius", "color", "lifespan"])
ScorePopup = namedtuple("ScorePopup", ["pos", "text", "color", "lifespan"])
Platform = namedtuple("Platform", ["rect", "score"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Tilt the platform to aim your ball. Charge and release powerful jumps to hit scoring platforms and become the Bounce King."
    user_guide = "Controls: Use ←→ to tilt the platform. Hold space to charge a jump, then release to launch the ball."
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (20, 15, 40)
    COLOR_BALL = (80, 255, 80)
    COLOR_BALL_GLOW = (180, 255, 180)
    COLOR_PLATFORM = (0, 100, 255)
    COLOR_PLAYER_PLATFORM = (255, 50, 150)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_CHARGE_BAR_BG = (50, 50, 50)
    COLOR_CHARGE_BAR_FILL = (255, 200, 0)
    
    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game Physics
    GRAVITY = 0.2
    BALL_RADIUS = 10
    PLAYER_PLATFORM_WIDTH = 120
    PLAYER_PLATFORM_HEIGHT = 15
    PLAYER_PLATFORM_Y = 380
    TILT_SPEED = 1.5
    MAX_TILT = 35  # degrees
    BOUNCE_DAMPING = 0.85 # Energy lost on regular platform bounce
    JUMP_CHARGE_RATE = 0.1
    MAX_JUMP_CHARGE = 5.0
    MIN_JUMP_POWER = 4.0
    MAX_JUMP_POWER = 12.0
    
    # Game Rules
    WIN_SCORE = 100
    MAX_STEPS = 3600 # 60 seconds at 60 FPS
    WALL_PENALTY = -10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_popup = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # State variables are initialized in reset()
        self.ball_pos = None
        self.ball_vel = None
        self.player_platform_angle = None
        self.jump_charge = None
        self.ball_on_player_platform = None
        self.prev_space_held = None
        self.platforms = []
        self.particles = []
        self.score_popups = []
        self.ball_trail = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ball_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.PLAYER_PLATFORM_Y - self.PLAYER_PLATFORM_HEIGHT / 2 - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.player_platform_angle = 0
        self.jump_charge = 0
        self.ball_on_player_platform = True
        self.prev_space_held = False
        
        self.particles = []
        self.score_popups = []
        self.ball_trail = deque(maxlen=15)
        
        self._generate_platforms()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held)
        self._update_ball_physics()
        step_reward = self._handle_collisions()
        
        self._update_effects()

        terminated = self._check_termination()
        
        if terminated and self.score >= self.WIN_SCORE:
            step_reward += 100 # Goal-oriented reward

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # --- Tilt Control ---
        if movement == 3: # Left
            self.player_platform_angle = max(-self.MAX_TILT, self.player_platform_angle - self.TILT_SPEED)
        elif movement == 4: # Right
            self.player_platform_angle = min(self.MAX_TILT, self.player_platform_angle + self.TILT_SPEED)

        # --- Jump Charge & Release ---
        if self.ball_on_player_platform:
            if space_held:
                self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + self.JUMP_CHARGE_RATE)
            elif self.prev_space_held: # Space was just released
                jump_power = self.MIN_JUMP_POWER + (self.MAX_JUMP_POWER - self.MIN_JUMP_POWER) * (self.jump_charge / self.MAX_JUMP_CHARGE)
                jump_vector = pygame.Vector2(0, -1).rotate(-self.player_platform_angle)
                self.ball_vel = jump_vector * jump_power
                self.ball_on_player_platform = False
                self.jump_charge = 0
                # sfx: jump_release
        
        self.prev_space_held = space_held
        
    def _update_ball_physics(self):
        if not self.ball_on_player_platform:
            self.ball_vel.y += self.GRAVITY
            self.ball_pos += self.ball_vel
        
        self.ball_trail.append(self.ball_pos.copy())

    def _handle_collisions(self):
        reward = 0
        
        # --- Wall Collisions ---
        if self.ball_pos.x - self.BALL_RADIUS < 0:
            self.ball_pos.x = self.BALL_RADIUS
            self.ball_vel.x *= -0.5 # Dampen horizontal velocity
            self.ball_vel.y = 0 # Reset vertical momentum
            reward += self.WALL_PENALTY
            self.score += self.WALL_PENALTY
            self._create_score_popup(self.ball_pos, str(self.WALL_PENALTY), (255, 80, 80))
            self._create_particle_burst(self.ball_pos, self.COLOR_BALL, 20)
            # sfx: wall_hit
        elif self.ball_pos.x + self.BALL_RADIUS > self.SCREEN_WIDTH:
            self.ball_pos.x = self.SCREEN_WIDTH - self.BALL_RADIUS
            self.ball_vel.x *= -0.5
            self.ball_vel.y = 0
            reward += self.WALL_PENALTY
            self.score += self.WALL_PENALTY
            self._create_score_popup(self.ball_pos, str(self.WALL_PENALTY), (255, 80, 80))
            self._create_particle_burst(self.ball_pos, self.COLOR_BALL, 20)
            # sfx: wall_hit

        # --- Ceiling Collision ---
        if self.ball_pos.y - self.BALL_RADIUS < 0:
            self.ball_pos.y = self.BALL_RADIUS
            self.ball_vel.y *= -0.5

        # --- Player Platform Collision ---
        player_rect = self._get_player_platform_rect()
        if not self.ball_on_player_platform and self.ball_vel.y > 0 and player_rect.collidepoint(self.ball_pos):
             # A simple approximation for landing
            if self.ball_pos.y > player_rect.top - self.BALL_RADIUS:
                self.ball_on_player_platform = True
                self.ball_vel = pygame.Vector2(0, 0)
                self.ball_pos.y = player_rect.top - self.BALL_RADIUS
                self._create_particle_burst(self.ball_pos, self.COLOR_PLAYER_PLATFORM, 15)
                # sfx: land_on_player_platform
        
        # --- Static Platform Collisions ---
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        for p in self.platforms:
            if self.ball_vel.y > 0 and ball_rect.colliderect(p.rect):
                # Check if the ball's bottom is intersecting the platform's top surface
                if ball_rect.bottom > p.rect.top and ball_rect.centery < p.rect.centery:
                    self.ball_pos.y = p.rect.top - self.BALL_RADIUS
                    self.ball_vel.y *= -self.BOUNCE_DAMPING
                    self.ball_vel.x *= 0.98 # Air friction on bounce
                    
                    reward += p.score + 0.1 # Point for platform + continuous reward
                    self.score += p.score
                    self._create_score_popup(self.ball_pos, f"+{p.score}", self.COLOR_UI_TEXT)
                    self._create_particle_burst(self.ball_pos, self.COLOR_PLATFORM, 10)
                    # sfx: platform_bounce
                    break
        
        # Out of bounds check
        if self.ball_pos.y > self.SCREEN_HEIGHT + 50:
             self.ball_on_player_platform = True
             self.ball_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.PLAYER_PLATFORM_Y - self.PLAYER_PLATFORM_HEIGHT / 2 - self.BALL_RADIUS)
             self.ball_vel = pygame.Vector2(0,0)

        return reward

    def _update_effects(self):
        # Update particles
        self.particles = [
            Particle(p.pos + p.vel, p.vel + pygame.Vector2(0, 0.1), p.radius, p.color, p.lifespan - 1)
            for p in self.particles if p.lifespan > 0
        ]
        # Update score popups
        self.score_popups = [
            ScorePopup(s.pos - pygame.Vector2(0, 0.5), s.text, s.color, s.lifespan - 1)
            for s in self.score_popups if s.lifespan > 0
        ]

    def _check_termination(self):
        time_ran_out = self.steps >= self.MAX_STEPS
        won_game = self.score >= self.WIN_SCORE
        if time_ran_out or won_game:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": (self.MAX_STEPS - self.steps) / 60.0
        }

    def _generate_platforms(self):
        self.platforms.clear()
        # Ensure platforms are reachable and provide variety
        y_levels = [300, 220, 140]
        for i, y in enumerate(y_levels):
            num_platforms = 2 if i < 2 else 1
            for j in range(num_platforms):
                width = 100 - i * 20
                x = (self.SCREEN_WIDTH / (num_platforms + 1)) * (j + 1) + self.np_random.uniform(-50, 50)
                x = np.clip(x, width / 2, self.SCREEN_WIDTH - width / 2)
                score = (i + 1) * 5
                self.platforms.append(Platform(pygame.Rect(x - width / 2, y, width, 10), score))

    def _create_particle_burst(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            radius = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append(Particle(pos.copy(), vel, radius, color, lifespan))
            
    def _create_score_popup(self, pos, text, color):
        self.score_popups.append(ScorePopup(pos.copy(), text, color, 60))
        
    def _get_player_platform_rect(self):
        # This is simplified for collision, actual drawing is rotated.
        return pygame.Rect(
            self.SCREEN_WIDTH / 2 - self.PLAYER_PLATFORM_WIDTH / 2,
            self.PLAYER_PLATFORM_Y - self.PLAYER_PLATFORM_HEIGHT / 2,
            self.PLAYER_PLATFORM_WIDTH,
            self.PLAYER_PLATFORM_HEIGHT
        )

    # --- RENDERING METHODS ---
    
    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.SCREEN_WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Static Platforms
        for p in self.platforms: pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p.rect, border_radius=3)

        # Player Platform
        self._render_player_platform()
        
        # Ball Trail and Ball
        self._render_ball_and_trail()
        
        # Effects
        self._render_effects()

        # UI
        self._render_ui()

    def _render_player_platform(self):
        center_x, center_y = self.SCREEN_WIDTH / 2, self.PLAYER_PLATFORM_Y
        angle_rad = math.radians(self.player_platform_angle)
        
        points = []
        w, h = self.PLAYER_PLATFORM_WIDTH, self.PLAYER_PLATFORM_HEIGHT
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        
        for x, y in corners:
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad) + center_x
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad) + center_y
            points.append((rx, ry))
            
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_PLATFORM)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_PLATFORM)

    def _render_ball_and_trail(self):
        # Trail
        if self.ball_trail:
            for i, pos in enumerate(self.ball_trail):
                alpha = int(255 * (i / len(self.ball_trail)) * 0.3)
                color = self.COLOR_BALL_GLOW
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.BALL_RADIUS, (*color, alpha))
            
        # Ball Glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS + 4, (*self.COLOR_BALL_GLOW, 100))
        # Ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_effects(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / 40.0))
            color_with_alpha = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), color_with_alpha)

        # Score Popups
        for s in self.score_popups:
            alpha = int(255 * (s.lifespan / 60.0))
            if alpha > 0:
                text_surf = self.font_popup.render(s.text, True, s.color)
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, (s.pos.x - text_surf.get_width()/2, s.pos.y - text_surf.get_height()/2))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 60.0)
        time_text = self.font_ui.render(f"TIME: {time_left:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Jump Charge Meter
        if self.ball_on_player_platform and self.jump_charge > 0:
            bar_width = 100
            bar_height = 10
            bar_x = self.SCREEN_WIDTH / 2 - bar_width / 2
            bar_y = self.PLAYER_PLATFORM_Y + 20
            
            fill_width = (self.jump_charge / self.MAX_JUMP_CHARGE) * bar_width
            
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
            if fill_width > 0:
                pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=3)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # The following code requires a display environment and is not compatible with headless execution
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bounce King")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0

    # --- Manual Control Mapping ---
    # A/D or Left/Right Arrow for tilt
    # Space to charge/release jump
    
    while not done:
        # Default action is "do nothing"
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action[0] = 3 # Left
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action[0] = 4 # Right
            
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Space held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            done = True
        
        clock.tick(60) # Limit to 60 FPS
        
    env.close()