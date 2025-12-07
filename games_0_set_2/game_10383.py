import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:23:32.987490
# Source Brief: brief_00383.md
# Brief Index: 383
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    GameEnv: Control a bouncing ball, strategically collecting multipliers
    and transforming for bonus momentum to reach a target score within a time limit.

    - Action Space: MultiDiscrete([5, 2, 2])
        - [0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        - [1]: Space (unused)
        - [2]: Shift (unused)
    - Observation Space: Box(0, 255, (400, 640, 3), uint8) - RGB array of the game screen.
    - Reward Structure:
        - +0.01 per step (survival)
        - -0.2 for hitting a wall
        - +1.0 for collecting a x2 orb
        - +2.0 for collecting a x3 orb
        - +5.0 for collecting a star
        - +100 for winning (reaching 5000 momentum)
        - -100 for losing (time running out)
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to collect score-multiplying orbs and power-ups, "
        "aiming to reach a target score before time runs out."
    )
    user_guide = "Use the arrow keys (↑↓←→) to apply force to the ball."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 60

    # Colors
    COLOR_BG_OUTER = (10, 20, 40)
    COLOR_BG_INNER = (30, 40, 60)
    COLOR_WALL = (80, 90, 110)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_TRANSFORMED = (255, 223, 0)
    COLOR_ORB_2X = (0, 255, 128)
    COLOR_ORB_3X = (0, 128, 255)
    COLOR_STAR = (255, 223, 0)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # Game Parameters
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * TARGET_FPS
    WIN_SCORE = 5000
    INITIAL_MOMENTUM = 200
    PLAYER_BASE_RADIUS = 12
    PLAYER_FORCE = 0.25
    WALL_DAMPING = 0.8
    STAR_SPAWN_INTERVAL = 15
    TRANSFORM_DURATION = 5

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

        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 48)
            self.font_medium = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_large = pygame.font.SysFont("monospace", 48)
            self.font_medium = pygame.font.SysFont("monospace", 24)

        # Initialize state variables to avoid attribute errors before reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.time_left = 0
        self.orbs = []
        self.star = None
        self.particles = []
        self.is_transformed = False
        self.transform_timer = 0
        self.star_spawn_timer = 0
        self.orb_spawn_timer = 0
        self.orb_spawn_rate = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = self.INITIAL_MOMENTUM
        self.game_over = False

        self.ball_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        angle = self.np_random.uniform(0, 2 * math.pi)
        initial_speed = 2.5
        self.ball_vel = pygame.Vector2(math.cos(angle) * initial_speed, math.sin(angle) * initial_speed)

        self.time_left = self.GAME_DURATION_SECONDS * self.TARGET_FPS

        self.orbs = []
        self.star = None
        self.particles = []

        self.is_transformed = False
        self.transform_timer = 0

        self.star_spawn_timer = self.STAR_SPAWN_INTERVAL * self.TARGET_FPS
        self.orb_spawn_timer = 0
        self.orb_spawn_rate = 1.0

        for _ in range(5): # Start with a few orbs
            self._spawn_orb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action # space and shift are unused

        reward = 0.01 # Small reward for staying alive

        # --- Update Timers ---
        self.steps += 1
        self.time_left -= 1

        if self.is_transformed:
            self.transform_timer -= 1
            if self.transform_timer <= 0:
                self.is_transformed = False
                # SFX: Power-down

        self.star_spawn_timer -= 1
        if self.star_spawn_timer <= 0:
            self._spawn_star()
            self.star_spawn_timer = self.STAR_SPAWN_INTERVAL * self.TARGET_FPS

        self.orb_spawn_timer -= 1
        if self.orb_spawn_timer <= 0:
            self._spawn_orb()
            # Difficulty progression: orbs spawn faster over time
            self.orb_spawn_rate = max(0.2, self.orb_spawn_rate * 0.995)
            self.orb_spawn_timer = self.orb_spawn_rate * self.TARGET_FPS

        # --- Handle Player Input ---
        if movement == 1: self.ball_vel.y -= self.PLAYER_FORCE
        elif movement == 2: self.ball_vel.y += self.PLAYER_FORCE
        elif movement == 3: self.ball_vel.x -= self.PLAYER_FORCE
        elif movement == 4: self.ball_vel.x += self.PLAYER_FORCE

        max_speed = 10
        if self.ball_vel.magnitude() > max_speed:
            self.ball_vel.scale_to_length(max_speed)

        # --- Update Ball Position ---
        self.ball_pos += self.ball_vel
        ball_radius = self._get_ball_radius()

        # --- Wall Collisions ---
        wall_hit = False
        if self.ball_pos.x < ball_radius:
            self.ball_pos.x = ball_radius
            self.ball_vel.x *= -1
            wall_hit = True
        elif self.ball_pos.x > self.SCREEN_WIDTH - ball_radius:
            self.ball_pos.x = self.SCREEN_WIDTH - ball_radius
            self.ball_vel.x *= -1
            wall_hit = True

        if self.ball_pos.y < ball_radius:
            self.ball_pos.y = ball_radius
            self.ball_vel.y *= -1
            wall_hit = True
        elif self.ball_pos.y > self.SCREEN_HEIGHT - ball_radius:
            self.ball_pos.y = self.SCREEN_HEIGHT - ball_radius
            self.ball_vel.y *= -1
            wall_hit = True

        if wall_hit:
            self.score *= self.WALL_DAMPING
            reward -= 0.2
            self._create_particles(self.ball_pos, self.COLOR_WALL, 20)
            # SFX: Wall bounce

        # --- Orb Collisions ---
        for orb in self.orbs[:]:
            dist = self.ball_pos.distance_to(orb['pos'])
            if dist < ball_radius + orb['radius']:
                if orb['type'] == 'x2':
                    self.score *= 2
                    reward += 1.0
                else: # x3
                    self.score *= 3
                    reward += 2.0
                self.orbs.remove(orb)
                self._create_particles(orb['pos'], orb['color'], 30, 2.0)
                # SFX: Orb collect

        # --- Star Collision ---
        if self.star:
            dist = self.ball_pos.distance_to(self.star['pos'])
            if dist < ball_radius + self.star['radius']:
                self.score *= 2
                self.ball_vel *= 0.5
                reward += 5.0
                self.is_transformed = True
                self.transform_timer = self.TRANSFORM_DURATION * self.TARGET_FPS
                self._create_particles(self.star['pos'], self.COLOR_STAR, 50, 3.0)
                self.star = None
                # SFX: Star collect / Power-up

        self._update_particles()

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            # SFX: Win
        elif self.time_left <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            self.game_over = True
            # SFX: Lose / Time out

        self.score = max(1, self.score)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_ball_radius(self):
        return self.PLAYER_BASE_RADIUS + math.log(max(1, self.score/50)) * 2

    def _spawn_orb(self):
        radius = 10
        pos = pygame.Vector2(
            self.np_random.uniform(radius * 2, self.SCREEN_WIDTH - radius * 2),
            self.np_random.uniform(radius * 2, self.SCREEN_HEIGHT - radius * 2)
        )
        orb_type = 'x2' if self.np_random.random() < 0.7 else 'x3'
        color = self.COLOR_ORB_2X if orb_type == 'x2' else self.COLOR_ORB_3X
        self.orbs.append({'pos': pos, 'type': orb_type, 'radius': radius, 'color': color})

    def _spawn_star(self):
        if self.star is None:
            radius = 15
            self.star = {
                'pos': pygame.Vector2(
                    self.np_random.uniform(radius * 2, self.SCREEN_WIDTH - radius * 2),
                    self.np_random.uniform(radius * 2, self.SCREEN_HEIGHT - radius * 2)
                ),
                'radius': radius
            }

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left / self.TARGET_FPS,
        }

    def _render_game(self):
        self._draw_radial_gradient()

        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            size = max(1, int(p['lifespan'] / 8))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

        if self.star:
            self._draw_glow_circle(self.star['pos'], self.COLOR_STAR, self.star['radius'])

        for orb in self.orbs:
            self._draw_glow_circle(orb['pos'], orb['color'], orb['radius'])

        ball_radius = self._get_ball_radius()
        ball_color = self.COLOR_PLAYER_TRANSFORMED if self.is_transformed else self.COLOR_PLAYER
        self._draw_glow_circle(self.ball_pos, ball_color, ball_radius, glow_factor=1.8 if self.is_transformed else 1.5)

        self._render_ui()

    def _draw_radial_gradient(self):
        center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        max_radius = int(math.sqrt(center[0]**2 + center[1]**2))
        for r in range(max_radius, 0, -5):
            t = r / max_radius
            color = (
                int(self.COLOR_BG_OUTER[0] * t + self.COLOR_BG_INNER[0] * (1-t)),
                int(self.COLOR_BG_OUTER[1] * t + self.COLOR_BG_INNER[1] * (1-t)),
                int(self.COLOR_BG_OUTER[2] * t + self.COLOR_BG_INNER[2] * (1-t)),
            )
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], r, color)

    def _draw_glow_circle(self, pos, color, radius, glow_factor=1.5):
        x, y = int(pos.x), int(pos.y)

        for i in range(int(radius * glow_factor), int(radius), -1):
            alpha = int(50 * (1 - (i - radius) / (radius * (glow_factor - 1))))
            s = pygame.Surface((i*2, i*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (i, i), i)
            self.screen.blit(s, (x - i, y - i))

        pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, x, y, int(radius), color)

    def _render_ui(self):
        score_text = f"{int(self.score)}"
        self._render_text(score_text, self.font_large, (self.SCREEN_WIDTH / 2, 30))

        time_str = f"{max(0, self.time_left / self.TARGET_FPS):.1f}"
        self._render_text(time_str, self.font_medium, (self.SCREEN_WIDTH - 60, 25))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            message = "VICTORY!" if self.score >= self.WIN_SCORE else "TIME'S UP"
            self._render_text(message, self.font_large, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))

    def _render_text(self, text, font, center_pos):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=center_pos)

        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        shadow_rect = shadow_surf.get_rect(center=(center_pos[0]+2, center_pos[1]+2))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the game and play it manually.
    # It will not be executed by the autograder.
    
    # Un-comment the line below to run with a visible display
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")

    env = GameEnv()
    obs, info = env.reset()

    keys_held = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Momentum Ball")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement = 0
        if keys_held[pygame.K_UP]: movement = 1
        if keys_held[pygame.K_DOWN]: movement = 2
        if keys_held[pygame.K_LEFT]: movement = 3
        if keys_held[pygame.K_RIGHT]: movement = 4

        action = [movement, 0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.0f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.TARGET_FPS)

    env.close()