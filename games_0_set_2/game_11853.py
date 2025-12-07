import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a horizontal platform to guide
    bouncing balls into a goal area. The game emphasizes visual polish and "game feel"
    through smooth animations, particle effects, and responsive controls.

    **Gameplay:**
    - Control a white platform at the bottom of the screen.
    - Guide 5 colored balls into the green goal area at the top.
    - Avoid letting balls fall off the bottom of the screen.
    - Hold the 'space' key to activate a magnet mode, which slows balls
      and attracts randomly appearing bonus stars for extra points.

    **Termination:**
    - **Win:** All 5 balls are in the goal area.
    - **Timeout:** The episode exceeds 1000 steps.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = (
        "Control a platform to bounce balls into a goal area at the top of the screen. "
        "Use the platform's magnet to collect bonus stars and guide the balls."
    )
    user_guide = "Use ←→ arrow keys to move the platform. Hold space to activate the magnet."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Assumed frame rate for physics simulation

    # Colors
    COLOR_BG = (15, 19, 32)
    COLOR_PLATFORM = (220, 220, 255)
    COLOR_GOAL = (40, 180, 120)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_MAGNET_AURA = (80, 120, 255)
    BALL_COLORS = [
        (0, 200, 255),  # Cyan
        (255, 100, 200), # Pink
        (255, 220, 0),  # Yellow
        (120, 255, 120), # Light Green
        (255, 140, 0)   # Orange
    ]

    # Physics & Game Parameters
    GRAVITY = 0.4
    BALL_RADIUS = 10
    PLATFORM_SPEED = 10
    PLATFORM_HEIGHT = 10
    PLATFORM_Y_POS = SCREEN_HEIGHT - 30
    BOUNCE_DAMPENING = 0.95
    MAGNET_DRAG = 0.96
    MAX_STEPS = 1000
    NUM_BALLS = 5

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
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State Initialization ---
        self.platform = None
        self.balls = []
        self.particles = []
        self.bonus_star = None
        self.steps = 0
        self.score = 0
        self.prev_ball_distances_to_goal = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0

        # Player Platform
        platform_width = 120
        self.platform = pygame.Rect(
            (self.SCREEN_WIDTH - platform_width) / 2,
            self.PLATFORM_Y_POS,
            platform_width,
            self.PLATFORM_HEIGHT
        )

        # Balls
        self.balls = []
        for i in range(self.NUM_BALLS):
            self.balls.append(self._create_ball())

        # Bonus Star
        self.bonus_star = None

        # Particles
        self.particles = []

        # Reward calculation helper
        self.prev_ball_distances_to_goal = self._get_ball_distances()

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused in this design

        self._handle_input(movement)
        self._update_physics(space_held)
        self._update_bonus_star(space_held)
        self._update_particles()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        self.steps += 1

        return (
            self._get_observation(space_held=space_held),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.platform.x -= self.PLATFORM_SPEED
        elif movement == 4:  # Right
            self.platform.x += self.PLATFORM_SPEED

        # Clamp platform position to screen bounds
        self.platform.x = max(0, min(self.SCREEN_WIDTH - self.platform.width, self.platform.x))

    def _update_physics(self, magnet_on):
        for ball in self.balls:
            if ball['in_goal']:
                continue

            # Apply gravity
            ball['vel'].y += self.GRAVITY

            # Apply magnet drag if active
            if magnet_on and self.platform.colliderect(
                pygame.Rect(ball['pos'].x - 50, ball['pos'].y - 50, 100, 100)
            ):
                ball['vel'] *= self.MAGNET_DRAG

            # Update position
            ball['pos'] += ball['vel']

            # Wall collisions
            if ball['pos'].x - self.BALL_RADIUS < 0 or ball['pos'].x + self.BALL_RADIUS > self.SCREEN_WIDTH:
                ball['pos'].x = max(self.BALL_RADIUS, min(self.SCREEN_WIDTH - self.BALL_RADIUS, ball['pos'].x))
                ball['vel'].x *= -self.BOUNCE_DAMPENING
                self._create_particles(ball['pos'], ball['color'], 5)

            # Ceiling collision
            if ball['pos'].y - self.BALL_RADIUS < 0:
                ball['pos'].y = self.BALL_RADIUS
                ball['vel'].y *= -self.BOUNCE_DAMPENING

            # Platform collision
            platform_rect = self.platform
            ball_rect = pygame.Rect(ball['pos'].x - self.BALL_RADIUS, ball['pos'].y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if platform_rect.colliderect(ball_rect) and ball['vel'].y > 0:
                ball['pos'].y = platform_rect.top - self.BALL_RADIUS
                ball['vel'].y *= -self.BOUNCE_DAMPENING
                impact_offset = (ball['pos'].x - platform_rect.centerx) / (platform_rect.width / 2)
                ball['vel'].x += impact_offset * 2.0
                self._create_particles(pygame.Vector2(ball['pos'].x, platform_rect.top), self.COLOR_PLATFORM, 10)

            # Goal check
            goal_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 50)
            if goal_rect.collidepoint(ball['pos']):
                if not ball['in_goal']:
                    ball['in_goal'] = True
                    ball['vel'] = pygame.Vector2(0, 0)
                    self._create_particles(ball['pos'], self.COLOR_GOAL, 20)

    def _update_bonus_star(self, magnet_on):
        if self.bonus_star is None and self.np_random.random() < 0.01:
            self.bonus_star = {
                'pos': pygame.Vector2(
                    self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                    self.np_random.uniform(100, self.SCREEN_HEIGHT - 100)
                ),
                'collected': False
            }

        if self.bonus_star:
            if magnet_on:
                direction = pygame.Vector2(self.platform.center) - self.bonus_star['pos']
                dist = direction.length()
                if dist > 1 and dist < 150:
                    direction.normalize_ip()
                    self.bonus_star['pos'] += direction * 3.0
            
            star_rect = pygame.Rect(self.bonus_star['pos'].x - 15, self.bonus_star['pos'].y - 15, 30, 30)
            if self.platform.colliderect(star_rect):
                self.bonus_star['collected'] = True
                self._create_particles(self.bonus_star['pos'], (255, 255, 100), 30)

    def _calculate_reward(self):
        reward = 0
        
        # Check for lost balls and respawn them, applying a penalty
        for ball in self.balls:
            if not ball['in_goal'] and ball['pos'].y > self.SCREEN_HEIGHT + self.BALL_RADIUS:
                reward -= 10
                ball['pos'] = pygame.Vector2(
                    self.np_random.integers(50, self.SCREEN_WIDTH - 50),
                    self.np_random.integers(80, 150)
                )
                ball['vel'] = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(1, 3))
        
        # Anti-softlock: respawn balls stuck below platform
        for ball in self.balls:
             if not ball['in_goal'] and ball['pos'].y > self.platform.y + 20 and abs(ball['vel'].y) < 0.1:
                ball['pos'] = pygame.Vector2(self.np_random.integers(50, self.SCREEN_WIDTH - 50), 80)
                ball['vel'] = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(1, 3))

        # Event: Bonus collected
        if self.bonus_star and self.bonus_star['collected']:
            reward += 5
            self.score += 500
            self.bonus_star = None
        
        # Continuous reward: balls getting closer to goal center
        current_ball_distances = self._get_ball_distances()
        for i in range(len(current_ball_distances)):
            dist_change = self.prev_ball_distances_to_goal[i] - current_ball_distances[i]
            reward += dist_change * 0.1
        self.prev_ball_distances_to_goal = current_ball_distances
        
        return reward

    def _check_termination(self):
        # Win condition
        if all(ball['in_goal'] for ball in self.balls):
            self.score += 10000
            return True
        return False

    def _get_observation(self, space_held=False):
        self.screen.fill(self.COLOR_BG)
        self._render_game(space_held=space_held)
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, space_held=False):
        goal_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 50)
        s = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        s.fill((*self.COLOR_GOAL, 60))
        self.screen.blit(s, (0, 0))
        pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_rect, 2)

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        if self.bonus_star:
            self._draw_star(self.screen, (255, 220, 50), self.bonus_star['pos'], 15)
            self._draw_star(self.screen, (255, 255, 255), self.bonus_star['pos'], 10)

        for ball in self.balls:
            self._draw_glowing_circle(self.screen, ball['pos'], self.BALL_RADIUS, ball['color'])

        if space_held:
            aura_radius = self.platform.width * 0.7 + 10 * math.sin(self.steps * 0.3)
            self._draw_glowing_circle(self.screen, self.platform.center, int(aura_radius), self.COLOR_MAGNET_AURA, max_alpha=30)
        
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, self.platform, border_radius=5)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        balls_in_play = sum(1 for b in self.balls if not b['in_goal'])
        balls_in_goal = sum(1 for b in self.balls if b['in_goal'])

        text = self.font.render(f"BALLS: {balls_in_play}", True, self.COLOR_UI_TEXT)
        text_rect = text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text, text_rect)

        for i in range(balls_in_goal):
             pygame.draw.circle(self.screen, self.COLOR_GOAL, (text_rect.left - 20 - i*20, 22), 6)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_in_goal": sum(1 for b in self.balls if b['in_goal']),
        }

    # --- Helper Functions ---

    def _create_ball(self):
        return {
            'pos': pygame.Vector2(
                self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                self.np_random.uniform(80, 200)
            ),
            'vel': pygame.Vector2(
                self.np_random.uniform(-2, 2),
                self.np_random.uniform(-2, 2)
            ),
            'color': random.choice(self.BALL_COLORS),
            'in_goal': False,
        }

    def _get_ball_distances(self):
        goal_center_x = self.SCREEN_WIDTH / 2
        distances = []
        for ball in self.balls:
            if not ball['in_goal']:
                distances.append(abs(ball['pos'].x - goal_center_x))
            else:
                distances.append(0)
        return distances

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 1)),
                'radius': self.np_random.uniform(2, 5),
                'lifespan': self.np_random.uniform(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.1
            p['lifespan'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    @staticmethod
    def _draw_glowing_circle(surface, center, radius, color, max_alpha=60):
        center_int = (int(center[0]), int(center[1]))
        radius = int(radius)
        if radius <= 0: return

        for i in range(4):
            alpha = max_alpha * (1 - i / 4)
            rad = radius + i * 2
            pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], rad, (*color, alpha))
        
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)

    @staticmethod
    def _draw_star(surface, color, center, size):
        points = []
        for i in range(10):
            angle = math.radians(i * 36)
            radius = size if i % 2 == 0 else size * 0.5
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncing Ball Synchronizer")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("----------------\n")
    
    while True:
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Unused
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        quit_game = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("\n--- Resetting Environment ---")
                obs, info = env.reset()
                done = False

        if quit_game:
            break

        if done:
            print("\n--- GAME OVER ---")
            print(f"Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

        clock.tick(GameEnv.FPS)
        
    pygame.quit()