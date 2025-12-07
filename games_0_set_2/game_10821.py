import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:01:35.779788
# Source Brief: brief_00821.md
# Brief Index: 821
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An arcade-style Gymnasium environment where the agent controls a swarm of
    bouncing balls to collect orbs in a circular arena. Each collected orb
    spawns a new ball, escalating the challenge. The goal is to collect 100
    orbs before the time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a swarm of bouncing balls to collect orbs in a circular arena. "
        "Each collected orb spawns a new ball, increasing the challenge."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to apply force to all balls in the swarm."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_ARENA_BORDER = (100, 100, 150)
    COLOR_TEXT = (220, 220, 240)
    NEON_COLORS = [
        (50, 255, 255),  # Cyan
        (255, 50, 255),  # Magenta
        (255, 255, 50),  # Yellow
        (50, 255, 50),   # Lime Green
        (255, 100, 50),  # Orange
    ]
    COLOR_ORB = (255, 255, 255)

    # Game parameters
    MAX_STEPS = 1000
    WIN_SCORE = 100
    ARENA_CENTER = pygame.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    ARENA_RADIUS = 180
    BALL_RADIUS = 8
    ORB_RADIUS = 6
    INITIAL_ORBS = 5
    BALL_ACCELERATION = 0.4
    BALL_MAX_SPEED = 6.0
    BALL_DAMPING = 0.995 # Slight friction
    ORB_PROXIMITY_THRESHOLD = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.render_mode = render_mode

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.balls = []
        self.orbs = []
        self.particles = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        
        self.balls.clear()
        self.orbs.clear()
        self.particles.clear()

        # Spawn initial ball
        self._spawn_ball()

        # Spawn initial orbs
        for _ in range(self.INITIAL_ORBS):
            self._spawn_orb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0.0

        # --- 1. APPLY ACTION ---
        movement, _, _ = action[0], action[1] == 1, action[2] == 1
        
        acceleration_vector = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            acceleration_vector.y = -self.BALL_ACCELERATION
        elif movement == 2:  # Down
            acceleration_vector.y = self.BALL_ACCELERATION
        elif movement == 3:  # Left
            acceleration_vector.x = -self.BALL_ACCELERATION
        elif movement == 4:  # Right
            acceleration_vector.x = self.BALL_ACCELERATION

        for ball in self.balls:
            ball['vel'] += acceleration_vector

        # --- 2. UPDATE GAME STATE ---
        # Update balls
        for ball in self.balls:
            # Clamp speed
            speed = ball['vel'].length()
            if speed > self.BALL_MAX_SPEED:
                ball['vel'].scale_to_length(self.BALL_MAX_SPEED)
            
            # Apply damping
            ball['vel'] *= self.BALL_DAMPING

            # Update position
            ball['pos'] += ball['vel']

            # Arena collision
            dist_from_center = ball['pos'].distance_to(self.ARENA_CENTER)
            if dist_from_center > self.ARENA_RADIUS - self.BALL_RADIUS:
                # # sfx: bounce
                normal = (ball['pos'] - self.ARENA_CENTER).normalize()
                ball['vel'] = ball['vel'].reflect(normal) * 0.95 # Lose some energy on bounce
                # Ensure ball is inside arena to prevent sticking
                ball['pos'] = self.ARENA_CENTER + normal * (self.ARENA_RADIUS - self.BALL_RADIUS)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- 3. CHECK INTERACTIONS & REWARD ---
        collected_orbs_indices = set()
        for i, orb in enumerate(self.orbs):
            for ball in self.balls:
                if ball['pos'].distance_to(orb['pos']) < self.BALL_RADIUS + self.ORB_RADIUS:
                    if i not in collected_orbs_indices:
                        # # sfx: collect_orb
                        collected_orbs_indices.add(i)
                        self.score += 1
                        reward += 1.0
                        self._create_particles(orb['pos'], ball['color'])
                        self._spawn_ball() # Spawn a new ball on collection
                        break # Next orb

        # Remove collected orbs and spawn new ones
        if collected_orbs_indices:
            self.orbs = [orb for i, orb in enumerate(self.orbs) if i not in collected_orbs_indices]
            for _ in range(len(collected_orbs_indices)):
                self._spawn_orb()
        
        # Proximity reward
        min_dist_to_orb = float('inf')
        if self.orbs and self.balls:
            for ball in self.balls:
                for orb in self.orbs:
                    min_dist_to_orb = min(min_dist_to_orb, ball['pos'].distance_to(orb['pos']))
        
        if min_dist_to_orb < self.ORB_PROXIMITY_THRESHOLD:
            reward += 0.01 * (1 - min_dist_to_orb / self.ORB_PROXIMITY_THRESHOLD)


        # --- 4. CHECK TERMINATION ---
        terminated = (self.score >= self.WIN_SCORE) or (self.steps >= self.MAX_STEPS)
        truncated = self.steps >= self.MAX_STEPS
        if self.score >= self.WIN_SCORE:
            reward += 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Draw arena
        pygame.gfxdraw.aacircle(self.screen, int(self.ARENA_CENTER.x), int(self.ARENA_CENTER.y), self.ARENA_RADIUS, self.COLOR_ARENA_BORDER)
        pygame.gfxdraw.aacircle(self.screen, int(self.ARENA_CENTER.x), int(self.ARENA_CENTER.y), self.ARENA_RADIUS-1, self.COLOR_ARENA_BORDER)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            p_color = (*p['color'], alpha)
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                self._draw_glowing_circle(self.screen, p_color, p['pos'], radius, 0.5)

        # Draw orbs (pulsating)
        pulse = (math.sin(self.steps * 0.1) + 1) / 2  # 0 to 1
        orb_glow = 0.4 + pulse * 0.6
        for orb in self.orbs:
            self._draw_glowing_circle(self.screen, self.COLOR_ORB, orb['pos'], self.ORB_RADIUS, orb_glow)

        # Draw balls
        for ball in self.balls:
            self._draw_glowing_circle(self.screen, ball['color'], ball['pos'], self.BALL_RADIUS, 0.8)

        # Draw UI
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls": len(self.balls),
        }

    def _render_ui(self):
        score_text = f"Orbs: {self.score}/{self.WIN_SCORE}"
        text_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (15, 10))

        steps_text = f"Time: {self.MAX_STEPS - self.steps}"
        text_surface = self.font.render(steps_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(text_surface, text_rect)

    def _spawn_ball(self):
        new_color = self.NEON_COLORS[len(self.balls) % len(self.NEON_COLORS)]
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(1.0, 2.0)
        
        self.balls.append({
            'pos': self.ARENA_CENTER.copy(),
            'vel': vel,
            'color': new_color,
        })

    def _spawn_orb(self):
        # Spawn in a random location within the arena
        for _ in range(100): # Attempt to find a clear spot
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(0, self.ARENA_RADIUS - self.ORB_RADIUS - 20)
            pos = self.ARENA_CENTER + pygame.Vector2(math.cos(angle), math.sin(angle)) * radius
            
            # Ensure it's not too close to other orbs or balls
            too_close = False
            for orb in self.orbs:
                if pos.distance_to(orb['pos']) < self.ORB_RADIUS * 4:
                    too_close = True
                    break
            if not too_close:
                for ball in self.balls:
                    if pos.distance_to(ball['pos']) < self.BALL_RADIUS * 4:
                        too_close = True
                        break
            
            if not too_close:
                self.orbs.append({'pos': pos})
                return
        
        # Failsafe if no clear spot found
        self.orbs.append({'pos': self.ARENA_CENTER.copy()})


    def _create_particles(self, pos, color):
        num_particles = 20
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 4.0)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'life': life,
                'max_life': life,
                'radius': self.np_random.uniform(2, 5)
            })

    @staticmethod
    def _draw_glowing_circle(surface, color, pos, radius, glow_factor):
        """Draws a circle with a soft glow effect."""
        x, y = int(pos.x), int(pos.y)
        r, g, b = color[:3]

        # Draw multiple layers for the glow
        for i in range(int(radius * 2 * glow_factor), 0, -2):
            alpha = int(30 * (1 - (i / (radius * 2 * glow_factor)))**2)
            if alpha > 0:
                try:
                    pygame.gfxdraw.aacircle(surface, x, y, int(radius + i / 2), (r, g, b, alpha))
                except TypeError: # Sometimes alpha is not accepted, fallback
                    pygame.gfxdraw.aacircle(surface, x, y, int(radius + i / 2), (r, g, b))


        # Draw the main circle
        pygame.gfxdraw.aacircle(surface, x, y, int(radius), color)
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius), color)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncing Balls Collector")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0.0

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    print("R: Reset")

    while not terminated and not truncated:
        movement_action = 0 # None
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement_action, space_action, shift_action]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    terminated = False
                    truncated = False
                    print("--- Game Reset ---")

        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        truncated = trunc
        total_reward += reward
        
        # Draw the observation to the display screen
        draw_surface = pygame.surfarray.make_surface(env.render())
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

        clock.tick(30) # Run at 30 FPS

    env.close()