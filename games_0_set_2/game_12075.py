import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:16:32.068343
# Source Brief: brief_02075.md
# Brief Index: 2075
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-crafted Gymnasium environment where a bouncing ball collects valuable
    orbs while avoiding penalty orbs in a gravity-based arena.

    This environment prioritizes visual quality and satisfying gameplay feel,
    featuring smooth physics, particle effects, and a clear, minimalist aesthetic.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) - Unused
    - actions[2]: Shift button (0=released, 1=held) - Unused
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Control a bouncing ball in a physics-based arena. Collect valuable green orbs for points "
        "while avoiding the red penalty orbs."
    )
    user_guide = "Controls: Use ← and → arrow keys to apply horizontal force to the ball."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    WIN_CONDITION_ORBS = 100

    # Colors (Bright interactive, dark background)
    COLOR_BG_TOP = (20, 25, 35)
    COLOR_BG_BOTTOM = (40, 45, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_LARGE_ORB = (80, 255, 120)
    COLOR_LARGE_ORB_GLOW = (80, 255, 120, 50)
    COLOR_SMALL_ORB = (255, 80, 80)
    COLOR_TEXT = (230, 230, 230)
    COLOR_PARTICLE_GOOD = (200, 255, 220)
    COLOR_PARTICLE_BAD = (255, 200, 200)

    # Physics
    GRAVITY = 0.3
    HORIZONTAL_FORCE = 0.6
    FRICTION = 0.95
    BOUNCE_DAMPENING = 0.85
    MAX_VELOCITY_X = 8
    MAX_VELOCITY_Y = 12

    # Player and Orb properties
    PLAYER_RADIUS = 15
    LARGE_ORB_RADIUS = 20
    SMALL_ORB_RADIUS = 7
    MAX_LARGE_ORBS = 10
    MAX_SMALL_ORBS = 50
    INITIAL_PENALTY_SPAWN_CHANCE = 0.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_big = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_big = pygame.font.SysFont(None, 32)
            self.font_small = pygame.font.SysFont(None, 24)

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.total_orbs_collected = 0
        self.penalty_spawn_chance = 0.0
        self.player_pos = None
        self.player_vel = None
        self.large_orbs = []
        self.small_orbs = []
        self.particles = []
        
        # Create a static background surface for performance
        self.background_surface = self._create_background()

        # self.reset() is called by the wrapper/runner
        # self.validate_implementation() is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.total_orbs_collected = 0
        self.penalty_spawn_chance = self.INITIAL_PENALTY_SPAWN_CHANCE

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)

        self.large_orbs = []
        self.small_orbs = []
        self.particles = []

        self._spawn_large_orb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, _, _ = action  # space_held and shift_held are unused

        # --- Continuous Reward Calculation (Part 1) ---
        prev_dist_to_orb = self._get_dist_to_nearest_large_orb()

        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_player()
        self._update_orbs()
        self._update_particles()
        
        # --- Handle Collisions & Spawning ---
        event_reward = self._handle_collisions()
        self._ensure_large_orb_exists()
        
        # --- Continuous Reward Calculation (Part 2) ---
        new_dist_to_orb = self._get_dist_to_nearest_large_orb()
        
        # Reward for moving closer to the nearest large orb
        continuous_reward = 0
        if prev_dist_to_orb is not None and new_dist_to_orb is not None:
            distance_delta = prev_dist_to_orb - new_dist_to_orb
            continuous_reward = distance_delta * 0.01  # Small reward for closing distance

        # --- Finalize Step ---
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        win_reward = 100.0 if self.total_orbs_collected >= self.WIN_CONDITION_ORBS else 0.0
        reward = event_reward + continuous_reward + win_reward
        self.score += event_reward # Score only reflects orb collection

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.player_vel[0] -= self.HORIZONTAL_FORCE
        elif movement == 4:  # Right
            self.player_vel[0] += self.HORIZONTAL_FORCE
    
    def _update_player(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        # Apply friction
        self.player_vel[0] *= self.FRICTION
        # Clamp velocity
        self.player_vel[0] = np.clip(self.player_vel[0], -self.MAX_VELOCITY_X, self.MAX_VELOCITY_X)
        self.player_vel[1] = np.clip(self.player_vel[1], -self.MAX_VELOCITY_Y, self.MAX_VELOCITY_Y)

        # Update position
        self.player_pos += self.player_vel

        # Wall bouncing
        if self.player_pos[0] - self.PLAYER_RADIUS < 0:
            self.player_pos[0] = self.PLAYER_RADIUS
            self.player_vel[0] *= -self.BOUNCE_DAMPENING
        elif self.player_pos[0] + self.PLAYER_RADIUS > self.SCREEN_WIDTH:
            self.player_pos[0] = self.SCREEN_WIDTH - self.PLAYER_RADIUS
            self.player_vel[0] *= -self.BOUNCE_DAMPENING

        if self.player_pos[1] - self.PLAYER_RADIUS < 0:
            self.player_pos[1] = self.PLAYER_RADIUS
            self.player_vel[1] *= -self.BOUNCE_DAMPENING
        elif self.player_pos[1] + self.PLAYER_RADIUS > self.SCREEN_HEIGHT:
            self.player_pos[1] = self.SCREEN_HEIGHT - self.PLAYER_RADIUS
            self.player_vel[1] *= -self.BOUNCE_DAMPENING

    def _update_orbs(self):
        for orb in self.large_orbs + self.small_orbs:
            orb['pos'] += orb['vel']
            # Wall bouncing for orbs
            if orb['pos'][0] - orb['radius'] < 0 or orb['pos'][0] + orb['radius'] > self.SCREEN_WIDTH:
                orb['vel'][0] *= -1
            if orb['pos'][1] - orb['radius'] < 0 or orb['pos'][1] + orb['radius'] > self.SCREEN_HEIGHT:
                orb['vel'][1] *= -1
            # Add slight random jitter to small orbs
            if orb['radius'] == self.SMALL_ORB_RADIUS:
                orb['vel'] += self.np_random.uniform(-0.1, 0.1, size=2)
                speed = np.linalg.norm(orb['vel'])
                if speed > 3: orb['vel'] = (orb['vel'] / speed) * 3
                if speed < 1.5: orb['vel'] = (orb['vel'] / max(speed, 0.1)) * 1.5


    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Large Orbs
        for orb in self.large_orbs[:]:
            dist = np.linalg.norm(self.player_pos - orb['pos'])
            if dist < self.PLAYER_RADIUS + orb['radius']:
                self.large_orbs.remove(orb)
                reward += 10.0
                self.total_orbs_collected += 1
                # sfx: positive collection sound
                self._create_particle_burst(orb['pos'], 20, self.COLOR_PARTICLE_GOOD, 5)

                if self.np_random.random() < self.penalty_spawn_chance:
                    self._spawn_penalty_orbs(orb['pos'], 3)
                
                # Increase difficulty
                self.penalty_spawn_chance = min(0.8, self.penalty_spawn_chance + 0.005)

        # Player vs Small Orbs
        for orb in self.small_orbs[:]:
            dist = np.linalg.norm(self.player_pos - orb['pos'])
            if dist < self.PLAYER_RADIUS + orb['radius']:
                self.small_orbs.remove(orb)
                reward -= 1.0
                # sfx: negative collection sound
                self._create_particle_burst(orb['pos'], 10, self.COLOR_PARTICLE_BAD, 2)
        
        return reward

    def _ensure_large_orb_exists(self):
        if not self.large_orbs and self.total_orbs_collected < self.WIN_CONDITION_ORBS:
            self._spawn_large_orb()

    def _spawn_large_orb(self):
        if len(self.large_orbs) < self.MAX_LARGE_ORBS:
            pos = self._get_safe_spawn_pos(self.LARGE_ORB_RADIUS)
            vel = self.np_random.uniform(-0.5, 0.5, size=2)
            self.large_orbs.append({'pos': pos, 'vel': vel, 'radius': self.LARGE_ORB_RADIUS})
            # sfx: spawn sound
            self._create_particle_burst(pos, 15, self.COLOR_LARGE_ORB, 3, is_spawn=True)

    def _spawn_penalty_orbs(self, position, count):
        for _ in range(count):
            if len(self.small_orbs) < self.MAX_SMALL_ORBS:
                pos = self._get_safe_spawn_pos(self.SMALL_ORB_RADIUS, near_pos=position)
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1.5, 3.0)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
                self.small_orbs.append({'pos': pos, 'vel': vel, 'radius': self.SMALL_ORB_RADIUS})
                # sfx: penalty spawn sound

    def _get_safe_spawn_pos(self, radius, near_pos=None):
        max_attempts = 100
        for _ in range(max_attempts):
            if near_pos is not None:
                 pos = near_pos + self.np_random.uniform(-50, 50, size=2)
            else:
                pos = self.np_random.uniform(
                    [radius, radius], 
                    [self.SCREEN_WIDTH - radius, self.SCREEN_HEIGHT - radius]
                )
            
            # Clamp to be safe
            pos[0] = np.clip(pos[0], radius, self.SCREEN_WIDTH - radius)
            pos[1] = np.clip(pos[1], radius, self.SCREEN_HEIGHT - radius)
            
            # Check collision with player
            if np.linalg.norm(pos - self.player_pos) < radius + self.PLAYER_RADIUS + 20:
                continue

            # Check collision with other orbs
            is_overlapping = False
            for orb in self.large_orbs + self.small_orbs:
                if np.linalg.norm(pos - orb['pos']) < radius + orb['radius'] + 10:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                return pos
        
        # Fallback if no safe spot is found
        return np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])

    def _get_dist_to_nearest_large_orb(self):
        if not self.large_orbs:
            return None
        distances = [np.linalg.norm(self.player_pos - orb['pos']) for orb in self.large_orbs]
        return min(distances)

    def _check_termination(self):
        return self.total_orbs_collected >= self.WIN_CONDITION_ORBS

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "orbs_collected": self.total_orbs_collected,
        }
        
    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            radius_int = int(max(0, p['radius']))
            if radius_int > 0:
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, (*p['color'], alpha))
        
        # Draw orbs
        for orb in self.large_orbs:
            self._draw_circle_with_glow(self.screen, orb['pos'], self.LARGE_ORB_RADIUS, self.COLOR_LARGE_ORB, self.COLOR_LARGE_ORB_GLOW)
        for orb in self.small_orbs:
            pos_int = (int(orb['pos'][0]), int(orb['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.SMALL_ORB_RADIUS, self.COLOR_SMALL_ORB)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.SMALL_ORB_RADIUS, self.COLOR_SMALL_ORB)
        
        # Draw player
        self._draw_circle_with_glow(self.screen, self.player_pos, self.PLAYER_RADIUS, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_ui(self):
        orbs_text = self.font_big.render(f"Orbs: {self.total_orbs_collected}/{self.WIN_CONDITION_ORBS}", True, self.COLOR_TEXT)
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(orbs_text, (15, 10))
        self.screen.blit(score_text, (15, 40))

    def _draw_circle_with_glow(self, surface, pos, radius, color, glow_color):
        pos_int = (int(pos[0]), int(pos[1]))
        # Draw multiple layers for a softer glow
        for i in range(3):
            glow_radius = int(radius + (4 - i) * 3)
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], glow_radius, glow_color)
        # Draw main circle
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius, color)

    def _create_particle_burst(self, pos, count, color, max_speed, is_spawn=False):
        lifespan = 30 if not is_spawn else 20
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            radius = self.np_random.uniform(2, 5) if not is_spawn else self.np_random.uniform(4, 8)
            self.particles.append({
                'pos': pos.copy(), 
                'vel': vel, 
                'radius': radius, 
                'lifespan': lifespan, 
                'max_lifespan': lifespan,
                'color': color
            })
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    # To see the game, unset SDL_VIDEODRIVER or set it to a valid driver.
    if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
        print("Cannot run main in a headless environment. Unset SDL_VIDEODRIVER to play manually.")
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Bouncing Orb Collector")
        clock = pygame.time.Clock()
        
        terminated = False
        truncated = False
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print("Left/Right Arrow Keys: Move")
        print("Q: Quit")
        
        while not (terminated or truncated):
            movement = 0 # No-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            if keys[pygame.K_RIGHT]:
                movement = 4

            action = [movement, 0, 0] # space and shift are not used

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    terminated = True

            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            
            # Convert observation back to a Pygame surface for display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Run at 30 FPS

        print(f"\nGame Over!")
        print(f"Final Score: {int(info['score'])}")
        print(f"Total Orbs Collected: {info['orbs_collected']}")
        print(f"Total Steps: {info['steps']}")
        print(f"Total Reward (for RL): {total_reward:.2f}")

        env.close()