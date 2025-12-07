import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:47:33.091835
# Source Brief: brief_00034.md
# Brief Index: 34
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent controls gravity wells to capture asteroids.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Well activation (0=none, 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right)
    - `actions[1]`: Unused (space button)
    - `actions[2]`: Unused (shift button)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game screen.

    **Reward Structure:**
    - +1.0 for each asteroid captured.
    - +0.01 for each asteroid within the influence radius of any well.
    - -0.001 time penalty per step.
    - +10 per captured asteroid as a bonus at the end of the episode.

    **Termination:**
    - The episode ends after 5400 steps (90 seconds at 60 FPS).
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Control powerful gravity wells to capture drifting asteroids. "
        "Activate the correct well to pull asteroids into its capture zone before time runs out."
    )
    user_guide = (
        "Activate one of the four corner gravity wells to capture asteroids. "
        "A no-op action is also available. The other actions (space/shift) are unused."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 5400  # 90 seconds * 60 FPS
    NUM_ASTEROIDS = 8
    NUM_STARS = 100

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_STAR = (100, 100, 120)
    COLOR_ASTEROID = (180, 180, 190)
    COLOR_ASTEROID_OUTLINE = (120, 120, 130)
    COLOR_WELL = (0, 150, 255)
    COLOR_WELL_INACTIVE = (60, 80, 100)
    COLOR_WELL_ACTIVE_GLOW = (180, 220, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CAPTURE_FLASH = (255, 255, 255)

    # Physics
    INITIAL_ASTEROID_SPEED = 1.0
    GRAVITY_CONSTANT = 0.05
    DIFFICULTY_INTERVAL = 300 # steps
    DIFFICULTY_SPEED_INCREASE = 0.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_well = pygame.font.SysFont("Consolas", 16, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wells = []
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.active_well_idx = -1
        self.base_asteroid_speed = self.INITIAL_ASTEROID_SPEED
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.active_well_idx = -1
        self.base_asteroid_speed = self.INITIAL_ASTEROID_SPEED

        # --- Initialize Game Elements ---
        self._create_stars()
        self._initialize_wells()
        self._initialize_asteroids()
        self.particles = []
        
        # --- Return initial observation and info ---
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.001  # Small time penalty

        # --- Handle Action ---
        well_action = action[0]
        self.active_well_idx = well_action - 1 if well_action > 0 else -1
        
        # Visual feedback for activating a well
        if self.active_well_idx != -1:
            self._spawn_activation_effect(self.wells[self.active_well_idx]['pos'])

        # --- Update Game Logic ---
        self._update_asteroids()
        self._update_particles()
        reward += self._handle_captures()
        reward += self._calculate_proximity_reward()
        self._increase_difficulty()

        # --- Check Termination ---
        terminated = self.steps >= self.MAX_STEPS
        truncated = False
        if terminated:
            self.game_over = True
            reward += self.score * 10 # Final bonus reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / self.metadata['render_fps'],
            "asteroid_speed": self.base_asteroid_speed
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    # ==========================================================================
    # Private Helper Methods: Game Logic
    # ==========================================================================

    def _initialize_wells(self):
        self.wells = []
        margin_x, margin_y = 120, 100
        positions = [
            (margin_x, margin_y),
            (self.WIDTH - margin_x, margin_y),
            (margin_x, self.HEIGHT - margin_y),
            (self.WIDTH - margin_x, self.HEIGHT - margin_y),
        ]
        for pos in positions:
            self.wells.append({
                'pos': pygame.math.Vector2(pos),
                'strength': 30, # Base influence radius
                'capture_radius': 15,
                'captures': 0
            })

    def _initialize_asteroids(self):
        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            self._spawn_asteroid()

    def _spawn_asteroid(self):
        # Spawn on an edge
        edge = self.np_random.integers(4)
        if edge == 0: # top
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -20)
        elif edge == 1: # bottom
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
        elif edge == 2: # left
            pos = pygame.math.Vector2(-20, self.np_random.uniform(0, self.HEIGHT))
        else: # right
            pos = pygame.math.Vector2(self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.base_asteroid_speed * self.np_random.uniform(0.8, 1.2)
        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed

        # Generate a random polygonal shape
        num_points = self.np_random.integers(5, 9)
        points = []
        base_radius = self.np_random.uniform(8, 12)
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            radius = base_radius + self.np_random.uniform(-2, 2)
            points.append((math.cos(angle) * radius, math.sin(angle) * radius))

        self.asteroids.append({
            'pos': pos,
            'vel': vel,
            'shape': points,
            'angle': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-2, 2)
        })

    def _update_asteroids(self):
        active_well = self.wells[self.active_well_idx] if self.active_well_idx != -1 else None

        for asteroid in self.asteroids:
            # Apply gravity from the active well
            if active_well:
                dist_vec = active_well['pos'] - asteroid['pos']
                dist = dist_vec.length()
                if 1 < dist < active_well['strength'] * 2:
                    # Force weakens with distance but is capped to avoid extreme pulls
                    force_magnitude = (active_well['strength'] / 10) * self.GRAVITY_CONSTANT * (1 - dist / (active_well['strength'] * 2))
                    force = dist_vec.normalize() * force_magnitude
                    asteroid['vel'] += force
            
            # Update position and rotation
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] = (asteroid['angle'] + asteroid['rot_speed']) % 360

            # Screen bounce
            if asteroid['pos'].x < 0 or asteroid['pos'].x > self.WIDTH:
                asteroid['vel'].x *= -1
                asteroid['pos'].x = max(0, min(self.WIDTH, asteroid['pos'].x))
            if asteroid['pos'].y < 0 or asteroid['pos'].y > self.HEIGHT:
                asteroid['vel'].y *= -1
                asteroid['pos'].y = max(0, min(self.HEIGHT, asteroid['pos'].y))

    def _handle_captures(self):
        capture_reward = 0
        active_well = self.wells[self.active_well_idx] if self.active_well_idx != -1 else None
        
        if not active_well:
            return 0

        for i in range(len(self.asteroids) - 1, -1, -1):
            asteroid = self.asteroids[i]
            dist = (active_well['pos'] - asteroid['pos']).length()
            if dist < active_well['capture_radius']:
                # --- CAPTURE ---
                # SFX: Capture sound
                self._spawn_capture_particles(asteroid['pos'])
                self.asteroids.pop(i)
                self._spawn_asteroid()

                self.score += 1
                active_well['captures'] += 1
                active_well['strength'] += 2 # Increase influence radius
                capture_reward += 1.0
        
        return capture_reward
    
    def _calculate_proximity_reward(self):
        proximity_reward = 0
        for asteroid in self.asteroids:
            for well in self.wells:
                dist = (well['pos'] - asteroid['pos']).length()
                if dist < well['strength']:
                    proximity_reward += 0.01
        return proximity_reward

    def _increase_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.base_asteroid_speed += self.DIFFICULTY_SPEED_INCREASE

    # ==========================================================================
    # Private Helper Methods: Visuals & Rendering
    # ==========================================================================
    
    def _create_stars(self):
        if self.np_random is None:
            self.reset()
        self.stars = []
        for _ in range(self.NUM_STARS):
            self.stars.append({
                'pos': (self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                'size': self.np_random.uniform(0.5, 1.5)
            })

    def _spawn_activation_effect(self, pos):
        # SFX: Well activation hum
        for i in range(3):
            self.particles.append({
                'type': 'ring', 'pos': pos, 'radius': 10, 'max_radius': 60,
                'lifespan': 20, 'max_lifespan': 20, 'speed': 2 + i * 0.5,
                'color': self.COLOR_WELL_ACTIVE_GLOW
            })

    def _spawn_capture_particles(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'type': 'spark', 'pos': pos.copy(), 'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': random.choice([self.COLOR_CAPTURE_FLASH, self.COLOR_WELL, self.COLOR_WELL_ACTIVE_GLOW])
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.pop(i)
                continue
            
            if p['type'] == 'ring':
                p['radius'] += p['speed']
            elif p['type'] == 'spark':
                p['pos'] += p['vel']
                p['vel'] *= 0.95 # friction

    def _render_all(self):
        # --- Clear screen with background ---
        self.screen.fill(self.COLOR_BG)

        # --- Render game elements ---
        self._render_stars()
        self._render_particles()
        self._render_wells()
        self._render_asteroids()

        # --- Render UI overlay ---
        self._render_ui()
        
        self.clock.tick(self.metadata["render_fps"])

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, star['pos'], star['size'])

    def _render_wells(self):
        for i, well in enumerate(self.wells):
            pos_int = (int(well['pos'].x), int(well['pos'].y))
            is_active = (i == self.active_well_idx)
            
            # Draw influence radius (semi-transparent)
            influence_radius = int(well['strength'])
            if influence_radius > 0:
                # Use a surface for transparency
                s = pygame.Surface((influence_radius*2, influence_radius*2), pygame.SRCALPHA)
                color = self.COLOR_WELL if is_active else self.COLOR_WELL_INACTIVE
                pygame.draw.circle(s, (*color, 30), (influence_radius, influence_radius), influence_radius)
                self.screen.blit(s, (pos_int[0] - influence_radius, pos_int[1] - influence_radius))
            
            # Draw main well circle
            color = self.COLOR_WELL if is_active else self.COLOR_WELL_INACTIVE
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(well['capture_radius']), color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(well['capture_radius']), color)

            # Draw strength text
            strength_text = self.font_well.render(str(well['captures']), True, self.COLOR_TEXT)
            text_rect = strength_text.get_rect(center=pos_int)
            self.screen.blit(strength_text, text_rect)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            # Rotate shape points
            rotated_points = []
            angle_rad = math.radians(asteroid['angle'])
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            for p in asteroid['shape']:
                x = p[0] * cos_a - p[1] * sin_a + asteroid['pos'].x
                y = p[0] * sin_a + p[1] * cos_a + asteroid['pos'].y
                rotated_points.append((x, y))
            
            if len(rotated_points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_ASTEROID_OUTLINE)

    def _render_particles(self):
        for p in self.particles:
            if p['type'] == 'ring':
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                color = (*p['color'], alpha)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)
            elif p['type'] == 'spark':
                alpha = 255 * (p['lifespan'] / 40)
                color = (*p['color'], alpha)
                s = pygame.Surface((4,4), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (2,2), 2)
                self.screen.blit(s, (int(p['pos'].x)-2, int(p['pos'].y)-2))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.metadata['render_fps'])
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        text_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, text_rect)


# Example usage
if __name__ == '__main__':
    # This validation section is not part of the required fix, but was in the original code.
    # It's helpful for local testing.
    def validate_implementation(env_instance):
        """Call this at the end of __init__ to verify implementation."""
        print("Validating implementation...")
        # Test action space
        assert env_instance.action_space.shape == (3,)
        assert env_instance.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs, _ = env_instance.reset()
        assert test_obs.shape == (env_instance.HEIGHT, env_instance.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = env_instance.reset()
        assert obs.shape == (env_instance.HEIGHT, env_instance.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = env_instance.action_space.sample()
        obs, reward, term, trunc, info = env_instance.step(test_action)
        assert obs.shape == (env_instance.HEIGHT, env_instance.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    env = GameEnv(render_mode="rgb_array")
    validate_implementation(env)

    # --- Manual Play Loop ---
    # Un-comment the block below to run the game with manual controls
    # Note: This requires a display and will not run in a pure headless environment
    
    # from gymnasium.utils.play import play
    # play(GameEnv(), keys_to_action={
    #     "q": np.array([1, 0, 0]),
    #     "w": np.array([2, 0, 0]),
    #     "a": np.array([3, 0, 0]),
    #     "s": np.array([4, 0, 0]),
    # }, noop=np.array([0, 0, 0]))

    # The original manual play loop is preserved below for reference
    # It requires `SDL_VIDEODRIVER` to be something other than "dummy"
    # os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS", etc.
    # env_manual = GameEnv(render_mode="rgb_array")
    # obs, info = env_manual.reset()
    # done = False
    
    # pygame.display.set_caption("Gravity Well - Manual Control")
    # screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    # action = np.array([0, 0, 0])
    
    # while not done:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_q: action[0] = 1
    #             if event.key == pygame.K_w: action[0] = 2
    #             if event.key == pygame.K_a: action[0] = 3
    #             if event.key == pygame.K_s: action[0] = 4
    #         if event.type == pygame.KEYUP:
    #             if event.key in [pygame.K_q, pygame.K_w, pygame.K_a, pygame.K_s]:
    #                 action[0] = 0

    #     obs, reward, terminated, truncated, info = env_manual.step(action)
    #     done = terminated or truncated

    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    # print(f"Game Over. Final Info: {info}")
    # env_manual.close()