import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a magnet to attract and collect all the metallic spheres before time runs out. "
        "Collecting spheres increases your magnetic pull."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move the magnet around the screen."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 75 * FPS  # 75 seconds

    # Colors
    COLOR_BG = (44, 62, 80)  # Dark Blue/Gray
    COLOR_MAGNET = (231, 76, 60)  # Red
    COLOR_MAGNET_PULL_RADIUS = (231, 76, 60, 50) # Semi-transparent Red
    COLOR_SPHERE = (189, 195, 199)  # Silver
    COLOR_PARTICLE = (241, 196, 15) # Yellow/Gold
    COLOR_TRAIL = (236, 240, 241) # Light Gray
    COLOR_UI_TEXT = (236, 240, 241) # Light Gray

    # Game Parameters
    TOTAL_SPHERES = 30
    MAGNET_START_PULL_RADIUS = 50.0
    MAGNET_SIZE = 12
    SPHERE_SIZE = 5
    MAGNET_ACCELERATION = 0.6
    MAGNET_DRAG = 0.96
    SPHERE_DRAG = 0.98
    ATTRACTION_STRENGTH = 0.05
    CHAIN_REACTION_THRESHOLD = 5
    CHAIN_REACTION_BONUS_SPHERES = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0

        self.magnet_pos = pygame.Vector2(0, 0)
        self.magnet_vel = pygame.Vector2(0, 0)
        self.magnet_pull_radius = 0

        self.spheres = []
        self.particles = []
        self.trail = deque(maxlen=20)
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS

        self.magnet_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.magnet_vel = pygame.Vector2(0, 0)
        self.magnet_pull_radius = self.MAGNET_START_PULL_RADIUS

        self.spheres = []
        for _ in range(self.TOTAL_SPHERES):
            self.spheres.append({
                "pos": pygame.Vector2(
                    self.np_random.uniform(20, self.SCREEN_WIDTH - 20),
                    self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)
                ),
                "vel": pygame.Vector2(0, 0)
            })

        self.particles = []
        self.trail.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        reward = 0.0
        
        # --- 1. Handle Input & Update Magnet ---
        movement = action[0]
        acc = pygame.Vector2(0, 0)
        if movement == 1: acc.y = -1 # Up
        elif movement == 2: acc.y = 1  # Down
        elif movement == 3: acc.x = -1 # Left
        elif movement == 4: acc.x = 1  # Right
        
        self.magnet_vel += acc * self.MAGNET_ACCELERATION
        self.magnet_vel *= self.MAGNET_DRAG
        self.magnet_pos += self.magnet_vel

        # Clamp magnet to screen
        self.magnet_pos.x = np.clip(self.magnet_pos.x, self.MAGNET_SIZE, self.SCREEN_WIDTH - self.MAGNET_SIZE)
        self.magnet_pos.y = np.clip(self.magnet_pos.y, self.MAGNET_SIZE, self.SCREEN_HEIGHT - self.MAGNET_SIZE)
        
        self.trail.append(self.magnet_pos.copy())

        # --- 2. Update Spheres & Check Collection ---
        spheres_collected_this_step = 0
        
        for sphere in reversed(self.spheres):
            dist_vec = self.magnet_pos - sphere["pos"]
            dist = dist_vec.length()

            # Attraction physics
            if dist > 1 and dist < self.magnet_pull_radius:
                force = dist_vec.normalize() * (self.magnet_pull_radius - dist) * self.ATTRACTION_STRENGTH
                sphere["vel"] += force
            
            sphere["vel"] *= self.SPHERE_DRAG
            sphere["pos"] += sphere["vel"]

            # Collection check
            if dist < self.MAGNET_SIZE + self.SPHERE_SIZE:
                self.spheres.remove(sphere)
                spheres_collected_this_step += 1
                self.score += 1
                reward += 0.1
                self.magnet_pull_radius += 0.5
                self._spawn_particles(sphere["pos"], 15)
                # Play collection sound

        # --- 3. Chain Reaction Logic ---
        if spheres_collected_this_step >= self.CHAIN_REACTION_THRESHOLD:
            reward += 1.0
            # Play chain reaction sound
            bonus_spheres_to_collect = min(self.CHAIN_REACTION_BONUS_SPHERES, len(self.spheres))
            for _ in range(bonus_spheres_to_collect):
                if self.spheres:
                    idx = self.np_random.integers(0, len(self.spheres))
                    bonus_sphere = self.spheres.pop(idx)
                    self.score += 1
                    reward += 0.1 # Reward for bonus spheres too
                    self.magnet_pull_radius += 0.5
                    self._spawn_particles(bonus_sphere["pos"], 25, self.COLOR_MAGNET)

        # --- 4. Update Particles ---
        self._update_particles()
        
        # --- 5. Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.score >= self.TOTAL_SPHERES:
            terminated = True
            reward += 100.0
            # Play victory sound
        elif self.time_remaining <= 0:
            terminated = True
            reward -= 100.0
            # Play failure sound
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_particles(self, pos, count, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifetime": lifetime, "color": color})

    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Particle drag
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Magnet Trail
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                alpha = int(255 * (i / len(self.trail)))
                pygame.draw.line(
                    self.screen, 
                    (*self.COLOR_TRAIL, alpha),
                    (int(self.trail[i].x), int(self.trail[i].y)),
                    (int(self.trail[i+1].x), int(self.trail[i+1].y)),
                    max(1, int(self.MAGNET_SIZE * 0.5 * (i / len(self.trail))))
                )

        # Render Magnet Pull Radius
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.magnet_pos.x), int(self.magnet_pos.y),
            int(self.magnet_pull_radius), self.COLOR_MAGNET_PULL_RADIUS
        )

        # Render Spheres
        for sphere in self.spheres:
            pygame.gfxdraw.filled_circle(
                self.screen, int(sphere["pos"].x), int(sphere["pos"].y),
                self.SPHERE_SIZE, self.COLOR_SPHERE
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(sphere["pos"].x), int(sphere["pos"].y),
                self.SPHERE_SIZE, self.COLOR_SPHERE
            )

        # Render Magnet
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.magnet_pos.x), int(self.magnet_pos.y),
            self.MAGNET_SIZE, self.COLOR_MAGNET
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.magnet_pos.x), int(self.magnet_pos.y),
            self.MAGNET_SIZE, self.COLOR_MAGNET
        )

        # Render Particles
        for p in self.particles:
            size = max(1, int(3 * (p["lifetime"] / 30)))
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), size)

    def _render_ui(self):
        # Score Display
        score_text = self.font_large.render(f"Collected: {self.score}/{self.TOTAL_SPHERES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time Display
        time_sec = self.time_remaining / self.FPS
        time_text = self.font_large.render(f"Time: {time_sec:.1f}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game Over Message
        if self.game_over:
            if self.score >= self.TOTAL_SPHERES:
                msg = "SUCCESS!"
                color = (46, 204, 113) # Green
            else:
                msg = "TIME UP!"
                color = (231, 76, 60) # Red
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "spheres_left": len(self.spheres),
        }

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
        
        # Test game-specific assertions
        self.reset()
        assert self.magnet_pos.x > 0 and self.magnet_pos.x < self.SCREEN_WIDTH
        assert self.magnet_pos.y > 0 and self.magnet_pos.y < self.SCREEN_HEIGHT
        assert len(self.spheres) == self.TOTAL_SPHERES
        assert self.time_remaining == self.MAX_STEPS

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This part of the code is for manual play and will not run in the headless evaluation
    # so we can re-enable the display here.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Magnet Sphere Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = np.array([0, 0, 0]) # No-op
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Key mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    
    # Keep window open for a few seconds to see final score
    pygame.time.wait(3000)
    env.close()