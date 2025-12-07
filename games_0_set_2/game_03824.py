import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ↑↓←→ to fire. Hit 15 targets before time runs out."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced, top-down target practice game. Hit moving targets to score points against the clock."

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 100  # Effective steps per second for time calculation
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds
        self.WIN_SCORE = 15
        self.NUM_TARGETS = 8
        self.TARGET_RADIUS = 15
        self.TARGET_SPEED_RANGE = (1.5, 3.0)
        self.PROJECTILE_SPEED = 12.0
        self.PROJECTILE_RADIUS = 4
        self.HIT_MARKER_DURATION = 15 # steps
        self.HIT_MARKER_MAX_RADIUS = 30

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PLAYER = (200, 200, 220)
        self.COLOR_PLAYER_OUTLINE = (120, 120, 140)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.TARGET_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
        ]
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HIT_MARKER = (255, 255, 255)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        self.center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.targets = []
        self.projectile = None
        self.hit_markers = []
        self.np_random = None # Will be seeded in reset
        
    def _spawn_target(self, index):
        edge = self.np_random.integers(0, 4)
        pos = pygame.Vector2(0, 0)
        vel = pygame.Vector2(0, 0)
        speed = self.np_random.uniform(self.TARGET_SPEED_RANGE[0], self.TARGET_SPEED_RANGE[1])
        
        if edge == 0:  # Top
            pos.x = self.np_random.uniform(0, self.WIDTH)
            pos.y = -self.TARGET_RADIUS
            angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
        elif edge == 1:  # Bottom
            pos.x = self.np_random.uniform(0, self.WIDTH)
            pos.y = self.HEIGHT + self.TARGET_RADIUS
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        elif edge == 2:  # Left
            pos.x = -self.TARGET_RADIUS
            pos.y = self.np_random.uniform(0, self.HEIGHT)
            angle = self.np_random.uniform(-math.pi * 0.25, math.pi * 0.25)
        else:  # Right
            pos.x = self.WIDTH + self.TARGET_RADIUS
            pos.y = self.np_random.uniform(0, self.HEIGHT)
            angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25)
            
        vel.from_polar((speed, math.degrees(angle)))
        
        return {
            "pos": pos,
            "vel": vel,
            "color": self.TARGET_COLORS[index % len(self.TARGET_COLORS)],
            "radius": self.TARGET_RADIUS
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.projectile = None
        self.hit_markers = []
        
        self.targets = [self._spawn_target(i) for i in range(self.NUM_TARGETS)]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        # --- Unpack Action ---
        fire_direction = action[0]
        
        # 1. Player Action (Fire Projectile)
        if fire_direction != 0 and self.projectile is None:
            vel = pygame.Vector2(0, 0)
            if fire_direction == 1: vel.y = -self.PROJECTILE_SPEED # Up
            elif fire_direction == 2: vel.y = self.PROJECTILE_SPEED # Down
            elif fire_direction == 3: vel.x = -self.PROJECTILE_SPEED # Left
            elif fire_direction == 4: vel.x = self.PROJECTILE_SPEED # Right
            
            self.projectile = {"pos": pygame.Vector2(self.center), "vel": vel}
            # SFX: Pew!
        
        # 2. Update Projectile
        if self.projectile:
            self.projectile["pos"] += self.projectile["vel"]
            p_pos = self.projectile["pos"]
            if not (0 <= p_pos.x < self.WIDTH and 0 <= p_pos.y < self.HEIGHT):
                self.projectile = None
        
        # 3. Update Targets and Check Collisions
        for i, target in enumerate(self.targets):
            target["pos"] += target["vel"]
            
            # Projectile collision
            if self.projectile and self.projectile["pos"].distance_to(target["pos"]) < target["radius"] + self.PROJECTILE_RADIUS:
                self.score += 1
                reward += 0.1
                # SFX: Target Hit!
                self.hit_markers.append({
                    "pos": pygame.Vector2(target["pos"]),
                    "timer": self.HIT_MARKER_DURATION,
                })
                self.projectile = None
                self.targets[i] = self._spawn_target(i) # Respawn
                # Break because projectile is gone
                break

            # Out of bounds check for target
            t_pos = target["pos"]
            if (t_pos.x < -self.TARGET_RADIUS * 2 or t_pos.x > self.WIDTH + self.TARGET_RADIUS * 2 or
                t_pos.y < -self.TARGET_RADIUS * 2 or t_pos.y > self.HEIGHT + self.TARGET_RADIUS * 2):
                self.targets[i] = self._spawn_target(i)

        # 4. Update Hit Markers
        self.hit_markers = [m for m in self.hit_markers if m["timer"] > 0]
        for marker in self.hit_markers:
            marker["timer"] -= 1

        # 5. Check Termination
        terminated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100.0
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 100.0
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_game(self):
        # Render targets
        for target in self.targets:
            x, y = int(target["pos"].x), int(target["pos"].y)
            pygame.gfxdraw.aacircle(self.screen, x, y, target["radius"], target["color"])
            pygame.gfxdraw.filled_circle(self.screen, x, y, target["radius"], target["color"])

        # Render projectile
        if self.projectile:
            x, y = int(self.projectile["pos"].x), int(self.projectile["pos"].y)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        # Render player turret
        center_x, center_y = int(self.center.x), int(self.center.y)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 8, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 8, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 6, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 6, self.COLOR_PLAYER)

        # Render hit markers
        for marker in self.hit_markers:
            progress = 1.0 - (marker["timer"] / self.HIT_MARKER_DURATION)
            radius = int(self.HIT_MARKER_MAX_RADIUS * progress)
            alpha = int(255 * (1.0 - progress))
            color = (*self.COLOR_HIT_MARKER, alpha)
            
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, color)
            self.screen.blit(temp_surf, (int(marker["pos"].x) - radius, int(marker["pos"].y) - radius))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # Game Over Message
        if self.game_over:
            message = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME UP!"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually.
    # It will not open a window because the environment is configured to be headless.
    # To play interactively, you would need to modify the environment's __init__
    # to handle a "human" render_mode that uses pygame.display.
    
    # The following code is for demonstration and will run headlessly.
    # You can see the print statements in the console.
    
    # Temporarily unset the headless environment variable to allow display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # Set up the display window
    pygame.display.set_caption("Target Practice")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            
        clock.tick(60) # Limit frame rate for playability

    env.close()