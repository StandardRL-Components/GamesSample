import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:25:04.207331
# Source Brief: brief_01685.md
# Brief Index: 1685
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent manipulates gravity to guide falling orbs
    into a collector zone. The game is inspired by minimalist neon aesthetics.

    **Objective:** Collect as many orbs as possible and survive for 12 levels.
    **Actions:** Control the direction of horizontal gravity (left, right, or neutral).
    **Rewards:** Awarded for surviving, collecting orbs, and leveling up.
    **Termination:** The episode ends if an orb hits the ground, the agent completes
    all 12 levels, or the maximum step count is reached.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    game_description = (
        "Manipulate horizontal gravity to guide falling orbs into a collector. "
        "Survive for as long as possible and complete all levels."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to control the direction of horizontal gravity. "
        "Press ↓ to set neutral gravity."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 385
    MAX_STEPS = 10000
    MAX_LEVEL = 12
    ORBS_PER_LEVEL = 10
    MAX_ACTIVE_ORBS = 50

    # --- Colors ---
    COLOR_BG = (10, 10, 20)
    COLOR_GROUND = (100, 100, 120)
    COLOR_COLLECTOR_GLOW = (80, 80, 200)
    COLOR_TEXT = (220, 220, 240)
    COLOR_INDICATOR = (255, 255, 255)

    # --- Physics ---
    VERTICAL_GRAVITY = 0.03
    HORIZONTAL_GRAVITY_STRENGTH = 0.08
    FRICTION = 0.99
    INITIAL_ORB_SPEED = 1.0
    SPEED_INCREASE_ON_COLLECT = 1.05 # 5% faster

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        if self.render_mode == "human":
            pygame.display.set_caption("Gravity Well")
            self.display_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.level = 1
        self.orbs_collected_for_level_up = 0
        self.total_orbs_collected = 0
        self.gravity_direction = 0  # -1 for left, 0 for neutral, 1 for right
        self.orbs = []
        self.particles = deque()
        self.np_random = None

        # --- Collector Zone ---
        collector_width = 120
        collector_height = 30
        self.collector_rect = pygame.Rect(
            (self.SCREEN_WIDTH - collector_width) / 2,
            self.GROUND_Y - collector_height,
            collector_width,
            collector_height,
        )


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed) # Python's random for compatibility with existing code

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.level = 1
        self.orbs_collected_for_level_up = 0
        self.total_orbs_collected = 0
        self.gravity_direction = 0
        self.orbs.clear()
        self.particles.clear()
        
        self._spawn_initial_orbs()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # 1. Unpack action and update gravity
        movement = action[0]
        if movement == 3:  # Left
            self.gravity_direction = -1
        elif movement == 4: # Right
            self.gravity_direction = 1
        else: # None, Up, Down
            self.gravity_direction = 0
        
        # 2. Update game state
        reward += self._update_orbs()
        self._update_particles()
        
        # 3. Calculate rewards and check for termination
        if not self.game_over:
            reward += 0.01  # Survival reward

        self.score += reward
        self.steps += 1
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_initial_orbs(self):
        self.orbs.clear()
        num_to_spawn = 1 + (self.level - 1) * 2
        for _ in range(num_to_spawn):
            if len(self.orbs) < self.MAX_ACTIVE_ORBS:
                self._spawn_orb(self.INITIAL_ORB_SPEED)

    def _spawn_orb(self, base_speed):
        hue = (self.total_orbs_collected * 13) % 360
        color = pygame.Color(0, 0, 0)
        color.hsva = (hue, 90, 100, 100)
        
        orb = {
            "pos": pygame.Vector2(random.uniform(50, self.SCREEN_WIDTH - 50), random.uniform(-50, -20)),
            "vel": pygame.Vector2(0, 0),
            "radius": random.uniform(7, 10),
            "color": tuple(color)[:3],
            "base_speed": base_speed,
        }
        self.orbs.append(orb)

    def _update_orbs(self):
        reward = 0.0
        
        for orb in self.orbs[:]:
            # Apply gravity
            orb["vel"].y += self.VERTICAL_GRAVITY * orb["base_speed"]
            orb["vel"].x += self.gravity_direction * self.HORIZONTAL_GRAVITY_STRENGTH
            
            # Apply friction
            orb["vel"].x *= self.FRICTION
            
            # Update position
            orb["pos"] += orb["vel"]
            
            # Screen wrap (horizontal)
            if orb["pos"].x < -orb["radius"]:
                orb["pos"].x = self.SCREEN_WIDTH + orb["radius"]
            elif orb["pos"].x > self.SCREEN_WIDTH + orb["radius"]:
                orb["pos"].x = -orb["radius"]

            # Check for collection
            if self.collector_rect.collidepoint(orb["pos"]):
                # Sound: Orb collection
                self._create_particles(orb["pos"], orb["color"], 20)
                self.orbs.remove(orb)
                
                reward += 1.0
                self.total_orbs_collected += 1
                self.orbs_collected_for_level_up += 1
                
                # Spawn new, faster orb
                if len(self.orbs) < self.MAX_ACTIVE_ORBS:
                    self._spawn_orb(orb["base_speed"] * self.SPEED_INCREASE_ON_COLLECT)
                
                if self.orbs_collected_for_level_up >= self.ORBS_PER_LEVEL:
                    self.level += 1
                    self.orbs_collected_for_level_up = 0
                    reward += 100.0
                    # Sound: Level up
                    if self.level > self.MAX_LEVEL:
                        self.game_over = True
                        reward += 1000.0 # Victory bonus
                    else:
                        self._spawn_initial_orbs()

            # Check for loss condition
            elif orb["pos"].y > self.GROUND_Y - orb["radius"]:
                # Sound: Fail
                self._create_particles(orb["pos"], self.COLOR_GROUND, 40, is_failure=True)
                self.game_over = True
                reward = -100.0
                break # End updates immediately on loss

        return reward

    def _create_particles(self, pos, color, count, is_failure=False):
        for _ in range(count):
            if is_failure:
                angle = random.uniform(math.pi, 2 * math.pi) # Upwards explosion
                speed = random.uniform(1, 4)
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.5, 2.5)
            
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "color": color,
                "lifespan": random.randint(20, 40)
            })

    def _update_particles(self):
        for i in range(len(self.particles) -1, -1, -1):
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Particle friction
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_to_surface()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "orbs_on_screen": len(self.orbs),
            "total_orbs_collected": self.total_orbs_collected,
        }

    def _render_to_surface(self):
        self.screen.fill(self.COLOR_BG)
        self._render_collector_zone()
        self._render_particles()
        self._render_orbs()
        self._render_ground()
        self._render_ui()

    def _render_collector_zone(self):
        # Draw glowing layers for the collector
        for i in range(10, 0, -1):
            alpha = 15 - i
            color = (*self.COLOR_COLLECTOR_GLOW, alpha)
            rect = self.collector_rect.inflate(i * 2, i * 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BG, self.collector_rect, border_radius=5)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p["lifespan"] * 6)))
            color = (*p["color"], alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (1, 1), 1)
            self.screen.blit(s, (int(p["pos"].x - 1), int(p["pos"].y - 1)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_orbs(self):
        pulse = math.sin(self.steps * 0.1) * 2
        for orb in self.orbs:
            pos = (int(orb["pos"].x), int(orb["pos"].y))
            radius = int(orb["radius"])
            
            # Glow effect
            glow_radius = int(radius * 1.8 + pulse)
            if glow_radius > 0:
                glow_color = (*orb["color"], 100)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Orb body
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, orb["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, orb["color"])

    def _render_ground(self):
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 2)

    def _render_ui(self):
        # Level Display
        level_text = self.font_large.render(f"LEVEL {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (20, 10))

        # Orb Count Display
        count_text = self.font_small.render(f"ORBS: {len(self.orbs)}/{self.MAX_ACTIVE_ORBS}", True, self.COLOR_TEXT)
        self.screen.blit(count_text, (self.SCREEN_WIDTH - count_text.get_width() - 20, 20))
        
        # Gravity Indicator
        indicator_pos = (self.SCREEN_WIDTH // 2, 30)
        if self.gravity_direction == -1: # Left
            points = [(indicator_pos[0] - 10, indicator_pos[1]), (indicator_pos[0] + 5, indicator_pos[1] - 8), (indicator_pos[0] + 5, indicator_pos[1] + 8)]
            pygame.draw.polygon(self.screen, self.COLOR_INDICATOR, points)
        elif self.gravity_direction == 1: # Right
            points = [(indicator_pos[0] + 10, indicator_pos[1]), (indicator_pos[0] - 5, indicator_pos[1] - 8), (indicator_pos[0] - 5, indicator_pos[1] + 8)]
            pygame.draw.polygon(self.screen, self.COLOR_INDICATOR, points)
        else: # Neutral
            pygame.draw.circle(self.screen, self.COLOR_INDICATOR, indicator_pos, 5, 2)
            
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            end_text_str = "VICTORY!" if self.level > self.MAX_LEVEL else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        # human mode is handled in step/reset
    
    def _render_frame(self):
        if self.render_mode == "human":
            self._render_to_surface()
            self.display_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("←: Gravity Left")
    print("→: Gravity Right")
    print("↓: Neutral Gravity")
    print("R: Reset")
    print("Q: Quit")
    print("----------------------\n")

    while not (terminated or truncated):
        action = [0, 0, 0] # Default: no movement, no buttons
        
        # Manual keyboard control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3 # Left
        elif keys[pygame.K_RIGHT]:
            action[0] = 4 # Right
        elif keys[pygame.K_DOWN]:
            action[0] = 0 # None
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False
                print(f"Game Reset. Initial Info: {info}")

        if not (terminated or truncated):
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term
            truncated = trunc
            if terminated or truncated:
                print(f"Episode Finished. Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
                # Wait for reset or quit
                pass

    env.close()