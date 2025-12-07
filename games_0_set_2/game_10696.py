import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    """
    A steampunk-inspired puzzle game where the player controls the speed of two
    interconnected gears to manipulate platforms. The goal is to position these
    platforms to catch falling resources and collect a target amount before
    time runs out.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control two steampunk gears to position platforms and catch falling resources. "
        "Align platforms for bonuses and reach the target score before time runs out."
    )
    user_guide = (
        "Controls: Use ↑/↓ to control the speed of the first gear and ←/→ for the second gear. "
        "Position platforms to catch falling resources over the central bin."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (220, 220, 225)  # Light Grey
    COLOR_SCAFFOLD = (180, 180, 185)
    COLOR_GEAR = (205, 127, 50)  # Bronze
    COLOR_GEAR_DARK = (165, 97, 30)
    COLOR_PLATFORM = (80, 80, 90) # Dark Grey
    COLOR_RESOURCE = (0, 191, 255)  # Deep Sky Blue
    COLOR_RESOURCE_GLOW = (135, 206, 250, 100) # Light Sky Blue, translucent
    COLOR_BIN = (60, 60, 65)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_TIMER_BAR = (100, 200, 100)
    COLOR_TIMER_BAR_BG = (200, 100, 100)
    COLOR_PARTICLE = (255, 255, 100)

    # Game Parameters
    GAME_DURATION_SECONDS = 90
    WIN_SCORE = 200
    MAX_STEPS = GAME_DURATION_SECONDS * FPS + 10 # A bit of buffer
    
    GEAR_SPEED_ADJUST = 0.005 # Rate of speed change per step
    GEAR_MAX_SPEED = 0.1
    
    NUM_PLATFORMS = 10
    PLATFORM_WIDTH = 80
    PLATFORM_HEIGHT = 10
    
    RESOURCE_SPAWN_RATE = 0.1 # Chance to spawn per step
    RESOURCE_SPEED = 2.0
    RESOURCE_RADIUS = 5
    
    ALIGNMENT_BONUS_COUNT = 5
    ALIGNMENT_TOLERANCE = 10
    ALIGNMENT_FREEZE_DURATION = 1.0 # in seconds
    
    # Reward Structure
    REWARD_COLLECT = 0.5
    REWARD_ALIGNMENT = 5.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables to be defined in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.gear_speeds = None
        self.gear_angles = None
        self.platforms = None
        self.resources = None
        self.particles = None
        self.alignment_bonus_timer = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS
        
        self.gear_speeds = [0.01, -0.01]
        self.gear_angles = [0.0, math.pi]
        self.alignment_bonus_timer = 0
        
        self._init_platforms()
        self.resources = []
        self.particles = []
        
        # Collection bin definition
        self.collection_bin = pygame.Rect(
            self.SCREEN_WIDTH // 2 - 100, self.SCREEN_HEIGHT - 30, 200, 30
        )
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.time_remaining -= 1 / self.FPS
        
        if self.alignment_bonus_timer > 0:
            self.alignment_bonus_timer -= 1 / self.FPS
        
        self._handle_action(action)
        self._update_gears()
        self._update_platforms()
        self._update_resources()
        self._update_particles()
        
        collection_reward = self._check_collections()
        alignment_reward = self._check_alignment_bonus()
        reward += collection_reward + alignment_reward
        
        terminated = self.score >= self.WIN_SCORE or self.time_remaining <= 0
        truncated = self.steps >= self.MAX_STEPS
        
        self.game_over = terminated or truncated

        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += self.REWARD_WIN
            else: # Time ran out
                reward += self.REWARD_LOSE

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, _, _ = action
        
        # movement: 0=no-op, 1=up, 2=down, 3=right, 4=left
        # up/down control gear 1, left/right control gear 2
        
        if movement == 1: # Increase gear 1 speed
            self.gear_speeds[0] += self.GEAR_SPEED_ADJUST
        elif movement == 2: # Decrease gear 1 speed
            self.gear_speeds[0] -= self.GEAR_SPEED_ADJUST
        elif movement == 3: # Increase gear 2 speed
            self.gear_speeds[1] += self.GEAR_SPEED_ADJUST
        elif movement == 4: # Decrease gear 2 speed
            self.gear_speeds[1] -= self.GEAR_SPEED_ADJUST
            
        # Clamp speeds
        self.gear_speeds[0] = np.clip(self.gear_speeds[0], -self.GEAR_MAX_SPEED, self.GEAR_MAX_SPEED)
        self.gear_speeds[1] = np.clip(self.gear_speeds[1], -self.GEAR_MAX_SPEED, self.GEAR_MAX_SPEED)

    def _init_platforms(self):
        self.platforms = []
        y_spacing = (self.SCREEN_HEIGHT - 150) / self.NUM_PLATFORMS
        for i in range(self.NUM_PLATFORMS):
            gear_index = i % 2
            center_x = self.SCREEN_WIDTH / 2
            amplitude = self.SCREEN_WIDTH / 2 - self.PLATFORM_WIDTH / 2 - 20
            y_pos = 100 + i * y_spacing
            phase = (math.pi * 2 / self.NUM_PLATFORMS) * i
            
            self.platforms.append({
                "rect": pygame.Rect(0, 0, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT),
                "gear_index": gear_index,
                "center_x": center_x,
                "amplitude": amplitude,
                "y_pos": y_pos,
                "phase": phase
            })

    def _update_gears(self):
        if self.alignment_bonus_timer <= 0:
            self.gear_angles[0] += self.gear_speeds[0] * (self.FPS / 30)
            self.gear_angles[1] += self.gear_speeds[1] * (self.FPS / 30)

    def _update_platforms(self):
        for p in self.platforms:
            angle = self.gear_angles[p["gear_index"]] + p["phase"]
            p["rect"].centerx = p["center_x"] + p["amplitude"] * math.sin(angle)
            p["rect"].centery = p["y_pos"]
    
    def _update_resources(self):
        # Spawn new resources
        if self.np_random.random() < self.RESOURCE_SPAWN_RATE:
            spawn_x = self.np_random.uniform(self.SCREEN_WIDTH * 0.2, self.SCREEN_WIDTH * 0.8)
            self.resources.append({"pos": [spawn_x, -self.RESOURCE_RADIUS], "vel": self.RESOURCE_SPEED})
            
        # Move and remove old resources
        for res in self.resources[:]:
            res["pos"][1] += res["vel"]
            if res["pos"][1] > self.SCREEN_HEIGHT + self.RESOURCE_RADIUS:
                self.resources.remove(res)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_collections(self):
        collected_reward = 0
        for res in self.resources[:]:
            res_rect = pygame.Rect(res["pos"][0] - self.RESOURCE_RADIUS, res["pos"][1] - self.RESOURCE_RADIUS, self.RESOURCE_RADIUS * 2, self.RESOURCE_RADIUS * 2)
            for p in self.platforms:
                if p["rect"].colliderect(res_rect):
                    # Check if platform is over the collection bin
                    if p["rect"].left < self.collection_bin.right and p["rect"].right > self.collection_bin.left:
                        self.score += 1
                        collected_reward += self.REWARD_COLLECT
                        self.resources.remove(res)
                        self._spawn_particles(res["pos"])
                        break # Move to next resource
        return collected_reward

    def _check_alignment_bonus(self):
        if self.alignment_bonus_timer > 0:
            return 0
        
        sorted_platforms = sorted(self.platforms, key=lambda p: p["rect"].centerx)
        
        for i in range(len(sorted_platforms) - self.ALIGNMENT_BONUS_COUNT + 1):
            cluster = sorted_platforms[i:i+self.ALIGNMENT_BONUS_COUNT]
            max_dist = cluster[-1]["rect"].centerx - cluster[0]["rect"].centerx
            if max_dist < self.ALIGNMENT_TOLERANCE:
                self.alignment_bonus_timer = self.ALIGNMENT_FREEZE_DURATION * self.FPS
                self._spawn_particles((self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), count=50, color=(255,215,0))
                return self.REWARD_ALIGNMENT
        return 0

    def _spawn_particles(self, pos, count=10, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color})
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_gears()
        self._render_collection_bin()
        self._render_platforms()
        self._render_resources()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "gear_speeds": self.gear_speeds
        }
    
    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_SCAFFOLD, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_SCAFFOLD, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_gears(self):
        gear_positions = [(60, 60), (self.SCREEN_WIDTH - 60, 60)]
        gear_radii = [40, 40]
        
        for i, pos in enumerate(gear_positions):
            radius = gear_radii[i]
            angle = self.gear_angles[i]
            
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_GEAR)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_GEAR_DARK)
            
            for j in range(8):
                spoke_angle = angle + (j * math.pi / 4)
                start_x = pos[0] + (radius * 0.2) * math.cos(spoke_angle)
                start_y = pos[1] + (radius * 0.2) * math.sin(spoke_angle)
                end_x = pos[0] + radius * math.cos(spoke_angle)
                end_y = pos[1] + radius * math.sin(spoke_angle)
                pygame.draw.line(self.screen, self.COLOR_GEAR_DARK, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 5)

    def _render_platforms(self):
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p["rect"], border_radius=3)

    def _render_resources(self):
        for res in self.resources:
            pos = (int(res["pos"][0]), int(res["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.RESOURCE_RADIUS + 3, self.COLOR_RESOURCE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.RESOURCE_RADIUS, self.COLOR_RESOURCE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.RESOURCE_RADIUS, self.COLOR_RESOURCE)
            
    def _render_collection_bin(self):
        pygame.draw.rect(self.screen, self.COLOR_BIN, self.collection_bin, border_radius=5)
        
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 20.0))
            color = p["color"] + (int(alpha),)
            
            particle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(particle_surf, color, (0, 0, 2, 2))
            self.screen.blit(particle_surf, (int(p["pos"][0]), int(p["pos"][1])))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        timer_pct = max(0, self.time_remaining / self.GAME_DURATION_SECONDS)
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (bar_x, bar_y, bar_width * timer_pct, bar_height), border_radius=5)
        
        for i in range(2):
            speed_text = self.font_ui.render(f"G{i+1}: {self.gear_speeds[i]:.3f}", True, self.COLOR_UI_TEXT)
            text_x = 10 if i == 0 else self.SCREEN_WIDTH - speed_text.get_width() - 10
            self.screen.blit(speed_text, (text_x, self.SCREEN_HEIGHT - 30))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.score >= self.WIN_SCORE:
            message = "VICTORY!"
            color = (152, 251, 152) # Pale Green
        else:
            message = "TIME UP"
            color = (255, 105, 97) # Salmon
            
        text_surf = self.font_big.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Example of how to run the environment with a display
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS", etc.
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display window
    pygame.display.set_caption("Gear Master")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = truncated = False

        if not (terminated or truncated):
            keys = pygame.key.get_pressed()
            # Map keys to MultiDiscrete actions
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                action[0] = 1 # up
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                action[0] = 2 # down
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action[0] = 3 # right
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action[0] = 4 # left
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()