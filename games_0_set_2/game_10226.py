import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set SDL to dummy mode for headless operation, which is required by the test environment.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment for an arcade action game.
    The player controls a launcher at the bottom of the screen, aiming to shoot
    projectiles at falling targets. Hitting targets increases score and projectile
    speed. Missing targets decreases projectile speed. The goal is to reach a
    score of 100 within 120 seconds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = (
        "Control a launcher to shoot projectiles at falling targets. "
        "Hitting targets increases your score and projectile speed."
    )
    user_guide = "Use ↑ and ↓ arrow keys to aim the launcher. Press space to fire."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30  # Assumed frame rate for game logic and rendering
        self.GAME_DURATION_SECONDS = 120
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS

        # --- Colors ---
        self.COLOR_BG_TOP = (10, 10, 30)
        self.COLOR_BG_BOTTOM = (40, 40, 80)
        self.COLOR_LAUNCHER = (0, 255, 128)
        self.COLOR_PROJECTILE_CORE = (255, 255, 255)
        self.COLOR_PROJECTILE_GLOW = (255, 255, 0)
        self.COLOR_TARGET = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_UI_ACCENT = (255, 255, 0)

        # --- Game Mechanics ---
        self.WIN_SCORE = 100
        self.GRAVITY = 0.2
        self.LAUNCHER_POS = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20)
        self.MIN_ANGLE_DEG = 5
        self.MAX_ANGLE_DEG = 175
        self.ANGLE_CHANGE_RATE = 1.5
        self.INITIAL_PROJECTILE_SPEED = 10.0
        self.INITIAL_TARGET_FALL_SPEED = 1.5
        self.TARGET_SPAWN_RATE = int(0.8 * self.FPS) # Every 0.8 seconds
        self.TARGET_RADIUS = 15

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_speed = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.launcher_angle = 90.0
        self.projectile = None
        self.projectile_speed_multiplier = 1.0
        self.targets = []
        self.particles = []
        self.target_spawn_timer = 0
        self.target_fall_speed = self.INITIAL_TARGET_FALL_SPEED
        self.hits_since_speed_increase = 0
        self.consecutive_misses = 0
        self.prev_space_held = False
        
        # self.reset() # Not needed here, will be called by runner
        # self.validate_implementation() # Not needed for final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        
        self.launcher_angle = 90.0
        self.projectile = None
        self.projectile_speed_multiplier = 1.0
        self.targets = []
        self.particles = []
        
        self.target_spawn_timer = 0
        self.target_fall_speed = self.INITIAL_TARGET_FALL_SPEED
        self.hits_since_speed_increase = 0
        self.consecutive_misses = 0
        self.prev_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0

        self._handle_input(action)
        reward += self._update_projectile()
        self._update_targets()
        self._update_particles()
        self._spawn_targets()

        self.steps += 1
        self.time_remaining -= 1

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                # Goal-oriented reward for winning
                reward += 100.0

        # Small penalty for every step to encourage efficiency
        reward -= 0.01

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held_action, _ = action
        space_held = space_held_action == 1

        # Movement affects launcher angle
        if movement == 1:  # Up
            self.launcher_angle += self.ANGLE_CHANGE_RATE
        elif movement == 2:  # Down
            self.launcher_angle -= self.ANGLE_CHANGE_RATE
        self.launcher_angle = np.clip(self.launcher_angle, self.MIN_ANGLE_DEG, self.MAX_ANGLE_DEG)

        # Space button launches projectile on press
        if space_held and not self.prev_space_held and self.projectile is None:
            self._launch_projectile()

        self.prev_space_held = space_held

    def _launch_projectile(self):
        angle_rad = math.radians(self.launcher_angle)
        speed = self.INITIAL_PROJECTILE_SPEED * self.projectile_speed_multiplier
        velocity = pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * speed
        
        self.projectile = {
            "pos": self.LAUNCHER_POS.copy(),
            "vel": velocity,
            "radius": 8,
            "trail": []
        }

    def _update_projectile(self):
        if self.projectile is None:
            return 0.0

        reward = 0.0
        
        # --- Continuous Shaping Reward ---
        min_dist_before = float('inf')
        if self.targets:
            proj_pos = self.projectile["pos"]
            distances_before = [proj_pos.distance_to(t["pos"]) for t in self.targets]
            min_dist_before = min(distances_before)
        
        # Update projectile physics
        self.projectile["vel"].y += self.GRAVITY
        self.projectile["pos"] += self.projectile["vel"]
        
        # Update trail
        self.projectile["trail"].append(self.projectile["pos"].copy())
        if len(self.projectile["trail"]) > 15:
            self.projectile["trail"].pop(0)

        if self.targets:
            distances_after = [self.projectile["pos"].distance_to(t["pos"]) for t in self.targets]
            min_dist_after = min(distances_after)
            # Reward for getting closer to the nearest target
            reward += (min_dist_before - min_dist_after) * 0.02
            
        # Check for target collision
        for target in self.targets[:]:
            dist = self.projectile["pos"].distance_to(target["pos"])
            if dist < self.projectile["radius"] + target["radius"]:
                # --- HIT ---
                self._create_particles(target["pos"], self.COLOR_PROJECTILE_GLOW, 30)
                self.targets.remove(target)
                self.projectile = None
                
                self.score += 10
                reward += 10.0
                
                self.consecutive_misses = 0
                self.projectile_speed_multiplier *= 1.05 # Speed up by 5%
                self.projectile_speed_multiplier = min(self.projectile_speed_multiplier, 3.0) # Cap speed
                
                self.hits_since_speed_increase += 1
                if self.hits_since_speed_increase >= 20:
                    self.target_fall_speed += 0.05
                    self.hits_since_speed_increase = 0
                return reward

        # Check for out of bounds
        p = self.projectile["pos"]
        if not (0 < p.x < self.SCREEN_WIDTH and p.y < self.SCREEN_HEIGHT):
            # --- MISS ---
            self.projectile = None
            reward -= 5.0
            
            self.consecutive_misses += 1
            if self.consecutive_misses >= 3:
                self.projectile_speed_multiplier *= 0.8 # Slow down by 20%
                self.projectile_speed_multiplier = max(self.projectile_speed_multiplier, 0.1) # Floor speed
                self.consecutive_misses = 0
            return reward

        return reward

    def _update_targets(self):
        for target in self.targets[:]:
            target["pos"].y += self.target_fall_speed
            if target["pos"].y > self.SCREEN_HEIGHT + target["radius"]:
                self.targets.remove(target)

    def _spawn_targets(self):
        self.target_spawn_timer -= 1
        if self.target_spawn_timer <= 0:
            self.target_spawn_timer = self.TARGET_SPAWN_RATE
            
            x_pos = random.uniform(self.TARGET_RADIUS, self.SCREEN_WIDTH - self.TARGET_RADIUS)
            y_pos = -self.TARGET_RADIUS
            
            self.targets.append({
                "pos": pygame.Vector2(x_pos, y_pos),
                "radius": self.TARGET_RADIUS
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, position, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": position.copy(),
                "vel": pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3)),
                "life": random.randint(10, 20),
                "color": color,
                "radius": random.uniform(1, 4)
            })

    def _check_termination(self):
        current_speed = self.INITIAL_PROJECTILE_SPEED * self.projectile_speed_multiplier
        if self.score >= self.WIN_SCORE:
            return True
        if self.time_remaining <= 0:
            return True
        if current_speed <= 0.1 and self.projectile is None: # Effectively zero speed
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining / self.FPS,
            "projectile_speed_mult": self.projectile_speed_multiplier,
        }

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_targets()
        self._render_launcher()
        self._render_projectile()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_launcher(self):
        # Platform
        platform_rect = pygame.Rect(self.LAUNCHER_POS.x - 40, self.LAUNCHER_POS.y, 80, 20)
        pygame.draw.rect(self.screen, self.COLOR_LAUNCHER, platform_rect, border_radius=5)
        
        # Aiming line
        angle_rad = math.radians(self.launcher_angle)
        end_pos = self.LAUNCHER_POS + pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * 50
        pygame.draw.line(self.screen, self.COLOR_LAUNCHER, self.LAUNCHER_POS, end_pos, 3)

    def _render_projectile(self):
        if self.projectile is None:
            return
            
        # Trail
        if len(self.projectile["trail"]) > 1:
            for i in range(len(self.projectile["trail"]) - 1):
                alpha = int(255 * (i / len(self.projectile["trail"])))
                color = (*self.COLOR_PROJECTILE_GLOW, alpha)
                start_pos = self.projectile["trail"][i]
                end_pos = self.projectile["trail"][i+1]
                
                # FIX: Use pygame.Surface with SRALPHA flag for headless rendering with alpha.
                # .convert_alpha() requires a display to be set.
                temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                temp_surf.fill((0,0,0,0))
                pygame.draw.line(temp_surf, color, start_pos, end_pos, int(self.projectile["radius"] * (i / len(self.projectile["trail"]))))
                self.screen.blit(temp_surf, (0,0))


        # Glow effect
        pos = (int(self.projectile["pos"].x), int(self.projectile["pos"].y))
        radius = self.projectile["radius"]
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 4, (*self.COLOR_PROJECTILE_GLOW, 60))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 2, (*self.COLOR_PROJECTILE_GLOW, 120))
        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PROJECTILE_CORE)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PROJECTILE_CORE)

    def _render_targets(self):
        for target in self.targets:
            pos = (int(target["pos"].x), int(target["pos"].y))
            radius = target["radius"]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (255,150,150))

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            alpha = int(255 * (p["life"] / 20.0))
            color = (*p["color"], alpha)
            
            # FIX: Use pygame.Surface with SRALPHA flag for headless rendering with alpha.
            # .convert_alpha() requires a display to be set.
            temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            temp_surf.fill((0,0,0,0))
            pygame.draw.circle(temp_surf, color, pos, int(p["radius"] * (p["life"] / 20.0)))
            self.screen.blit(temp_surf, (0,0))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Time
        time_seconds = max(0, self.time_remaining // self.FPS)
        time_text = self.font_ui.render(f"Time: {time_seconds}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Projectile Speed
        speed_mult = self.projectile_speed_multiplier
        speed_text = self.font_speed.render(f"Speed: {speed_mult:.2f}x", True, self.COLOR_TEXT)
        text_rect = speed_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 45))
        
        bar_rect = text_rect.inflate(20, 10)
        s = pygame.Surface(bar_rect.size, pygame.SRCALPHA)
        s.fill((*self.COLOR_BG_TOP, 128))
        self.screen.blit(s, bar_rect.topleft)
        self.screen.blit(speed_text, text_rect)

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_UI_ACCENT
            else:
                msg = "GAME OVER"
                color = self.COLOR_TARGET
            
            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It is not run by the test suite, but is useful for development.
    # Re-enable the display driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # We need to create a real display window here
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Momentum Launcher")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            terminated = True

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()