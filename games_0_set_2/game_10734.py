import gymnasium as gym
import os
import pygame
import numpy as np
import math
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A neon-drenched arcade shooter. Aim and fire at moving targets while mastering a "
        "rhythm-based energy system to keep your weapon charged."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim. Press space to fire and to play the "
        "timing mini-game for energy."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500 # Approx 25s at 60fps
    NUM_TARGETS = 20
    TARGET_BASE_SPEED = 1.0
    TARGET_SPEED_INCREMENT = 0.05
    PROJECTILE_SPEED = 8
    PROJECTILE_COST = 10
    MAX_ENERGY = 100

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_TARGET = (255, 50, 50)
    COLOR_TARGET_GLOW = (150, 0, 0)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_PROJECTILE_GLOW = (50, 100, 200)
    COLOR_ENERGY_BAR = (50, 255, 50)
    COLOR_ENERGY_BAR_BG = (50, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_AIM = (255, 255, 255, 100)
    
    # Timing Circle
    TIMING_CENTER = (SCREEN_WIDTH // 2, 100)
    TIMING_MAX_RADIUS = 70
    TIMING_TARGET_RADIUS = 20
    TIMING_WINDOW_PERFECT = 4
    TIMING_WINDOW_GOOD = 12
    TIMING_SHRINK_SPEED = 0.75
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = 0
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.targets_destroyed_count = 0
        self.current_target_speed = self.TARGET_BASE_SPEED
        self.aim_direction = np.array([1.0, 0.0])
        self.prev_space_held = False
        
        # Timing mechanic state
        self.timing_circle_radius = self.TIMING_MAX_RADIUS
        self.timing_flash = {"color": (0,0,0), "alpha": 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = 0
        self.targets_destroyed_count = 0
        self.current_target_speed = self.TARGET_BASE_SPEED
        self.aim_direction = np.array([1.0, 0.0])
        self.prev_space_held = False
        
        self.projectiles.clear()
        self.particles.clear()
        self.targets.clear()

        self.timing_circle_radius = self.TIMING_MAX_RADIUS
        self.timing_flash = {"color": (0,0,0), "alpha": 0}

        self._spawn_targets(self.NUM_TARGETS)
        
        return self._get_observation(), self._get_info()

    def _spawn_targets(self, num_targets):
        for _ in range(num_targets):
            self.targets.append({
                "pos": np.array([
                    self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                    self.np_random.uniform(180, self.SCREEN_HEIGHT - 50)
                ], dtype=np.float32),
                "radius": self.np_random.uniform(8, 15),
                "freq": self.np_random.uniform(0.01, 0.05),
                "amp": self.np_random.uniform(20, 60),
                "phase": self.np_random.uniform(0, 2 * math.pi),
                "base_y": self.np_random.uniform(180, self.SCREEN_HEIGHT - 50),
                "direction": self.np_random.choice([-1, 1])
            })
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # --- 1. Handle Input & Actions ---
        movement, space_held, _ = action
        space_pressed = (space_held == 1) and not self.prev_space_held
        
        # Update aim direction
        if movement == 1: self.aim_direction = np.array([0., -1.]) # Up
        elif movement == 2: self.aim_direction = np.array([0., 1.])  # Down
        elif movement == 3: self.aim_direction = np.array([-1., 0.]) # Left
        elif movement == 4: self.aim_direction = np.array([1., 0.])  # Right

        # Handle space press for timing game and firing
        if space_pressed:
            # Firing
            if self.energy >= self.PROJECTILE_COST:
                self.energy -= self.PROJECTILE_COST
                # sfx: Fire projectile
                self.projectiles.append({
                    "pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32),
                    "vel": self.aim_direction * self.PROJECTILE_SPEED
                })
            
            # Timing game
            diff = abs(self.timing_circle_radius - self.TIMING_TARGET_RADIUS)
            energy_change = 0
            if diff <= self.TIMING_WINDOW_PERFECT:
                energy_change = 20
                self.timing_flash = {"color": (50, 255, 50), "alpha": 255} # Green
                # sfx: Perfect timing
            elif diff <= self.TIMING_WINDOW_GOOD:
                energy_change = 10
                self.timing_flash = {"color": (255, 255, 50), "alpha": 255} # Yellow
                # sfx: Good timing
            else:
                energy_change = -10
                self.timing_flash = {"color": (255, 50, 50), "alpha": 255} # Red
                # sfx: Bad timing
            
            self.energy = max(0, min(self.MAX_ENERGY, self.energy + energy_change))
            reward += energy_change if energy_change > 0 else energy_change * 0.5
            self.timing_circle_radius = self.TIMING_MAX_RADIUS # Reset circle

        self.prev_space_held = (space_held == 1)

        # --- 2. Update Game State ---
        self._update_timing_circle()
        self._update_projectiles()
        self._update_targets()
        reward += self._handle_collisions()
        self._update_particles()
        
        self.steps += 1
        
        # --- 3. Check Termination ---
        terminated = False
        truncated = False
        if not self.targets:
            terminated = True
            reward += 100 # Win bonus
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 50 # Loss penalty
            self.game_over = True
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_timing_circle(self):
        self.timing_circle_radius -= self.TIMING_SHRINK_SPEED
        if self.timing_circle_radius < 0:
            self.timing_circle_radius = self.TIMING_MAX_RADIUS
        
        if self.timing_flash["alpha"] > 0:
            self.timing_flash["alpha"] -= 15
            self.timing_flash["alpha"] = max(0, self.timing_flash["alpha"])

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            if not (0 <= p["pos"][0] < self.SCREEN_WIDTH and 0 <= p["pos"][1] < self.SCREEN_HEIGHT):
                self.projectiles.remove(p)

    def _update_targets(self):
        for t in self.targets:
            t["pos"][0] += self.current_target_speed * t["direction"]
            t["pos"][1] = t["base_y"] + t["amp"] * math.sin(t["freq"] * self.steps + t["phase"])
            if t["pos"][0] < t["radius"] or t["pos"][0] > self.SCREEN_WIDTH - t["radius"]:
                t["direction"] *= -1

    def _handle_collisions(self):
        collision_reward = 0
        for p in self.projectiles[:]:
            for t in self.targets[:]:
                dist = np.linalg.norm(p["pos"] - t["pos"])
                if dist < t["radius"]:
                    self.projectiles.remove(p)
                    self.targets.remove(t)
                    collision_reward += 5
                    self.targets_destroyed_count += 1
                    # sfx: Target explosion
                    self._create_explosion(t["pos"], self.COLOR_TARGET, 30)
                    
                    # Difficulty scaling
                    if self.targets_destroyed_count > 0 and self.targets_destroyed_count % 5 == 0:
                        self.current_target_speed += self.TARGET_SPEED_INCREMENT
                    break # a projectile can only hit one target
        return collision_reward

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": self.np_random.integers(20, 40),
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
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
        # Render aim indicator
        start_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        end_pos = (
            int(start_pos[0] + self.aim_direction[0] * 40),
            int(start_pos[1] + self.aim_direction[1] * 40)
        )
        pygame.draw.line(self.screen, self.COLOR_AIM, start_pos, end_pos, 2)
        pygame.gfxdraw.aacircle(self.screen, start_pos[0], start_pos[1], 5, self.COLOR_AIM)

        # Render targets
        for t in self.targets:
            pos_int = (int(t["pos"][0]), int(t["pos"][1]))
            radius_int = int(t["radius"])
            glow_radius = int(radius_int * 1.8)
            
            # Glow effect using a surface for alpha blending
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_TARGET_GLOW, 80))
            self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))

            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, self.COLOR_TARGET)


        # Render projectiles
        for p in self.projectiles:
            start_pos = (int(p["pos"][0]), int(p["pos"][1]))
            end_pos = (int(p["pos"][0] - p["vel"][0]*0.8), int(p["pos"][1] - p["vel"][1]*0.8))
            # Glow
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE_GLOW, start_pos, end_pos, 7)
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 3)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 40.0))
            color = (*p["color"], alpha)
            if p["size"] > 1:
                # Use a surface for per-particle alpha
                part_surf = pygame.Surface((int(p["size"]), int(p["size"])), pygame.SRCALPHA)
                part_surf.fill(color)
                self.screen.blit(part_surf, (int(p["pos"][0]), int(p["pos"][1])))

        # Render timing circle
        tc_pos = self.TIMING_CENTER
        # Target ring
        pygame.gfxdraw.aacircle(self.screen, tc_pos[0], tc_pos[1], int(self.TIMING_TARGET_RADIUS), (255,255,255))
        pygame.gfxdraw.aacircle(self.screen, tc_pos[0], tc_pos[1], int(self.TIMING_TARGET_RADIUS - self.TIMING_WINDOW_PERFECT), (150,255,150))
        pygame.gfxdraw.aacircle(self.screen, tc_pos[0], tc_pos[1], int(self.TIMING_TARGET_RADIUS + self.TIMING_WINDOW_PERFECT), (150,255,150))
        # Shrinking circle
        pygame.gfxdraw.aacircle(self.screen, tc_pos[0], tc_pos[1], int(self.timing_circle_radius), (200,200,255))
        # Flash effect
        if self.timing_flash["alpha"] > 0:
            flash_color = (*self.timing_flash["color"], self.timing_flash["alpha"])
            s = pygame.Surface((self.TIMING_MAX_RADIUS*2, self.TIMING_MAX_RADIUS*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, self.TIMING_MAX_RADIUS, self.TIMING_MAX_RADIUS, self.TIMING_MAX_RADIUS, flash_color)
            self.screen.blit(s, (tc_pos[0]-self.TIMING_MAX_RADIUS, tc_pos[1]-self.TIMING_MAX_RADIUS))


    def _render_ui(self):
        # Energy Bar
        bar_width = 200
        bar_height = 20
        energy_pct = self.energy / self.MAX_ENERGY
        fill_width = int(bar_width * energy_pct)
        
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, (10, 10, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (10, 10, fill_width, bar_height))
        energy_text = self.font_small.render(f"ENERGY: {int(self.energy)}/{self.MAX_ENERGY}", True, self.COLOR_TEXT)
        self.screen.blit(energy_text, (15, 12))

        # Targets Remaining
        targets_text = self.font_main.render(f"TARGETS: {len(self.targets)}", True, self.COLOR_TEXT)
        text_rect = targets_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(targets_text, text_rect)

        # Time Remaining
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_main.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        text_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 35))
        self.screen.blit(time_text, text_rect)
        
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "targets_remaining": len(self.targets),
            "targets_destroyed": self.targets_destroyed_count
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to re-enable the display driver
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Target Practice")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # The observation is (H, W, C), but pygame wants (W, H) for surfarray
        # So we need to transpose it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to play again or close the window to exit.")
            # Wait for 'R' to be pressed to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting environment.")
                        obs, info = env.reset()
                        total_reward = 0.0
                        wait_for_reset = False
                clock.tick(15) # Don't burn CPU while waiting
        
        clock.tick(60) # Run at 60 FPS
        
    env.close()