import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:59:30.637877
# Source Brief: brief_00146.md
# Brief Index: 146
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a splitting slime blob.
    The goal is to merge smaller blob pieces to reach a target size within a time limit
    across four consecutive levels.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) for aiming
    - actions[1]: Space button (0=released, 1=held) for launching
    - actions[2]: Shift button (0=released, 1=held) - unused

    **Observation Space:** Box(shape=(400, 640, 3), dtype=uint8)
    - An RGB image of the game screen.

    **Reward Structure:**
    - +0.1 for each successful merge of blobs.
    - +1.0 for completing a level.
    - +100 for winning the game (completing all 4 levels).
    - -100 for losing (timer runs out on any level).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a slime blob by aiming and launching it. Collide with walls to split, and merge the pieces to grow and reach the target mass for each level."
    )
    user_guide = (
        "Use ←→ arrow keys to adjust launch power and ↑↓ to adjust the angle. Press space to launch the slime."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_LEVELS = 4
    LEVEL_TIME = 60.0  # seconds

    # Colors
    COLOR_BG_TOP = (15, 20, 45)
    COLOR_BG_BOTTOM = (30, 40, 70)
    COLOR_SLIME = (50, 255, 150)
    COLOR_SLIME_CORE = (200, 255, 220)
    COLOR_PARTICLE = (220, 255, 240)
    COLOR_UI_TEXT = (230, 230, 255)
    COLOR_UI_BAR_BG = (50, 60, 100)
    COLOR_UI_BAR_FILL = (0, 180, 255)
    COLOR_TIMER_WARN = (255, 80, 80)
    COLOR_AIM_LINE = (255, 255, 255, 150)

    # Physics & Gameplay
    GRAVITY = 0.0
    FRICTION = 0.995
    MIN_LAUNCH_POWER = 3.0
    MAX_LAUNCH_POWER = 15.0
    RADIUS_SCALAR = 3.5
    SPLIT_MASS_FACTOR = 0.4
    SPLIT_VEL_DIVERGENCE = 1.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # State variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 0
        self.timer = 0.0
        self.target_mass = 0
        self.total_mass = 0
        self.game_phase = "aiming"  # 'aiming' or 'simulating'
        self.launch_angle = 0.0
        self.launch_power = 0.0
        self.blobs = []
        self.particles = []
        self.last_space_held = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.particles = []
        self.last_space_held = False
        self._setup_level()
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.game_phase = "aiming"
        self.timer = self.LEVEL_TIME
        self.target_mass = 150 + (self.level - 1) * 50
        self.launch_angle = -math.pi / 4
        self.launch_power = (self.MIN_LAUNCH_POWER + self.MAX_LAUNCH_POWER) / 2

        initial_mass = 100
        initial_blob = {
            "pos": np.array([50.0, self.SCREEN_HEIGHT - 50.0]),
            "vel": np.array([0.0, 0.0]),
            "mass": initial_mass,
            "radius": self._mass_to_radius(initial_mass),
        }
        self.blobs = [initial_blob]
        self.total_mass = self._calculate_total_mass()
        
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        
        self._handle_input(action)

        self.timer -= 1.0 / self.FPS
        
        if self.game_phase == "simulating":
            reward += self._update_physics()

        self._update_particles()
        
        # --- Check for level/game end conditions ---
        if self.total_mass >= self.target_mass:
            # --- Sound: Level Complete ---
            reward += 1.0
            self.level += 1
            if self.level > self.MAX_LEVELS:
                # --- Game Won ---
                reward += 100.0
                terminated = True
                self.game_over = True
            else:
                self._setup_level()
        
        if self.timer <= 0 and not terminated:
            # --- Game Lost ---
            reward -= 100.0
            terminated = True
            self.game_over = True
            # --- Sound: Game Over ---

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held_raw, _ = action
        space_held = space_held_raw == 1

        if self.game_phase == "aiming":
            # Adjust angle
            if movement == 1: self.launch_angle -= 0.05
            if movement == 2: self.launch_angle += 0.05
            self.launch_angle = np.clip(self.launch_angle, -math.pi * 0.9, -math.pi * 0.1)

            # Adjust power
            if movement == 4: self.launch_power += 0.2
            if movement == 3: self.launch_power -= 0.2
            self.launch_power = np.clip(self.launch_power, self.MIN_LAUNCH_POWER, self.MAX_LAUNCH_POWER)

            # Launch on space press (rising edge)
            if space_held and not self.last_space_held:
                # --- Sound: Launch ---
                launch_vel = np.array([math.cos(self.launch_angle), math.sin(self.launch_angle)])
                base_speed = 6.0 + (self.level - 1) * 1.0 # Difficulty scaling
                self.blobs[0]["vel"] = launch_vel * self.launch_power * (base_speed / 10.0)
                self.game_phase = "simulating"
        
        self.last_space_held = space_held

    def _update_physics(self):
        # Update blob positions
        for blob in self.blobs:
            blob["vel"] *= self.FRICTION
            blob["pos"] += blob["vel"]

        self._handle_wall_collisions()
        merge_reward = self._handle_blob_merging()
        self.total_mass = self._calculate_total_mass()
        return merge_reward

    def _handle_wall_collisions(self):
        blobs_to_add = []
        blobs_to_remove = []

        for blob in self.blobs:
            split = False
            if blob["pos"][0] - blob["radius"] < 0 or blob["pos"][0] + blob["radius"] > self.SCREEN_WIDTH:
                blob["vel"][0] *= -1
                blob["pos"][0] = np.clip(blob["pos"][0], blob["radius"], self.SCREEN_WIDTH - blob["radius"])
                split = True
            if blob["pos"][1] - blob["radius"] < 0 or blob["pos"][1] + blob["radius"] > self.SCREEN_HEIGHT:
                blob["vel"][1] *= -1
                blob["pos"][1] = np.clip(blob["pos"][1], blob["radius"], self.SCREEN_HEIGHT - blob["radius"])
                split = True

            if split and blob["mass"] > 10: # Don't split tiny blobs
                # --- Sound: Split ---
                blobs_to_remove.append(blob)
                new_mass = blob["mass"] * self.SPLIT_MASS_FACTOR
                if new_mass < 1: continue

                # Create two new blobs
                vel1 = np.copy(blob["vel"]) + self.np_random.uniform(-1, 1, size=2) * self.SPLIT_VEL_DIVERGENCE
                vel2 = np.copy(blob["vel"]) + self.np_random.uniform(-1, 1, size=2) * self.SPLIT_VEL_DIVERGENCE
                
                blobs_to_add.append(self._create_blob(blob["pos"], vel1, new_mass))
                blobs_to_add.append(self._create_blob(blob["pos"], vel2, new_mass))
                self._create_particles(blob["pos"], 20, 2.0)

        if blobs_to_remove:
            self.blobs = [b for b in self.blobs if b not in blobs_to_remove]
            self.blobs.extend(blobs_to_add)

    def _handle_blob_merging(self):
        merged_in_pass = True
        reward = 0
        
        while merged_in_pass:
            merged_in_pass = False
            if len(self.blobs) < 2:
                break
            
            new_blobs = []
            merged_indices = set()
            
            for i in range(len(self.blobs)):
                if i in merged_indices:
                    continue
                
                b1 = self.blobs[i]
                found_merge = False
                
                for j in range(i + 1, len(self.blobs)):
                    if j in merged_indices:
                        continue
                        
                    b2 = self.blobs[j]
                    dist_sq = np.sum((b1["pos"] - b2["pos"]) ** 2)
                    radii_sum = b1["radius"] + b2["radius"]
                    
                    if dist_sq < (radii_sum * 0.9) ** 2: # Merge if overlapping significantly
                        # --- Sound: Merge ---
                        total_mass = b1["mass"] + b2["mass"]
                        new_pos = (b1["pos"] * b1["mass"] + b2["pos"] * b2["mass"]) / total_mass
                        new_vel = (b1["vel"] * b1["mass"] + b2["vel"] * b2["mass"]) / total_mass
                        
                        new_blobs.append(self._create_blob(new_pos, new_vel, total_mass))
                        self._create_particles(new_pos, 15, 1.5, self.COLOR_UI_BAR_FILL)
                        
                        merged_indices.add(i)
                        merged_indices.add(j)
                        merged_in_pass = True
                        reward += 0.1
                        found_merge = True
                        break # Merge b1 with only one other blob per pass
                
                if not found_merge:
                    new_blobs.append(b1)
            
            self.blobs = new_blobs
        return reward

    def _create_blob(self, pos, vel, mass):
        return {
            "pos": np.copy(pos),
            "vel": np.copy(vel),
            "mass": mass,
            "radius": self._mass_to_radius(mass),
        }
    
    def _mass_to_radius(self, mass):
        return max(1.0, math.sqrt(mass) * self.RADIUS_SCALAR)

    def _calculate_total_mass(self):
        return sum(b["mass"] for b in self.blobs)

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_blobs()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_blobs(self):
        for blob in self.blobs:
            x, y = int(blob["pos"][0]), int(blob["pos"][1])
            radius = int(blob["radius"])
            if radius <= 0: continue

            # Outer glow/body
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_SLIME)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_SLIME)
            
            # Inner core for 3D effect
            core_radius = max(0, int(radius * 0.7))
            pygame.gfxdraw.filled_circle(self.screen, x, y, core_radius, self.COLOR_SLIME_CORE)
            pygame.gfxdraw.aacircle(self.screen, x, y, core_radius, self.COLOR_SLIME_CORE)
            
    def _render_ui(self):
        # Level Text
        level_text = self.font_main.render(f"Level: {self.level}/{self.MAX_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Timer Text
        timer_color = self.COLOR_UI_TEXT if self.timer > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_main.render(f"Time: {max(0, self.timer):.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Score Text
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        # Mass Progress Bar
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 20
        progress = min(1.0, self.total_mass / self.target_mass)
        fill_width = int(bar_width * progress)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, self.SCREEN_HEIGHT - 35, bar_width, bar_height), border_radius=5)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (10, self.SCREEN_HEIGHT - 35, fill_width, bar_height), border_radius=5)
        
        mass_text = self.font_small.render(f"Mass: {int(self.total_mass)} / {int(self.target_mass)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(mass_text, (15, self.SCREEN_HEIGHT - 34))

        # Aiming UI
        if self.game_phase == "aiming":
            self._render_aim_indicator()

    def _render_aim_indicator(self):
        start_pos = self.blobs[0]["pos"]
        angle = self.launch_angle
        power = self.launch_power
        
        # Power bar
        power_bar_width = 100
        power_progress = (power - self.MIN_LAUNCH_POWER) / (self.MAX_LAUNCH_POWER - self.MIN_LAUNCH_POWER)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (start_pos[0] - 50, start_pos[1] + 20, power_bar_width, 10), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_SLIME, (start_pos[0] - 50, start_pos[1] + 20, power_bar_width * power_progress, 10), border_radius=3)

        # Trajectory line
        line_len = 50 + power * 5
        end_pos = start_pos + np.array([math.cos(angle), math.sin(angle)]) * line_len
        pygame.draw.line(self.screen, self.COLOR_AIM_LINE, start_pos.astype(int), end_pos.astype(int), 2)

    def _create_particles(self, pos, count, max_speed, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": np.copy(pos), "vel": vel, "lifespan": lifespan, "max_life": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = p["lifespan"] / p["max_life"]
            radius = int(alpha * 3)
            if radius < 1: continue
            color = (*p["color"], int(alpha * 255))
            
            # Create a temporary surface for the particle to handle alpha
            particle_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (radius, radius), radius)
            self.screen.blit(particle_surf, (int(p["pos"][0] - radius), int(p["pos"][1] - radius)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer,
            "total_mass": self.total_mass,
            "target_mass": self.target_mass,
            "blobs": len(self.blobs),
        }
    
    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # The validation code has been removed from the main block
    # as it's not needed for the final runnable script.
    
    # To play manually, you need a display.
    # The environment itself runs headlessly, but to see it,
    # you'd need to unset the dummy video driver.
    # For example:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    try:
        pygame.display.init()
        pygame.display.set_caption("Slime Split Gym Environment")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        display_available = True
    except pygame.error:
        print("No display available. Running headlessly.")
        display_available = False

    total_reward = 0
    
    while not terminated:
        # Map keyboard inputs to actions
        action = [0, 0, 0] # Default no-op
        if display_available:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if display_available:
            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated:
            print("Game terminated.")

        env.clock.tick(GameEnv.FPS)

    print(f"Game Over. Final Score: {total_reward:.2f}")
    env.close()