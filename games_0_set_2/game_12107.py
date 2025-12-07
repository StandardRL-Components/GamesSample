import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:19:40.757399
# Source Brief: brief_02107.md
# Brief Index: 2107
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a water cannon to extinguish a spreading wildfire.
    The agent must manage the position, angle, and size of water droplets to efficiently
    put out the fire before it consumes the forest or time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a water cannon to extinguish a spreading wildfire before it consumes the forest. "
        "Upgrade your water to fight the flames more effectively."
    )
    user_guide = (
        "Controls: Use ←→ to move the cannon and ↑↓ to aim. "
        "Press space to fire a water droplet. Hold shift to increase the droplet size."
    )
    auto_advance = False

    # --- Class-level variables for persistent state ---
    cumulative_score = 0
    UNLOCKED_MAGNETIC_THRESHOLD = 1000
    UNLOCKED_SUPER_THRESHOLD = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS # 60 seconds

        # Fire Grid
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_W = self.WIDTH / self.GRID_W
        self.CELL_H = self.HEIGHT / self.GRID_H
        self.INITIAL_FIRE_CHANCE = 0.015
        self.SPREAD_CHANCE_INCREASE_PER_SEC = 0.005

        # Player/Launcher
        self.LAUNCHER_Y = self.HEIGHT - 15
        self.LAUNCHER_SPEED = 8
        self.MIN_ANGLE, self.MAX_ANGLE = -85, -5 # In degrees
        self.ANGLE_STEP = 2
        self.MIN_DROPLET_SIZE, self.MAX_DROPLET_SIZE = 5, 25
        self.SIZE_STEP = 0.5
        self.LAUNCH_VELOCITY = 10

        # Physics
        self.GRAVITY = 0.15
        self.EXTINGUISH_POWER = 0.1

        # Colors
        self.COLOR_BG = (10, 25, 15)
        self.COLOR_TREE = (15, 40, 25)
        self.COLOR_CHAR = (20, 15, 10)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WATER = (100, 150, 255)
        self.COLOR_WATER_GLOW = (150, 200, 255, 60)
        self.COLOR_MAGNETIC = (100, 255, 150)
        self.COLOR_MAGNETIC_GLOW = (150, 255, 200, 60)
        self.COLOR_SUPER = (200, 100, 255)
        self.COLOR_SUPER_GLOW = (220, 150, 255, 60)
        self.FIRE_COLORS = [(255, 220, 0), (255, 150, 0), (255, 60, 0), (200, 0, 0)]

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_water_type = pygame.font.Font(None, 22)
        self.font_unlock = pygame.font.Font(None, 36)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.fire_map = np.zeros((self.GRID_H, self.GRID_W))
        self.char_map = np.zeros((self.GRID_H, self.GRID_W))
        self.launcher_x = self.WIDTH / 2
        self.launcher_angle = -45
        self.current_droplet_size = (self.MIN_DROPLET_SIZE + self.MAX_DROPLET_SIZE) / 2
        self.droplets = []
        self.particles = []
        self.was_space_held = False
        self.unlocked_magnetic = self.cumulative_score >= self.UNLOCKED_MAGNETIC_THRESHOLD
        self.unlocked_super = self.cumulative_score >= self.UNLOCKED_SUPER_THRESHOLD
        self.current_water_type = self._get_best_water_type()
        self.unlock_message_timer = 0
        self.unlock_message_text = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._initialize_fire()
        self.char_map = np.zeros((self.GRID_H, self.GRID_W))

        self.launcher_x = self.WIDTH / 2
        self.launcher_angle = -45
        self.current_droplet_size = (self.MIN_DROPLET_SIZE + self.MAX_DROPLET_SIZE) / 2
        
        self.droplets = []
        self.particles = []
        self.was_space_held = False
        
        # Check for unlocks based on cumulative score
        newly_unlocked = False
        if not self.unlocked_magnetic and self.cumulative_score >= self.UNLOCKED_MAGNETIC_THRESHOLD:
            self.unlocked_magnetic = True
            self.unlock_message_text = "Magnetic Water Unlocked!"
            self.unlock_message_timer = self.FPS * 3
            newly_unlocked = True
        if not self.unlocked_super and self.cumulative_score >= self.UNLOCKED_SUPER_THRESHOLD:
            self.unlocked_super = True
            self.unlock_message_text = "Super Water Unlocked!"
            self.unlock_message_timer = self.FPS * 3 # Overwrites previous message if unlocked simultaneously
            newly_unlocked = True
        
        if newly_unlocked:
            self.current_water_type = self._get_best_water_type()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        self._handle_actions(action)
        
        reward_from_droplets = self._update_droplets()
        reward += reward_from_droplets

        self._update_fire()
        self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not truncated: # Win
            reward += 100
        elif terminated and truncated and np.sum(self.fire_map) > 0: # Loss by timeout
            reward -= 100
            
        if terminated:
            GameEnv.cumulative_score += self.score # Update persistent score

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _initialize_fire(self):
        self.fire_map = np.zeros((self.GRID_H, self.GRID_W))
        start_y, start_x = self.np_random.integers(0, self.GRID_H), self.np_random.integers(0, self.GRID_W)
        self.fire_map[start_y, start_x] = 0.5
        for _ in range(5): # Start with a small cluster
            self._spread_fire_from(start_y, start_x, initial_spread=True)

    def _handle_actions(self, action):
        movement, space_held_action, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement: Launcher Position & Angle ---
        if movement == 1: # Up
            self.launcher_angle = min(self.MAX_ANGLE, self.launcher_angle + self.ANGLE_STEP)
        elif movement == 2: # Down
            self.launcher_angle = max(self.MIN_ANGLE, self.launcher_angle - self.ANGLE_STEP)
        elif movement == 3: # Left
            self.launcher_x = max(0, self.launcher_x - self.LAUNCHER_SPEED)
        elif movement == 4: # Right
            self.launcher_x = min(self.WIDTH, self.launcher_x + self.LAUNCHER_SPEED)

        # --- Shift: Droplet Size ---
        if shift_held:
            self.current_droplet_size = min(self.MAX_DROPLET_SIZE, self.current_droplet_size + self.SIZE_STEP)
        else:
            self.current_droplet_size = max(self.MIN_DROPLET_SIZE, self.current_droplet_size - self.SIZE_STEP)
            
        # --- Space: Launch Droplet ---
        if space_held_action and not self.was_space_held:
            # // SFX: Water launch
            angle_rad = math.radians(self.launcher_angle)
            vx = self.LAUNCH_VELOCITY * math.cos(angle_rad)
            vy = self.LAUNCH_VELOCITY * math.sin(angle_rad)
            self.droplets.append({
                "x": self.launcher_x, "y": self.LAUNCHER_Y,
                "vx": vx, "vy": vy,
                "size": self.current_droplet_size,
                "type": self.current_water_type,
                "trail": []
            })
        self.was_space_held = space_held_action

    def _update_droplets(self):
        step_reward = 0
        for droplet in self.droplets[:]:
            droplet["trail"].append((droplet["x"], droplet["y"]))
            if len(droplet["trail"]) > 10:
                droplet["trail"].pop(0)

            droplet["vy"] += self.GRAVITY
            droplet["x"] += droplet["vx"]
            droplet["y"] += droplet["vy"]

            # Magnetic attraction
            if droplet["type"] != "magnetic":
                for other in self.droplets:
                    if other["type"] == "magnetic":
                        dx = other["x"] - droplet["x"]
                        dy = other["y"] - droplet["y"]
                        dist_sq = dx*dx + dy*dy
                        if dist_sq > 1 and dist_sq < 150**2:
                            force = 30 / dist_sq
                            droplet["vx"] += dx * force
                            droplet["vy"] += dy * force

            # Collision detection
            if 0 <= droplet["x"] < self.WIDTH and 0 <= droplet["y"] < self.HEIGHT:
                grid_x = int(droplet["x"] / self.CELL_W)
                grid_y = int(droplet["y"] / self.CELL_H)
                
                if self.fire_map[grid_y, grid_x] > 0:
                    extinguish_amount, patches_extinguished, total_fire_removed = self._extinguish_fire(droplet)
                    reward = 0.1 * total_fire_removed + 1.0 * patches_extinguished
                    if patches_extinguished >= 3: # Chain reaction
                        reward += 5
                    if droplet["type"] != "standard" and total_fire_removed >= 5: # Effective special use
                        reward += 10
                    
                    step_reward += reward
                    self.score += reward
                    self.droplets.remove(droplet)
                    self._create_splash_particles(droplet["x"], droplet["y"], droplet["size"])
                    continue

            if droplet["y"] > self.HEIGHT or droplet["x"] < 0 or droplet["x"] > self.WIDTH:
                self.droplets.remove(droplet)
        return step_reward

    def _extinguish_fire(self, droplet):
        extinguished_patches = 0
        total_fire_removed = 0
        
        radius = droplet["size"] / self.CELL_W
        if droplet["type"] == "super":
            radius *= 2.5 # Super water has a larger explosion radius
        
        center_x = int(droplet["x"] / self.CELL_W)
        center_y = int(droplet["y"] / self.CELL_H)

        for r in range(center_y - int(radius), center_y + int(radius) + 1):
            for c in range(center_x - int(radius), center_x + int(radius) + 1):
                if 0 <= r < self.GRID_H and 0 <= c < self.GRID_W:
                    dist_sq = (r - center_y)**2 + (c - center_x)**2
                    if dist_sq <= radius**2:
                        if self.fire_map[r, c] > 0:
                            amount = droplet["size"] * self.EXTINGUISH_POWER
                            removed = min(self.fire_map[r, c], amount)
                            total_fire_removed += removed
                            self.fire_map[r, c] -= amount
                            if self.fire_map[r, c] <= 0:
                                self.fire_map[r, c] = 0
                                self.char_map[r, c] = 1 # Mark as charred
                                extinguished_patches += 1
        return droplet["size"], extinguished_patches, total_fire_removed

    def _update_fire(self):
        spread_chance = self.INITIAL_FIRE_CHANCE + (self.steps / self.FPS) * self.SPREAD_CHANCE_INCREASE_PER_SEC
        fire_coords = np.argwhere(self.fire_map > 0)
        self.np_random.shuffle(fire_coords)

        for y, x in fire_coords:
            if self.fire_map[y, x] > 0:
                self._spread_fire_from(y, x, spread_chance)
                self.fire_map[y, x] = min(1.0, self.fire_map[y, x] + 0.001) # Fire intensifies slowly

    def _spread_fire_from(self, y, x, chance=1.0, initial_spread=False):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.GRID_H and 0 <= nx < self.GRID_W:
                    if self.fire_map[ny, nx] == 0 and self.char_map[ny, nx] == 0:
                        if self.np_random.random() < chance:
                            self.fire_map[ny, nx] = 0.1 if not initial_spread else 0.3

    def _update_particles(self):
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if np.sum(self.fire_map) == 0:
            return True
        fire_cell_count = np.sum(self.fire_map > 0)
        if fire_cell_count / self.fire_map.size >= 0.95: # 95% of forest is on fire
            return True
        return False
        
    def _get_best_water_type(self):
        if self.unlocked_super:
            return "super"
        if self.unlocked_magnetic:
            return "magnetic"
        return "standard"

    def _get_observation(self):
        # --- Render Background ---
        self.screen.fill(self.COLOR_BG)
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                rect = pygame.Rect(c * self.CELL_W, r * self.CELL_H, self.CELL_W, self.CELL_H)
                if self.char_map[r, c] > 0:
                    pygame.draw.rect(self.screen, self.COLOR_CHAR, rect)
                else:
                    # Subtle tree pattern
                    if (r + c) % 2 == 0:
                        pygame.draw.rect(self.screen, self.COLOR_TREE, rect)

        # --- Render Fire and Particles ---
        self._render_fire_and_particles()

        # --- Render Droplets ---
        self._render_droplets()

        # --- Render Launcher ---
        self._render_launcher()
        
        # --- Render UI ---
        self._render_ui()

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_fire_and_particles(self):
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                intensity = self.fire_map[r, c]
                if intensity > 0:
                    cx = (c + 0.5) * self.CELL_W
                    cy = (r + 0.5) * self.CELL_H
                    num_particles = int(2 + intensity * 8)
                    for _ in range(num_particles):
                        px = cx + self.np_random.uniform(-self.CELL_W / 2, self.CELL_W / 2)
                        py = cy + self.np_random.uniform(-self.CELL_H / 2, self.CELL_H / 2)
                        size = int(1 + intensity * 3 * self.np_random.random())
                        color_idx = min(len(self.FIRE_COLORS) - 1, int(intensity * len(self.FIRE_COLORS)))
                        color = self.FIRE_COLORS[color_idx]
                        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), size, color)
        
        for p in self.particles:
             pygame.gfxdraw.aacircle(self.screen, int(p["x"]), int(p["y"]), int(p["size"]), p["color"])


    def _render_droplets(self):
        for droplet in self.droplets:
            x, y, size = int(droplet["x"]), int(droplet["y"]), int(droplet["size"])
            
            # Trail
            if len(droplet["trail"]) > 1:
                pygame.draw.aalines(self.screen, (200, 200, 220, 100), False, droplet["trail"], 1)

            # Color based on type
            if droplet["type"] == "magnetic":
                color, glow_color = self.COLOR_MAGNETIC, self.COLOR_MAGNETIC_GLOW
            elif droplet["type"] == "super":
                color, glow_color = self.COLOR_SUPER, self.COLOR_SUPER_GLOW
            else:
                color, glow_color = self.COLOR_WATER, self.COLOR_WATER_GLOW
            
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(size * 1.5), glow_color)
            pygame.gfxdraw.aacircle(self.screen, x, y, int(size * 1.5), glow_color)
            # Main droplet
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color)

    def _render_launcher(self):
        # --- Aiming line ---
        angle_rad = math.radians(self.launcher_angle)
        end_x = self.launcher_x + 60 * math.cos(angle_rad)
        end_y = self.LAUNCHER_Y + 60 * math.sin(angle_rad)
        pygame.draw.aaline(self.screen, (255, 255, 255, 50), (self.launcher_x, self.LAUNCHER_Y), (end_x, end_y))

        # --- Launcher base ---
        base_rect = pygame.Rect(0, 0, 40, 10)
        base_rect.center = (self.launcher_x, self.LAUNCHER_Y + 5)
        pygame.draw.rect(self.screen, (100, 120, 130), base_rect, border_radius=3)
        
        # --- Water type indicator on launcher ---
        if self.current_water_type == "magnetic":
            l_color = self.COLOR_MAGNETIC
        elif self.current_water_type == "super":
            l_color = self.COLOR_SUPER
        else:
            l_color = self.COLOR_WATER
        pygame.gfxdraw.filled_circle(self.screen, int(self.launcher_x), self.LAUNCHER_Y + 5, 3, l_color)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_ui.render(f"Time: {max(0, time_left):.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Water type
        type_str = f"Water Type: {self.current_water_type.upper()}"
        type_text = self.font_water_type.render(type_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(type_text, (self.WIDTH / 2 - type_text.get_width()/2, self.HEIGHT - 35))

        # Unlock Message
        if self.unlock_message_timer > 0:
            unlock_surf = self.font_unlock.render(self.unlock_message_text, True, (255, 255, 100))
            unlock_rect = unlock_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(unlock_surf, unlock_rect)
            self.unlock_message_timer -= 1


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cumulative_score": self.cumulative_score,
            "fire_percentage": np.sum(self.fire_map > 0) / self.fire_map.size,
            "water_type": self.current_water_type,
        }
        
    def _create_splash_particles(self, x, y, strength):
        num_particles = int(strength * 2)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(10, 20),
                "size": self.np_random.integers(1, 3),
                "color": (150, 200, 255, 150)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Controls ---
    # Arrows: Move launcher & aim
    # Space: Launch (press, not hold)
    # Shift: Hold to increase droplet size, release to decrease
    # Q: Quit
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Wildfire Water")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Shift
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE:
                    action[1] = 1 # Space is a press event
                if event.key == pygame.K_r: # Reset on R
                    obs, info = env.reset()
                    total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart or 'Q' to quit.")

        clock.tick(env.FPS)
        
    env.close()