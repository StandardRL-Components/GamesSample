import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:46:57.126330
# Source Brief: brief_00623.md
# Brief Index: 623
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Tempest Cannon
    Master the tempest's fury by aiming and firing a powerful cannonball through a 
    destructible network of tubes, utilizing gravity and ricochets to reach the target.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Master the tempest's fury by aiming and firing a powerful cannonball through a "
        "destructible network of tubes, utilizing gravity and ricochets to reach the target."
    )
    user_guide = (
        "Controls: ↑/↓ to adjust power, ←/→ to aim. Press space to fire. Press shift to reset aim."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.GRAVITY = 0.2
        self.DAMPING = 0.85
        
        # --- Colors ---
        self.COLOR_BG_TOP = (20, 10, 40)
        self.COLOR_BG_BOTTOM = (40, 20, 70)
        self.COLOR_CANNON = (180, 180, 200)
        self.COLOR_CANNONBALL = (0, 191, 255) # Deep Sky Blue
        self.COLOR_TARGET = (50, 255, 50) # Bright Green
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_POWER_BAR_BG = (50, 50, 50)
        self.COLOR_POWER_BAR_FILL = (255, 165, 0) # Orange
        self.TUBE_COLORS = {
            'intact': (150, 150, 150),
            'damaged': (255, 165, 0),
            'critical': (255, 69, 0)
        }

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Consolas", 16)
            self.font_large = pygame.font.SysFont("Consolas", 24)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 30)
            
        # --- Game State Initialization ---
        self.game_phase = "AIMING"
        self.cannon_pos = np.array([60.0, self.HEIGHT - 50.0])
        self.cannon_angle = -45.0
        self.cannon_power = 50.0
        self.cannonball = None
        self.tubes = []
        self.target = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_phase = "AIMING"
        
        self.cannon_angle = -45.0
        self.cannon_power = 50.0
        
        self._generate_level()
        
        self.cannonball = {
            "pos": self.cannon_pos.copy(),
            "vel": np.array([0.0, 0.0]),
            "radius": 8,
            "active": False,
            "stuck_timer": 0,
            "last_pos": self.cannon_pos.copy(),
            "passed_tubes": set()
        }
        
        self.particles = []
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action
        reward = 0.0
        terminated = False

        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if self.game_phase == "AIMING":
            # --- Handle Aiming Input ---
            if movement == 1: self.cannon_power = min(100.0, self.cannon_power + 2.0)
            elif movement == 2: self.cannon_power = max(10.0, self.cannon_power - 2.0)
            elif movement == 3: self.cannon_angle = max(-90.0, self.cannon_angle - 2.0)
            elif movement == 4: self.cannon_angle = min(0.0, self.cannon_angle + 2.0)
                
            if shift_pressed:
                self.cannon_angle = -45.0
                self.cannon_power = 50.0
            
            if space_pressed:
                self.game_phase = "FIRING"
                self.cannonball["active"] = True
                rad = math.radians(self.cannon_angle)
                launch_speed = self.cannon_power / 5.0
                self.cannonball["vel"] = np.array([
                    math.cos(rad) * launch_speed,
                    math.sin(rad) * launch_speed
                ])
                # SFX: cannon_fire.wav
                self._create_particles(self.cannon_pos, 20, (255, 200, 100), 5)

        elif self.game_phase == "FIRING":
            # --- Update Physics ---
            tick_reward, shot_ended, terminal_reason = self._update_physics_tick()
            reward += tick_reward
            if shot_ended:
                terminated = True
                # Assign terminal rewards
                if terminal_reason == "TARGET":
                    reward += 100.0
                    self.score += 100
                elif terminal_reason == "OOB":
                    reward -= 10.0
                elif terminal_reason == "STALLED":
                    reward -= 1.0
                elif terminal_reason == "STUCK":
                    reward -= 10.0

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            reward -= 1.0 # Penalty for running out of time

        self._update_particles()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_physics_tick(self):
        if not self.cannonball["active"]:
            return 0.0, False, None

        # Apply gravity
        self.cannonball["vel"][1] += self.GRAVITY
        
        # Update position
        self.cannonball["pos"] += self.cannonball["vel"]
        
        reward = 0.0
        
        # --- Check for Terminal Conditions ---
        pos = self.cannonball["pos"]
        radius = self.cannonball["radius"]
        if not (0 < pos[0] < self.WIDTH and -50 < pos[1] < self.HEIGHT):
            # SFX: fail_whiff.wav
            return -10.0, True, "OOB"

        dist_to_target = np.linalg.norm(pos - self.target["pos"])
        if dist_to_target < self.target["radius"] + radius:
            # SFX: win_chime.wav
            self._create_particles(self.target["pos"], 50, self.COLOR_TARGET, 10)
            return 100.0, True, "TARGET"

        # --- Collision with Tubes ---
        for i, tube in enumerate(self.tubes):
            if not tube["active"]: continue
            
            closest_point = np.clip(self.cannonball["pos"], tube["rect"].topleft, tube["rect"].bottomright)
            dist_vec = self.cannonball["pos"] - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)
            
            if dist_sq < radius**2:
                # Collision detected
                # SFX: ricochet.wav
                self._create_particles(closest_point, 10, self.TUBE_COLORS['intact'], 3)
                
                # Damage tube
                tube["health"] -= 25
                reward -= 0.5
                self.score -= 1
                if tube["health"] <= 0:
                    tube["active"] = False
                    # SFX: explosion.wav
                    self._create_particles(tube["rect"].center, 30, self.TUBE_COLORS['critical'], 6)
                
                # Simple bounce physics
                dist = math.sqrt(dist_sq)
                overlap = radius - dist
                
                # Move ball out of collision
                if dist > 1e-6:
                    self.cannonball["pos"] += (dist_vec / dist) * overlap
                
                # Reflect velocity
                normal = dist_vec / dist
                vel_component = np.dot(self.cannonball["vel"], normal)
                self.cannonball["vel"] -= 2 * vel_component * normal
                self.cannonball["vel"] *= self.DAMPING
                
                reward += 1.0 # Ricochet reward
                self.score += 5
                
                break # Only handle one collision per frame

        # --- Passthrough Reward ---
        for i, tube in enumerate(self.tubes):
            if tube["active"] and i not in self.cannonball["passed_tubes"]:
                if tube["rect"].collidepoint(self.cannonball["pos"]):
                    reward += 0.1
                    self.score += 1
                    self.cannonball["passed_tubes"].add(i)

        # --- Check for Stalled/Stuck ---
        speed = np.linalg.norm(self.cannonball["vel"])
        if speed < 0.1 and self.cannonball["pos"][1] > self.HEIGHT - 20:
            # SFX: fizzle.wav
            return -1.0, True, "STALLED"
        
        pos_change = np.linalg.norm(self.cannonball["pos"] - self.cannonball["last_pos"])
        if pos_change < 0.5:
            self.cannonball["stuck_timer"] += 1
        else:
            self.cannonball["stuck_timer"] = 0
            self.cannonball["last_pos"] = self.cannonball["pos"].copy()
        
        if self.cannonball["stuck_timer"] > 60: # Stuck for 2 seconds
            # SFX: fail_stuck.wav
            return -10.0, True, "STUCK"

        return reward, False, None

    def _generate_level(self):
        self.tubes = []
        path_points = []
        
        start_pos = self.cannon_pos + np.array([80, -50])
        path_points.append(start_pos)
        
        current_pos = start_pos.copy()
        direction = np.array([1.0, 0.0]) # Start moving right
        
        for _ in range(self.np_random.integers(5, 10)):
            length = self.np_random.uniform(80, 150)
            end_pos = current_pos + direction * length
            
            # Clamp to screen bounds
            end_pos[0] = np.clip(end_pos[0], 50, self.WIDTH - 50)
            end_pos[1] = np.clip(end_pos[1], 50, self.HEIGHT - 50)
            
            path_points.append(end_pos)
            
            # Create tube segment from rect
            min_x, max_x = sorted((current_pos[0], end_pos[0]))
            min_y, max_y = sorted((current_pos[1], end_pos[1]))
            
            if direction[0] != 0: # Horizontal
                rect = pygame.Rect(min_x, min_y - 15, max_x - min_x, 30)
            else: # Vertical
                rect = pygame.Rect(min_x - 15, min_y, 30, max_y - min_y)
            
            self.tubes.append({"rect": rect, "health": 100, "max_health": 100, "active": True})
            
            current_pos = end_pos
            # Turn 90 degrees
            if self.np_random.random() < 0.7: # Prefer downward turns
                direction = np.array([0.0, 1.0]) if direction[0] != 0 else np.array([-direction[1], 0.0])
            else:
                direction = np.array([0.0, -1.0]) if direction[0] != 0 else np.array([direction[1], 0.0])

        # Place target near the end of the path
        target_pos = path_points[-1] + self.np_random.uniform(-50, 50, 2)
        target_pos[0] = np.clip(target_pos[0], self.WIDTH/2, self.WIDTH - 50)
        target_pos[1] = np.clip(target_pos[1], 50, self.HEIGHT - 50)
        self.target = {"pos": target_pos, "radius": 20}
        
    def _create_particles(self, pos, count, color, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Drag
            p["lifespan"] -= 1

    def _get_observation(self):
        self._render_background()
        self._render_tubes()
        self._render_target()
        self._render_cannon()
        if self.cannonball and self.cannonball["active"]:
            self._render_cannonball()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "phase": self.game_phase}

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_tubes(self):
        for tube in self.tubes:
            if not tube["active"]: continue
            health_ratio = tube["health"] / tube["max_health"]
            if health_ratio <= 0.33: color = self.TUBE_COLORS['critical']
            elif health_ratio <= 0.66: color = self.TUBE_COLORS['damaged']
            else: color = self.TUBE_COLORS['intact']
            
            pygame.draw.rect(self.screen, color, tube["rect"], border_radius=5)
            pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in color), tube["rect"], width=2, border_radius=5)

    def _render_target(self):
        pos = self.target["pos"].astype(int)
        radius = self.target["radius"]
        for i in range(radius, 0, -2):
            alpha = 255 * (1 - i / radius)**2
            color = (*self.COLOR_TARGET, int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, color)

    def _render_cannon(self):
        # Base
        base_rect = pygame.Rect(0,0, 40, 20)
        base_rect.center = (self.cannon_pos[0], self.cannon_pos[1] + 10)
        pygame.draw.rect(self.screen, self.COLOR_CANNON, base_rect, border_radius=5)
        
        # Barrel
        rad = math.radians(self.cannon_angle)
        barrel_length = 40
        end_pos = self.cannon_pos + np.array([math.cos(rad) * barrel_length, math.sin(rad) * barrel_length])
        pygame.draw.line(self.screen, self.COLOR_CANNON, self.cannon_pos.astype(int), end_pos.astype(int), 12)
        pygame.gfxdraw.filled_circle(self.screen, int(self.cannon_pos[0]), int(self.cannon_pos[1]), 8, self.COLOR_CANNON)
        
    def _render_cannonball(self):
        pos = self.cannonball["pos"].astype(int)
        radius = self.cannonball["radius"]
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_CANNONBALL)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (200, 230, 255))

    def _render_particles(self):
        for p in self.particles:
            pos = p["pos"].astype(int)
            alpha = 255 * (p["lifespan"] / p["max_lifespan"])
            color = (*p["color"], int(alpha))
            size = int(3 * (p["lifespan"] / p["max_lifespan"]))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 40))
        
        # Power Bar
        bar_x, bar_y = self.cannon_pos[0] - 40, self.HEIGHT - 120
        bar_height = 100
        power_ratio = (self.cannon_power - 10) / 90.0
        
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, 20, bar_height))
        fill_height = bar_height * power_ratio
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (bar_x, bar_y + bar_height - fill_height, 20, fill_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, 20, bar_height), 1)
        power_label = self.font_small.render("PWR", True, self.COLOR_UI_TEXT)
        self.screen.blit(power_label, (bar_x - 4, bar_y - 20))
        
    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # The following code is for local testing and visualization.
    # It will not be executed in the evaluation environment.
    
    # Un-comment the line below to run with a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Tempest Cannon")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    total_reward = 0
    
    # Remove the validation call for manual play
    # env.validate_implementation()
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before next round

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        env.clock.tick(env.metadata["render_fps"])
        
    env.close()