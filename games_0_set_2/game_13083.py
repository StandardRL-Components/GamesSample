import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:54:10.844843
# Source Brief: brief_03083.md
# Brief Index: 3083
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your planetary system from encroaching black holes by firing a repulsor ball to deflect them."
    )
    user_guide = (
        "Use ←→ arrow keys to aim and ↑↓ to adjust power. Press space to launch the repulsor ball. Press shift to reset your aim."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array", level=1):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.LEVEL = level

        # Visuals
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLANET = (60, 140, 220)
        self.COLOR_PLANET_GLOW = (30, 70, 110)
        self.COLOR_BLACK_HOLE = (20, 0, 30)
        self.COLOR_ACCRETION_DISK = (120, 40, 200)
        self.COLOR_GOLF_BALL = (255, 255, 0)
        self.COLOR_GOLF_BALL_GLOW = (200, 200, 0)
        self.COLOR_AIM = (255, 255, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_ENERGY_HIGH = (0, 255, 0)
        self.COLOR_ENERGY_MID = (255, 255, 0)
        self.COLOR_ENERGY_LOW = (255, 0, 0)

        # Physics
        self.BALL_REPEL_FORCE = 50000
        self.BH_ATTRACT_SPEED = 0.5
        self.BALL_MAX_POWER = 15
        self.BALL_MIN_POWER = 3
        self.ENERGY_PER_SHOT_BASE = 10
        self.ENERGY_PER_SHOT_POWER_SCALAR = 1.5

        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.planets = []
        self.black_holes = []
        self.golf_ball = None
        self.particles = []
        self.stars = []
        self.energy = 0
        self.max_energy = 0
        self.game_phase = "AIMING"  # "AIMING" or "ACTION"
        self.aim_angle = 0.0
        self.aim_power = 0.0
        self.last_space_held = False
        self.last_shift_held = False
        self.launcher_pos = (self.WIDTH // 2, self.HEIGHT - 30)
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is a helper, not part of the standard API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "AIMING"
        self.last_space_held = False
        self.last_shift_held = False

        self._generate_level()
        self._reset_aim()

        self.golf_ball = None
        self.particles = []
        
        # Generate a static starfield for performance
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.uniform(0.5, 1.5),
            )
            for _ in range(200)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        if self.game_phase == "AIMING":
            # Adjust aim
            if movement == 1: self.aim_power = min(self.BALL_MAX_POWER, self.aim_power + 0.2)
            if movement == 2: self.aim_power = max(self.BALL_MIN_POWER, self.aim_power - 0.2)
            if movement == 3: self.aim_angle -= 0.05
            if movement == 4: self.aim_angle += 0.05
            self.aim_angle %= (2 * math.pi)

            # Reset aim
            if shift_pressed:
                self._reset_aim()
                # sound: aim_reset.wav

            # Launch ball
            if space_pressed and self.energy > 0:
                self._launch_ball()
                reward += -0.1 # Small penalty for taking a shot
                self.game_phase = "ACTION"
                # sound: launch_ball.wav
        
        # --- Update Game State ---
        if self.game_phase == "ACTION":
            prev_bh_distances = self._get_bh_distances()
            self._update_physics()
            new_bh_distances = self._get_bh_distances()

            # Continuous reward for deflecting black holes
            for i in range(len(self.black_holes)):
                if i < len(prev_bh_distances) and i < len(new_bh_distances):
                    dist_change = new_bh_distances[i] - prev_bh_distances[i]
                    reward += dist_change * 0.1 # +reward if distance increases
            
            # Check if ball is out of bounds
            if self.golf_ball and (
                self.golf_ball["pos"][0] < 0 or self.golf_ball["pos"][0] > self.WIDTH or
                self.golf_ball["pos"][1] < 0 or self.golf_ball["pos"][1] > self.HEIGHT
            ):
                self.golf_ball = None
                self.game_phase = "AIMING"
                # End of "turn", now black holes move naturally
                turn_reward, turn_terminated = self._process_turn_end()
                reward += turn_reward
                self.game_over = self.game_over or turn_terminated

        # --- Update Particles ---
        self._update_particles()

        # --- Check Termination Conditions ---
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            reward -= 50 # Penalty for running out of time
        
        if not self.black_holes: # Win condition
            self.game_over = True
            reward += 100
            self.score += 100
            # sound: level_win.wav

        if self.energy <= 0 and self.game_phase == "AIMING" and self.golf_ball is None:
            self.game_over = True
            reward -= 100
            # sound: game_over_energy.wav

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _generate_level(self):
        self.planets = []
        self.black_holes = []
        
        num_bhs = 1 + (self.LEVEL - 1) // 5
        planet_health = max(1, 10 - (self.LEVEL - 1) // 2)
        self.max_energy = 100 + num_bhs * 20
        self.energy = self.max_energy

        # Add a central planet
        self.planets.append({
            "pos": np.array([self.WIDTH / 2, self.HEIGHT / 3], dtype=float),
            "radius": 20,
            "health": planet_health,
            "max_health": planet_health
        })
        
        # Spawn black holes
        for _ in range(num_bhs):
            side = self.np_random.integers(4)
            if side == 0: x, y = self.np_random.uniform(0, self.WIDTH), -20
            elif side == 1: x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20
            elif side == 2: x, y = -20, self.np_random.uniform(0, self.HEIGHT)
            else: x, y = self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)
            
            self.black_holes.append({
                "pos": np.array([x, y], dtype=float),
                "vel": np.array([0.0, 0.0], dtype=float),
                "radius": 12,
                "swirl_angle": self.np_random.uniform(0, 2 * math.pi)
            })

    def _reset_aim(self):
        self.aim_angle = -math.pi / 2
        self.aim_power = (self.BALL_MIN_POWER + self.BALL_MAX_POWER) / 2

    def _launch_ball(self):
        cost = self.ENERGY_PER_SHOT_BASE + self.aim_power * self.ENERGY_PER_SHOT_POWER_SCALAR
        self.energy -= cost
        
        vel_x = self.aim_power * math.cos(self.aim_angle)
        vel_y = self.aim_power * math.sin(self.aim_angle)
        
        self.golf_ball = {
            "pos": np.array(self.launcher_pos, dtype=float),
            "vel": np.array([vel_x, vel_y], dtype=float),
            "radius": 6,
            "trail": []
        }

    def _update_physics(self):
        # Update ball and apply forces
        if self.golf_ball:
            self.golf_ball["pos"] += self.golf_ball["vel"]
            self.golf_ball["trail"].append(tuple(self.golf_ball["pos"]))
            if len(self.golf_ball["trail"]) > 20:
                self.golf_ball["trail"].pop(0)

            for bh in self.black_holes:
                dist_vec = bh["pos"] - self.golf_ball["pos"]
                dist_sq = np.dot(dist_vec, dist_vec)
                if dist_sq < 1: dist_sq = 1
                
                force_mag = self.BALL_REPEL_FORCE / dist_sq
                force_vec = (dist_vec / np.sqrt(dist_sq)) * force_mag
                
                # Apply force scaled by time (1/FPS)
                bh["vel"] += force_vec / self.FPS
        
        # Update black holes based on their velocity
        for bh in self.black_holes:
            bh["pos"] += bh["vel"] / self.FPS
            bh["swirl_angle"] += 0.1
            # Dampen velocity
            bh["vel"] *= 0.98

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _process_turn_end(self):
        """Called when a shot is finished, advancing the game turn."""
        reward = 0
        terminated = False
        bhs_to_remove = []

        for i, bh in enumerate(self.black_holes):
            # Natural movement towards nearest planet
            if not self.planets: break
            nearest_planet = min(self.planets, key=lambda p: np.linalg.norm(p["pos"] - bh["pos"]))
            direction = nearest_planet["pos"] - bh["pos"]
            dist = np.linalg.norm(direction)
            if dist > 1:
                bh["pos"] += (direction / dist) * self.BH_ATTRACT_SPEED * 5 # Move 5 steps worth

            # Check for planet collision
            if dist < bh["radius"] + nearest_planet["radius"]:
                nearest_planet["health"] -= 1
                reward -= 10
                self.score -= 10
                # sound: planet_hit.wav
                self._create_explosion(nearest_planet["pos"], self.COLOR_PLANET, 30)
                if nearest_planet["health"] <= 0:
                    reward -= 100
                    terminated = True
                    # sound: game_over_planet_destroyed.wav
                bhs_to_remove.append(bh)
                continue

            # Check if BH is deflected out of bounds
            if not (-50 < bh["pos"][0] < self.WIDTH + 50 and -50 < bh["pos"][1] < self.HEIGHT + 50):
                reward += 5
                self.score += 5
                # sound: bh_deflected.wav
                bhs_to_remove.append(bh)

        self.black_holes = [bh for bh in self.black_holes if bh not in bhs_to_remove]
        return reward, terminated

    def _get_bh_distances(self):
        if not self.planets or not self.black_holes:
            return []
        return [min(np.linalg.norm(bh["pos"] - p["pos"]) for p in self.planets) for bh in self.black_holes]
    
    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": color
            })
    
    def _get_observation(self):
        self._render_background()
        self._render_game_objects()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x, y, size in self.stars:
            c = int(size * 50)
            self.screen.set_at((int(x), int(y)), (c, c, c))

    def _render_game_objects(self):
        # Render planets
        for p in self.planets:
            pos_i = (int(p["pos"][0]), int(p["pos"][1]))
            # Glow effect
            for i in range(p["radius"], p["radius"] + 10):
                alpha = 1 - (i - p["radius"]) / 10
                color = (*self.COLOR_PLANET_GLOW, int(alpha * 100))
                pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], i, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], p["radius"], self.COLOR_PLANET)
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], p["radius"], self.COLOR_PLANET)

        # Render black holes
        for bh in self.black_holes:
            pos_i = (int(bh["pos"][0]), int(bh["pos"][1]))
            # Accretion disk
            for i in range(int(bh["radius"] * 1.2), int(bh["radius"] * 2.5)):
                angle_offset = i * 0.5
                start_angle = bh["swirl_angle"] + angle_offset
                end_angle = start_angle + math.pi * 0.8
                alpha = 1 - (i - bh["radius"] * 1.2) / (bh["radius"] * 1.3)
                color = (*self.COLOR_ACCRETION_DISK, int(alpha * 200))
                pygame.draw.arc(self.screen, color, (pos_i[0]-i, pos_i[1]-i, i*2, i*2), start_angle, end_angle, 2)
            pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], bh["radius"], self.COLOR_BLACK_HOLE)

        # Render golf ball
        if self.golf_ball:
            pos_i = (int(self.golf_ball["pos"][0]), int(self.golf_ball["pos"][1]))
            # Trail
            for i, trail_pos in enumerate(self.golf_ball["trail"]):
                alpha = i / len(self.golf_ball["trail"])
                radius = int(self.golf_ball["radius"] * alpha * 0.8)
                if radius > 0:
                    color = (*self.COLOR_GOLF_BALL_GLOW, int(alpha * 150))
                    pygame.gfxdraw.filled_circle(self.screen, int(trail_pos[0]), int(trail_pos[1]), radius, color)
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], self.golf_ball["radius"] + 3, (*self.COLOR_GOLF_BALL_GLOW, 100))
            # Ball
            pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], self.golf_ball["radius"], self.COLOR_GOLF_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], self.golf_ball["radius"], self.COLOR_GOLF_BALL)

        # Render particles
        for p in self.particles:
            alpha = p["life"] / 30.0
            color = (*p["color"], int(alpha * 255))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, color)

    def _render_ui(self):
        # Render launcher base
        pygame.draw.circle(self.screen, (100, 100, 100), self.launcher_pos, 10)
        pygame.draw.circle(self.screen, (150, 150, 150), self.launcher_pos, 10, 2)

        # Render aiming guide
        if self.game_phase == "AIMING":
            # Aiming line
            end_x = self.launcher_pos[0] + 30 * math.cos(self.aim_angle)
            end_y = self.launcher_pos[1] + 30 * math.sin(self.aim_angle)
            pygame.draw.aaline(self.screen, self.COLOR_AIM, self.launcher_pos, (end_x, end_y))
            # Power indicator
            power_ratio = (self.aim_power - self.BALL_MIN_POWER) / (self.BALL_MAX_POWER - self.BALL_MIN_POWER)
            r = int(255 * (1-power_ratio))
            g = int(255 * power_ratio)
            pygame.draw.circle(self.screen, (r, g, 0), self.launcher_pos, 4)

        # Render Energy Bar
        energy_ratio = max(0, self.energy / self.max_energy)
        bar_width = 200
        bar_height = 15
        bar_x, bar_y = self.WIDTH // 2 - bar_width // 2, 10
        if energy_ratio > 0.5:
            color = self.COLOR_ENERGY_HIGH
        elif energy_ratio > 0.2:
            color = self.COLOR_ENERGY_MID
        else:
            color = self.COLOR_ENERGY_LOW
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, int(bar_width * energy_ratio), bar_height))
        pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Render Text
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        level_text = self.font_small.render(f"LEVEL: {self.LEVEL}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

        if self.game_over:
            win_lose_text = "SYSTEM SAVED" if not any(p['health'] <= 0 for p in self.planets) and not self.black_holes else "SYSTEM LOST"
            text_surface = self.font_large.render(win_lose_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "level": self.LEVEL,
            "black_holes_remaining": len(self.black_holes),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block is for manual testing and visualization.
    # It will not be run by the evaluation server.
    # You can remove the `os.environ` line to run with a visible window.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(level=1)
    obs, info = env.reset()
    done = False
    
    # Manual play loop
    # Keys: ARROWS for aim, SPACE to launch, LSHIFT to reset aim
    
    action = [0, 0, 0] # no-op, no-space, no-shift
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Display the game screen
        rgb_array = env.render()
        
        # The observation is (H, W, C), but pygame surface wants (W, H)
        # and surfarray.make_surface expects (W, H, C)
        surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
        
        # Create a display if one doesn't exist
        try:
            display_surface = pygame.display.get_surface()
            if display_surface is None:
                raise AttributeError
        except (pygame.error, AttributeError):
            display_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

        display_surface.blit(surface, (0, 0))
        pygame.display.flip()
        env.clock.tick(env.FPS)

    env.close()