import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:02:57.470657
# Source Brief: brief_00848.md
# Brief Index: 848
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cyberpunk-themed, color-matching arcade shooter.
    The player controls a cursor to shoot down color-coded drones, utilizing a
    combo system and a slow-motion ability.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A cyberpunk-themed, color-matching arcade shooter. Shoot down color-coded drones, "
        "build combos, and use a slow-motion ability to survive."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to shoot and hold shift to activate slow-motion."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    WIN_WAVE = 20

    # Colors (Cyberpunk Neon)
    COLOR_BG = (10, 0, 25)
    COLOR_GRID = (30, 20, 50)
    COLOR_WHITE = (255, 255, 255)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_SLOWMO_TINT = (50, 100, 255, 100)
    DRONE_COLORS = {
        "red": (255, 50, 50),
        "green": (50, 255, 50),
        "blue": (50, 100, 255),
        "yellow": (255, 255, 0),
    }
    COLOR_NAMES = list(DRONE_COLORS.keys())
    
    # Game Parameters
    CURSOR_SPEED = 250.0  # pixels per second
    PROJECTILE_SPEED = 600.0
    FIRE_COOLDOWN = 0.2  # seconds
    PLAYER_MAX_HEALTH = 100
    SLOWMO_MAX_CHARGE = 100.0
    SLOWMO_DEPLETION_RATE = 30.0 # per second
    SLOWMO_CHARGE_PER_KILL = 15.0
    SLOWMO_FACTOR = 0.25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_combo = pygame.font.SysFont("Verdana", 28, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 50, bold=True)

        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.cursor_pos = np.array([0.0, 0.0])
        self.drones = []
        self.projectiles = []
        self.particles = []
        self.current_wave = 0
        self.combo = 0
        self.slow_mo_charge = 0.0
        self.slow_mo_active = False
        self.fire_cooldown_timer = 0.0
        self.weapon_color_index = 0
        self.prev_space_held = False
        self.reward_this_step = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.PLAYER_MAX_HEALTH
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        
        self.drones = []
        self.projectiles = []
        self.particles = []
        
        self.current_wave = 0
        self.combo = 0
        self.slow_mo_charge = self.SLOWMO_MAX_CHARGE / 2
        self.slow_mo_active = False
        self.fire_cooldown_timer = 0.0
        self.weapon_color_index = 0
        self.prev_space_held = False
        
        self._spawn_next_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Time Management ---
        delta_time = self.clock.tick(self.metadata["render_fps"]) / 1000.0
        
        self.reward_this_step = 0.0
        self.steps += 1

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_slow_mo(shift_held, delta_time)
        effective_delta_time = delta_time * (self.SLOWMO_FACTOR if self.slow_mo_active else 1.0)
        
        self._handle_input(movement, space_held, effective_delta_time)
        
        # --- Update Game State ---
        self._update_projectiles(effective_delta_time)
        self._update_drones(effective_delta_time)
        self._update_particles(effective_delta_time)
        
        self._handle_collisions()

        # --- Game Flow ---
        if not self.drones:
            self._spawn_next_wave()
        
        # --- Termination and Reward ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.player_health <= 0:
                self.reward_this_step += -100.0 # Loss penalty
            elif self.current_wave > self.WIN_WAVE:
                self.reward_this_step += 100.0 # Win bonus
            self.game_over = True

        self.score = max(0, self.score) # Ensure score doesn't go negative

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_slow_mo(self, shift_held, delta_time):
        if shift_held and self.slow_mo_charge > 0:
            if not self.slow_mo_active:
                # Strategic activation reward
                if len(self.drones) >= 3:
                    self.reward_this_step += 1.0
                # sfx: slow_mo_activate
            self.slow_mo_active = True
            self.slow_mo_charge -= self.SLOWMO_DEPLETION_RATE * delta_time
        else:
            if self.slow_mo_active:
                pass # sfx: slow_mo_deactivate
            self.slow_mo_active = False

        self.slow_mo_charge = max(0, self.slow_mo_charge)

    def _handle_input(self, movement, space_held, delta_time):
        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1.0  # Up
        elif movement == 2: move_vec[1] = 1.0   # Down
        elif movement == 3: move_vec[0] = -1.0  # Left
        elif movement == 4: move_vec[0] = 1.0   # Right
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        self.cursor_pos += move_vec * self.CURSOR_SPEED * delta_time
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # Shooting (on button press, not hold)
        if self.fire_cooldown_timer > 0:
            self.fire_cooldown_timer -= delta_time

        if space_held and not self.prev_space_held and self.fire_cooldown_timer <= 0:
            self.fire_cooldown_timer = self.FIRE_COOLDOWN
            
            proj_color_name = self.COLOR_NAMES[self.weapon_color_index]
            proj_color_rgb = self.DRONE_COLORS[proj_color_name]
            
            self.projectiles.append({
                "pos": self.cursor_pos.copy(),
                "color_name": proj_color_name,
                "color_rgb": proj_color_rgb,
                "trail": []
            })
            # sfx: shoot
            
            self.weapon_color_index = (self.weapon_color_index + 1) % len(self.COLOR_NAMES)
        
        self.prev_space_held = space_held
        
    def _update_projectiles(self, delta_time):
        for p in self.projectiles:
            p["trail"].append(p["pos"].copy())
            if len(p["trail"]) > 10:
                p["trail"].pop(0)
            p["pos"][1] -= self.PROJECTILE_SPEED * delta_time
        self.projectiles = [p for p in self.projectiles if p["pos"][1] > -10]

    def _update_drones(self, delta_time):
        for d in self.drones:
            d["age"] += delta_time
            # Pattern-based movement
            if d["pattern"] == "vertical":
                d["pos"][1] += d["speed"] * delta_time
            elif d["pattern"] == "sine":
                d["pos"][0] = d["start_x"] + math.sin(d["age"] * d["freq"]) * d["amp"]
                d["pos"][1] += d["speed"] * delta_time
            elif d["pattern"] == "bounce":
                d["pos"] += d["vel"] * delta_time
                if d["pos"][0] < d["radius"] or d["pos"][0] > self.SCREEN_WIDTH - d["radius"]:
                    d["vel"][0] *= -1
                    d["pos"][0] = np.clip(d["pos"][0], d["radius"], self.SCREEN_WIDTH - d["radius"])

            if d["pos"][1] > self.SCREEN_HEIGHT + d["radius"]:
                self.player_health -= 10
                self.combo = 0 # Reaching bottom breaks combo
                # sfx: player_damage
        
        self.drones = [d for d in self.drones if d["pos"][1] <= self.SCREEN_HEIGHT + d["radius"]]

    def _update_particles(self, delta_time):
        for p in self.particles:
            p["pos"] += p["vel"] * delta_time
            p["lifespan"] -= delta_time
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _handle_collisions(self):
        projectiles_to_remove = []
        drones_to_remove = []

        for i, proj in enumerate(self.projectiles):
            for j, drone in enumerate(self.drones):
                if j in drones_to_remove: continue
                
                dist = np.linalg.norm(proj["pos"] - drone["pos"])
                if dist < drone["radius"]:
                    projectiles_to_remove.append(i)
                    self.reward_this_step += 0.1 # Hit reward
                    # sfx: hit_drone

                    if proj["color_name"] == drone["color_name"]:
                        drone["health"] -= 50
                        self.score += 10
                    else: # Mismatched color
                        drone["health"] -= 25
                        self.score += 5
                        if self.combo > 0:
                            # sfx: combo_break
                            self.combo = 0

                    if drone["health"] <= 0:
                        drones_to_remove.append(j)
                        self.score += 50
                        self.reward_this_step += 0.5 # Destroy reward
                        
                        if proj["color_name"] == drone["color_name"]:
                            self.combo += 1
                            if self.combo > 1:
                                self.score += self.combo * 10
                                self.reward_this_step += self.combo # Combo reward
                                # sfx: combo_increase
                        
                        self.slow_mo_charge = min(self.SLOWMO_MAX_CHARGE, self.slow_mo_charge + self.SLOWMO_CHARGE_PER_KILL)
                        self._create_explosion(drone["pos"], drone["color_rgb"])
                        # sfx: explosion

                    break # Projectile can only hit one drone

        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        self.drones = [d for j, d in enumerate(self.drones) if j not in drones_to_remove]
    
    def _create_explosion(self, pos, color):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": random.uniform(0.3, 0.8),
                "radius": random.uniform(1, 4),
                "color": color
            })
            
    def _spawn_next_wave(self):
        if self.current_wave > 0:
            self.reward_this_step += 5.0 # Wave complete reward
            self.score += self.current_wave * 100

        self.current_wave += 1
        if self.current_wave > self.WIN_WAVE:
            return

        wave_speed = 100 + self.current_wave * 7.5
        num_drones = 3 + self.current_wave // 2

        patterns = ["vertical", "sine", "bounce"]
        pattern = self.np_random.choice(patterns)

        for i in range(num_drones):
            color_name = self.np_random.choice(self.COLOR_NAMES)
            radius = 15
            start_y = -radius - i * 60
            
            drone = {
                "pos": np.array([0.0, float(start_y)]),
                "radius": radius,
                "color_name": color_name,
                "color_rgb": self.DRONE_COLORS[color_name],
                "health": 100,
                "speed": self.np_random.uniform(0.9, 1.1) * wave_speed,
                "pattern": pattern,
                "age": 0,
            }

            if pattern == "vertical":
                drone["pos"][0] = self.np_random.uniform(radius, self.SCREEN_WIDTH - radius)
            elif pattern == "sine":
                drone["start_x"] = self.np_random.uniform(100, self.SCREEN_WIDTH - 100)
                drone["pos"][0] = drone["start_x"]
                drone["amp"] = self.np_random.uniform(50, 150)
                drone["freq"] = self.np_random.uniform(0.8, 1.5)
            elif pattern == "bounce":
                drone["pos"][0] = self.np_random.uniform(radius, self.SCREEN_WIDTH - radius)
                angle = self.np_random.uniform(math.pi / 4, 3 * math.pi / 4)
                drone["vel"] = np.array([math.cos(angle), math.sin(angle)]) * drone["speed"]

            self.drones.append(drone)
    
    def _check_termination(self):
        return (
            self.player_health <= 0 or
            self.current_wave > self.WIN_WAVE or
            self.steps >= self.MAX_STEPS
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "health": self.player_health,
            "combo": self.combo,
            "slow_mo_charge": self.slow_mo_charge
        }

    def _get_observation(self):
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_projectiles()
        self._render_drones()
        self._render_player_cursor()
        if self.slow_mo_active: self._render_slow_mo_effect()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering Helpers ---
    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 0.8))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

    def _render_projectiles(self):
        for p in self.projectiles:
            # Trail
            for i, pos in enumerate(p["trail"]):
                alpha = int(150 * (i / len(p["trail"])))
                color = (*p["color_rgb"], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 2, color)
            # Main projectile
            self._draw_glowing_circle(self.screen, p["pos"], p["color_rgb"], 4)

    def _render_drones(self):
        for d in self.drones:
            # Health indicates brightness
            health_ratio = d["health"] / 100.0
            color = tuple(int(c * health_ratio + (255 - c) * (1 - health_ratio)) for c in d["color_rgb"])
            self._draw_glowing_circle(self.screen, d["pos"], color, d["radius"])
            # Draw a small white core
            pygame.gfxdraw.aacircle(self.screen, int(d["pos"][0]), int(d["pos"][1]), int(d["radius"]*0.3), self.COLOR_WHITE)

    def _render_player_cursor(self):
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        current_weapon_color = self.DRONE_COLORS[self.COLOR_NAMES[self.weapon_color_index]]
        
        # Glow
        glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*current_weapon_color, 50), (20, 20), 20)
        self.screen.blit(glow_surf, (pos[0] - 20, pos[1] - 20))
        
        # Crosshair
        pygame.draw.line(self.screen, self.COLOR_WHITE, (pos[0] - 10, pos[1]), (pos[0] - 5, pos[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_WHITE, (pos[0] + 5, pos[1]), (pos[0] + 10, pos[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_WHITE, (pos[0], pos[1] - 10), (pos[0], pos[1] - 5), 2)
        pygame.draw.line(self.screen, self.COLOR_WHITE, (pos[0], pos[1] + 5), (pos[0], pos[1] + 10), 2)
        
        # Center dot
        pygame.draw.circle(self.screen, current_weapon_color, pos, 3)

    def _render_slow_mo_effect(self):
        tint_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        tint_surface.fill(self.COLOR_SLOWMO_TINT)
        self.screen.blit(tint_surface, (0, 0))

    def _render_ui(self):
        # Score and Wave
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.WIN_WAVE}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 5))
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 5))

        # Health Bar
        health_bar_bg = pygame.Rect(10, 30, 200, 15)
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        health_bar_fg = pygame.Rect(10, 30, 200 * health_ratio, 15)
        pygame.draw.rect(self.screen, (255, 0, 0, 50), health_bar_bg)
        pygame.draw.rect(self.screen, (255, 0, 0), health_bar_fg)
        
        # Slow-mo Bar
        slowmo_bar_bg = pygame.Rect(self.SCREEN_WIDTH - 210, 30, 200, 15)
        slowmo_ratio = self.slow_mo_charge / self.SLOWMO_MAX_CHARGE
        slowmo_bar_fg = pygame.Rect(self.SCREEN_WIDTH - 210, 30, 200 * slowmo_ratio, 15)
        pygame.draw.rect(self.screen, (*self.COLOR_SLOWMO_TINT[:3], 50), slowmo_bar_bg)
        pygame.draw.rect(self.screen, self.COLOR_SLOWMO_TINT[:3], slowmo_bar_fg)
        
        # Combo Meter
        if self.combo > 1:
            combo_text = self.font_combo.render(f"{self.combo}x COMBO!", True, self.COLOR_PLAYER)
            text_rect = combo_text.get_rect(center=(self.SCREEN_WIDTH / 2, 50))
            self.screen.blit(combo_text, text_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        msg = "LEVEL COMPLETE" if self.current_wave > self.WIN_WAVE else "GAME OVER"
        color = (0, 255, 0) if self.current_wave > self.WIN_WAVE else (255, 0, 0)
        
        text = self.font_game_over.render(msg, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _draw_glowing_circle(self, surface, pos, color, radius):
        int_pos = (int(pos[0]), int(pos[1]))
        int_radius = int(radius)
        if int_radius <= 0: return

        # Glow effect
        glow_radius = int(int_radius * 1.8)
        glow_alpha = 60
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int_pos[0] - glow_radius, int_pos[1] - glow_radius))

        # Main circle
        pygame.gfxdraw.aacircle(surface, int_pos[0], int_pos[1], int_radius, color)
        pygame.gfxdraw.filled_circle(surface, int_pos[0], int_pos[1], int_radius, color)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # The original code had a validation check in __init__ which is not standard
    # for gym.Env and can cause issues with env registration. It's removed from __init__
    # but could be run manually after instantiation if needed.
    
    # To run validation:
    # env = GameEnv()
    # try:
    #     # This is a simplified version of the original validation logic
    #     print("Running validation...")
    #     obs, info = env.reset()
    #     assert env.observation_space.contains(obs)
    #     action = env.action_space.sample()
    #     obs, reward, term, trunc, info = env.step(action)
    #     assert env.observation_space.contains(obs)
    #     print("✓ Implementation validated successfully")
    # except Exception as e:
    #     print(f"Validation failed: {e}")
    # env.close()
    
    # To play manually:
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Cyber Drone Shooter - Manual Control")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000) # Pause before closing
            
        clock.tick(env.metadata["render_fps"])

    env.close()