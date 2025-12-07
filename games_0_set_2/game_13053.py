import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:31:17.336579
# Source Brief: brief_03053.md
# Brief Index: 3053
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
    A Gymnasium environment for a visually-rich, arcade-puzzle game.
    The player aims a slingshot to launch colored stars, matching them with
    cosmic debris to trigger satisfying chain reactions. The goal is to
    clear all debris from the screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Aim your cosmic slingshot to launch stars and clear the screen of matching debris. "
        "Create chain reactions for a higher score."
    )
    user_guide = (
        "Controls: ←→ to aim, ↑↓ to adjust power. Press space to launch a star and shift to launch a "
        "powerful gravity star."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000

    # --- Colors ---
    COLOR_BG = (15, 10, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SLINGSHOT = (100, 110, 130)
    COLOR_SLINGSHOT_BAND = (250, 240, 200)
    STAR_COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 120, 255),
        "yellow": (255, 255, 80),
    }
    GRAVITY_STAR_COLOR = (255, 255, 255)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 16)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        
        self.slingshot_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30)
        self.slingshot_angle = -math.pi / 2
        self.launch_power = 5.0
        self.min_power, self.max_power = 2.0, 15.0
        
        self.stars = []
        self.debris = []
        self.particles = []
        self.shockwaves = []
        
        self.next_star_color_key = "red"
        self.gravity_stars_available = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.background_stars = []
        self._generate_background_stars()

        # self.reset() is called implicitly by some wrappers, but we call it here
        # to ensure the environment is in a valid state after __init__.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        
        self.slingshot_angle = -math.pi / 2
        self.launch_power = 7.0
        
        self.stars.clear()
        self.debris.clear()
        self.particles.clear()
        self.shockwaves.clear()
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._spawn_level()
        self._update_gravity_star_availability()
        self._prepare_next_star()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._handle_input(movement, space_held, shift_held)
        
        self._update_stars()
        self._update_debris()
        self._update_particles()
        reward += self._update_shockwaves()
        
        collision_reward, chain_reaction_bonus, gravity_star_bonus = self._handle_collisions()
        reward += collision_reward + chain_reaction_bonus + gravity_star_bonus
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            reward -= 100 # Penalty for losing
            
        if not self.debris and not self.game_over:
            reward += 100 # Bonus for clearing level
            self.level += 1
            self._spawn_level()
            self._update_gravity_star_availability()
            self._prepare_next_star()
            # SFX: level_complete_chime.wav
        
        return self._get_observation(), reward, terminated or truncated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Adjust angle
        if movement == 3: # Left
            self.slingshot_angle -= 0.05
        elif movement == 4: # Right
            self.slingshot_angle += 0.05
        self.slingshot_angle = self.slingshot_angle % (2 * math.pi)

        # Adjust power
        if movement == 1: # Up
            self.launch_power += 0.2
        elif movement == 2: # Down
            self.launch_power -= 0.2
        self.launch_power = np.clip(self.launch_power, self.min_power, self.max_power)

        # Launch star on PRESS (not hold)
        if space_held and not self.prev_space_held:
            self._launch_star('normal')
        if shift_held and not self.prev_shift_held and self.gravity_stars_available > 0:
            self._launch_star('gravity')

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _launch_star(self, star_type):
        if star_type == 'normal':
            pos = list(self.slingshot_pos)
            vel = [self.launch_power * math.cos(self.slingshot_angle), 
                   self.launch_power * math.sin(self.slingshot_angle)]
            self.stars.append({
                "pos": pos, "vel": vel, "color": self.STAR_COLORS[self.next_star_color_key],
                "color_key": self.next_star_color_key, "type": "normal", "radius": 7
            })
            self._prepare_next_star()
            # SFX: launch_normal.wav
        elif star_type == 'gravity':
            self.gravity_stars_available -= 1
            pos = list(self.slingshot_pos)
            vel = [self.launch_power * math.cos(self.slingshot_angle), 
                   self.launch_power * math.sin(self.slingshot_angle)]
            self.stars.append({
                "pos": pos, "vel": vel, "color": self.GRAVITY_STAR_COLOR,
                "color_key": "gravity", "type": "gravity", "radius": 10
            })
            # SFX: launch_gravity.wav

    def _update_stars(self):
        for star in self.stars:
            star["pos"][0] += star["vel"][0]
            star["pos"][1] += star["vel"][1]
        self.stars = [s for s in self.stars if 0 < s["pos"][0] < self.SCREEN_WIDTH and 0 < s["pos"][1] < self.SCREEN_HEIGHT]

    def _update_debris(self):
        for d in self.debris:
            d["angle"] += d["rot_speed"]

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _update_shockwaves(self):
        reward = 0
        destroyed_in_wave = set()
        for sw in self.shockwaves:
            sw["radius"] += sw["speed"]
            sw["life"] -= 1
            
            for i, d in enumerate(self.debris):
                if i not in destroyed_in_wave:
                    dist = math.hypot(d["pos"][0] - sw["pos"][0], d["pos"][1] - sw["pos"][1])
                    if dist < sw["radius"] + d["radius"] and d["color_key"] == sw["color_key"]:
                        self._create_explosion(d["pos"], d["color"], 30)
                        destroyed_in_wave.add(i)
                        reward += 0.1 # Reward for chain reaction destruction
                        # SFX: chain_destroy.wav
        
        if destroyed_in_wave:
            self.debris = [d for i, d in enumerate(self.debris) if i not in destroyed_in_wave]
        
        self.shockwaves = [sw for sw in self.shockwaves if sw["life"] > 0]
        return reward

    def _handle_collisions(self):
        reward = 0
        chain_reaction_bonus = 0
        gravity_star_bonus = 0
        
        stars_to_remove = []
        debris_to_remove = set()
        
        for i, star in enumerate(self.stars):
            for j, d in enumerate(self.debris):
                if j in debris_to_remove: continue
                
                dist = math.hypot(star["pos"][0] - d["pos"][0], star["pos"][1] - d["pos"][1])
                if dist < star["radius"] + d["radius"]:
                    if star["type"] == 'normal' and star["color_key"] == d["color_key"]:
                        stars_to_remove.append(i)
                        debris_to_remove.add(j)
                        self.score += 1
                        reward += 0.1
                        self._create_explosion(d["pos"], d["color"], 50)
                        self._create_shockwave(d["pos"], d["color_key"])
                        # SFX: match_destroy.wav
                        break # Star can only hit one debris
                    elif star["type"] == 'gravity':
                        stars_to_remove.append(i)
                        destroyed_by_gravity = self._trigger_gravity_explosion(star["pos"])
                        gravity_debris_count = len(destroyed_by_gravity)
                        debris_to_remove.update(destroyed_by_gravity)
                        self.score += gravity_debris_count
                        reward += gravity_debris_count * 0.1
                        if gravity_debris_count >= 5:
                            gravity_star_bonus += 5
                        break
        
        # Check for chain reaction bonus (initial hit + shockwave hits)
        if len(debris_to_remove) >= 3:
            chain_reaction_bonus += 1
            
        if debris_to_remove:
            self.debris = [d for j, d in enumerate(self.debris) if j not in debris_to_remove]
        if stars_to_remove:
            self.stars = [s for i, s in enumerate(self.stars) if i not in stars_to_remove]
            
        return reward, chain_reaction_bonus, gravity_star_bonus

    def _trigger_gravity_explosion(self, pos):
        radius = 120
        self._create_explosion(pos, self.GRAVITY_STAR_COLOR, 200, is_gravity=True)
        # SFX: gravity_explosion.wav
        
        destroyed_indices = set()
        for i, d in enumerate(self.debris):
            dist = math.hypot(d["pos"][0] - pos[0], d["pos"][1] - pos[1])
            if dist < radius:
                destroyed_indices.add(i)
                self._create_explosion(d["pos"], d["color"], 20, is_gravity=False)
        return destroyed_indices

    def _create_explosion(self, pos, color, num_particles, is_gravity=False):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5 if not is_gravity else 8)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            life = self.np_random.integers(20, 40)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color})
            
    def _create_shockwave(self, pos, color_key):
        self.shockwaves.append({
            "pos": pos, "radius": 15, "speed": 3, "life": 15, "color_key": color_key
        })

    def _spawn_level(self):
        self.debris.clear()
        num_debris = 10 + (self.level - 1) * 2
        rot_speed = 0.005 + (self.level // 5) * 0.005
        
        for _ in range(num_debris):
            while True:
                pos = (self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                       self.np_random.uniform(50, self.SCREEN_HEIGHT - 120))
                
                # Ensure no overlap with other debris
                if not any(math.hypot(pos[0]-d["pos"][0], pos[1]-d["pos"][1]) < 40 for d in self.debris):
                    break
            
            color_key = self.np_random.choice(list(self.STAR_COLORS.keys()))
            self.debris.append({
                "pos": pos, "radius": self.np_random.uniform(12, 18),
                "color": self.STAR_COLORS[color_key], "color_key": color_key,
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "rot_speed": rot_speed * self.np_random.choice([-1, 1]),
                "shape_points": self._generate_asteroid_shape()
            })

    def _generate_asteroid_shape(self):
        points = []
        num_vertices = self.np_random.integers(5, 9)
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(0.7, 1.3)
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
        return points

    def _prepare_next_star(self):
        if not self.debris:
            self.next_star_color_key = self.np_random.choice(list(self.STAR_COLORS.keys()))
            return
        
        available_colors = {d["color_key"] for d in self.debris}
        self.next_star_color_key = self.np_random.choice(list(available_colors))
        
    def _update_gravity_star_availability(self):
        if self.level >= 15:
            self.gravity_stars_available = 3
        elif self.level >= 10:
            self.gravity_stars_available = 2
        elif self.level >= 5:
            self.gravity_stars_available = 1
        else:
            self.gravity_stars_available = 0

    def _check_termination(self):
        for d in self.debris:
            if d["pos"][1] + d["radius"] > self.slingshot_pos[1] - 20:
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_debris()
        self._render_stars()
        self._render_slingshot()
        self._render_particles()
        self._render_shockwaves()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.background_stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['radius'])

    def _render_slingshot(self):
        # Base
        base_rect = pygame.Rect(0, 0, 80, 20)
        base_rect.center = self.slingshot_pos
        pygame.draw.rect(self.screen, self.COLOR_SLINGSHOT, base_rect, border_radius=5)

        # Aiming line and loaded star
        end_x = self.slingshot_pos[0] + (self.launch_power * 5) * math.cos(self.slingshot_angle)
        end_y = self.slingshot_pos[1] + (self.launch_power * 5) * math.sin(self.slingshot_angle)
        
        # Band
        pygame.draw.aaline(self.screen, self.COLOR_SLINGSHOT_BAND, (base_rect.left, base_rect.centery), (end_x, end_y))
        pygame.draw.aaline(self.screen, self.COLOR_SLINGSHOT_BAND, (base_rect.right, base_rect.centery), (end_x, end_y))

        # Loaded star
        color = self.STAR_COLORS[self.next_star_color_key]
        pygame.gfxdraw.filled_circle(self.screen, int(end_x), int(end_y), 7, color)
        pygame.gfxdraw.aacircle(self.screen, int(end_x), int(end_y), 7, color)

    def _render_debris(self):
        for d in self.debris:
            points = []
            for p in d["shape_points"]:
                rotated_x = p[0] * math.cos(d["angle"]) - p[1] * math.sin(d["angle"])
                rotated_y = p[0] * math.sin(d["angle"]) + p[1] * math.cos(d["angle"])
                points.append((d["pos"][0] + rotated_x * d["radius"], 
                               d["pos"][1] + rotated_y * d["radius"]))
            
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, d["color"])
                pygame.gfxdraw.aapolygon(self.screen, points, d["color"])

    def _render_stars(self):
        for star in self.stars:
            pos = (int(star["pos"][0]), int(star["pos"][1]))
            radius = star["radius"]
            color = star["color"]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            if star["type"] == "gravity":
                # Pulsating glow effect
                pulse_alpha = 100 + 80 * math.sin(self.steps * 0.2)
                glow_color = (*color, int(pulse_alpha))
                s = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
                pygame.draw.circle(s, glow_color, (radius * 2, radius * 2), radius * 2)
                self.screen.blit(s, (pos[0] - radius * 2, pos[1] - radius * 2))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 40.0))
            color = (*p["color"], alpha)
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (2, 2), 2)
            self.screen.blit(s, (int(p["pos"][0]) - 2, int(p["pos"][1]) - 2))

    def _render_shockwaves(self):
        for sw in self.shockwaves:
            alpha = max(0, 255 * (sw["life"] / 15.0))
            color = (*self.STAR_COLORS[sw["color_key"]], int(alpha))
            s = pygame.Surface((sw["radius"] * 2, sw["radius"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (sw["radius"], sw["radius"]), sw["radius"], width=2)
            self.screen.blit(s, (int(sw["pos"][0] - sw["radius"]), int(sw["pos"][1] - sw["radius"])))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        level_text = self.font_level.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))

        if self.gravity_stars_available > 0:
            grav_text = self.font_level.render(f"GRAVITY STARS: {self.gravity_stars_available}", True, self.GRAVITY_STAR_COLOR)
            self.screen.blit(grav_text, (10, self.SCREEN_HEIGHT - grav_text.get_height() - 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            game_over_text = self.font_main.render("GAME OVER", True, (255, 50, 50))
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _generate_background_stars(self):
        self.background_stars.clear()
        for _ in range(150):
            pos = (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT))
            radius = random.choice([0, 0, 0, 1, 1, 2])
            brightness = random.randint(50, 150)
            color = (brightness, brightness, int(brightness * 1.1))
            self.background_stars.append({'pos': pos, 'radius': radius, 'color': color})

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not work in a headless environment
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cosmic Chains")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Convert the observation (H, W, C) back to a Pygame surface (W, H)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"GAME OVER! Final Score: {info['score']}")
            pygame.time.wait(2000)
            env.reset()

        clock.tick(GameEnv.FPS)
        
    env.close()