import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:46:28.505977
# Source Brief: brief_02371.md
# Brief Index: 2371
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Absorb nutrients and smaller rivals to grow your mass and level up. "
        "Shoot projectiles to stun opponents and avoid being consumed by larger ones."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire a projectile."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.MAX_LEVEL = 20
        self.FONT_SIZE_UI = 24
        self.FONT_SIZE_PLAYER = 18

        # Colors
        self.COLOR_BG = (15, 20, 45)
        self.COLOR_BG_GRID = (25, 30, 55)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 30)
        self.COLOR_PROJECTILE = (100, 200, 255)
        self.COLOR_NUTRIENT_PALETTE = [(255, 255, 100), (255, 100, 255), (100, 255, 255)]
        self.COLOR_TEXT = (220, 220, 240)
        self.RIVAL_BASE_COLOR = (255, 50, 50)

        # Gameplay
        self.PLAYER_INITIAL_MASS = 200
        self.PLAYER_ACCELERATION = 0.8
        self.PLAYER_FRICTION = 0.92
        self.PLAYER_MAX_SPEED = 8
        self.MASS_TO_RADIUS_SCALE = 0.2
        self.LEVEL_UP_MASS_FACTOR = 1.5
        
        self.NUTRIENT_COUNT = 25
        self.NUTRIENT_MASS = 50
        
        self.PROJECTILE_SPEED = 12
        self.PROJECTILE_MASS_COST = 10
        self.PROJECTILE_LIFESPAN = 40 # steps
        self.PROJECTILE_STUN_DURATION = 90 # steps
        self.SHOOT_COOLDOWN = 10 # steps
        
        self.RIVAL_INITIAL_COUNT = 2
        self.RIVAL_INITIAL_MASS = 150
        self.RIVAL_MAX_COUNT = 10
        self.RIVAL_WANDER_STRENGTH = 0.4
        
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
        self.font_ui = pygame.font.Font(None, self.FONT_SIZE_UI)
        self.font_player = pygame.font.Font(None, self.FONT_SIZE_PLAYER)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        
        self.player = {}
        self.rivals = []
        self.nutrients = []
        self.projectiles = []
        self.particles = []
        
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player = {
            "pos": np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float),
            "vel": np.zeros(2, dtype=float),
            "mass": self.PLAYER_INITIAL_MASS,
            "level": 1,
            "radius": self._mass_to_radius(self.PLAYER_INITIAL_MASS),
            "last_move_dir": np.array([1.0, 0.0]),
            "shoot_cooldown": 0
        }
        
        # Entity lists
        self.rivals = []
        self.nutrients = []
        self.projectiles = []
        self.particles = []
        
        # Initial spawns
        for _ in range(self.RIVAL_INITIAL_COUNT):
            self._spawn_rival()
        for _ in range(self.NUTRIENT_COUNT):
            self._spawn_nutrient()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = -0.01 # Time penalty
        
        self._handle_input(action)
        self._update_player()
        self._update_rivals()
        self._update_projectiles()
        self._update_particles()
        
        self._handle_collisions()
        
        self._cleanup_entities()
        self._spawn_entities()
        
        self.steps += 1
        # The score is cumulative, so add the reward for this step
        # Note: reward_this_step can be negative
        self.score += self.reward_this_step

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False # Not using truncation based on time limit, but termination.
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        move_dir = np.zeros(2, dtype=float)
        if movement == 1: move_dir[1] = -1 # Up
        elif movement == 2: move_dir[1] = 1 # Down
        elif movement == 3: move_dir[0] = -1 # Left
        elif movement == 4: move_dir[0] = 1 # Right

        if np.linalg.norm(move_dir) > 0:
            self.player["vel"] += move_dir * self.PLAYER_ACCELERATION
            self.player["last_move_dir"] = move_dir
        
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1
            
        if space_held and self.player["shoot_cooldown"] == 0 and self.player["mass"] > self.PLAYER_INITIAL_MASS / 2:
            self._fire_projectile()

    def _fire_projectile(self):
        self.player["shoot_cooldown"] = self.SHOOT_COOLDOWN
        self.player["mass"] -= self.PROJECTILE_MASS_COST
        
        start_pos = self.player["pos"] + self.player["last_move_dir"] * (self.player["radius"] + 5)
        
        self.projectiles.append({
            "pos": start_pos,
            "vel": self.player["last_move_dir"] * self.PROJECTILE_SPEED,
            "lifespan": self.PROJECTILE_LIFESPAN,
            "max_lifespan": self.PROJECTILE_LIFESPAN
        })

    def _update_player(self):
        # Apply friction
        self.player["vel"] *= self.PLAYER_FRICTION
        
        # Clamp speed
        speed = np.linalg.norm(self.player["vel"])
        if speed > self.PLAYER_MAX_SPEED:
            self.player["vel"] = self.player["vel"] / speed * self.PLAYER_MAX_SPEED
            
        # Update position and enforce boundaries
        self.player["pos"] += self.player["vel"]
        self.player["pos"][0] = np.clip(self.player["pos"][0], self.player["radius"], self.WIDTH - self.player["radius"])
        self.player["pos"][1] = np.clip(self.player["pos"][1], self.player["radius"], self.HEIGHT - self.player["radius"])

        # Update radius and level based on mass
        self.player["radius"] = self._mass_to_radius(self.player["mass"])
        new_level = 1 + int(math.log(max(1, self.player["mass"] / self.PLAYER_INITIAL_MASS), self.LEVEL_UP_MASS_FACTOR))
        if new_level > self.player["level"]:
            self.player["level"] = new_level
            self._create_particle_burst(self.player["pos"], self.COLOR_PLAYER, 30)
        
        if self.player["level"] >= self.MAX_LEVEL:
            self.reward_this_step += 100 # Win bonus
            self.game_over = True

    def _update_rivals(self):
        for rival in self.rivals:
            if rival["stun_timer"] > 0:
                rival["stun_timer"] -= 1
                rival["vel"] *= 0.8 # Slow down when stunned
            else:
                # Simple wander behavior
                if self.np_random.random() < 0.05:
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    rival["wander_dir"] = np.array([math.cos(angle), math.sin(angle)])
                
                rival["vel"] += rival["wander_dir"] * self.RIVAL_WANDER_STRENGTH
            
            # Difficulty scaling from player level
            speed_multiplier = 1 + (self.player["level"] * 0.1)
            
            rival["vel"] *= self.PLAYER_FRICTION
            speed = np.linalg.norm(rival["vel"])
            max_rival_speed = self.PLAYER_MAX_SPEED * 0.7 * speed_multiplier
            if speed > max_rival_speed:
                rival["vel"] = rival["vel"] / speed * max_rival_speed
            
            rival["pos"] += rival["vel"]
            rival["pos"][0] = np.clip(rival["pos"][0], rival["radius"], self.WIDTH - rival["radius"])
            rival["pos"][1] = np.clip(rival["pos"][1], rival["radius"], self.HEIGHT - rival["radius"])
            rival["radius"] = self._mass_to_radius(rival["mass"])

    def _update_projectiles(self):
        for p in self.projectiles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] *= 0.95
            
    def _handle_collisions(self):
        # Player vs Nutrients
        for nutrient in self.nutrients[:]:
            if np.linalg.norm(self.player["pos"] - nutrient["pos"]) < self.player["radius"]:
                self.player["mass"] += self.NUTRIENT_MASS
                self.reward_this_step += 0.1
                self.nutrients.remove(nutrient)
                self._create_particle_burst(self.player["pos"], nutrient["color"], 5)

        # Rivals vs Nutrients
        for rival in self.rivals:
            for nutrient in self.nutrients[:]:
                if np.linalg.norm(rival["pos"] - nutrient["pos"]) < rival["radius"]:
                    rival["mass"] += self.NUTRIENT_MASS
                    self.nutrients.remove(nutrient)

        # Projectiles vs Rivals
        for p in self.projectiles[:]:
            for rival in self.rivals:
                if np.linalg.norm(p["pos"] - rival["pos"]) < rival["radius"]:
                    rival["stun_timer"] = self.PROJECTILE_STUN_DURATION
                    self.reward_this_step += 1.0
                    self.projectiles.remove(p)
                    self._create_particle_burst(rival["pos"], self.COLOR_PROJECTILE, 10)
                    break
        
        # Player vs Rivals
        for rival in self.rivals[:]:
            dist = np.linalg.norm(self.player["pos"] - rival["pos"])
            if dist < self.player["radius"] + rival["radius"]:
                absorb = False
                if rival["stun_timer"] > 0 and self.player["mass"] > rival["mass"] * 0.5:
                    absorb = True
                elif self.player["mass"] > rival["mass"] * 1.1:
                    absorb = True
                
                if absorb:
                    self.player["mass"] += rival["mass"]
                    self.reward_this_step += 5.0
                    self.rivals.remove(rival)
                    self._create_particle_burst(self.player["pos"], self._get_rival_color(rival['mass']), 20)
                elif rival["mass"] > self.player["mass"] * 1.1:
                    self.reward_this_step -= 100 # Game over penalty
                    self.game_over = True
                    self._create_particle_burst(self.player["pos"], self.COLOR_PLAYER, 50)
                else: # Bounce
                    overlap = self.player["radius"] + rival["radius"] - dist
                    direction = (self.player["pos"] - rival["pos"]) / (dist + 1e-6) # Avoid division by zero
                    self.player["pos"] += direction * overlap / 2
                    rival["pos"] -= direction * overlap / 2

    def _cleanup_entities(self):
        self.projectiles = [p for p in self.projectiles if p["lifespan"] > 0 and 0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT]
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _spawn_entities(self):
        while len(self.nutrients) < self.NUTRIENT_COUNT:
            self._spawn_nutrient()
        
        max_rivals = min(self.RIVAL_MAX_COUNT, self.RIVAL_INITIAL_COUNT + self.player["level"] // 2)
        while len(self.rivals) < max_rivals:
            self._spawn_rival()

    def _spawn_nutrient(self):
        self.nutrients.append({
            "pos": self.np_random.uniform([10, 10], [self.WIDTH - 10, self.HEIGHT - 10]),
            "color": self.COLOR_NUTRIENT_PALETTE[self.np_random.integers(len(self.COLOR_NUTRIENT_PALETTE))]
        })

    def _spawn_rival(self):
        # Spawn away from the player
        while True:
            pos = self.np_random.uniform([20, 20], [self.WIDTH - 20, self.HEIGHT - 20])
            if np.linalg.norm(pos - self.player["pos"]) > self.player["radius"] + 100:
                break
        
        base_mass = self.RIVAL_INITIAL_MASS + self.player["level"] * 20
        mass = self.np_random.uniform(base_mass * 0.8, base_mass * 1.2)
        
        self.rivals.append({
            "pos": pos,
            "vel": np.zeros(2, dtype=float),
            "mass": mass,
            "radius": self._mass_to_radius(mass),
            "stun_timer": 0,
            "wander_dir": np.array([1.0, 0.0])
        })
        
    def _mass_to_radius(self, mass):
        return math.sqrt(mass) * self.MASS_TO_RADIUS_SCALE

    def _get_rival_color(self, mass):
        intensity = min(1.0, (mass - self.RIVAL_INITIAL_MASS) / (self.PLAYER_INITIAL_MASS * 5))
        r = int(self.RIVAL_BASE_COLOR[0])
        g = int(self.RIVAL_BASE_COLOR[1] * (1 - intensity) + 200 * intensity)
        b = int(self.RIVAL_BASE_COLOR[2] * (1 - intensity) + 50 * intensity)
        return (r, g, b)

    def _create_particle_burst(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "color": color,
                "lifespan": self.np_random.integers(15, 30)
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_texture()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_texture(self):
        for x in range(0, self.WIDTH, 20):
            for y in range(0, self.HEIGHT, 20):
                pygame.gfxdraw.pixel(self.screen, x, y, self.COLOR_BG_GRID)

    def _render_game(self):
        # Draw nutrients
        for n in self.nutrients:
            pygame.draw.rect(self.screen, n["color"], (int(n["pos"][0]-2), int(n["pos"][1]-2), 4, 4))
            
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            self._draw_circle(p["pos"], p["radius"], color, blend=pygame.BLEND_RGBA_ADD)

        # Draw projectiles
        for p in self.projectiles:
            radius = (p["lifespan"] / p["max_lifespan"]) * 6
            self._draw_glowing_circle(p["pos"], radius, self.COLOR_PROJECTILE, (100,200,255,50))

        # Draw rivals
        for r in self.rivals:
            color = self._get_rival_color(r["mass"])
            glow_color = (*color, 30)
            radius = r["radius"] + math.sin(self.steps * 0.15 + r["pos"][0]) * 1.5
            self._draw_glowing_circle(r["pos"], radius, color, glow_color)
            if r["stun_timer"] > 0:
                # Stun effect
                angle = (self.steps % 10 / 10) * 2 * math.pi
                for i in range(3):
                    p_angle = angle + i * 2 * math.pi / 3
                    pos = r["pos"] + np.array([math.cos(p_angle), math.sin(p_angle)]) * (radius + 5)
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 3, (200, 200, 255))
        
        # Draw player
        if not self.game_over:
            radius = self.player["radius"] + math.sin(self.steps * 0.2) * 2
            self._draw_glowing_circle(self.player["pos"], radius, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
            # Draw aiming direction
            dir_pos = self.player["pos"] + self.player["last_move_dir"] * (radius + 5)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(dir_pos[0]), int(dir_pos[1])), 3)

    def _draw_circle(self, pos, radius, color, blend=0):
        # Custom drawing to handle alpha blending
        radius = max(1, radius)
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (radius, radius), radius)
        self.screen.blit(surf, (int(pos[0] - radius), int(pos[1] - radius)), special_flags=blend)

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        int_pos = (int(pos[0]), int(pos[1]))
        int_radius = max(1, int(radius))

        # Glow effect
        glow_radius = int(int_radius * 2.0)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (int_pos[0] - glow_radius, int_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Main circle with anti-aliasing
        pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], int_radius, color)
        pygame.gfxdraw.aacircle(self.screen, int_pos[0], int_pos[1], int_radius, color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        # Player level
        if not self.game_over:
            level_text = self.font_player.render(f"LVL {self.player['level']}", True, self.COLOR_TEXT)
            text_rect = level_text.get_rect(center=(int(self.player["pos"][0]), int(self.player["pos"][1])))
            self.screen.blit(level_text, text_rect)
            
        if self.game_over:
            end_text_str = "YOU WON!" if self.player['level'] >= self.MAX_LEVEL else "GAME OVER"
            end_text = pygame.font.Font(None, 60).render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.player.get("level", 1),
            "mass": self.player.get("mass", 0)
        }

    def close(self):
        pygame.quit()
        
    def render(self):
        return self._get_observation()

# Example usage for testing
if __name__ == '__main__':
    # This block is for local testing and will not be executed by the grader.
    # It allows you to play the game with keyboard controls.
    
    # Un-comment the line below to run with a display window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("GameEnv")
    
    running = True
    while running:
        # Human player control
        action = [0, 0, 0] # [movement, space, shift]
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                if event.key == pygame.K_q:
                    running = False

        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(env.metadata['render_fps'])
        
    env.close()