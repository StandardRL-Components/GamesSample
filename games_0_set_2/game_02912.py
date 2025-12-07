
# Generated: 2025-08-27T21:48:25.019543
# Source Brief: brief_02912.md
# Brief Index: 2912

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to allocate resources to a defense (N,S,W,E). Space to allocate to all. Shift to save resources."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of zombies. Allocate resources to bolster your defenses and survive until dawn."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.MAX_SURVIVORS = 10
        self.INITIAL_RESOURCES = 10
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.SysFont("Consolas", 20)
        self.title_font = pygame.font.SysFont("Consolas", 40, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 30, 25)
        self.COLOR_BASE = (40, 60, 50)
        self.COLOR_BASE_INNER = (50, 80, 65)
        self.COLOR_DEFENSE = (60, 120, 200)
        self.COLOR_DEFENSE_LOW = (100, 30, 30)
        self.COLOR_SURVIVOR = (100, 220, 100)
        self.COLOR_ZOMBIE_PALETTE = [(200, 50, 50), (220, 80, 80), (180, 100, 100), (240, 60, 60), (210, 90, 90)]
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_RESOURCE = (255, 200, 0)
        self.COLOR_LASER = (255, 100, 100)
        
        # Game element definitions
        self.BASE_RECT = pygame.Rect(120, 80, self.WIDTH - 240, self.HEIGHT - 160)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.survivors = 0
        self.resources = 0
        self.zombies = []
        self.defenses = {}
        self.particles = []
        self.zombie_spawn_rate = 0
        self.zombie_base_health = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.survivors = self.MAX_SURVIVORS
        self.resources = self.INITIAL_RESOURCES
        
        self.zombies = []
        self.particles = []
        
        self.zombie_spawn_rate = 1.0
        self.zombie_base_health = 1
        
        defense_radius = 20
        self.defenses = {
            "N": {"pos": (self.BASE_RECT.centerx, self.BASE_RECT.top), "health": 100, "max_health": 100, "strength": 1, "range": 150, "cooldown": 0, "max_cooldown": 30, "radius": defense_radius},
            "S": {"pos": (self.BASE_RECT.centerx, self.BASE_RECT.bottom), "health": 100, "max_health": 100, "strength": 1, "range": 150, "cooldown": 0, "max_cooldown": 30, "radius": defense_radius},
            "W": {"pos": (self.BASE_RECT.left, self.BASE_RECT.centery), "health": 100, "max_health": 100, "strength": 1, "range": 150, "cooldown": 0, "max_cooldown": 30, "radius": defense_radius},
            "E": {"pos": (self.BASE_RECT.right, self.BASE_RECT.centery), "health": 100, "max_health": 100, "strength": 1, "range": 150, "cooldown": 0, "max_cooldown": 30, "radius": defense_radius},
        }
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # 1. Handle player action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        resource_gain = 1

        if shift_held:
            self.resources += resource_gain
            # SFX: resource_banked
            self._create_text_particle(f"+{resource_gain} Res", (self.WIDTH - 150, 40), self.COLOR_RESOURCE)
        else:
            action_taken = False
            if space_held and self.resources >= 4:
                self.resources -= 4
                for d in self.defenses.values():
                    d["strength"] += 1
                action_taken = True
                # SFX: upgrade_all
                self._create_text_particle("All Defenses++", (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_DEFENSE)
            elif movement > 0 and self.resources >= 1:
                target_defense = None
                if movement == 1: target_defense = "N" # Up
                elif movement == 2: target_defense = "S" # Down
                elif movement == 3: target_defense = "W" # Left
                elif movement == 4: target_defense = "E" # Right
                
                if target_defense:
                    self.resources -= 1
                    self.defenses[target_defense]["strength"] += 1
                    action_taken = True
                    # SFX: upgrade_single
                    pos = (self.defenses[target_defense]['pos'][0], self.defenses[target_defense]['pos'][1] - 20)
                    self._create_text_particle("Str+1", pos, self.COLOR_DEFENSE)

        # 2. Update game state
        self.steps += 1
        self.zombie_spawn_rate = 1.0 + self.steps / 200.0
        self.zombie_base_health = 1 + self.steps // 400

        # 3. Spawn zombies
        num_to_spawn = self.np_random.poisson(self.zombie_spawn_rate / 10) # Slower spawn rate for turn-based
        for _ in range(num_to_spawn):
            self._spawn_zombie()

        # 4. Defense logic
        for key, defense in self.defenses.items():
            if defense["health"] <= 0: continue
            defense["cooldown"] = max(0, defense["cooldown"] - 1)
            if defense["cooldown"] == 0:
                target = self._find_closest_zombie(defense["pos"], defense["range"])
                if target:
                    target["health"] -= defense["strength"]
                    defense["cooldown"] = defense["max_cooldown"]
                    # SFX: defense_fire
                    self._create_particle(defense["pos"], target["pos"], self.COLOR_LASER, 2, 10)
                    if target["health"] <= 0:
                        # SFX: zombie_die
                        reward += 1.0
                        self._create_splatter_effect(target["pos"], self.COLOR_ZOMBIE_PALETTE[target["type"]], 15)

        # 5. Zombie logic
        for z in self.zombies[:]:
            if z["health"] <= 0:
                self.zombies.remove(z)
                continue

            attacking = False
            for key, defense in self.defenses.items():
                if defense["health"] > 0:
                    dist = math.hypot(z["pos"][0] - defense["pos"][0], z["pos"][1] - defense["pos"][1])
                    if dist < defense["radius"] + z["radius"]:
                        attacking = True
                        defense["health"] -= 1
                        # SFX: defense_hit
                        self._create_splatter_effect(defense["pos"], (100,100,100), 3, speed=1)
                        if defense["health"] <= 0:
                            # SFX: defense_destroyed
                            self._create_splatter_effect(defense["pos"], (255,255,255), 50, speed=3)
                        break
            
            if not attacking:
                angle = math.atan2(self.BASE_RECT.centery - z["pos"][1], self.BASE_RECT.centerx - z["pos"][0])
                z["pos"] = (z["pos"][0] + math.cos(angle) * z["speed"], z["pos"][1] + math.sin(angle) * z["speed"])
            
            if self.BASE_RECT.collidepoint(z["pos"]):
                self.survivors -= 1
                self.zombies.remove(z)
                # SFX: survivor_lost
                self._create_splatter_effect(z["pos"], self.COLOR_SURVIVOR, 30, speed=2)
                if self.survivors <= 0:
                    self.game_over = True

        # 6. Particle logic
        for p in self.particles[:]:
            p["pos"] = (p["pos"][0] + p["vel"][0], p["pos"][1] + p["vel"][1])
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

        # 7. Calculate rewards and check termination
        reward += self.survivors * 0.1
        self.score += reward
        
        terminated = self.survivors <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.survivors > 0:
                reward += 100 # Victory bonus
            else:
                reward -= 100 # Defeat penalty
            self.score += 100 if self.survivors > 0 else -100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "survivors": self.survivors, "resources": self.resources}

    def _render_game(self):
        # Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.BASE_RECT, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE_INNER, self.BASE_RECT.inflate(-20, -20), border_radius=5)

        # Defenses
        for key, d in self.defenses.items():
            health_ratio = max(0, d["health"] / d["max_health"])
            color = self.COLOR_DEFENSE if health_ratio > 0.2 else self.COLOR_DEFENSE_LOW
            pygame.draw.circle(self.screen, color, (int(d["pos"][0]), int(d["pos"][1])), d["radius"])
            pygame.draw.circle(self.screen, self.COLOR_TEXT, (int(d["pos"][0]), int(d["pos"][1])), d["radius"], 1)
            strength_text = self.game_font.render(str(d["strength"]), True, self.COLOR_TEXT)
            self.screen.blit(strength_text, strength_text.get_rect(center=d["pos"]))
            # Health bar
            if health_ratio < 1.0:
                bar_w, bar_h = 30, 4
                bar_x = d["pos"][0] - bar_w / 2
                bar_y = d["pos"][1] + d["radius"] + 4
                pygame.draw.rect(self.screen, self.COLOR_DEFENSE_LOW, (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_DEFENSE, (bar_x, bar_y, bar_w * health_ratio, bar_h))

        # Survivors
        for i in range(self.survivors):
            angle = (i / self.MAX_SURVIVORS) * 2 * math.pi + (self.steps * 0.01)
            dist = 20 + (i % 3) * 10
            pos = (self.BASE_RECT.centerx + math.cos(angle) * dist, self.BASE_RECT.centery + math.sin(angle) * dist)
            pygame.draw.circle(self.screen, self.COLOR_SURVIVOR, (int(pos[0]), int(pos[1])), 3)
            
        # Zombies
        for z in self.zombies:
            wobble_x = math.sin(self.steps * 0.2 + z["id"]) * 2
            wobble_y = math.cos(self.steps * 0.2 + z["id"]) * 2
            pos = (int(z["pos"][0] + wobble_x), int(z["pos"][1] + wobble_y))
            color = self.COLOR_ZOMBIE_PALETTE[z["type"]]
            pygame.draw.circle(self.screen, color, pos, z["radius"])

        # Particles
        for p in self.particles:
            if p.get("is_text", False):
                alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
                p["surface"].set_alpha(alpha)
                self.screen.blit(p["surface"], p["pos"])
            else:
                alpha = 255 * (p["lifetime"] / p["max_lifetime"])
                color = (*p["color"], alpha)
                if p["lifetime"] > 0:
                    try:
                        pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), int(p["radius"]))
                    except TypeError: # Color might not support alpha
                        pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(p["radius"]))


    def _render_ui(self):
        # Top bar UI
        ui_texts = [
            f"TIME: {self.steps}/{self.MAX_STEPS}",
            f"SURVIVORS: {self.survivors}",
            f"RESOURCES: {self.resources}",
            f"SCORE: {self.score:.1f}"
        ]
        for i, text in enumerate(ui_texts):
            rendered_text = self.game_font.render(text, True, self.COLOR_TEXT)
            self.screen.blit(rendered_text, (10 + i * 160, 10))

        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.survivors > 0:
                msg = "DAWN HAS COME. YOU SURVIVED."
                color = self.COLOR_SURVIVOR
            else:
                msg = "THE HORDE HAS OVERRUN THE BASE."
                color = self.COLOR_ZOMBIE_PALETTE[0]
                
            title_surf = self.title_font.render(msg, True, color)
            self.screen.blit(title_surf, title_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2)))

    def _spawn_zombie(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = (self.np_random.uniform(0, self.WIDTH), -20)
        elif edge == 1: # Bottom
            pos = (self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
        elif edge == 2: # Left
            pos = (-20, self.np_random.uniform(0, self.HEIGHT))
        else: # Right
            pos = (self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))
            
        z_type = self.np_random.integers(0, 5)
        speeds = [0.5, 0.7, 0.9, 1.1, 1.3]
        healths = [1, 2, 3, 4, 5]
        radii = [5, 6, 7, 8, 9]

        self.zombies.append({
            "id": self.np_random.uniform(0, 1000),
            "pos": pos,
            "health": self.zombie_base_health * healths[z_type],
            "speed": speeds[z_type],
            "type": z_type,
            "radius": radii[z_type]
        })

    def _find_closest_zombie(self, pos, range_):
        closest_zombie = None
        min_dist_sq = range_ ** 2
        for z in self.zombies:
            dist_sq = (z["pos"][0] - pos[0])**2 + (z["pos"][1] - pos[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_zombie = z
        return closest_zombie

    def _create_particle(self, start_pos, end_pos, color, radius, lifetime):
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        dist = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        speed = dist / lifetime
        self.particles.append({
            "pos": list(start_pos),
            "vel": (math.cos(angle) * speed, math.sin(angle) * speed),
            "color": color,
            "radius": radius,
            "lifetime": lifetime,
            "max_lifetime": lifetime
        })

    def _create_splatter_effect(self, pos, color, count, speed=1.5):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 1.0) * speed
            lifetime = self.np_random.integers(10, 20)
            self.particles.append({
                "pos": list(pos),
                "vel": (math.cos(angle) * vel_mag, math.sin(angle) * vel_mag),
                "color": color,
                "radius": self.np_random.uniform(1, 3),
                "lifetime": lifetime,
                "max_lifetime": lifetime
            })
    
    def _create_text_particle(self, text, pos, color):
        surface = self.game_font.render(text, True, color)
        lifetime = 30
        self.particles.append({
            "is_text": True,
            "surface": surface,
            "pos": [pos[0] - surface.get_width()//2, pos[1] - surface.get_height()//2],
            "vel": [0, -0.5],
            "lifetime": lifetime,
            "max_lifetime": lifetime
        })

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up the display window
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)

    total_reward = 0
    
    # Game loop
    while not done:
        movement, space, shift = 0, 0, 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # The game only advances on an action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # The brief says auto_advance=False, so we don't use a fixed FPS,
        # but we'll cap the manual play loop to avoid it running too fast.
        clock.tick(30)
        
    print(f"Game Over! Final Score: {info['score']:.2f}")
    
    # Keep the final screen visible for a moment
    pygame.time.wait(3000)
    env.close()