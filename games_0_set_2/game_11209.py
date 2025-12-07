import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:40:35.026133
# Source Brief: brief_01209.md
# Brief Index: 1209
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your base from waves of enemies by launching elemental spells from a catapult. "
        "Combine elements for powerful effects and survive as long as you can."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to adjust angle and ←→ to adjust power. "
        "Press space to fire. Use shift to cycle through unlocked spells."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_UI_BG = (30, 35, 50, 180)
    COLOR_TEXT = (220, 220, 240)
    COLOR_BASE = (87, 58, 46)
    COLOR_CATAPULT = (120, 100, 90)
    COLOR_AIMER = (255, 255, 255, 100)

    ELEMENT_COLORS = {
        "fire": (255, 100, 50),
        "ice": (100, 200, 255),
        "lightning": (255, 255, 150),
        "nature": (100, 255, 100),
    }

    # Game Parameters
    MAX_STEPS = 5000
    MAX_WAVES = 50
    BASE_MAX_HEALTH = 1000
    MANA_MAX = 100
    MANA_REGEN_RATE = 0.2
    PROJECTILE_COST = 20
    CATAPULT_POS = (WIDTH // 2, HEIGHT - 30)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.projectiles = []
        self.enemies = []
        self.particles = []
        self.explosions = []
        self.floating_texts = []
        
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        self.wave_transition_timer = self.FPS * 3 # 3 second delay before first wave

        self.base_health = self.BASE_MAX_HEALTH
        self.mana = self.MANA_MAX
        
        self.launch_angle = -math.pi / 2
        self.launch_power = 50

        self.spells = ["fire"]
        self.current_spell_idx = 0

        self.projectiles.clear()
        self.enemies.clear()
        self.particles.clear()
        self.explosions.clear()
        self.floating_texts.clear()
        
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_raw, shift_raw = action
        space_held = space_raw == 1
        shift_held = shift_raw == 1
        
        self.steps += 1
        step_reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self.mana = min(self.MANA_MAX, self.mana + self.MANA_REGEN_RATE)
        
        reward_from_updates = self._update_game_objects()
        step_reward += reward_from_updates

        # --- Wave Management ---
        reward_from_wave = self._update_wave_system()
        step_reward += reward_from_wave

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.base_health <= 0:
            step_reward = -100.0
            terminated = True
            self.game_over = True
            self._add_floating_text("BASE DESTROYED", (self.WIDTH/2, self.HEIGHT/2), (255, 50, 50), 3.0, size=2)
        elif self.wave > self.MAX_WAVES:
            step_reward = 100.0
            terminated = True
            self.game_over = True
            self._add_floating_text("VICTORY!", (self.WIDTH/2, self.HEIGHT/2), (50, 255, 50), 3.0, size=2)
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Internal Logic Methods ---

    def _handle_input(self, movement, space_held, shift_held):
        # Angle adjustment
        if movement == 1: self.launch_angle -= 0.05
        if movement == 2: self.launch_angle += 0.05
        self.launch_angle = max(-math.pi, min(0, self.launch_angle))

        # Power adjustment
        if movement == 3: self.launch_power -= 2
        if movement == 4: self.launch_power += 2
        self.launch_power = max(10, min(100, self.launch_power))

        # Spell cycling
        if shift_held and not self.last_shift_held:
            self.current_spell_idx = (self.current_spell_idx + 1) % len(self.spells)
            # sfx: spell_cycle.wav

        # Launch projectile
        if not space_held and self.last_space_held:
            self._fire_projectile()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
    def _fire_projectile(self):
        spell_type = self.spells[self.current_spell_idx]
        if self.mana >= self.PROJECTILE_COST:
            self.mana -= self.PROJECTILE_COST
            power_scaled = self.launch_power * 0.15
            vel = (power_scaled * math.cos(self.launch_angle), power_scaled * math.sin(self.launch_angle))
            
            self.projectiles.append({
                "pos": list(self.CATAPULT_POS),
                "vel": list(vel),
                "size": 12,
                "max_size": 12,
                "element": spell_type,
                "life": 200
            })
            # sfx: catapult_launch.wav

    def _update_game_objects(self):
        reward = 0
        
        # Update projectiles
        for p in self.projectiles[:]:
            p["vel"][1] += 0.1  # Gravity
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["size"] = max(2, p["max_size"] * (p["life"] / 200))
            p["life"] -= 1

            if p["life"] <= 0 or p["pos"][1] > self.HEIGHT - 20:
                self.projectiles.remove(p)
                self._create_explosion(p["pos"], p["element"], p["max_size"])
                # Handle nature projectile hitting ground near base
                if p["element"] == 'nature' and p["pos"][1] > self.HEIGHT - 40:
                    heal_amount = 50
                    self.base_health = min(self.BASE_MAX_HEALTH, self.base_health + heal_amount)
                    self._add_floating_text(f"+{heal_amount}", (p["pos"][0], self.HEIGHT - 40), (100, 255, 100), 1.0)
                continue

            for e in self.enemies[:]:
                dist = math.hypot(p["pos"][0] - e["pos"][0], p["pos"][1] - e["pos"][1])
                if dist < p["size"] + e["size"]:
                    self.projectiles.remove(p)
                    self._create_explosion(p["pos"], p["element"], p["max_size"])
                    reward += 0.1 # Hit reward
                    break
        
        # Update enemies
        for e in self.enemies[:]:
            speed_multiplier = 0.5 if e.get("status") == "frozen" else 1.0
            e["pos"][1] += e["speed"] * speed_multiplier
            
            if e.get("status_timer", 0) > 0:
                e["status_timer"] -= 1
                if e["status_timer"] <= 0:
                    e["status"] = None
            
            if e["pos"][1] > self.HEIGHT - 20:
                self.base_health -= e["damage"]
                self._create_damage_particles(e["pos"])
                self.enemies.remove(e)
                # sfx: base_hit.wav
                continue
        
        # Update explosions and apply effects
        for explosion in self.explosions[:]:
            explosion["radius"] += explosion["expand_rate"]
            explosion["alpha"] -= 10
            if explosion["alpha"] <= 0:
                self.explosions.remove(explosion)
                continue

            if not explosion.get("has_damaged", False):
                explosion_reward = self._apply_explosion_effects(explosion)
                reward += explosion_reward
                explosion["has_damaged"] = True

        # Update particles
        for part in self.particles[:]:
            part["pos"][0] += part["vel"][0]
            part["pos"][1] += part["vel"][1]
            part["life"] -= 1
            if part["life"] <= 0:
                self.particles.remove(part)

        # Update floating texts
        for ft in self.floating_texts[:]:
            ft["pos"][1] -= 0.5
            ft["life"] -= 1 / self.FPS
            if ft["life"] <= 0:
                self.floating_texts.remove(ft)

        return reward

    def _apply_explosion_effects(self, explosion):
        reward = 0
        is_chain_reaction = False

        for e in self.enemies[:]:
            dist = math.hypot(explosion["pos"][0] - e["pos"][0], explosion["pos"][1] - e["pos"][1])
            if dist < explosion["radius"] + e["size"]:
                
                # Elemental interaction check
                if explosion["element"] == 'fire' and e.get("status") == 'frozen':
                    is_chain_reaction = True
                    e["health"] -= explosion["damage"] * 2 # Shatter bonus damage
                    e["status"] = None
                    e["status_timer"] = 0
                else:
                    e["health"] -= explosion["damage"]

                # Apply new status
                if explosion["element"] == 'ice' and e.get("status") != 'frozen':
                    e["status"] = "frozen"
                    e["status_timer"] = self.FPS * 3 # 3 seconds
                
                if e["health"] <= 0:
                    reward += 1.0 # Kill reward
                    self._add_floating_text("+1", e["pos"], (255, 255, 100), 0.7)
                    self.enemies.remove(e)
                    # sfx: enemy_die.wav
        
        # Lightning Chain
        if explosion["element"] == "lightning":
            targets = sorted([e for e in self.enemies if math.hypot(explosion["pos"][0] - e["pos"][0], explosion["pos"][1] - e["pos"][1]) < explosion["radius"] * 2], 
                             key=lambda en: math.hypot(explosion["pos"][0] - en["pos"][0], explosion["pos"][1] - en["pos"][1]))
            
            last_pos = explosion["pos"]
            for i, target in enumerate(targets[:4]): # Chain up to 4 enemies
                if i > 0: is_chain_reaction = True
                
                chain_damage = explosion["damage"] / (i + 1)
                target["health"] -= chain_damage
                self._create_lightning_arc(last_pos, target["pos"])
                last_pos = target["pos"]
                
                if target["health"] <= 0 and target in self.enemies:
                    reward += 1.0
                    self._add_floating_text("+1", target["pos"], (255, 255, 100), 0.7)
                    self.enemies.remove(target)
        
        if is_chain_reaction:
            reward += 0.5 # Chain reaction reward
        
        return reward

    def _update_wave_system(self):
        if not self.enemies and self.wave <= self.MAX_WAVES and not self.game_over:
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
            else:
                self.wave += 1
                if self.wave > self.MAX_WAVES: return 0
                
                self._spawn_wave()
                self._unlock_spells()
                self.wave_transition_timer = self.FPS * 5 # 5 seconds between waves
                self._add_floating_text(f"WAVE {self.wave}", (self.WIDTH/2, self.HEIGHT/2), (200, 200, 255), 2.0, size=2)
                # sfx: wave_start.wav
                return 5.0 # Wave completion reward
        return 0

    def _spawn_wave(self):
        num_enemies = 5 + self.wave
        base_health = 10 + (self.wave // 10)
        base_speed = 0.5 + ((self.wave // 5) * 0.02)
        
        for _ in range(num_enemies):
            self.enemies.append({
                "pos": [random.uniform(50, self.WIDTH - 50), random.uniform(-100, -20)],
                "health": base_health,
                "max_health": base_health,
                "speed": random.uniform(base_speed * 0.8, base_speed * 1.2),
                "size": 10,
                "damage": 50,
                "status": None,
                "status_timer": 0
            })

    def _unlock_spells(self):
        if self.wave == 15 and "ice" not in self.spells:
            self.spells.append("ice")
            self._add_floating_text("ICE SPELL UNLOCKED", (self.WIDTH/2, 100), self.ELEMENT_COLORS["ice"], 3)
        if self.wave == 30 and "lightning" not in self.spells:
            self.spells.append("lightning")
            self._add_floating_text("LIGHTNING SPELL UNLOCKED", (self.WIDTH/2, 100), self.ELEMENT_COLORS["lightning"], 3)
        if self.wave == 45 and "nature" not in self.spells:
            self.spells.append("nature")
            self._add_floating_text("NATURE SPELL UNLOCKED", (self.WIDTH/2, 100), self.ELEMENT_COLORS["nature"], 3)

    # --- Particle and Effect Creation ---

    def _create_explosion(self, pos, element, size):
        self.explosions.append({
            "pos": pos,
            "element": element,
            "radius": size,
            "max_radius": size * 3 + (20 if element == "fire" else 10),
            "expand_rate": 2 + (2 if element == "fire" else 0),
            "alpha": 255,
            "color": self.ELEMENT_COLORS[element],
            "damage": 2 + size / 4
        })
        # sfx: explosion.wav
        for _ in range(int(size * 3)):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(15, 30),
                "color": self.ELEMENT_COLORS[element],
                "size": random.uniform(1, 3)
            })

    def _create_damage_particles(self, pos):
        for _ in range(20):
            angle = random.uniform(math.pi, 2*math.pi) # Upwards
            speed = random.uniform(1, 3)
            self.particles.append({
                "pos": [pos[0], self.HEIGHT - 20],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(20, 40),
                "color": self.COLOR_BASE,
                "size": random.uniform(2, 4)
            })
            
    def _create_lightning_arc(self, start_pos, end_pos):
        # A simple particle trail to simulate a lightning arc
        dist = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        for i in range(0, int(dist), 4):
            t = i / dist
            pos = [start_pos[0] * (1-t) + end_pos[0] * t, start_pos[1] * (1-t) + end_pos[1] * t]
            pos[0] += random.uniform(-3, 3)
            pos[1] += random.uniform(-3, 3)
            self.particles.append({
                "pos": pos,
                "vel": [0,0],
                "life": 5,
                "color": self.ELEMENT_COLORS["lightning"],
                "size": random.uniform(1, 2)
            })

    def _add_floating_text(self, text, pos, color, life, size=1):
        font = self.font_large if size == 2 else self.font_small
        self.floating_texts.append({"text": text, "pos": list(pos), "color": color, "life": life, "font": font})
        
    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (0, self.HEIGHT - 20, self.WIDTH, 20))
        pygame.draw.rect(self.screen, (0,0,0), (0, self.HEIGHT - 20, self.WIDTH, 20), 2)

        # Catapult and Aimer
        self._render_catapult()
        
        # Projectiles
        for p in self.projectiles:
            self._render_glowing_circle(p["pos"], p["size"], self.ELEMENT_COLORS[p["element"]])

        # Enemies
        for e in self.enemies:
            health_ratio = e["health"] / e["max_health"]
            color = (120 + 100 * (1 - health_ratio), 120 - 100 * (1-health_ratio), 120 - 100 * (1-health_ratio))
            pygame.draw.rect(self.screen, color, (e["pos"][0] - e["size"], e["pos"][1] - e["size"], e["size"] * 2, e["size"] * 2))
            if e.get("status") == "frozen":
                pygame.gfxdraw.box(self.screen, (int(e["pos"][0] - e["size"]), int(e["pos"][1] - e["size"]), int(e["size"] * 2), int(e["size"] * 2)), (150, 220, 255, 100))
        
        # Explosions
        for ex in self.explosions:
            if ex["alpha"] > 0:
                self._render_glowing_circle(ex["pos"], ex["radius"], ex["color"], ex["alpha"])
        
        # Particles
        for part in self.particles:
            alpha = max(0, 255 * (part["life"] / 30))
            color = (*part["color"], alpha)
            s = pygame.Surface((part["size"]*2, part["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (part["size"], part["size"]), part["size"])
            self.screen.blit(s, (int(part["pos"][0] - part["size"]), int(part["pos"][1] - part["size"])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_catapult(self):
        # Base
        pygame.draw.circle(self.screen, self.COLOR_CATAPULT, (int(self.CATAPULT_POS[0]), int(self.CATAPULT_POS[1])), 15)
        
        # Aimer trajectory
        pos = list(self.CATAPULT_POS)
        vel = [self.launch_power * 0.15 * math.cos(self.launch_angle), self.launch_power * 0.15 * math.sin(self.launch_angle)]
        for i in range(20):
            vel[1] += 0.1
            pos[0] += vel[0]
            pos[1] += vel[1]
            if i % 2 == 0:
                pygame.draw.circle(self.screen, self.COLOR_AIMER, (int(pos[0]), int(pos[1])), 2)

    def _render_ui(self):
        # Health Bar
        bar_width = 200
        health_ratio = self.base_health / self.BASE_MAX_HEALTH
        pygame.draw.rect(self.screen, (80, 0, 0), (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, (200, 0, 0), (10, 10, bar_width * health_ratio, 20))
        health_text = self.font_small.render(f"BASE: {int(self.base_health)}/{self.BASE_MAX_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Mana Bar
        mana_ratio = self.mana / self.MANA_MAX
        pygame.draw.rect(self.screen, (0, 0, 80), (10, 35, bar_width, 20))
        pygame.draw.rect(self.screen, (0, 100, 200), (10, 35, bar_width * mana_ratio, 20))
        mana_text = self.font_small.render(f"MANA: {int(self.mana)}/{self.MANA_MAX}", True, self.COLOR_TEXT)
        self.screen.blit(mana_text, (15, 37))

        # Wave and Score
        wave_text = self.font_large.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Spell indicator
        spell_name = self.spells[self.current_spell_idx].upper()
        spell_color = self.ELEMENT_COLORS[self.spells[self.current_spell_idx]]
        spell_text = self.font_small.render(f"SPELL: {spell_name}", True, self.COLOR_TEXT)
        self.screen.blit(spell_text, (self.WIDTH - spell_text.get_width() - 10, 40))
        pygame.draw.rect(self.screen, spell_color, (self.WIDTH - 15, 60, 10, 10))

        # Floating text
        for ft in self.floating_texts:
            alpha = max(0, 255 * ft["life"])
            text_surf = ft["font"].render(ft["text"], True, ft["color"])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (ft["pos"][0] - text_surf.get_width() / 2, ft["pos"][1] - text_surf.get_height() / 2))

    def _render_glowing_circle(self, pos, radius, color, alpha=255):
        # Draw multiple transparent circles for a glow effect
        center_x, center_y = int(pos[0]), int(pos[1])
        max_radius = int(radius)
        if max_radius <= 0: return

        # Glow
        glow_surf = pygame.Surface((max_radius * 4, max_radius * 4), pygame.SRCALPHA)
        for i in range(max_radius, 0, -2):
            current_alpha = int(alpha * (1 - i / max_radius)**2 * 0.5)
            pygame.gfxdraw.filled_circle(glow_surf, max_radius*2, max_radius*2, i + max_radius, (*color, current_alpha))
        self.screen.blit(glow_surf, (center_x - max_radius*2, center_y - max_radius*2), special_flags=pygame.BLEND_RGBA_ADD)

        # Core circle
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, max_radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, max_radius, color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "base_health": self.base_health}

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("Validating implementation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # Arrow Keys: Move aimer
    # Spacebar: Fire (on release)
    # Left Shift: Cycle spell
    
    action = [0, 0, 0] # [movement, space, shift]
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Spell Slingers")
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Get Key Presses for Continuous Actions ---
        keys = pygame.key.get_pressed()
        
        # Movement
        movement_action = 0 # none
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        # Space and Shift
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # --- Rendering ---
        # The observation is already the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()