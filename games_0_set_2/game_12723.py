import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:20:40.011847
# Source Brief: brief_02723.md
# Brief Index: 2723
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend the sentient Garden Core from waves of relentless pests. "
        "Use your energy to fire projectiles and unlock powerful new abilities."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim. Press space to fire your "
        "currently selected weapon. Press shift to cycle through unlocked abilities."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (12, 20, 33)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (150, 255, 200)
    COLOR_CORE = (255, 180, 50)
    COLOR_CORE_GLOW = (255, 220, 150)
    COLOR_PEST = (255, 80, 80)
    COLOR_PEST_GLOW = (255, 150, 150)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR = (40, 60, 80)
    COLOR_ENERGY = (100, 150, 255)
    COLOR_HEALTH = (255, 100, 100)
    
    # Player Physics
    PLAYER_RECOIL = 15.0
    PLAYER_DRAG = 0.92
    PLAYER_RADIUS = 12

    # Core
    CORE_RADIUS = 40
    CORE_MAX_HEALTH = 100.0

    # Pest
    PEST_RADIUS = 8
    PEST_SPEED = 1.2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # Ability definitions
        self.abilities = {
            "NORMAL": {"cost": 5, "cooldown": 5, "damage": 20, "speed": 10, "type": "normal"},
            "PIERCING": {"cost": 15, "cooldown": 15, "damage": 15, "speed": 12, "type": "piercing"},
            "BURST": {"cost": 25, "cooldown": 25, "damage": 10, "speed": 8, "type": "burst"},
        }

        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_aim_dir = pygame.Vector2(1, 0)
        self.player_energy = 0
        self.player_max_energy = 100
        self.player_energy_regen = 0.5
        self.player_fire_cooldown = 0
        self.player_switch_cooldown = 0
        self.player_unlocked_abilities = []
        self.player_current_ability_idx = 0
        self.player_recoil_anim = 0

        self.core_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.core_health = 0
        
        self.pests = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.pest_spawn_rate = 0.02
        self.pest_base_health = 20

        self.shift_was_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        
        # Player State
        self.player_pos = self.core_pos.copy()
        self.player_vel = pygame.Vector2(0, 0)
        self.player_aim_dir = pygame.Vector2(1, 0)
        self.player_energy = self.player_max_energy
        self.player_fire_cooldown = 0
        self.player_switch_cooldown = 0
        self.player_unlocked_abilities = ["NORMAL"]
        self.player_current_ability_idx = 0
        self.shift_was_held = False
        self.player_recoil_anim = 0

        # Game State
        self.core_health = self.CORE_MAX_HEALTH
        self.pests.clear()
        self.projectiles.clear()
        self.particles.clear()

        # Difficulty State
        self.pest_spawn_rate = 0.02
        self.pest_base_health = 20

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0

        # --- 1. HANDLE INPUT & COOLDOWNS ---
        self._handle_input(action)
        self.player_fire_cooldown = max(0, self.player_fire_cooldown - 1)
        self.player_switch_cooldown = max(0, self.player_switch_cooldown - 1)
        self.player_recoil_anim = max(0, self.player_recoil_anim - 0.1)

        # --- 2. UPDATE GAME LOGIC ---
        self._update_player()
        self._update_projectiles()
        self._update_pests()
        self._update_particles()
        
        # --- 3. HANDLE COLLISIONS ---
        reward += self._handle_collisions()

        # --- 4. SPAWN & PROGRESSION ---
        self._spawn_pests()
        self._update_progression()
        
        # --- 5. CALCULATE TERMINATION & FINAL REWARDS ---
        self.steps += 1
        terminated = self.core_health <= 0 or self.steps >= self.MAX_STEPS
        truncated = False
        
        if terminated:
            if self.steps >= self.MAX_STEPS and self.core_health > 0:
                reward += 100  # Victory reward
            else:
                reward -= 100  # Defeat penalty

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Aiming
        if movement == 1: self.player_aim_dir = pygame.Vector2(0, -1)
        elif movement == 2: self.player_aim_dir = pygame.Vector2(0, 1)
        elif movement == 3: self.player_aim_dir = pygame.Vector2(-1, 0)
        elif movement == 4: self.player_aim_dir = pygame.Vector2(1, 0)
        elif movement == 0:
            if (self.core_pos - self.player_pos).length() > 1:
                self.player_aim_dir = (self.core_pos - self.player_pos).normalize()

        # Firing
        ability_name = self.player_unlocked_abilities[self.player_current_ability_idx]
        ability = self.abilities[ability_name]
        if space_held and self.player_fire_cooldown == 0 and self.player_energy >= ability["cost"]:
            # SFX: Shoot
            self.player_energy -= ability["cost"]
            self.player_fire_cooldown = ability["cooldown"]
            self.player_vel -= self.player_aim_dir * self.PLAYER_RECOIL
            self.player_recoil_anim = 1.0

            if ability["type"] == "burst":
                for i in range(-1, 2):
                    angle = math.radians(i * 15)
                    rotated_dir = self.player_aim_dir.rotate_rad(angle)
                    self._create_projectile(self.player_pos.copy(), rotated_dir, ability)
            else:
                self._create_projectile(self.player_pos.copy(), self.player_aim_dir.copy(), ability)

        # Ability Switching
        if shift_held and not self.shift_was_held and self.player_switch_cooldown == 0:
            # SFX: Switch ability
            self.player_current_ability_idx = (self.player_current_ability_idx + 1) % len(self.player_unlocked_abilities)
            self.player_switch_cooldown = 10
        self.shift_was_held = shift_held

    def _update_player(self):
        self.player_vel *= self.PLAYER_DRAG
        if self.player_vel.length() < 0.1: self.player_vel = pygame.Vector2(0, 0)
        self.player_pos += self.player_vel
        self.player_energy = min(self.player_max_energy, self.player_energy + self.player_energy_regen)

        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj["pos"] += proj["vel"]
            if not self.screen.get_rect().collidepoint(proj["pos"]):
                self.projectiles.remove(proj)
            else:
                self._create_particle(proj["pos"], self.np_random.integers(1, 3), proj["color"], speed_scale=0.1, life_scale=0.5)

    def _update_pests(self):
        for pest in self.pests:
            if (self.core_pos - pest["pos"]).length() > 0:
                direction = (self.core_pos - pest["pos"]).normalize()
                pest["pos"] += direction * self.PEST_SPEED
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] = max(0, p["size"] - 0.2)
            if p["life"] <= 0 or p["size"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        for proj in self.projectiles[:]:
            for pest in self.pests[:]:
                if proj["pos"].distance_to(pest["pos"]) < self.PEST_RADIUS:
                    # SFX: Pest hit
                    pest["health"] -= proj["damage"]
                    reward += 0.1
                    self._create_particle(pest["pos"], 10, self.COLOR_PEST, speed_scale=2, life_scale=0.5)
                    
                    if proj["type"] != "piercing":
                        if proj in self.projectiles: self.projectiles.remove(proj)
                    
                    if pest["health"] <= 0:
                        # SFX: Pest defeat
                        reward += 1.0
                        self._create_particle(pest["pos"], 30, self.COLOR_PEST_GLOW, speed_scale=4, life_scale=1)
                        self.pests.remove(pest)
                    break 

        for pest in self.pests[:]:
            if pest["pos"].distance_to(self.core_pos) < self.CORE_RADIUS:
                # SFX: Core damage
                self.core_health -= 10
                self._create_particle(self.core_pos, 20, self.COLOR_CORE, speed_scale=5, life_scale=0.8)
                self.pests.remove(pest)
                reward -= 5.0

            if pest["pos"].distance_to(self.player_pos) < self.PLAYER_RADIUS + self.PEST_RADIUS:
                # SFX: Player hit
                reward -= 0.1
                if (self.player_pos - pest["pos"]).length() > 0:
                    self.player_vel += (self.player_pos - pest["pos"]).normalize() * 5
                self._create_particle(self.player_pos, 10, self.COLOR_PLAYER_GLOW, speed_scale=2, life_scale=0.5)
        
        return reward

    def _spawn_pests(self):
        if self.np_random.random() < self.pest_spawn_rate:
            edge = self.np_random.integers(4)
            if edge == 0: pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.PEST_RADIUS)
            elif edge == 1: pos = pygame.Vector2(self.WIDTH + self.PEST_RADIUS, self.np_random.uniform(0, self.HEIGHT))
            elif edge == 2: pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.PEST_RADIUS)
            else: pos = pygame.Vector2(-self.PEST_RADIUS, self.np_random.uniform(0, self.HEIGHT))
            
            self.pests.append({
                "pos": pos,
                "health": self.pest_base_health,
                "max_health": self.pest_base_health
            })
    
    def _update_progression(self):
        # Increase difficulty
        self.pest_spawn_rate = min(0.1, self.pest_spawn_rate + 0.0001)
        if self.steps > 0 and self.steps % 500 == 0:
            self.pest_base_health += 10
        
        # Unlock abilities
        if self.steps == 250 and "PIERCING" not in self.player_unlocked_abilities:
            self.player_unlocked_abilities.append("PIERCING")
            self._create_announcement("Piercing Shot Unlocked!")
        if self.steps == 500 and "BURST" not in self.player_unlocked_abilities:
            self.player_unlocked_abilities.append("BURST")
            self._create_announcement("Burst Shot Unlocked!")

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background foliage
        for i in range(15):
            r = (i * 37) % 255
            g = (i * 51) % 255
            b = (i * 42) % 255
            color = (max(0, self.COLOR_BG[0] - r%10), max(0, self.COLOR_BG[1] - g%15), max(0, self.COLOR_BG[2] - b%10))
            radius = 100 + (i*23)%50
            offset_x = math.sin(i) * 200
            offset_y = math.cos(i) * 150
            pos = (int(self.core_pos.x + offset_x), int(self.core_pos.y + offset_y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        self._render_particles()
        self._render_core()
        self._render_pests()
        self._render_projectiles()
        self._render_player()

    def _render_core(self):
        pos = (int(self.core_pos.x), int(self.core_pos.y))
        # Glow
        for i in range(self.CORE_RADIUS, self.CORE_RADIUS + 15, 2):
            alpha = 100 * (1 - (i - self.CORE_RADIUS) / 15)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (*self.COLOR_CORE_GLOW, int(alpha)))
        # Base
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CORE_RADIUS, self.COLOR_CORE)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CORE_RADIUS, self.COLOR_CORE_GLOW)

    def _render_pests(self):
        for pest in self.pests:
            pos = (int(pest["pos"].x), int(pest["pos"].y))
            # Glow
            for i in range(self.PEST_RADIUS, self.PEST_RADIUS + 5, 1):
                alpha = 80 * (1 - (i - self.PEST_RADIUS) / 5)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (*self.COLOR_PEST_GLOW, int(alpha)))
            # Base
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PEST_RADIUS, self.COLOR_PEST)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PEST_RADIUS, self.COLOR_PEST_GLOW)
            # Health bar
            if pest["health"] < pest["max_health"]:
                hp_ratio = pest["health"] / pest["max_health"]
                bar_w = self.PEST_RADIUS * 2
                bar_h = 4
                bar_x = pos[0] - self.PEST_RADIUS
                bar_y = pos[1] - self.PEST_RADIUS - 8
                pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH, (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj["pos"].x), int(proj["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], proj["size"], (*proj["color"], 100))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], proj["size"] // 2, proj["color"])

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Recoil animation
        squash = 1.0 - 0.3 * self.player_recoil_anim
        stretch = 1.0 + 0.3 * self.player_recoil_anim
        radius_x = int(self.PLAYER_RADIUS * (self.player_aim_dir.y**2 * stretch + self.player_aim_dir.x**2 * squash))
        radius_y = int(self.PLAYER_RADIUS * (self.player_aim_dir.x**2 * stretch + self.player_aim_dir.y**2 * squash))

        # Glow
        for i in range(max(radius_x, radius_y), max(radius_x, radius_y) + 10):
            alpha = 120 * (1 - (i - max(radius_x, radius_y)) / 10)
            pygame.gfxdraw.filled_ellipse(self.screen, pos[0], pos[1], i, i, (*self.COLOR_PLAYER_GLOW, int(alpha)))
        
        # Body
        pygame.gfxdraw.filled_ellipse(self.screen, pos[0], pos[1], radius_x, radius_y, self.COLOR_PLAYER)
        pygame.gfxdraw.aaellipse(self.screen, pos[0], pos[1], radius_x, radius_y, self.COLOR_PLAYER_GLOW)
        
        # Aiming indicator
        aim_end = self.player_pos + self.player_aim_dir * (self.PLAYER_RADIUS + 10)
        pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, pos, (int(aim_end.x), int(aim_end.y)), 2)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            size = int(p["size"])
            if size > 0:
                alpha = p["life"] / p["max_life"] * 255
                color = (*p["color"], int(alpha))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

    def _render_ui(self):
        # Core Health Bar
        bar_w, bar_h = 200, 20
        bar_x, bar_y = self.WIDTH // 2 - bar_w // 2, 20
        health_ratio = max(0, self.core_health / self.CORE_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (bar_x, bar_y, int(bar_w * health_ratio), bar_h), border_radius=4)
        core_text = self.font_small.render("GARDEN CORE", True, self.COLOR_UI_TEXT)
        self.screen.blit(core_text, (bar_x + bar_w // 2 - core_text.get_width() // 2, bar_y + bar_h // 2 - core_text.get_height() // 2))

        # Player Energy Bar
        bar_w, bar_h = 100, 8
        bar_x, bar_y = self.player_pos.x - bar_w / 2, self.player_pos.y + self.PLAYER_RADIUS + 8
        energy_ratio = max(0, self.player_energy / self.player_max_energy)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_w, bar_h), border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (bar_x, bar_y, int(bar_w * energy_ratio), bar_h), border_radius=2)

        # Score and Steps
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        steps_text = self.font_medium.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 20, 20))

        # Current Ability
        ability_name = self.player_unlocked_abilities[self.player_current_ability_idx]
        ability_text = self.font_medium.render(f"MODE: {ability_name}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ability_text, (20, self.HEIGHT - ability_text.get_height() - 20))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "core_health": self.core_health,
            "player_energy": self.player_energy,
            "pests_remaining": len(self.pests),
            "current_ability": self.player_unlocked_abilities[self.player_current_ability_idx],
        }

    def _create_projectile(self, pos, direction, ability):
        color_map = {"normal": (100, 200, 255), "piercing": (255, 100, 255), "burst": (255, 255, 100)}
        self.projectiles.append({
            "pos": pos,
            "vel": direction * ability["speed"],
            "damage": ability["damage"],
            "type": ability["type"],
            "size": 5 if ability["type"] != "piercing" else 7,
            "color": color_map.get(ability["type"], (255,255,255))
        })

    def _create_particle(self, pos, count, color, speed_scale=1.0, life_scale=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = int(self.np_random.uniform(10, 20) * life_scale)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "color": color,
                "size": self.np_random.uniform(2, 5),
                "life": life,
                "max_life": life,
            })
    
    def _create_announcement(self, text):
        # This is a visual effect, particles can be used to represent it.
        # For simplicity, we'll just flash some particles from the center.
        self._create_particle(self.core_pos, 50, self.COLOR_PLAYER_GLOW, speed_scale=3, life_scale=1.5)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # You will need to unset SDL_VIDEODRIVER to play with a display.
    # E.g., `SDL_VIDEODRIVER="" python your_game_file.py`
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Sentient Plant Defender")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()