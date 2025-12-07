import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:06:23.659971
# Source Brief: brief_03251.md
# Brief Index: 3251
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for game objects
class GameObject:
    def __init__(self, x, y, radius):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.radius = radius

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Evolve from a simple organism into an apex predator in a primordial world. "
        "Consume biomass to grow, fight off hostile creatures, and unlock new evolutionary forms."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to attack with a pseudopod. "
        "Hold shift to consume nearby biomass."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 1280, 800
        self.FPS = 30
        self.MAX_STEPS = 5000
        
        # Colors
        self.COLOR_BG = (12, 23, 35)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_BIOMASS = (150, 255, 100)
        self.COLOR_HAZARD = (255, 50, 100)
        self.COLOR_ENEMY_WEAK = (255, 120, 50)
        self.COLOR_ENEMY_STRONG = (255, 60, 180)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (20, 40, 60, 180)
        self.COLOR_HEALTH = (255, 70, 70)
        self.COLOR_BIOMASS_BAR = (100, 220, 255)

        # Evolution Config
        self.EVOLUTION_CONFIG = {
            1: {"biomass_cost": 0, "max_health": 100, "radius": 15, "speed": 2.5, "attack_damage": 10, "attack_cooldown": 20, "consume_radius": 40},
            2: {"biomass_cost": 40, "max_health": 120, "radius": 17, "speed": 2.7, "attack_damage": 15, "attack_cooldown": 18, "consume_radius": 45},
            3: {"biomass_cost": 100, "max_health": 150, "radius": 20, "speed": 3.0, "attack_damage": 20, "attack_cooldown": 16, "consume_radius": 50},
            4: {"biomass_cost": 200, "max_health": 200, "radius": 23, "speed": 3.3, "attack_damage": 25, "attack_cooldown": 14, "consume_radius": 55},
            5: {"biomass_cost": 350, "max_health": 250, "radius": 25, "speed": 3.5, "attack_damage": 30, "attack_cooldown": 12, "consume_radius": 60},
        }
        self.MAX_EVOLUTION_LEVEL = 5

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- Game State Initialization ---
        self.player = None
        self.enemies = []
        self.biomass_particles = []
        self.hazards = []
        self.pseudopods = []
        self.particles = []
        self.background_elements = []
        self.camera_offset = pygame.Vector2(0, 0)
        
        # self.reset() # reset is called by the wrapper/runner
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Player State ---
        level_1_config = self.EVOLUTION_CONFIG[1]
        self.player = {
            "pos": pygame.Vector2(self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2),
            "level": 1,
            "biomass": 0,
            "last_move_dir": pygame.Vector2(1, 0),
            "attack_timer": 0,
            "damage_flash": 0,
            **level_1_config
        }
        self.player["health"] = self.player["max_health"]

        # --- Initialize Game World ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.enemies.clear()
        self.biomass_particles.clear()
        self.hazards.clear()
        self.pseudopods.clear()
        self.particles.clear()
        self.background_elements.clear()

        self._spawn_world_elements()

        return self._get_observation(), self._get_info()

    def _spawn_world_elements(self):
        # Spawn initial biomass
        for _ in range(50):
            self._spawn_biomass(1)
        
        # Spawn hazards
        for _ in range(10):
            pos = self._get_random_world_pos()
            radius = self.np_random.integers(30, 60)
            self.hazards.append(GameObject(pos.x, pos.y, radius))

        # Spawn initial enemies
        self._spawn_enemies_for_level(1)

        # Spawn decorative background elements
        for _ in range(100):
            pos = self._get_random_world_pos()
            radius = self.np_random.integers(10, 80)
            color = (
                self.np_random.integers(5, 15),
                self.np_random.integers(15, 30),
                self.np_random.integers(25, 45)
            )
            self.background_elements.append({"pos": pos, "radius": radius, "color": color})


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Player movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player["pos"] += move_vec * self.player["speed"]
            self.player["last_move_dir"] = move_vec.copy()
        
        # Clamp player position to world bounds
        self.player["pos"].x = np.clip(self.player["pos"].x, self.player["radius"], self.WORLD_WIDTH - self.player["radius"])
        self.player["pos"].y = np.clip(self.player["pos"].y, self.player["radius"], self.WORLD_HEIGHT - self.player["radius"])

        # Player attack
        self.player["attack_timer"] = max(0, self.player["attack_timer"] - 1)
        if space_held and self.player["attack_timer"] == 0:
            self.player["attack_timer"] = self.player["attack_cooldown"]
            start_pos = self.player["pos"] + self.player["last_move_dir"] * self.player["radius"]
            end_pos = start_pos + self.player["last_move_dir"] * 75 # Pseudopod length
            self.pseudopods.append({"start": start_pos, "end": end_pos, "life": 10, "damage": self.player["attack_damage"]})
            # Sound: PlayerAttack.wav

        # Player consume
        if shift_held:
            # Visual feedback for consumption
            self._create_particles(self.player["pos"], 10, (100, 200, 255, 100), 2, 0.5, 15)
            
            consumed_biomass = []
            for biomass in self.biomass_particles:
                if self.player["pos"].distance_to(biomass.pos) < self.player["consume_radius"]:
                    self.player["biomass"] += biomass.value
                    reward += 0.1
                    consumed_biomass.append(biomass)
                    self._create_particles(biomass.pos, 5, self.COLOR_BIOMASS, 3, 1.5, 10, target=self.player["pos"])
                    # Sound: Consume.wav
            self.biomass_particles = [b for b in self.biomass_particles if b not in consumed_biomass]
            
            # Respawn biomass to keep the world populated
            if self.np_random.random() < 0.1:
                self._spawn_biomass(1)

        # --- Update Game Logic ---
        self._update_enemies()
        self._update_pseudopods()
        self._update_particles()
        reward += self._handle_collisions()
        
        # --- Check for Evolution ---
        next_level = self.player["level"] + 1
        if next_level <= self.MAX_EVOLUTION_LEVEL and self.player["biomass"] >= self.EVOLUTION_CONFIG[next_level]["biomass_cost"]:
            self.player["level"] = next_level
            config = self.EVOLUTION_CONFIG[next_level]
            self.player.update(config)
            self.player["health"] = self.player["max_health"] # Heal on evolution
            reward += 5.0
            self._spawn_enemies_for_level(next_level)
            self._create_particles(self.player["pos"], 100, self.COLOR_PLAYER, 5, 3, 40)
            # Sound: Evolve.wav

        self.player["damage_flash"] = max(0, self.player["damage_flash"] - 1)
        self.score += reward

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.player["health"] <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
            # Sound: PlayerDeath.wav
        elif self.player["level"] >= self.MAX_EVOLUTION_LEVEL:
            reward += 100.0
            terminated = True
            self.game_over = True
            # Sound: Victory.wav
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        # Gymnasium API requires `terminated` and `truncated` to be separate
        # `terminated` is for end-of-episode conditions (win/loss)
        # `truncated` is for external conditions (time limit)
        if truncated:
            terminated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_enemies(self):
        for enemy in self.enemies:
            dist_to_player = enemy.pos.distance_to(self.player["pos"])
            # AI: Chase if close, otherwise patrol
            if dist_to_player < enemy.aggro_radius:
                direction = (self.player["pos"] - enemy.pos).normalize()
                enemy.vel = direction * enemy.speed
            else:
                if enemy.pos.distance_to(enemy.home_pos) > enemy.patrol_radius or enemy.vel.length() == 0:
                    target_offset = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * enemy.patrol_radius
                    target = enemy.home_pos + target_offset
                    direction = (target - enemy.pos).normalize()
                    enemy.vel = direction * enemy.speed * 0.5 # Slower patrol
            
            enemy.pos += enemy.vel
            
            # Attack player if in range
            if dist_to_player < enemy.radius + self.player["radius"]:
                self.player["health"] -= enemy.damage
                self.player["damage_flash"] = 10
                self._create_particles(self.player["pos"], 10, self.COLOR_HAZARD, 4, 2, 15)
                # Sound: PlayerHit.wav

    def _update_pseudopods(self):
        for p in self.pseudopods:
            p["life"] -= 1
        self.pseudopods = [p for p in self.pseudopods if p["life"] > 0]

    def _update_particles(self):
        for p in self.particles:
            p["life"] -= 1
            if p.get("target"):
                p["vel"] += (p["target"] - p["pos"]).normalize() * 0.5
            p["pos"] += p["vel"]
            p["radius"] -= p["shrink_rate"]
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Pseudopod vs Enemies
        hit_enemies = []
        for p in self.pseudopods:
            for enemy in self.enemies:
                if enemy in hit_enemies: continue
                # Simple line-circle collision
                if self._line_circle_collision(p["start"], p["end"], enemy.pos, enemy.radius):
                    enemy.health -= p["damage"]
                    enemy.pos += (enemy.pos - self.player["pos"]).normalize() * 5 # Knockback
                    self._create_particles(enemy.pos, 5, self.COLOR_PLAYER, 3, 2, 10)
                    hit_enemies.append(enemy)
                    # Sound: EnemyHit.wav
        
        dead_enemies = []
        for enemy in self.enemies:
            if enemy.health <= 0:
                dead_enemies.append(enemy)
                self._spawn_biomass(enemy.biomass_drop, enemy.pos)
                self._create_particles(enemy.pos, 20, enemy.color, 5, 2, 20)
                # Sound: EnemyDie.wav
        self.enemies = [e for e in self.enemies if e not in dead_enemies]

        # Player vs Hazards
        for hazard in self.hazards:
            if self.player["pos"].distance_to(hazard.pos) < self.player["radius"] + hazard.radius:
                self.player["health"] -= 0.2 # Damage over time
                reward -= 1.0 / self.FPS # Small penalty per frame in hazard
                self.player["damage_flash"] = 5
                if self.np_random.random() < 0.2:
                    self._create_particles(self.player["pos"], 3, self.COLOR_HAZARD, 3, 1, 10)
        
        self.player["health"] = max(0, self.player["health"])
        return reward

    def _get_observation(self):
        # Center camera on player
        self.camera_offset.x = int(self.player["pos"].x - self.SCREEN_WIDTH / 2)
        self.camera_offset.y = int(self.player["pos"].y - self.SCREEN_HEIGHT / 2)

        # --- Render Game World ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        
        # --- Render UI Overlay ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background elements
        for bg_obj in self.background_elements:
            pygame.gfxdraw.filled_circle(
                self.screen,
                int(bg_obj["pos"].x - self.camera_offset.x),
                int(bg_obj["pos"].y - self.camera_offset.y),
                int(bg_obj["radius"]),
                bg_obj["color"]
            )

        # Hazards
        for hazard in self.hazards:
            pos_on_screen = hazard.pos - self.camera_offset
            # Bubbling effect
            temp_surface = pygame.Surface((hazard.radius * 2, hazard.radius * 2), pygame.SRCALPHA)
            alpha = 60 + 20 * math.sin(pygame.time.get_ticks() * 0.001 + hazard.pos.x)
            pygame.gfxdraw.filled_circle(
                temp_surface, int(hazard.radius), int(hazard.radius), int(hazard.radius), (*self.COLOR_HAZARD, int(alpha))
            )
            self.screen.blit(temp_surface, (int(pos_on_screen.x - hazard.radius), int(pos_on_screen.y - hazard.radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Biomass
        for biomass in self.biomass_particles:
            pos = biomass.pos - self.camera_offset
            glow_size = int(biomass.radius * (1.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.005 + biomass.pos.x)))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), glow_size, (*self.COLOR_BIOMASS, 20))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(biomass.radius), self.COLOR_BIOMASS)

        # Enemies
        for enemy in self.enemies:
            pos = enemy.pos - self.camera_offset
            if enemy.type == 'weak':
                points = []
                for i in range(5):
                    angle = (i / 5) * 2 * math.pi + pygame.time.get_ticks() * 0.002
                    p_radius = enemy.radius + 3 * math.sin((i*2) * math.pi * 0.4 + pygame.time.get_ticks() * 0.005)
                    points.append((int(pos.x + p_radius * math.cos(angle)), int(pos.y + p_radius * math.sin(angle))))
                pygame.gfxdraw.aapolygon(self.screen, points, enemy.color)
                pygame.gfxdraw.filled_polygon(self.screen, points, enemy.color)
            else: # Strong
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(enemy.radius), enemy.color)
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(enemy.radius), enemy.color)

        # Player
        player_pos_screen = self.player["pos"] - self.camera_offset
        # Glow
        if self.player["damage_flash"] > 0:
            flash_color = (255, 0, 0, 150)
            pygame.gfxdraw.filled_circle(self.screen, int(player_pos_screen.x), int(player_pos_screen.y), int(self.player["radius"] * 1.5), flash_color)
        else:
            pygame.gfxdraw.filled_circle(self.screen, int(player_pos_screen.x), int(player_pos_screen.y), int(self.player["radius"] * 1.8), self.COLOR_PLAYER_GLOW)
        # Body (wobble effect)
        num_blobs = 7
        for i in range(num_blobs):
            angle = (i / num_blobs) * 2 * math.pi
            offset_radius = self.player["radius"] * 0.2
            offset_angle = pygame.time.get_ticks() * 0.001 + angle * 3
            offset_x = offset_radius * math.cos(offset_angle)
            offset_y = offset_radius * math.sin(offset_angle)
            blob_radius = int(self.player["radius"] * 0.7)
            pygame.gfxdraw.filled_circle(self.screen, int(player_pos_screen.x + offset_x), int(player_pos_screen.y + offset_y), blob_radius, self.COLOR_PLAYER)
        
        # Pseudopods
        for p in self.pseudopods:
            life_ratio = p["life"] / 10.0
            current_end = p["start"].lerp(p["end"], 1.0 - abs(life_ratio * 2 - 1)) # Extend and retract
            start_pos_screen = p["start"] - self.camera_offset
            end_pos_screen = current_end - self.camera_offset
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (int(start_pos_screen.x), int(start_pos_screen.y)), (int(end_pos_screen.x), int(end_pos_screen.y)), max(3, int(self.player["radius"] * 0.5)))

        # Particles
        for p in self.particles:
            pos = p["pos"] - self.camera_offset
            color = (*p["color"][:3], int(255 * (p["life"] / p["max_life"])))
            if color[3] > 0:
                temp_surface = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surface, int(p["radius"]), int(p["radius"]), int(p["radius"]), color)
                self.screen.blit(temp_surface, (int(pos.x - p["radius"]), int(pos.y - p["radius"])), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_ui(self):
        # UI Background panels
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (5, 5, 210, 60), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH - 215, 5, 210, 35), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH // 2 - 150, self.SCREEN_HEIGHT - 45, 300, 40), border_radius=5)

        # Health Bar
        health_pct = self.player["health"] / self.player["max_health"]
        pygame.draw.rect(self.screen, (50, 0, 0), (15, 15, 190, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (15, 15, max(0, 190 * health_pct), 20))
        health_text = self.font_small.render(f"HEALTH: {int(self.player['health'])}/{self.player['max_health']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 40))

        # Biomass Bar
        next_level = self.player["level"] + 1
        biomass_needed = self.EVOLUTION_CONFIG.get(next_level, {}).get("biomass_cost", 1)
        biomass_pct = self.player["biomass"] / biomass_needed if biomass_needed > 0 else 1.0
        pygame.draw.rect(self.screen, (0, 0, 50), (self.SCREEN_WIDTH - 205, 15, 190, 20))
        pygame.draw.rect(self.screen, self.COLOR_BIOMASS_BAR, (self.SCREEN_WIDTH - 205, 15, max(0, 190 * biomass_pct), 20))
        biomass_text = self.font_small.render(f"BIOMASS: {self.player['biomass']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(biomass_text, (self.SCREEN_WIDTH - biomass_text.get_width() - 15, 40))

        # Evolution Stage
        evo_text = self.font_large.render(f"EVOLUTION STAGE: {self.player['level']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(evo_text, (self.SCREEN_WIDTH // 2 - evo_text.get_width() // 2, self.SCREEN_HEIGHT - 38))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "biomass": self.player["biomass"],
            "level": self.player["level"],
        }
    
    def _get_random_world_pos(self):
        return pygame.Vector2(
            self.np_random.uniform(50, self.WORLD_WIDTH - 50),
            self.np_random.uniform(50, self.WORLD_HEIGHT - 50)
        )

    def _spawn_biomass(self, count, pos=None):
        for _ in range(count):
            spawn_pos = pos if pos else self._get_random_world_pos()
            # Add some scatter if a position is provided
            if pos:
                spawn_pos = spawn_pos + pygame.Vector2(self.np_random.uniform(-20, 20), self.np_random.uniform(-20, 20))

            b = GameObject(spawn_pos.x, spawn_pos.y, self.np_random.integers(2, 5))
            b.value = 1
            self.biomass_particles.append(b)

    def _spawn_enemies_for_level(self, level):
        num_weak = 2 + level
        num_strong = max(0, level - 2)

        for _ in range(num_weak):
            e = GameObject(*self._get_random_world_pos(), radius=12)
            e.home_pos = e.pos.copy()
            e.type = 'weak'
            e.color = self.COLOR_ENEMY_WEAK
            e.health = 20 + 5 * level
            e.damage = 3 + level
            e.speed = 1.0 + 0.1 * level
            e.aggro_radius = 150
            e.patrol_radius = 100
            e.biomass_drop = 5
            self.enemies.append(e)

        for _ in range(num_strong):
            e = GameObject(*self._get_random_world_pos(), radius=18)
            e.home_pos = e.pos.copy()
            e.type = 'strong'
            e.color = self.COLOR_ENEMY_STRONG
            e.health = 40 + 10 * level
            e.damage = 5 + 2 * level
            e.speed = 0.8 + 0.1 * level
            e.aggro_radius = 200
            e.patrol_radius = 80
            e.biomass_drop = 15
            self.enemies.append(e)

    def _create_particles(self, pos, count, color, size, speed, lifetime, target=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(0.5, 1) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "color": color,
                "radius": self.np_random.uniform(size*0.5, size),
                "life": self.np_random.uniform(lifetime*0.5, lifetime),
                "max_life": lifetime,
                "shrink_rate": size / lifetime,
                "target": target
            })

    def _line_circle_collision(self, p1, p2, circle_center, circle_radius):
        # Check if either end is inside the circle
        if p1.distance_to(circle_center) < circle_radius or p2.distance_to(circle_center) < circle_radius:
            return True
        
        # Check for projection onto the line segment
        line_vec = p2 - p1
        if line_vec.length() == 0: return False
        
        point_vec = circle_center - p1
        t = point_vec.dot(line_vec) / line_vec.dot(line_vec)
        
        if 0 <= t <= 1:
            closest_point = p1 + t * line_vec
            if closest_point.distance_to(circle_center) < circle_radius:
                return True
        return False

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You might need to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Primordial Swamp")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Final Level: {info['level']}, Steps: {info['steps']}")
            done = True
            
        clock.tick(env.FPS)

    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()