import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓/←/→ to select a build location. Press Shift to cycle tower types. Press Space to build."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist tower defense game. Place towers to defend your base from waves of enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set headless mode for pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * 60 * 2 # Approx 2 minutes
        self.MAX_WAVES = 10

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PATH = (40, 40, 55)
        self.COLOR_BASE = (0, 150, 100)
        self.COLOR_BASE_STROKE = (0, 200, 130)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_ENEMY_STROKE = (255, 100, 100)
        self.COLOR_TOWER_ZONE = (60, 60, 80, 100)
        self.COLOR_TOWER_ZONE_SELECTED = (200, 200, 0, 200)
        self.COLOR_TEXT = (220, 220, 220)
        self.TOWER_COLORS = {
            1: ((0, 150, 255), (100, 200, 255)), # Blue
            2: ((255, 180, 0), (255, 220, 100)), # Yellow
            3: ((180, 0, 255), (220, 100, 255)), # Purple
        }

        # Fonts
        try:
            self.font_s = pygame.font.SysFont("Arial", 16, bold=True)
            self.font_m = pygame.font.SysFont("Arial", 24, bold=True)
            self.font_l = pygame.font.SysFont("Arial", 48, bold=True)
        except pygame.error:
            self.font_s = pygame.font.Font(None, 16)
            self.font_m = pygame.font.Font(None, 24)
            self.font_l = pygame.font.Font(None, 48)

        # Game configuration
        self.path = [(0, 200), (150, 200), (150, 100), (450, 100), (450, 300), (150, 300), (150, 200), (640, 200)]
        self.tower_zones = [
            {"pos": (100, 150), "occupied": False}, {"pos": (200, 150), "occupied": False}, {"pos": (300, 150), "occupied": False}, {"pos": (400, 150), "occupied": False},
            {"pos": (100, 250), "occupied": False}, {"pos": (200, 250), "occupied": False}, {"pos": (300, 250), "occupied": False}, {"pos": (400, 250), "occupied": False},
        ]
        # FIX: Renamed 'type' to 'attack_type' to avoid collision with the integer tower 'type' ID.
        self.TOWER_SPECS = {
            1: {"range": 70, "damage": 2.5, "fire_rate": 0.8, "cost": 30, "projectile_speed": 5, "attack_type": "single"},
            2: {"range": 90, "damage": 0.8, "fire_rate": 0.2, "cost": 40, "projectile_speed": 7, "attack_type": "single"},
            3: {"range": 60, "damage": 1.5, "fire_rate": 1.5, "cost": 60, "projectile_speed": 4, "attack_type": "aoe", "aoe_radius": 30},
        }

        # Initialize state variables
        # self.reset() is called by the API, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.max_base_health = 100
        self.resources = 80
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_spawn_timer = 0
        self.enemies_to_spawn = []
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        for zone in self.tower_zones:
            zone["occupied"] = False

        self.selected_zone_idx = 0
        self.selected_tower_type = 1
        
        self.last_space_held = False
        self.last_shift_held = False

        self._start_next_wave()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.001 # Small penalty for existing
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle rising edge of actions
        place_tower_action = space_held and not self.last_space_held
        cycle_tower_action = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        if not self.game_over:
            # Player input
            self._handle_input(movement, place_tower_action, cycle_tower_action)
            # Update game state
            step_rewards = self._update_game_state()
            reward += step_rewards
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward = 100
            else: # Loss
                reward = -100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, place_tower, cycle_tower):
        # Cycle through tower placement zones
        if movement in [1, 4]: # Up or Right
            self.selected_zone_idx = (self.selected_zone_idx + 1) % len(self.tower_zones)
        elif movement in [2, 3]: # Down or Left
            self.selected_zone_idx = (self.selected_zone_idx - 1 + len(self.tower_zones)) % len(self.tower_zones)
        
        # Cycle through tower types
        if cycle_tower:
            self.selected_tower_type = (self.selected_tower_type % 3) + 1

        # Place tower
        if place_tower:
            zone = self.tower_zones[self.selected_zone_idx]
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if not zone["occupied"] and self.resources >= spec["cost"]:
                self.resources -= spec["cost"]
                zone["occupied"] = True
                self.towers.append({
                    "pos": zone["pos"],
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    **spec
                })
                # sfx: build_tower

    def _update_game_state(self):
        rewards = 0
        
        self._spawn_enemies()
        rewards += self._update_enemies()
        self._update_towers()
        rewards += self._update_projectiles()
        self._update_particles()
        
        if not self.wave_in_progress and not self.enemies:
            if self.wave_number >= self.MAX_WAVES:
                self.win = True
            else:
                rewards += 1.0 # Wave clear bonus
                self._start_next_wave()
        
        return rewards

    def _start_next_wave(self):
        self.wave_number += 1
        self.wave_in_progress = True
        self.wave_spawn_timer = 0
        
        num_enemies = 5 + self.wave_number * 2
        base_health = 5 + self.wave_number * 2
        base_speed = 0.8 + self.wave_number * 0.05
        
        for i in range(num_enemies):
            self.enemies_to_spawn.append({
                "health": base_health * self.np_random.uniform(0.9, 1.1),
                "speed": base_speed * self.np_random.uniform(0.9, 1.1),
                "spawn_delay": i * (self.FPS / 2) # 0.5s between spawns
            })

    def _spawn_enemies(self):
        if not self.enemies_to_spawn:
            if not self.enemies:
                self.wave_in_progress = False
            return
            
        self.wave_spawn_timer += 1
        if self.wave_spawn_timer >= self.enemies_to_spawn[0]["spawn_delay"]:
            enemy_data = self.enemies_to_spawn.pop(0)
            self.enemies.append({
                "pos": list(self.path[0]),
                "max_health": enemy_data["health"],
                "health": enemy_data["health"],
                "speed": enemy_data["speed"],
                "path_idx": 0
            })
            # sfx: enemy_spawn

    def _update_enemies(self):
        rewards = 0
        for enemy in self.enemies[:]:
            if enemy["path_idx"] >= len(self.path) - 1:
                self.base_health -= enemy["health"]
                self.enemies.remove(enemy)
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 20)
                # sfx: base_damage
                continue

            target_pos = self.path[enemy["path_idx"] + 1]
            direction = np.array(target_pos) - np.array(enemy["pos"])
            distance = np.linalg.norm(direction)
            
            if distance < enemy["speed"]:
                enemy["pos"] = list(target_pos)
                enemy["path_idx"] += 1
            else:
                move_vec = (direction / distance) * enemy["speed"]
                enemy["pos"][0] += move_vec[0]
                enemy["pos"][1] += move_vec[1]
        
        self.base_health = max(0, self.base_health)
        return rewards

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] > 0:
                continue

            target = None
            min_dist = tower["range"]
            for enemy in self.enemies:
                dist = math.hypot(enemy["pos"][0] - tower["pos"][0], enemy["pos"][1] - tower["pos"][1])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower["cooldown"] = tower["fire_rate"] * self.FPS
                self.projectiles.append({
                    "pos": list(tower["pos"]),
                    "target": target,
                    "speed": tower["projectile_speed"],
                    "damage": tower["damage"],
                    # FIX: Use 'attack_type' for projectile logic, which is the string 'single' or 'aoe'.
                    "type": tower["attack_type"],
                    "color": self.TOWER_COLORS[tower["type"]][1],
                    "aoe_radius": tower.get("aoe_radius", 0)
                })
                # sfx: tower_shoot

    def _update_projectiles(self):
        rewards = 0
        for p in self.projectiles[:]:
            if p["target"] not in self.enemies:
                self.projectiles.remove(p)
                continue

            target_pos = p["target"]["pos"]
            direction = np.array(target_pos) - np.array(p["pos"])
            distance = np.linalg.norm(direction)

            if distance < p["speed"]:
                # Hit
                if p["type"] == "aoe":
                    # sfx: aoe_explosion
                    self._create_particles(p["target"]["pos"], p["color"], 25, 2)
                    for enemy in self.enemies[:]:
                        if math.hypot(enemy["pos"][0] - p["target"]["pos"][0], enemy["pos"][1] - p["target"]["pos"][1]) <= p["aoe_radius"]:
                            enemy["health"] -= p["damage"]
                            if enemy["health"] <= 0:
                                rewards += self._kill_enemy(enemy)
                else: # single target
                    # sfx: projectile_hit
                    p["target"]["health"] -= p["damage"]
                    self._create_particles(p["pos"], p["color"], 5)
                    if p["target"]["health"] <= 0:
                        rewards += self._kill_enemy(p["target"])
                
                self.projectiles.remove(p)
            else:
                move_vec = (direction / distance) * p["speed"]
                p["pos"][0] += move_vec[0]
                p["pos"][1] += move_vec[1]
        return rewards

    def _kill_enemy(self, enemy):
        if enemy in self.enemies:
            self.enemies.remove(enemy)
        self.score += 10
        self.resources += 5
        self._create_particles(enemy["pos"], self.COLOR_ENEMY_STROKE, 15)
        # sfx: enemy_die
        return 0.1 # Kill reward

    def _update_particles(self):
        for particle in self.particles[:]:
            particle["life"] -= 1
            particle["pos"][0] += particle["vel"][0]
            particle["pos"][1] += particle["vel"][1]
            if particle["life"] <= 0:
                self.particles.remove(particle)

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(10, 20),
                "color": color,
                "size": self.np_random.uniform(1, 3)
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.win:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            # This is a truncation condition, not termination
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)

        # Tower zones
        for i, zone in enumerate(self.tower_zones):
            pos = (int(zone["pos"][0]), int(zone["pos"][1]))
            if i == self.selected_zone_idx:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 22, (*self.COLOR_TOWER_ZONE_SELECTED[:3], 50))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 22, self.COLOR_TOWER_ZONE_SELECTED)
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 20, self.COLOR_TOWER_ZONE)

        # Base
        base_rect = pygame.Rect(self.WIDTH-40, 180, 40, 40)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, self.COLOR_BASE_STROKE, base_rect, 2)
        
        # Towers
        for tower in self.towers:
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            color, stroke_color = self.TOWER_COLORS[tower["type"]]
            if tower["type"] == 1: # Square
                pygame.draw.rect(self.screen, color, (pos[0]-10, pos[1]-10, 20, 20))
                pygame.draw.rect(self.screen, stroke_color, (pos[0]-10, pos[1]-10, 20, 20), 2)
            elif tower["type"] == 2: # Triangle
                points = [(pos[0], pos[1]-11), (pos[0]-10, pos[1]+8), (pos[0]+10, pos[1]+8)]
                pygame.gfxdraw.aapolygon(self.screen, points, stroke_color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif tower["type"] == 3: # Hexagon
                points = [(pos[0] + 12 * math.cos(math.pi/3 * i), pos[1] + 12 * math.sin(math.pi/3 * i)) for i in range(6)]
                pygame.gfxdraw.aapolygon(self.screen, points, stroke_color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Projectiles
        for p in self.projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, p["color"])
        
        # Particles
        for particle in self.particles:
            pos = (int(particle["pos"][0]), int(particle["pos"][1]))
            size = int(particle["size"] * (particle["life"] / 20.0))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, particle["color"])

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY_STROKE)
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (50,50,50), (pos[0]-10, pos[1]-18, 20, 4))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_STROKE, (pos[0]-10, pos[1]-18, int(20*health_pct), 4))

    def _render_ui(self):
        # Base Health
        self._draw_text("BASE HEALTH", self.font_s, (10, 10))
        health_pct = self.base_health / self.max_base_health
        health_color = (255 * (1-health_pct), 255 * health_pct, 0)
        pygame.draw.rect(self.screen, (50,50,50), (10, 30, 200, 20))
        pygame.draw.rect(self.screen, health_color, (10, 30, int(200*health_pct), 20))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 30, 200, 20), 1)

        # Wave
        self._draw_text(f"WAVE {self.wave_number}/{self.MAX_WAVES}", self.font_m, (self.WIDTH - 10, 10), align="topright")
        
        # Score
        self._draw_text(f"SCORE: {self.score}", self.font_m, (self.WIDTH - 10, self.HEIGHT - 10), align="bottomright")

        # Resources
        self._draw_text(f"RESOURCES: ${self.resources}", self.font_m, (10, self.HEIGHT - 10), align="bottomleft")
        
        # Selected Tower Info
        spec = self.TOWER_SPECS[self.selected_tower_type]
        info_text = f"TOWER {self.selected_tower_type} | COST: ${spec['cost']}"
        text_surf = self.font_s.render(info_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT - 30))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0,0))
        msg = "VICTORY!" if self.win else "GAME OVER"
        self._draw_text(msg, self.font_l, (self.WIDTH // 2, self.HEIGHT // 2), align="center")

    def _draw_text(self, text, font, pos, color=None, align="topleft"):
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        elif align == "bottomleft":
            text_rect.bottomleft = pos
        elif align == "bottomright":
            text_rect.bottomright = pos
        elif align == "center":
            text_rect.center = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "resources": self.resources,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Make sure to set the SDL_VIDEODRIVER to a supported backend.
    # For linux: "x11", for windows: "windows", for mac: "macOS"
    # This is not necessary for running the environment in a headless manner.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset(seed=123)
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_held = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0
    
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset(seed=123)
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()