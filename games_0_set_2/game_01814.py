
# Generated: 2025-08-27T18:24:15.258180
# Source Brief: brief_01814.md
# Brief Index: 1814

        
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


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, life, color_start, color_end, radius_start, radius_end):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.life = life
        self.max_life = life
        self.color_start = color_start
        self.color_end = color_end
        self.radius_start = radius_start
        self.radius_end = radius_end

    def update(self):
        self.pos += self.vel
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            progress = self.life / self.max_life
            current_radius = int(self.radius_start * progress + self.radius_end * (1 - progress))
            current_color = [
                int(c1 * progress + c2 * (1 - progress))
                for c1, c2 in zip(self.color_start, self.color_end)
            ]
            if current_radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), current_radius, current_color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to cycle tower plots. Shift to cycle tower types. Space to build."
    )

    game_description = (
        "Top-down tower defense. Strategically place towers to defend your base from waves of enemies."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Constants & Configs ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PATH = (40, 50, 80)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_SPOT_VALID = (0, 255, 0, 50)
        self.COLOR_SPOT_INVALID = (255, 0, 0, 50)
        self.COLOR_SPOT_SELECTED = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BG = (100, 0, 0)
        self.COLOR_HEALTH_FG = (0, 200, 0)
        
        self.MAX_STEPS = 30 * 180  # 3 minutes at 30fps
        self.MAX_WAVES = 20
        self.INITIAL_BASE_HEALTH = 100
        self.INITIAL_RESOURCES = 150
        
        self.font_s = pygame.font.SysFont("sans-serif", 16, bold=True)
        self.font_m = pygame.font.SysFont("sans-serif", 24, bold=True)
        
        self._define_layout()
        self._define_configs()
        
        self.reset()
        self.validate_implementation()

    def _define_layout(self):
        self.path = [
            pygame.Vector2(-20, 200), pygame.Vector2(100, 200),
            pygame.Vector2(100, 100), pygame.Vector2(300, 100),
            pygame.Vector2(300, 300), pygame.Vector2(540, 300),
            pygame.Vector2(540, 150), pygame.Vector2(self.WIDTH + 20, 150)
        ]
        self.base_pos = pygame.Vector2(self.WIDTH, 150)
        self.tower_spots = [
            (50, 150), (150, 150), (150, 50), (250, 50), (350, 150),
            (250, 250), (350, 250), (450, 250), (490, 220), (490, 350)
        ]

    def _define_configs(self):
        self.tower_configs = [
            {"name": "Gun", "cost": 50, "range": 80, "damage": 5, "fire_rate": 20, "color": (0, 200, 255), "shape": "circle"},
            {"name": "Cannon", "cost": 120, "range": 120, "damage": 25, "fire_rate": 60, "color": (255, 150, 0), "shape": "square"},
            {"name": "Sniper", "cost": 200, "range": 250, "damage": 50, "fire_rate": 90, "color": (255, 50, 255), "shape": "triangle"},
        ]
        self.enemy_configs = {
            "normal": {"health": 20, "speed": 1.5, "reward": 10, "color": (220, 50, 50)},
            "fast": {"health": 15, "speed": 2.5, "reward": 15, "color": (255, 100, 50)},
            "tank": {"health": 80, "speed": 1.0, "reward": 30, "color": (180, 50, 180)},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.inter_wave_timer = 90  # 3s
        
        self.wave_spawns = []
        self.spawn_cooldown = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.occupied_spots = [False] * len(self.tower_spots)
        
        self.selected_spot_idx = 0
        self.selected_tower_type_idx = 0
        self.available_tower_types = [0]
        
        self.space_was_held = False
        self.shift_was_held = False
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.001
        self.game_over = False
        
        self._handle_input(action)
        
        step_reward = self._update_game_state()
        reward += step_reward
        
        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated:
            if self.game_won:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cycle tower spots
        if movement in [1, 4]:  # Right/Up -> Next
            self.selected_spot_idx = (self.selected_spot_idx + 1) % len(self.tower_spots)
        elif movement in [2, 3]:  # Down/Left -> Previous
            self.selected_spot_idx = (self.selected_spot_idx - 1 + len(self.tower_spots)) % len(self.tower_spots)
            
        # Cycle tower types on rising edge of shift
        if shift_held and not self.shift_was_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.available_tower_types)
        self.shift_was_held = shift_held

        # Place tower on rising edge of space
        if space_held and not self.space_was_held:
            self._place_tower()
        self.space_was_held = space_held

    def _place_tower(self):
        spot_idx = self.selected_spot_idx
        tower_type_idx = self.available_tower_types[self.selected_tower_type_idx]
        config = self.tower_configs[tower_type_idx]

        if not self.occupied_spots[spot_idx] and self.resources >= config["cost"]:
            self.resources -= config["cost"]
            self.occupied_spots[spot_idx] = True
            pos = self.tower_spots[spot_idx]
            
            self.towers.append({
                "pos": pygame.Vector2(pos), "type_idx": tower_type_idx,
                "cooldown": 0, "target": None
            })
            # sfx: place_tower
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                self.particles.append(Particle(pos, vel, 20, config["color"], self.COLOR_BG, 5, 0))

    def _update_game_state(self):
        reward = 0
        
        # Wave Management
        if not self.wave_in_progress:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                if self.wave_number >= self.MAX_WAVES:
                    if not self.enemies:
                        self.game_won = True
                else:
                    self._start_next_wave()
        else:
            # Enemy Spawning
            self.spawn_cooldown -= 1
            if self.spawn_cooldown <= 0 and self.wave_spawns:
                enemy_type, spawn_delay = self.wave_spawns.pop(0)
                self._spawn_enemy(enemy_type)
                self.spawn_cooldown = spawn_delay
            
            if not self.wave_spawns and not self.enemies:
                self.wave_in_progress = False
                self.inter_wave_timer = 150 # 5s

        # Update Enemies
        for enemy in self.enemies[:]:
            if enemy["pos"].distance_to(self.path[enemy["waypoint_idx"]]) < enemy["speed"]:
                enemy["waypoint_idx"] += 1
                if enemy["waypoint_idx"] >= len(self.path):
                    self.base_health = max(0, self.base_health - 10)
                    reward -= 10
                    self.enemies.remove(enemy)
                    # sfx: base_hit
                    self._create_explosion(self.base_pos, (255, 0, 0), 30)
                    continue
            
            direction = (self.path[enemy["waypoint_idx"]] - enemy["pos"]).normalize()
            enemy["pos"] += direction * enemy["speed"]
            
        # Update Towers
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            config = self.tower_configs[tower["type_idx"]]
            if tower['cooldown'] <= 0:
                # Find target
                possible_targets = [e for e in self.enemies if tower["pos"].distance_to(e["pos"]) < config["range"]]
                if possible_targets:
                    target = min(possible_targets, key=lambda e: e["waypoint_idx"] + (1 - e["pos"].distance_to(self.path[e["waypoint_idx"]]) / 1000))
                    self.projectiles.append({
                        "pos": pygame.Vector2(tower["pos"]), "target_id": id(target),
                        "type_idx": tower["type_idx"], "speed": 10
                    })
                    tower["cooldown"] = config["fire_rate"]
                    # sfx: tower_shoot

        # Update Projectiles
        for proj in self.projectiles[:]:
            target = next((e for e in self.enemies if id(e) == proj["target_id"]), None)
            if not target:
                self.projectiles.remove(proj)
                continue
            
            if proj["pos"].distance_to(target["pos"]) < proj["speed"]:
                config = self.tower_configs[proj["type_idx"]]
                target["health"] -= config["damage"]
                reward += 0.1
                self.projectiles.remove(proj)
                self._create_explosion(target["pos"], config["color"], 5)
                # sfx: enemy_hit
                
                if target["health"] <= 0:
                    reward += 1
                    e_config = self.enemy_configs[target['type']]
                    self.score += e_config["reward"]
                    self.resources += e_config["reward"] // 2
                    self.enemies.remove(target)
                    self._create_explosion(target["pos"], e_config["color"], 15)
                    # sfx: enemy_destroy
            else:
                direction = (target["pos"] - proj["pos"]).normalize()
                proj["pos"] += direction * proj["speed"]

        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)
                
        return reward

    def _start_next_wave(self):
        self.wave_number += 1
        self.wave_in_progress = True
        
        if self.wave_number % 5 == 0 and len(self.available_tower_types) < len(self.tower_configs):
            self.available_tower_types.append(len(self.available_tower_types))
        
        num_normal = 3 + self.wave_number * 2
        num_fast = max(0, self.wave_number - 3)
        num_tank = max(0, (self.wave_number - 5) // 2)

        spawns = []
        for _ in range(num_normal): spawns.append(("normal", 30))
        for _ in range(num_fast): spawns.append(("fast", 45))
        for _ in range(num_tank): spawns.append(("tank", 60))
        
        random.shuffle(spawns)
        self.wave_spawns = spawns
        self.spawn_cooldown = 30

    def _spawn_enemy(self, enemy_type):
        config = self.enemy_configs[enemy_type]
        difficulty_mod = 1 + (self.wave_number * 0.05)
        
        self.enemies.append({
            "pos": pygame.Vector2(self.path[0]), "type": enemy_type,
            "health": config["health"] * difficulty_mod,
            "max_health": config["health"] * difficulty_mod,
            "speed": config["speed"], "waypoint_idx": 1
        })

    def _check_termination(self):
        return self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.game_won

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [tuple(p) for p in self.path], 30)
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos.x - 15), int(self.base_pos.y), 15, self.COLOR_PATH)

        # Tower spots
        for i, pos in enumerate(self.tower_spots):
            color = self.COLOR_SPOT_VALID
            if self.occupied_spots[i]:
                color = self.COLOR_SPOT_INVALID
            
            s = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (15, 15), 15)
            self.screen.blit(s, (pos[0] - 15, pos[1] - 15))

            if i == self.selected_spot_idx:
                pygame.draw.circle(self.screen, self.COLOR_SPOT_SELECTED, pos, 17, 2)
        
        # Particles
        for p in self.particles:
            p.draw(self.screen)

        # Towers
        for tower in self.towers:
            config = self.tower_configs[tower["type_idx"]]
            pos = (int(tower["pos"].x), int(tower["pos"].y))
            if config["shape"] == "circle":
                pygame.draw.circle(self.screen, config["color"], pos, 12)
                pygame.draw.circle(self.screen, self.COLOR_BG, pos, 8)
            elif config["shape"] == "square":
                pygame.draw.rect(self.screen, config["color"], (pos[0]-10, pos[1]-10, 20, 20))
                pygame.draw.rect(self.screen, self.COLOR_BG, (pos[0]-6, pos[1]-6, 12, 12))
            elif config["shape"] == "triangle":
                points = [(pos[0], pos[1]-12), (pos[0]-12, pos[1]+8), (pos[0]+12, pos[1]+8)]
                pygame.draw.polygon(self.screen, config["color"], points)

        # Projectiles
        for proj in self.projectiles:
            config = self.tower_configs[proj["type_idx"]]
            pygame.draw.circle(self.screen, config["color"], (int(proj["pos"].x), int(proj["pos"].y)), 4)
            
        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            config = self.enemy_configs[enemy["type"]]
            pygame.draw.rect(self.screen, config["color"], (pos[0] - 8, pos[1] - 8, 16, 16))
            
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_w = 20
            pygame.draw.rect(self.screen, (100,0,0), (pos[0] - bar_w/2, pos[1] - 15, bar_w, 4))
            pygame.draw.rect(self.screen, (0,255,0), (pos[0] - bar_w/2, pos[1] - 15, bar_w * health_ratio, 4))

        # Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.WIDTH - 15, self.base_pos.y - 15, 15, 30))

    def _render_ui(self):
        # Base Health Bar
        health_ratio = self.base_health / self.INITIAL_BASE_HEALTH
        bar_w, bar_h = 200, 20
        bar_x, bar_y = self.WIDTH // 2 - bar_w // 2, 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, bar_w * health_ratio, bar_h))
        health_text = self.font_s.render(f"BASE HEALTH: {self.base_health}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_x + bar_w/2 - health_text.get_width()/2, bar_y + bar_h/2 - health_text.get_height()/2))
        
        # Wave & Score
        wave_text = self.font_m.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        score_text = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Resources & Tower Selection
        res_text = self.font_m.render(f"RESOURCES: ${self.resources}", True, (255, 220, 100))
        self.screen.blit(res_text, (self.WIDTH // 2 - res_text.get_width() // 2, self.HEIGHT - 40))
        
        tower_type_idx = self.available_tower_types[self.selected_tower_type_idx]
        config = self.tower_configs[tower_type_idx]
        tower_text = self.font_s.render(f"Build: {config['name']} (${config['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (self.WIDTH // 2 - tower_text.get_width() // 2, self.HEIGHT - 60))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "resources": self.resources}
    
    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, life, color, self.COLOR_BG, 8, 0))

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")