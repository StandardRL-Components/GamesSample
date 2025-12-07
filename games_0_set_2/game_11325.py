import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A tower defense Gymnasium environment where the agent places musical towers
    to defend against waves of enemies. The goal is to survive all waves by
    strategically combining tower effects and using a powerful special attack.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A musical tower defense game where you place towers to stop waves of enemies. "
        "Combine tower abilities to survive as long as possible."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to place the selected tower. "
        "Press shift to cycle to the next tower type."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    GRID_W, GRID_H = SCREEN_WIDTH // GRID_SIZE, SCREEN_HEIGHT // GRID_SIZE
    UI_HEIGHT = 60

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_PATH = (40, 45, 70)
    COLOR_PATH_BORDER = (60, 68, 105)
    COLOR_GRID = (25, 30, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR_VALID = (255, 255, 255, 100)
    COLOR_CURSOR_INVALID = (255, 50, 50, 100)

    TOWER_SPECS = {
        "DRUM": {"color": (255, 80, 80), "cost": 100, "range": 80, "fire_rate": 45, "projectile_speed": 8, "damage": 12},
        "FLUTE": {"color": (80, 150, 255), "cost": 75, "range": 60, "fire_rate": 60, "projectile_speed": 6, "slow_duration": 90, "slow_amount": 0.5},
        "HARP": {"color": (255, 220, 80), "cost": 125, "range": 100, "fire_rate": 30, "resource_gain": 3, "projectile_speed": 10}
    }
    TOWER_TYPES = list(TOWER_SPECS.keys())

    MAX_STEPS = 15000
    MAX_WAVES = 20
    INITIAL_RESOURCES = 250

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 22)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # Path definition
        self.path_points = [
            (-20, 200), (100, 200), (100, 80), (300, 80), (300, 300),
            (540, 300), (540, 150), (self.SCREEN_WIDTH + 20, 150)
        ]
        self._precompute_path()

        # State variables are initialized in reset()
        self.cursor_pos = None
        self.grid = None
        self.resources = None
        self.wave_num = None
        self.wave_spawning_timer = None
        self.enemies_to_spawn = None
        self.enemies = None
        self.towers = None
        self.projectiles = None
        self.particles = None
        self.tower_queue = None
        self.special_attack_cooldown = None
        self.special_attack_active = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.pending_reward = None
        self.enemies_leaked = None

    def _precompute_path(self):
        self.path_segments = []
        self.total_path_length = 0
        for i in range(len(self.path_points) - 1):
            p1 = pygame.Vector2(self.path_points[i])
            p2 = pygame.Vector2(self.path_points[i+1])
            length = p1.distance_to(p2)
            direction = (p2 - p1).normalize() if length > 0 else pygame.Vector2(0)
            self.path_segments.append({"start": p1, "dir": direction, "len": length, "start_dist": self.total_path_length})
            self.total_path_length += length

    def _get_pos_on_path(self, dist):
        dist = max(0, dist)
        for seg in reversed(self.path_segments):
            if dist >= seg["start_dist"]:
                d = dist - seg["start_dist"]
                return seg["start"] + seg["dir"] * d
        return pygame.Vector2(self.path_points[0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cursor_pos = [self.GRID_W // 4, self.GRID_H // 2]
        self.resources = self.INITIAL_RESOURCES
        self.wave_num = 0
        self.wave_spawning_timer = 0
        self.enemies_to_spawn = []
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.tower_queue = deque(self._generate_tower_queue())
        self.special_attack_cooldown = 0
        self.special_attack_active = 0

        self.prev_space_held = 0
        self.prev_shift_held = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.pending_reward = 0.0
        self.enemies_leaked = 0

        self._create_grid_map()
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def _create_grid_map(self):
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                px, py = x * self.GRID_SIZE + self.GRID_SIZE / 2, y * self.GRID_SIZE + self.GRID_SIZE / 2
                for i in range(len(self.path_points) - 1):
                    p1 = pygame.Vector2(self.path_points[i])
                    p2 = pygame.Vector2(self.path_points[i+1])
                    point = pygame.Vector2(px, py)
                    
                    # Check distance to line segment
                    l2 = p1.distance_squared_to(p2)
                    if l2 == 0.0:
                        dist_sq = point.distance_squared_to(p1)
                    else:
                        t = max(0, min(1, (point - p1).dot(p2 - p1) / l2))
                        projection = p1 + t * (p2 - p1)
                        dist_sq = point.distance_squared_to(projection)
                    
                    if dist_sq < (self.GRID_SIZE * 1.2)**2:
                        self.grid[x][y] = 1  # Path tile, unbuildable
                        break

    def _generate_tower_queue(self):
        # Provides a predictable, repeating sequence of towers
        base_sequence = ["DRUM", "FLUTE", "DRUM", "HARP"]
        return [random.choice(base_sequence) for _ in range(100)]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.pending_reward = 0.0
        self.steps += 1

        # --- Handle Actions ---
        movement, space_action, shift_action = action[0], action[1], action[2]
        space_press = space_action and not self.prev_space_held
        shift_press = shift_action and not self.prev_shift_held

        self._handle_movement(movement)
        if space_press: self._action_place_tower()
        if shift_press: self._action_cycle_tower() # Adaptation: shift cycles tower type

        # --- Update Game State ---
        self._update_spawning()
        self._update_enemies()
        self._update_towers()
        self._update_projectiles()
        self._update_particles()
        
        self.pending_reward -= len(self.enemies) * 0.001 # Small penalty for existing enemies

        if not self.enemies and not self.enemies_to_spawn and self.wave_num > 0:
            if self.wave_num < self.MAX_WAVES:
                self.pending_reward += 10
                self.score += 100
                self._start_next_wave()
        
        # --- Update Cooldowns ---
        if self.special_attack_cooldown > 0: self.special_attack_cooldown -= 1
        if self.special_attack_active > 0: self.special_attack_active -= 1
        
        # --- Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.enemies_leaked > 0:
                self.pending_reward -= 50
            elif self.wave_num >= self.MAX_WAVES:
                self.pending_reward += 100
                self.score += 1000
            self.game_over = True

        self.prev_space_held = space_action
        self.prev_shift_held = shift_action

        return self._get_observation(), self.pending_reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, (self.SCREEN_HEIGHT - self.UI_HEIGHT) // self.GRID_SIZE - 1)

    def _action_place_tower(self):
        x, y = self.cursor_pos
        if self.grid[x][y] == 0:  # If not on path or another tower
            tower_type = self.tower_queue[0]
            cost = self.TOWER_SPECS[tower_type]["cost"]
            if self.resources >= cost:
                self.resources -= cost
                self.grid[x][y] = 2  # Mark as tower
                pos = (x * self.GRID_SIZE + self.GRID_SIZE / 2, y * self.GRID_SIZE + self.GRID_SIZE / 2)
                self.towers.append({
                    "type": tower_type, "pos": pos, "cooldown": 0,
                    "spec": self.TOWER_SPECS[tower_type], "anim_timer": 30
                })
                self.tower_queue.popleft()
                self.pending_reward += 0.5
                self.score += 10
                # sfx: tower_place.wav
                self._create_particles(pos, self.TOWER_SPECS[tower_type]['color'], 15, 2.0)

    def _action_cycle_tower(self):
        # Adaptation: Shift cycles the next tower in the queue
        self.tower_queue.rotate(-1)
        # sfx: ui_cycle.wav

    def _start_next_wave(self):
        self.wave_num += 1
        if self.wave_num > self.MAX_WAVES: return
        
        num_enemies = 5 + self.wave_num * 2
        base_health = 10 + (self.wave_num - 1) * 5
        base_speed = 1.0 + (self.wave_num - 1) * 0.02
        value = 5 + self.wave_num

        self.enemies_to_spawn = []
        for i in range(num_enemies):
            self.enemies_to_spawn.append({
                "health": base_health * (1 + random.uniform(-0.1, 0.1)),
                "max_health": base_health,
                "speed": base_speed * (1 + random.uniform(-0.1, 0.1)),
                "value": value,
                "dist": -i * self.GRID_SIZE * 1.5, # Stagger spawn
                "pos": self.path_points[0],
                "slow_timer": 0
            })
        self.wave_spawning_timer = 0

    def _update_spawning(self):
        if self.enemies_to_spawn:
            self.wave_spawning_timer -= 1
            if self.wave_spawning_timer <= 0:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.wave_spawning_timer = 30 # Time between spawns

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            speed_multiplier = 1.0
            if enemy["slow_timer"] > 0:
                enemy["slow_timer"] -= 1
                # Find the strongest slow effect if multiple exist
                strongest_slow = 1.0
                for tower_type in self.TOWER_TYPES:
                    if "slow_amount" in self.TOWER_SPECS[tower_type]:
                         if f"slow_{tower_type}" in enemy and enemy[f"slow_{tower_type}"] > 0:
                            enemy[f"slow_{tower_type}"] -= 1
                            strongest_slow = min(strongest_slow, self.TOWER_SPECS[tower_type]["slow_amount"])
                speed_multiplier = strongest_slow

            enemy["dist"] += enemy["speed"] * speed_multiplier
            enemy["pos"] = self._get_pos_on_path(enemy["dist"])

            if enemy["dist"] >= self.total_path_length:
                self.enemies.remove(enemy)
                self.enemies_leaked += 1
                # sfx: enemy_leak.wav

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
            else:
                target = self._find_target(tower)
                if target:
                    self._fire_projectile(tower, target)
                    tower["cooldown"] = tower["spec"]["fire_rate"]
                    tower["anim_timer"] = 15 # For visual feedback

    def _find_target(self, tower):
        targets = []
        for enemy in self.enemies:
            dist = pygame.Vector2(tower["pos"]).distance_to(enemy["pos"])
            if dist <= tower["spec"]["range"]:
                targets.append(enemy)
        
        if not targets: return None
        # Target enemy furthest along the path
        return max(targets, key=lambda e: e["dist"])

    def _fire_projectile(self, tower, target):
        self.projectiles.append({
            "start_pos": pygame.Vector2(tower["pos"]),
            "pos": pygame.Vector2(tower["pos"]),
            "target": target,
            "type": tower["type"],
            "spec": tower["spec"]
        })
        # sfx: fire_drum.wav or fire_flute.wav etc.

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            target_pos = pygame.Vector2(proj["target"]["pos"])
            proj_pos = pygame.Vector2(proj["pos"])
            direction = (target_pos - proj_pos).normalize() if target_pos != proj_pos else pygame.Vector2(0)
            proj["pos"] += direction * proj["spec"]["projectile_speed"]

            if proj["target"] not in self.enemies or pygame.Vector2(proj["pos"]).distance_to(target_pos) < 10:
                self._hit_target(proj)
                self.projectiles.remove(proj)

    def _hit_target(self, proj):
        target = proj["target"]
        if target not in self.enemies: return

        spec = proj["spec"]
        self.pending_reward += 0.01
        self._create_particles(target["pos"], spec["color"], 10, 1.5)

        if proj["type"] == "DRUM":
            target["health"] -= spec["damage"]
        elif proj["type"] == "FLUTE":
            target["slow_timer"] = max(target["slow_timer"], spec["slow_duration"])
            slow_key = f"slow_{proj['type']}"
            target[slow_key] = spec["slow_duration"]
        elif proj["type"] == "HARP":
            self.resources += spec["resource_gain"]
            self.score += spec["resource_gain"]
            self.pending_reward += 0.05

        if "health" in target and target["health"] <= 0:
            self.pending_reward += 1.0
            self.score += 25
            self.resources += target["value"]
            self._create_particles(target["pos"], (255, 255, 255), 25, 3.0)
            self.enemies.remove(target)
            # sfx: enemy_die.wav

    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["vel"] *= 0.95 # Drag
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5) * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": random.randint(15, 30),
                "color": color,
                "radius": random.uniform(1, 4)
            })

    def _check_termination(self):
        return (self.enemies_leaked > 0 or
                (self.wave_num >= self.MAX_WAVES and not self.enemies and not self.enemies_to_spawn) or
                self.steps >= self.MAX_STEPS)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_num, "resources": self.resources, "leaked": self.enemies_leaked}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_path()
        # self._render_grid() # Optional: for debugging
        for tower in self.towers: self._render_tower(tower)
        for proj in self.projectiles: self._render_projectile(proj)
        for enemy in self.enemies: self._render_enemy(enemy)
        for particle in self.particles: self._render_particle(particle)
        self._render_cursor()
        self._render_ui()

    def _render_path(self):
        for i in range(len(self.path_points) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, self.path_points[i], self.path_points[i+1], self.GRID_SIZE * 2 + 4)
        for i in range(len(self.path_points) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_points[i], self.path_points[i+1], self.GRID_SIZE * 2)

    def _render_grid(self):
        for x in range(self.GRID_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.GRID_SIZE, 0), (x * self.GRID_SIZE, self.SCREEN_HEIGHT - self.UI_HEIGHT))
        for y in range((self.SCREEN_HEIGHT - self.UI_HEIGHT) // self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.GRID_SIZE), (self.SCREEN_WIDTH, y * self.GRID_SIZE))
            
    def _render_tower(self, tower):
        pos = (int(tower["pos"][0]), int(tower["pos"][1]))
        spec = tower["spec"]
        color = spec["color"]
        
        # Glow effect
        glow_radius = int(self.GRID_SIZE * 0.6)
        if tower["anim_timer"] > 0:
            glow_radius = int(self.GRID_SIZE * 0.6 * (1 + (15 - tower["anim_timer"]) / 15 * 0.5))
            tower["anim_timer"] -= 1
        
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

        if tower["type"] == "DRUM":
            pygame.draw.circle(self.screen, color, pos, int(self.GRID_SIZE * 0.4))
        elif tower["type"] == "FLUTE":
            rect = pygame.Rect(0, 0, self.GRID_SIZE * 0.8, self.GRID_SIZE * 0.3)
            rect.center = pos
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
        elif tower["type"] == "HARP":
            points = [(pos[0], pos[1] - self.GRID_SIZE*0.4), (pos[0] - self.GRID_SIZE*0.3, pos[1] + self.GRID_SIZE*0.3), (pos[0] + self.GRID_SIZE*0.3, pos[1] + self.GRID_SIZE*0.3)]
            pygame.draw.polygon(self.screen, color, points)

    def _render_enemy(self, enemy):
        pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
        size = int(self.GRID_SIZE * 0.35)
        
        color = (200, 50, 150) # Base magenta color
        if enemy["slow_timer"] > 0:
            color = (100, 150, 220) # Blue when slowed

        pygame.draw.rect(self.screen, color, (pos[0]-size, pos[1]-size, size*2, size*2))
        
        # Health bar
        health_ratio = enemy["health"] / enemy["max_health"]
        bar_w = self.GRID_SIZE * 0.8
        pygame.draw.rect(self.screen, (80,0,0), (pos[0] - bar_w/2, pos[1] - size - 8, bar_w, 5))
        pygame.draw.rect(self.screen, (0,200,0), (pos[0] - bar_w/2, pos[1] - size - 8, bar_w * health_ratio, 5))

    def _render_projectile(self, proj):
        pos = (int(proj["pos"][0]), int(proj["pos"][1]))
        color = proj["spec"]["color"]
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, color)

    def _render_particle(self, p):
        life_ratio = p["life"] / 30.0
        radius = int(p["radius"] * life_ratio)
        if radius < 1: return
        pos = (int(p["pos"][0]), int(p["pos"][1]))
        color = p["color"]
        
        # Use alpha blending for fade effect
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, int(255 * life_ratio)), (radius, radius), radius)
        self.screen.blit(s, (pos[0] - radius, pos[1] - radius))

    def _render_cursor(self):
        gx, gy = self.cursor_pos
        is_valid = self.grid[gx][gy] == 0
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        rect = (gx * self.GRID_SIZE, gy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        s.fill(color)
        self.screen.blit(s, rect[:2])

    def _render_ui(self):
        ui_rect = (0, self.SCREEN_HEIGHT - self.UI_HEIGHT, self.SCREEN_WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, (10, 12, 22), ui_rect)
        pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, (0, self.SCREEN_HEIGHT - self.UI_HEIGHT), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.UI_HEIGHT), 2)
        
        # Wave info
        wave_text = self.font_medium.render(f"Wave: {self.wave_num}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (15, self.SCREEN_HEIGHT - self.UI_HEIGHT + 10))

        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, self.SCREEN_HEIGHT - self.UI_HEIGHT + 35))

        # Resources
        res_text = self.font_medium.render(f"Resources: {self.resources}", True, self.TOWER_SPECS["HARP"]["color"])
        res_rect = res_text.get_rect(right=self.SCREEN_WIDTH - 15, centery=self.SCREEN_HEIGHT - self.UI_HEIGHT / 2)
        self.screen.blit(res_text, res_rect)

        # Next Tower
        if self.tower_queue:
            next_tower_type = self.tower_queue[0]
            spec = self.TOWER_SPECS[next_tower_type]
            text = self.font_small.render("Next Tower:", True, self.COLOR_TEXT)
            self.screen.blit(text, (self.SCREEN_WIDTH / 2 - 100, self.SCREEN_HEIGHT - self.UI_HEIGHT + 10))
            
            # Draw a preview of the tower
            preview_pos = (self.SCREEN_WIDTH / 2 - 60, self.SCREEN_HEIGHT - self.UI_HEIGHT + 40)
            self._render_tower({"pos": preview_pos, "type": next_tower_type, "spec": spec, "anim_timer": 0})
            
            cost_text = self.font_medium.render(f"{spec['cost']}", True, self.COLOR_TEXT)
            self.screen.blit(cost_text, (self.SCREEN_WIDTH / 2 - 35, self.SCREEN_HEIGHT - self.UI_HEIGHT + 28))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "YOU WON" if self.enemies_leaked == 0 else "GAME OVER"
            text = self.font_large.render(msg, True, (255,255,255))
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Musical Tower Defense")
        clock = pygame.time.Clock()
        
        terminated = False
        while not terminated:
            movement = 0 # no-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit frame rate
            
        env.close()
    except pygame.error as e:
        print(f"Could not run graphical test: {e}")
        print("This is expected in a headless environment.")