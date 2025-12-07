import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend your village hall from waves of monsters by building walls and towers. Aim your towers and manage resources to survive the night."
    user_guide = "Use arrow keys to move the cursor. Press space to build the selected structure. Hold shift to aim and fire from the nearest tower. Press space and shift together to cycle between structures."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_popup = pygame.font.SysFont("Consolas", 14, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 30, 35)
        self.COLOR_GRASS = (46, 70, 54)
        self.COLOR_TREE = (25, 50, 30)
        self.COLOR_HALL = (160, 82, 45)
        self.COLOR_HALL_ROOF = (139, 69, 19)
        self.COLOR_PLAYER_CURSOR = (255, 255, 0)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_ENEMY_HEALTH = (220, 20, 60)
        self.COLOR_WALL = (100, 100, 100)
        self.COLOR_TOWER = (50, 100, 200)
        self.COLOR_PROJECTILE = (100, 180, 255)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BACK = (0, 0, 0, 128)
        self.COLOR_INVALID = (255, 0, 0, 100)
        
        # Grid settings for pathfinding
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE

        # Game constants
        self.MAX_STEPS = 1000
        self.PLAYER_CURSOR_SPEED = 8
        
        # Structure definitions
        self.structure_definitions = {
            "WALL": {"cost": 10, "health": 100, "size": self.GRID_SIZE, "color": self.COLOR_WALL},
            "TOWER": {"cost": 30, "health": 50, "size": self.GRID_SIZE, "color": self.COLOR_TOWER, "range": 120, "cooldown_max": 45, "damage": 10},
        }
        self.unlock_schedule = {
            # 250: "SLOW_TOWER", # Example for future expansion
        }
        
        # Initialize state variables
        self.cursor_pos = None
        self.steps = None
        self.score = None
        self.resources = None
        self.village_hall = None
        self.monsters = None
        self.structures = None
        self.projectiles = None
        self.particles = None
        self.popups = None
        self.available_structures = None
        self.selected_structure_idx = None
        self.grid = None
        self.flow_field = None
        self.last_action_was_cycle = False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0.0
        self.resources = 50
        self.cursor_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        
        # Game entities
        self.monsters = []
        self.structures = []
        self.projectiles = []
        self.particles = []
        self.popups = []

        # Village Hall setup
        hall_size = self.GRID_SIZE * 3
        self.village_hall = {
            "rect": pygame.Rect((self.WIDTH - hall_size) // 2, (self.HEIGHT - hall_size) // 2, hall_size, hall_size),
            "health": 200,
            "max_health": 200
        }
        
        # Pathfinding grid
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        hall_grid_x = int(self.village_hall['rect'].centerx / self.GRID_SIZE)
        hall_grid_y = int(self.village_hall['rect'].centery / self.GRID_SIZE)
        # Mark hall area as blocked
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= hall_grid_x + i < self.GRID_W and 0 <= hall_grid_y + j < self.GRID_H:
                    self.grid[hall_grid_x + i, hall_grid_y + j] = 1
        
        self._update_flow_field()

        # Procedural background elements
        self.procedural_trees = []
        for _ in range(30):
            x = self.np_random.integers(0, self.WIDTH)
            y = self.np_random.integers(0, self.HEIGHT)
            if not self.village_hall['rect'].collidepoint(x, y):
                self.procedural_trees.append((x, y, self.np_random.integers(5, 15)))

        # Structure selection
        self.available_structures = ["WALL", "TOWER"]
        self.selected_structure_idx = 0
        self.last_action_was_cycle = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0

        # --- 1. Handle Player Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._move_cursor(movement)
        
        # Combined action: Cycle selected structure
        if space_held and shift_held:
            if not self.last_action_was_cycle:
                self._cycle_structure()
                self.last_action_was_cycle = True
        # Place structure
        elif space_held:
            self._place_structure()
            self.last_action_was_cycle = False
        # Fire projectile
        elif shift_held:
            self._fire_projectile()
            self.last_action_was_cycle = False
        else:
            self.last_action_was_cycle = False

        # --- 2. Update Game Logic ---
        self.steps += 1
        
        # Monster spawning
        self._spawn_monsters()
        
        # Update structures (e.g., tower cooldowns)
        self._update_structures()
        
        # Update monsters (movement, attacks)
        reward += self._update_monsters()
        
        # Update projectiles and check for hits
        reward += self._update_projectiles()

        # Update visual effects
        self._update_particles()
        self._update_popups()

        # --- 3. Check for Termination ---
        terminated = False
        truncated = False
        if self.village_hall["health"] <= 0:
            terminated = True
            reward -= 100.0
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            if not terminated: # Don't give survival bonus if already lost
                reward += 100.0

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Action Handling ---
    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos.y -= self.PLAYER_CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.PLAYER_CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.PLAYER_CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.PLAYER_CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

    def _cycle_structure(self):
        self.selected_structure_idx = (self.selected_structure_idx + 1) % len(self.available_structures)

    def _place_structure(self):
        s_type = self.available_structures[self.selected_structure_idx]
        s_def = self.structure_definitions[s_type]
        
        if self.resources >= s_def["cost"]:
            grid_x = int(self.cursor_pos.x / self.GRID_SIZE)
            grid_y = int(self.cursor_pos.y / self.GRID_SIZE)
            
            if 0 <= grid_x < self.GRID_W and 0 <= grid_y < self.GRID_H and self.grid[grid_x, grid_y] == 0:
                self.resources -= s_def["cost"]
                new_struct = {
                    "type": s_type,
                    "pos": pygame.Vector2(grid_x * self.GRID_SIZE, grid_y * self.GRID_SIZE),
                    "health": s_def["health"],
                    "rect": pygame.Rect(grid_x * self.GRID_SIZE, grid_y * self.GRID_SIZE, s_def["size"], s_def["size"]),
                }
                if s_type == "TOWER":
                    new_struct["cooldown"] = 0
                
                self.structures.append(new_struct)
                self.grid[grid_x, grid_y] = 1
                self._update_flow_field()
    
    def _fire_projectile(self):
        ready_towers = [s for s in self.structures if s["type"] == "TOWER" and s["cooldown"] <= 0]
        if not ready_towers:
            return
            
        tower = min(ready_towers, key=lambda t: t["pos"].distance_to(self.cursor_pos))
        
        if tower["pos"].distance_to(self.cursor_pos) <= self.structure_definitions["TOWER"]["range"]:
            s_def = self.structure_definitions["TOWER"]
            tower["cooldown"] = s_def["cooldown_max"]
            
            start_pos = tower["pos"] + pygame.Vector2(self.GRID_SIZE/2, self.GRID_SIZE/2)
            direction_vec = self.cursor_pos - start_pos
            
            if direction_vec.length() > 0:
                direction = direction_vec.normalize()
                self.projectiles.append({
                    "pos": start_pos,
                    "vel": direction * 12,
                    "damage": s_def["damage"],
                    "lifespan": 60
                })

    # --- Game Logic Updates ---
    def _spawn_monsters(self):
        spawn_chance = 0.02 + (self.steps / self.MAX_STEPS) * 0.05
        if self.np_random.random() < spawn_chance:
            side = self.np_random.integers(0, 4)
            if side == 0: pos = pygame.Vector2(0, self.np_random.uniform(0, self.HEIGHT))
            elif side == 1: pos = pygame.Vector2(self.WIDTH, self.np_random.uniform(0, self.HEIGHT))
            elif side == 2: pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), 0)
            else: pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT)
            
            health_multiplier = 1.0 + (self.steps / 200) * 0.02
            
            self.monsters.append({
                "pos": pos,
                "health": int(20 * health_multiplier),
                "max_health": int(20 * health_multiplier),
                "speed": self.np_random.uniform(1.0, 1.5),
                "size": 8
            })

    def _update_structures(self):
        for s in self.structures:
            if s.get("cooldown", -1) > 0:
                s["cooldown"] -= 1

    def _update_monsters(self):
        reward = 0.0
        for m in self.monsters:
            if self.village_hall["rect"].collidepoint(m["pos"]):
                damage = 1
                self.village_hall["health"] -= damage
                reward -= 0.1 * damage
                m["health"] = 0
                self._create_particles(pygame.Vector2(self.village_hall["rect"].center), self.COLOR_ENEMY, 10)
            else:
                grid_x = int(m["pos"].x / self.GRID_SIZE)
                grid_y = int(m["pos"].y / self.GRID_SIZE)
                if 0 <= grid_x < self.GRID_W and 0 <= grid_y < self.GRID_H:
                    direction = self.flow_field[grid_x, grid_y]
                    if np.any(direction):
                        m["pos"] += pygame.Vector2(direction) * m["speed"]
        return reward

    def _update_projectiles(self):
        reward = 0.0
        projectiles_to_keep = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            
            hit = False
            if p["lifespan"] > 0 and (0 < p["pos"].x < self.WIDTH) and (0 < p["pos"].y < self.HEIGHT):
                for m in self.monsters:
                    if m["health"] > 0 and (m["pos"] - p["pos"]).length() < m["size"] + 4:
                        m["health"] -= p["damage"]
                        reward += 0.1
                        self._create_particles(p["pos"], self.COLOR_PROJECTILE, 5)
                        if m["health"] <= 0:
                            reward += 1.0
                            self.resources += 5
                            self._create_particles(m["pos"], self.COLOR_ENEMY, 20)
                            self.popups.append({"pos": m["pos"].copy(), "text": "+5", "life": 30, "color": self.COLOR_PLAYER_CURSOR})
                        hit = True
                        break
                if not hit:
                    projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep
        return reward
    
    def _update_flow_field(self):
        target_x = int(self.village_hall['rect'].centerx / self.GRID_SIZE)
        target_y = int(self.village_hall['rect'].centery / self.GRID_SIZE)

        q = deque([(target_x, target_y, 0)])
        distances = np.full((self.GRID_W, self.GRID_H), np.inf)
        if 0 <= target_x < self.GRID_W and 0 <= target_y < self.GRID_H:
            distances[target_x, target_y] = 0

        while q:
            x, y, dist = q.popleft()
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid[nx, ny] == 0 and distances[nx, ny] == np.inf:
                    distances[nx, ny] = dist + 1
                    q.append((nx, ny, dist + 1))
        
        flow = np.zeros((self.GRID_W, self.GRID_H, 2))
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if self.grid[x, y] == 0:
                    min_dist = np.inf
                    best_dir = (0, 0)
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and distances[nx, ny] < min_dist:
                            min_dist = distances[nx, ny]
                            best_dir = (dx, dy)
                    
                    length = math.sqrt(best_dir[0]**2 + best_dir[1]**2)
                    if length > 0:
                        flow[x, y] = (-best_dir[0] / length, -best_dir[1] / length)

        self.flow_field = flow

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_structures()
        self._render_monsters()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        self._render_popups()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    # --- Rendering ---
    def _render_background(self):
        self.screen.fill(self.COLOR_GRASS)
        for x, y, r in self.procedural_trees:
            pygame.draw.circle(self.screen, self.COLOR_TREE, (x, y), r)
        pygame.draw.rect(self.screen, self.COLOR_HALL_ROOF, self.village_hall["rect"])
        pygame.draw.rect(self.screen, self.COLOR_HALL, self.village_hall["rect"].inflate(-10, -10))

    def _render_structures(self):
        for s in self.structures:
            s_def = self.structure_definitions[s["type"]]
            pygame.draw.rect(self.screen, s_def["color"], s["rect"])
            pygame.draw.rect(self.screen, (0,0,0), s["rect"], 1)

    def _render_monsters(self):
        self.monsters = [m for m in self.monsters if m["health"] > 0]
        for m in self.monsters:
            pos = (int(m["pos"].x), int(m["pos"].y))
            size = int(m["size"] + math.sin(self.steps * 0.2 + m["pos"].x) * 2)
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, size)
            if m["health"] < m["max_health"]:
                bar_w = 20
                bar_h = 3
                health_pct = m["health"] / m["max_health"]
                pygame.draw.rect(self.screen, (50,0,0), (pos[0] - bar_w/2, pos[1] - size - 8, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (pos[0] - bar_w/2, pos[1] - size - 8, bar_w * health_pct, bar_h))

    def _render_projectiles(self):
        for p in self.projectiles:
            start = (int(p["pos"].x), int(p["pos"].y))
            end = (int(p["pos"].x - p["vel"].x * 0.8), int(p["pos"].y - p["vel"].y * 0.8))
            pygame.draw.line(self.screen, (*self.COLOR_PROJECTILE, 50), start, end, 8)
            pygame.gfxdraw.line(self.screen, *start, *end, self.COLOR_PROJECTILE)

    def _render_cursor(self):
        pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        
        s_type_name = self.available_structures[self.selected_structure_idx]
        s_def = self.structure_definitions[s_type_name]
        grid_x = int(self.cursor_pos.x / self.GRID_SIZE)
        grid_y = int(self.cursor_pos.y / self.GRID_SIZE)
        
        preview_rect = pygame.Rect(grid_x * self.GRID_SIZE, grid_y * self.GRID_SIZE, s_def["size"], s_def["size"])
        preview_surf = pygame.Surface((s_def["size"], s_def["size"]), pygame.SRCALPHA)

        is_valid_spot = 0 <= grid_x < self.GRID_W and 0 <= grid_y < self.GRID_H and self.grid[grid_x, grid_y] == 0
        has_res = self.resources >= s_def["cost"]

        if is_valid_spot and has_res:
            preview_surf.fill((*s_def["color"], 128))
        else:
            preview_surf.fill(self.COLOR_INVALID)
        
        self.screen.blit(preview_surf, preview_rect.topleft)

        pygame.gfxdraw.aacircle(self.screen, *pos, 10, self.COLOR_PLAYER_CURSOR)
        pygame.draw.line(self.screen, self.COLOR_PLAYER_CURSOR, (pos[0]-5, pos[1]), (pos[0]+5, pos[1]))
        pygame.draw.line(self.screen, self.COLOR_PLAYER_CURSOR, (pos[0], pos[1]-5), (pos[0], pos[1]+5))

    def _render_ui(self):
        panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        panel.fill(self.COLOR_UI_BACK)
        self.screen.blit(panel, (0, 0))

        health_text = self.font_small.render(f"HALL HP: {self.village_hall['health']}/{self.village_hall['max_health']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 12))

        res_text = self.font_small.render(f"RESOURCES: {self.resources}", True, self.COLOR_PLAYER_CURSOR)
        self.screen.blit(res_text, (220, 12))

        time_text = self.font_small.render(f"NIGHT: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (380, 12))

        s_type_name = self.available_structures[self.selected_structure_idx]
        s_def = self.structure_definitions[s_type_name]
        sel_text = self.font_small.render(f"BUILD: {s_type_name} (Cost: {s_def['cost']})", True, self.COLOR_UI_TEXT)
        self.screen.blit(sel_text, (self.WIDTH - sel_text.get_width() - 10, 12))

    # --- Visual Effects ---
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": 20, "color": color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 1]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["life"] -= 1

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = (*p["color"], alpha)
            size = int(p["life"] / 5)
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p["pos"].x-size), int(p["pos"].y-size)), special_flags=pygame.BLEND_RGBA_ADD)
    
    def _update_popups(self):
        self.popups = [p for p in self.popups if p['life'] > 0]
        for p in self.popups:
            p['pos'].y -= 0.5
            p['life'] -= 1

    def _render_popups(self):
        for p in self.popups:
            alpha = int(255 * (p['life'] / 30))
            text_surf = self.font_popup.render(p['text'], True, p['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(p['pos'].x), int(p['pos'].y)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "hall_health": self.village_hall["health"],
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Village Defense")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("----------------------\n")
    
    while not done:
        movement = 0
        space = 0
        shift = 0
        
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
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
        if done:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")

    env.close()