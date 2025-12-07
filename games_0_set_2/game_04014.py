
# Generated: 2025-08-28T01:07:19.231013
# Source Brief: brief_04014.md
# Brief Index: 4014

        
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


# Helper class for particles
class Particle:
    def __init__(self, pos, vel, color, size, lifetime):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.life = float(lifetime)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        self.vel[1] += 0.05 # a little gravity

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.lifetime))
            s = self.size * (self.life / self.lifetime)
            if s > 0:
                rect = pygame.Rect(self.pos[0] - s/2, self.pos[1] - s/2, s, s)
                shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, (*self.color, alpha), shape_surf.get_rect())
                surface.blit(shape_surf, rect)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor, Space to place tower, Shift to cycle tower type."
    )

    game_description = (
        "Defend your base from increasingly difficult waves of enemies by strategically placing defensive towers."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    GRID_CELL_SIZE = 40
    MAX_STEPS = 5000 # Increased to allow for 20 waves
    WIN_WAVE = 20

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_PATH = (60, 60, 80)
    COLOR_BASE = (255, 200, 0)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_PROJECTILE_MG = (0, 200, 255)
    COLOR_PROJECTILE_CANNON = (255, 150, 0)
    COLOR_TEXT = (230, 230, 230)
    COLOR_UI_BG = (30, 30, 45, 180)
    COLOR_VALID_CURSOR = (50, 255, 50)
    COLOR_INVALID_CURSOR = (255, 50, 50)

    # Tower Specs
    TOWER_SPECS = {
        "Machine Gun": {"cost": 100, "range": 80, "damage": 2, "fire_rate": 5, "proj_speed": 10, "color": (0, 180, 220)},
        "Cannon": {"cost": 250, "range": 120, "damage": 25, "fire_rate": 45, "proj_speed": 7, "color": (255, 120, 0)},
    }
    TOWER_TYPES = list(TOWER_SPECS.keys())

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.path = self._define_path()
        self.buildable_grid = self._create_buildable_grid()

        self.last_action = np.zeros(self.action_space.shape)
        
        self.reset()
        
        # self.validate_implementation() # Optional: run self-check

    def _define_path(self):
        path = []
        for i in range(20): path.append((-20 + i * 20, 180))
        for i in range(10): path.append((380, 180 - i * 10))
        for i in range(12): path.append((380 - i * 20, 80))
        for i in range(22): path.append((140, 80 + i * 10))
        for i in range(26): path.append((140 + i * 20, 300))
        return path

    def _create_buildable_grid(self):
        grid = np.ones((self.GRID_COLS, self.GRID_ROWS), dtype=bool)
        for i in range(self.GRID_COLS):
            for j in range(self.GRID_ROWS):
                gx, gy = i * self.GRID_CELL_SIZE, j * self.GRID_CELL_SIZE
                # Check if cell center is too close to path
                cell_center = pygame.Vector2(gx + self.GRID_CELL_SIZE / 2, gy + self.GRID_CELL_SIZE / 2)
                for k in range(len(self.path) - 1):
                    p1 = pygame.Vector2(self.path[k])
                    p2 = pygame.Vector2(self.path[k+1])
                    if self._dist_point_to_segment(cell_center, p1, p2) < self.GRID_CELL_SIZE * 0.8:
                        grid[i, j] = False
                        break
        # Make base area unbuildable
        base_gx, base_gy = int(self.path[-1][0] / self.GRID_CELL_SIZE), int(self.path[-1][1] / self.GRID_CELL_SIZE)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= base_gx + i < self.GRID_COLS and 0 <= base_gy + j < self.GRID_ROWS:
                    grid[base_gx + i, base_gy + j] = False
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.base_health = 100
        self.max_base_health = 100
        self.money = 300
        self.wave_number = 0
        self.game_state = "INTER_WAVE" # INTER_WAVE, WAVE_IN_PROGRESS
        self.wave_cooldown = 150 # 5 seconds at 30fps

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_idx = 0
        self.last_action = np.zeros(self.action_space.shape)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        
        # --- Handle player input (discrete press events) ---
        movement, space_action, shift_action = action[0], action[1], action[2]
        space_pressed = space_action == 1 and self.last_action[1] == 0
        shift_pressed = shift_action == 1 and self.last_action[2] == 0
        self.last_action = action

        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        if movement == 2 and self.cursor_pos[1] < self.GRID_ROWS - 1: self.cursor_pos[1] += 1
        if movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        if movement == 4 and self.cursor_pos[0] < self.GRID_COLS - 1: self.cursor_pos[0] += 1

        if shift_pressed:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_TYPES)

        if space_pressed:
            tower_type = self.TOWER_TYPES[self.selected_tower_idx]
            spec = self.TOWER_SPECS[tower_type]
            if self.money >= spec["cost"] and self.buildable_grid[self.cursor_pos[0], self.cursor_pos[1]]:
                self.money -= spec["cost"]
                # SFX: place_tower.wav
                self.towers.append({
                    "pos": (self.cursor_pos[0] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2, 
                            self.cursor_pos[1] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2),
                    "type": tower_type,
                    "cooldown": 0,
                    **spec
                })
                self.buildable_grid[self.cursor_pos[0], self.cursor_pos[1]] = False


        # --- Update game logic ---
        self._update_wave_manager()
        killed_enemies, reward_from_kills = self._update_projectiles_and_towers()
        reward += reward_from_kills
        
        leaked_enemies = self._update_enemies()
        self.base_health -= leaked_enemies * 10
        self.base_health = max(0, self.base_health)

        self._update_particles()
        
        # --- Check for wave completion ---
        if self.game_state == "WAVE_IN_PROGRESS" and not self.enemies:
            self.game_state = "INTER_WAVE"
            self.wave_cooldown = 240 # 8 seconds
            if self.wave_number > 0:
                reward += 1.0

        # --- Termination and final rewards ---
        terminated = False
        if self.base_health <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
        elif self.wave_number > self.WIN_WAVE:
            reward = 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.score += reward
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_wave_manager(self):
        if self.game_state == "INTER_WAVE":
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self.game_state = "WAVE_IN_PROGRESS"
                self.wave_number += 1
                self._spawn_wave()
    
    def _spawn_wave(self):
        # SFX: wave_start.wav
        num_enemies = 3 + (self.wave_number - 1) // 2
        health = 10 * (1.05 ** (self.wave_number - 1))
        speed = 1.0 * (1.02 ** (self.wave_number - 1))
        
        for i in range(num_enemies):
            self.enemies.append({
                "pos": pygame.Vector2(self.path[0]),
                "offset": (self.np_random.random() - 0.5) * 15,
                "path_idx": 0,
                "health": health,
                "max_health": health,
                "speed": speed,
                "dist_traveled": -i * 25, # Stagger enemies
            })
    
    def _update_projectiles_and_towers(self):
        # --- Update projectiles ---
        reward = 0
        killed_enemies = 0
        for p in self.projectiles[:]:
            target_pos = next((e["pos"] for e in self.enemies if e == p["target"]), None)
            if not target_pos:
                self.projectiles.remove(p)
                continue

            direction = (target_pos - p["pos"]).normalize()
            p["pos"] += direction * p["speed"]

            if p["pos"].distance_to(target_pos) < 5:
                # SFX: hit_sound.wav
                p["target"]["health"] -= p["damage"]
                self._create_hit_particles(p["pos"], p["color"])
                if p["target"]["health"] <= 0:
                    # SFX: enemy_die.wav
                    reward += 0.1
                    killed_enemies += 1
                    self.money += 20 + self.wave_number
                    self._create_explosion(p["target"]["pos"], self.COLOR_ENEMY)
                    self.enemies.remove(p["target"])
                self.projectiles.remove(p)

        # --- Update towers ---
        for t in self.towers:
            if t["cooldown"] > 0:
                t["cooldown"] -= 1
                continue
            
            # Find target
            potential_targets = [e for e in self.enemies if pygame.Vector2(t["pos"]).distance_to(e["pos"]) < t["range"]]
            if potential_targets:
                target = max(potential_targets, key=lambda e: e["dist_traveled"]) # Target enemy furthest along path
                t["cooldown"] = t["fire_rate"]
                # SFX: shoot.wav
                proj_color = self.COLOR_PROJECTILE_MG if t["type"] == "Machine Gun" else self.COLOR_PROJECTILE_CANNON
                self.projectiles.append({
                    "pos": pygame.Vector2(t["pos"]),
                    "target": target,
                    "speed": t["proj_speed"],
                    "damage": t["damage"],
                    "color": proj_color
                })
        return killed_enemies, reward

    def _update_enemies(self):
        leaked_count = 0
        for e in self.enemies[:]:
            e["dist_traveled"] += e["speed"]
            
            current_dist = 0
            for i in range(len(self.path) - 1):
                p1 = pygame.Vector2(self.path[i])
                p2 = pygame.Vector2(self.path[i+1])
                segment_len = p1.distance_to(p2)
                if current_dist + segment_len >= e["dist_traveled"]:
                    # Enemy is on this segment
                    part_way = (e["dist_traveled"] - current_dist) / segment_len
                    e["pos"] = p1.lerp(p2, part_way)
                    
                    # Apply perpendicular offset for visual spread
                    perp_vec = (p2-p1).normalize().rotate(90)
                    e["pos"] += perp_vec * e["offset"]
                    break
                current_dist += segment_len
            else:
                # Enemy reached the end
                leaked_count += 1
                self.enemies.remove(e)
                # SFX: base_damage.wav
                self._create_explosion(e["pos"], self.COLOR_BASE)
        return leaked_count

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(1, self.GRID_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * self.GRID_CELL_SIZE, 0), (i * self.GRID_CELL_SIZE, self.SCREEN_HEIGHT))
        for i in range(1, self.GRID_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i * self.GRID_CELL_SIZE), (self.SCREEN_WIDTH, i * self.GRID_CELL_SIZE))

        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)

        # Draw base
        base_pos = self.path[-1]
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos[0]), int(base_pos[1]), 15, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(base_pos[0]), int(base_pos[1]), 15, self.COLOR_BASE)

        # Draw towers
        for t in self.towers:
            x, y = int(t["pos"][0]), int(t["pos"][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 12, t["color"])
            pygame.gfxdraw.aacircle(self.screen, x, y, 12, t["color"])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 8, self.COLOR_BG)
            pygame.gfxdraw.aacircle(self.screen, x, y, 8, t["color"])

        # Draw enemies
        for e in self.enemies:
            x, y = int(e["pos"][0]), int(e["pos"][1])
            pygame.gfxdraw.filled_trigon(self.screen, x, y - 7, x - 6, y + 4, x + 6, y + 4, self.COLOR_ENEMY)
            pygame.gfxdraw.aatrigon(self.screen, x, y - 7, x - 6, y + 4, x + 6, y + 4, self.COLOR_ENEMY)
            # Health bar
            if e["health"] < e["max_health"]:
                health_pct = e["health"] / e["max_health"]
                pygame.draw.rect(self.screen, (255,0,0), (x - 8, y - 15, 16, 3))
                pygame.draw.rect(self.screen, (0,255,0), (x - 8, y - 15, 16 * health_pct, 3))

        # Draw projectiles
        for p in self.projectiles:
            x, y = int(p["pos"][0]), int(p["pos"][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, p["color"])
            pygame.gfxdraw.aacircle(self.screen, x, y, 3, p["color"])

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw cursor
        self._render_cursor()

    def _render_cursor(self):
        gx, gy = self.cursor_pos
        spec = self.TOWER_SPECS[self.TOWER_TYPES[self.selected_tower_idx]]
        can_build = self.money >= spec["cost"] and self.buildable_grid[gx, gy]
        cursor_color = self.COLOR_VALID_CURSOR if can_build else self.COLOR_INVALID_CURSOR
        
        # Draw range indicator
        cx, cy = int(gx * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE/2), int(gy * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE/2)
        range_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surf, cx, cy, int(spec["range"]), (*cursor_color, 30))
        pygame.gfxdraw.aacircle(range_surf, cx, cy, int(spec["range"]), (*cursor_color, 100))
        self.screen.blit(range_surf, (0,0))

        # Draw cursor box
        rect = (gx * self.GRID_CELL_SIZE, gy * self.GRID_CELL_SIZE, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
        pygame.draw.rect(self.screen, cursor_color, rect, 2)


    def _render_ui(self):
        # UI Panel
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 80), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, self.SCREEN_HEIGHT - 80))
        
        # Base Health
        health_text = self.font_large.render(f"Base: {self.base_health}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Wave Info
        if self.game_state == "INTER_WAVE":
            if self.wave_number >= self.WIN_WAVE:
                wave_text = self.font_large.render("VICTORY!", True, self.COLOR_BASE)
            else:
                wave_text = self.font_large.render(f"Wave {self.wave_number+1} in {self.wave_cooldown/30:.1f}s", True, self.COLOR_TEXT)
        else:
            wave_text = self.font_large.render(f"Wave {self.wave_number}/{self.WIN_WAVE}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        if self.game_over and self.base_health <= 0:
            end_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))

        # Bottom Panel Info
        money_text = self.font_large.render(f"${self.money}", True, self.COLOR_BASE)
        self.screen.blit(money_text, (20, self.SCREEN_HEIGHT - 55))
        
        # Selected Tower Info
        tower_type = self.TOWER_TYPES[self.selected_tower_idx]
        spec = self.TOWER_SPECS[tower_type]
        
        title_text = self.font_large.render(f"Build: {tower_type}", True, self.COLOR_TEXT)
        self.screen.blit(title_text, (200, self.SCREEN_HEIGHT - 65))
        
        cost_color = self.COLOR_BASE if self.money >= spec["cost"] else self.COLOR_ENEMY
        cost_text = self.font_small.render(f"Cost: ${spec['cost']}", True, cost_color)
        dmg_text = self.font_small.render(f"Damage: {spec['damage']}", True, self.COLOR_TEXT)
        range_text = self.font_small.render(f"Range: {spec['range']}", True, self.COLOR_TEXT)
        rate_text = self.font_small.render(f"Rate: {30/spec['fire_rate']:.1f}/s", True, self.COLOR_TEXT)
        
        self.screen.blit(cost_text, (200, self.SCREEN_HEIGHT - 35))
        self.screen.blit(dmg_text, (320, self.SCREEN_HEIGHT - 45))
        self.screen.blit(range_text, (320, self.SCREEN_HEIGHT - 25))
        self.screen.blit(rate_text, (440, self.SCREEN_HEIGHT - 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "money": self.money,
            "base_health": self.base_health,
        }

    # --- Helper functions ---
    def _dist_point_to_segment(self, p, a, b):
        if (b - a).length_squared() == 0:
            return p.distance_to(a)
        t = max(0, min(1, (p - a).dot(b - a) / (b - a).length_squared()))
        projection = a + t * (b - a)
        return p.distance_to(projection)

    def _create_hit_particles(self, pos, color):
        for _ in range(5):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 1.5 + 0.5
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(Particle(pos, vel, color, self.np_random.integers(2, 5), 10))

    def _create_explosion(self, pos, color):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(Particle(pos, vel, color, self.np_random.integers(3, 7), 20))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    terminated = False
    
    # --- Main Game Loop ---
    while not terminated:
        # --- Player Input ---
        # Convert keyboard input to a MultiDiscrete action
        keys = pygame.key.get_pressed()
        mov = 0 # none
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([mov, space, shift])

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
    
    env.close()
    print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")