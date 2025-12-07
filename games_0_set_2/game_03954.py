
# Generated: 2025-08-28T00:57:32.166772
# Source Brief: brief_03954.md
# Brief Index: 3954

        
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
        "Controls: Arrows to move cursor. Space to place turret. Shift to cycle turret type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of zombies by strategically placing turrets on the grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 50, 60)
    COLOR_PATH = (50, 60, 70)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_DMG = (255, 100, 100)
    
    COLOR_ZOMBIE = (200, 50, 50)
    COLOR_ZOMBIE_HEALTH_BG = (80, 20, 20)
    COLOR_ZOMBIE_HEALTH = (220, 60, 60)
    
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_INVALID = (255, 0, 0)
    
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    # Grid
    GRID_COLS, GRID_ROWS = 12, 6
    CELL_SIZE = 40
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_OFFSET_X = (WIDTH - GRID_WIDTH) // 2
    GRID_OFFSET_Y = 80
    
    # Game Logic
    MAX_WAVES = 20
    BASE_START_HEALTH = 100
    STARTING_RESOURCES = 150
    WAVE_PREP_TIME = 5 * FPS # 5 seconds
    MAX_EPISODE_STEPS = 30 * FPS * 3 # 3 minutes

    TURRET_SPECS = {
        0: {"name": "Gatling", "cost": 75, "range": 100, "damage": 4, "fire_rate": 8, "proj_speed": 10, "color": (0, 200, 255), "splash": 0},
        1: {"name": "Cannon", "cost": 150, "range": 160, "damage": 25, "fire_rate": 1, "proj_speed": 6, "color": (255, 150, 0), "splash": 35},
    }

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
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self._define_path()
        self.reset()
        
        self.validate_implementation()
    
    def _define_path(self):
        self.path_waypoints = []
        p = self.path_waypoints.append
        c2w = self._grid_to_world # Helper
        p(c2w(-1, 1))
        p(c2w(2, 1))
        p(c2w(2, 4))
        p(c2w(9, 4))
        p(c2w(9, 1))
        p(c2w(self.GRID_COLS + 1, 1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over_reason = ""
        
        self.base_health = self.BASE_START_HEALTH
        self.resources = self.STARTING_RESOURCES
        
        self.current_wave = 0
        self.wave_timer = self.WAVE_PREP_TIME // 2
        
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        self.zombies = []
        self.turrets = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_turret_type = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self.base_flash_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False

        # 1. Handle player input
        self._handle_input(action)

        # 2. Update game state
        self._update_wave_logic()
        reward += self._update_turrets()
        reward += self._update_projectiles()
        reward += self._update_zombies()
        self._update_particles()
        if self.base_flash_timer > 0: self.base_flash_timer -= 1
        
        self.steps += 1
        
        # 3. Check for termination conditions
        if self.base_health <= 0:
            terminated = True
            reward -= 10
            self.game_over_reason = "BASE DESTROYED"
        elif self.current_wave > self.MAX_WAVES:
            terminated = True
            reward += 100
            self.game_over_reason = "YOU SURVIVED!"
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over_reason = "TIME LIMIT REACHED"
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Cycle turret (on press)
        if shift_held and not self.last_shift_held:
            self.selected_turret_type = (self.selected_turret_type + 1) % len(self.TURRET_SPECS)
            # sfx: UI_CYCLE
        
        # Place turret (on press)
        if space_held and not self.last_space_held:
            self._try_place_turret()
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _try_place_turret(self):
        cx, cy = self.cursor_pos
        spec = self.TURRET_SPECS[self.selected_turret_type]
        
        if self.resources >= spec["cost"] and self.grid[cx, cy] == 0:
            self.resources -= spec["cost"]
            self.grid[cx, cy] = 1
            
            new_turret = {
                "grid_pos": [cx, cy],
                "type": self.selected_turret_type,
                "cooldown": 0,
                "angle": 0,
            }
            self.turrets.append(new_turret)
            # sfx: TURRET_PLACE
            self._create_particles(self._grid_to_world(cx, cy), 15, spec["color"], 2, 8, 0.5)

    def _update_wave_logic(self):
        if not self.zombies and self.current_wave <= self.MAX_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                if self.current_wave <= self.MAX_WAVES:
                    self._start_next_wave()
                    self.wave_timer = self.WAVE_PREP_TIME

    def _start_next_wave(self):
        zombie_count = 5 + (self.current_wave - 1)
        base_health = 50 * (1.05 ** (self.current_wave - 1))
        base_speed = 20 * (1.05 ** (self.current_wave - 1)) / self.FPS
        
        for i in range(zombie_count):
            offset = self.np_random.uniform(-15, 15)
            spawn_pos = [self.path_waypoints[0][0] - i * 25, self.path_waypoints[0][1] + offset]
            self.zombies.append({
                "pos": spawn_pos,
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed * self.np_random.uniform(0.9, 1.1),
                "path_index": 0,
            })
        # sfx: WAVE_START

    def _update_turrets(self):
        reward = 0
        for turret in self.turrets:
            spec = self.TURRET_SPECS[turret["type"]]
            turret["cooldown"] = max(0, turret["cooldown"] - 1)
            
            if turret["cooldown"] == 0:
                target = self._find_target(turret)
                if target:
                    turret_pos = self._grid_to_world(*turret["grid_pos"])
                    # sfx: GATLING_FIRE or CANNON_FIRE
                    self.projectiles.append({
                        "pos": list(turret_pos),
                        "type": turret["type"],
                        "target_zombie": target,
                        "damage": spec["damage"],
                        "splash": spec["splash"],
                    })
                    turret["cooldown"] = self.FPS / spec["fire_rate"]
                    
                    # Visual feedback: muzzle flash
                    dx = target["pos"][0] - turret_pos[0]
                    dy = target["pos"][1] - turret_pos[1]
                    turret["angle"] = math.atan2(dy, dx)
                    flash_pos = [turret_pos[0] + math.cos(turret["angle"]) * 20, turret_pos[1] + math.sin(turret["angle"]) * 20]
                    self._create_particles(flash_pos, 5, (255, 255, 100), 1, 5, 0.2)
        return reward

    def _find_target(self, turret):
        turret_pos = self._grid_to_world(*turret["grid_pos"])
        spec = self.TURRET_SPECS[turret["type"]]
        
        closest_zombie = None
        min_dist_sq = spec["range"] ** 2
        
        for zombie in self.zombies:
            dist_sq = (zombie["pos"][0] - turret_pos[0])**2 + (zombie["pos"][1] - turret_pos[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_zombie = zombie
        return closest_zombie

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            spec = self.TURRET_SPECS[proj["type"]]
            target = proj["target_zombie"]

            if target not in self.zombies: # Target already dead
                self.projectiles.remove(proj)
                continue

            # Move projectile towards target
            dx = target["pos"][0] - proj["pos"][0]
            dy = target["pos"][1] - proj["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < spec["proj_speed"]:
                # Hit!
                reward += self._deal_damage(proj)
                self.projectiles.remove(proj)
            else:
                proj["pos"][0] += (dx / dist) * spec["proj_speed"]
                proj["pos"][1] += (dy / dist) * spec["proj_speed"]
        return reward

    def _deal_damage(self, projectile):
        reward = 0
        spec = self.TURRET_SPECS[projectile["type"]]
        hit_pos = projectile["target_zombie"]["pos"]
        
        # sfx: HIT_NORMAL or HIT_SPLASH
        self._create_particles(hit_pos, 10, spec["color"], 2, 6, 0.3)

        zombies_hit = []
        if projectile["splash"] > 0:
            splash_radius_sq = projectile["splash"] ** 2
            for z in self.zombies:
                dist_sq = (z["pos"][0] - hit_pos[0])**2 + (z["pos"][1] - hit_pos[1])**2
                if dist_sq <= splash_radius_sq:
                    zombies_hit.append(z)
        else:
            if projectile["target_zombie"] in self.zombies:
                zombies_hit.append(projectile["target_zombie"])

        for zombie in zombies_hit:
            zombie["health"] -= projectile["damage"]
            reward += 0.1 # Reward for hitting a zombie
            if zombie["health"] <= 0:
                reward += 1 # Reward for killing a zombie
                self.resources += int(zombie["max_health"] / 10)
                # sfx: ZOMBIE_DEATH
                self._create_particles(zombie["pos"], 20, self.COLOR_ZOMBIE, 1, 10, 0.8)
                self.zombies.remove(zombie)
        return reward

    def _update_zombies(self):
        reward = 0
        for z in self.zombies[:]:
            if z["path_index"] >= len(self.path_waypoints):
                self.base_health = max(0, self.base_health - 10)
                self.base_flash_timer = 10
                self.zombies.remove(z)
                reward -= 0.01 # Penalty for base taking damage
                # sfx: BASE_DAMAGE
                self._create_particles((self.WIDTH-20, self.GRID_OFFSET_Y + self.GRID_HEIGHT//2), 20, self.COLOR_BASE_DMG, 3, 10, 0.7)
                continue

            target_pos = self.path_waypoints[z["path_index"]]
            dx = target_pos[0] - z["pos"][0]
            dy = target_pos[1] - z["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < z["speed"]:
                z["path_index"] += 1
            else:
                z["pos"][0] += (dx / dist) * z["speed"]
                z["pos"][1] += (dy / dist) * z["speed"]
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["size"] *= 0.95
            if p["life"] <= 0 or p["size"] < 0.5:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, min_size, max_size, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "size": self.np_random.uniform(min_size, max_size),
                "life": self.np_random.integers(15, 30),
                "color": color
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, self.CELL_SIZE)
        
        # Draw grid
        for x in range(self.GRID_COLS + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_ROWS + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw base
        base_color = self.COLOR_BASE_DMG if self.base_flash_timer > 0 else self.COLOR_BASE
        base_rect = pygame.Rect(self.WIDTH - 40, self.GRID_OFFSET_Y, 40, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, base_color, base_rect)
        pygame.gfxdraw.rectangle(self.screen, base_rect, tuple(c*0.8 for c in base_color))

        # Draw turrets
        for turret in self.turrets:
            spec = self.TURRET_SPECS[turret["type"]]
            pos = self._grid_to_world(*turret["grid_pos"])
            pygame.draw.circle(self.screen, spec["color"], pos, 15)
            pygame.draw.circle(self.screen, self.COLOR_BG, pos, 12)
            pygame.draw.circle(self.screen, spec["color"], pos, 8)
            # Barrel
            end_x = pos[0] + math.cos(turret["angle"]) * 20
            end_y = pos[1] + math.sin(turret["angle"]) * 20
            pygame.draw.line(self.screen, spec["color"], pos, (end_x, end_y), 5)

        # Draw projectiles
        for proj in self.projectiles:
            spec = self.TURRET_SPECS[proj["type"]]
            size = 3 if spec["splash"] == 0 else 6
            pygame.gfxdraw.filled_circle(self.screen, int(proj["pos"][0]), int(proj["pos"][1]), size, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, int(proj["pos"][0]), int(proj["pos"][1]), size, spec["color"])

        # Draw zombies
        for z in self.zombies:
            pos_int = (int(z["pos"][0]), int(z["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_ZOMBIE)
            # Health bar
            health_pct = z["health"] / z["max_health"]
            bar_w = 16
            bar_h = 3
            bar_x = pos_int[0] - bar_w // 2
            bar_y = pos_int[1] - 15
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_HEALTH, (bar_x, bar_y, int(bar_w * health_pct), bar_h))

        # Draw particles
        for p in self.particles:
            size = int(p["size"])
            if size > 0:
                alpha = int(255 * (p["life"] / 30))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (p["pos"][0] - size, p["pos"][1] - size))

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_world_pos = self._grid_to_world(cx, cy)
        spec = self.TURRET_SPECS[self.selected_turret_type]
        is_valid = self.resources >= spec["cost"] and self.grid[cx, cy] == 0
        color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        
        # Range indicator
        range_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(range_surf, cursor_world_pos[0], cursor_world_pos[1], spec["range"], (*color, 60))
        pygame.gfxdraw.filled_circle(range_surf, cursor_world_pos[0], cursor_world_pos[1], spec["range"], (*color, 30))
        self.screen.blit(range_surf, (0,0))
        
        # Cursor box
        rect = pygame.Rect(self.GRID_OFFSET_X + cx * self.CELL_SIZE, self.GRID_OFFSET_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect, 2)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color, center=False):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if center:
                text_rect.center = pos
            else:
                text_rect.topleft = pos
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        # Top bar
        draw_text(f"♥ {int(self.base_health)}/{self.BASE_START_HEALTH}", self.font_large, self.COLOR_TEXT, (20, 15), self.COLOR_TEXT_SHADOW)
        draw_text(f"$ {self.resources}", self.font_large, (255, 220, 100), (220, 15), self.COLOR_TEXT_SHADOW)
        draw_text(f"WAVE {self.current_wave}/{self.MAX_WAVES}", self.font_large, self.COLOR_TEXT, (400, 15), self.COLOR_TEXT_SHADOW)
        draw_text(f"SCORE: {int(self.score)}", self.font_large, self.COLOR_TEXT, (20, self.HEIGHT - 35), self.COLOR_TEXT_SHADOW)
        
        # Turret selection UI
        spec = self.TURRET_SPECS[self.selected_turret_type]
        draw_text(f"Selected: {spec['name']} (Cost: {spec['cost']})", self.font_large, spec['color'], (self.WIDTH - 280, self.HEIGHT - 35), self.COLOR_TEXT_SHADOW)

        # Wave timer
        if self.wave_timer > 0 and not self.zombies:
            seconds = math.ceil(self.wave_timer / self.FPS)
            draw_text(f"NEXT WAVE IN {seconds}", self.font_huge, self.COLOR_TEXT, (self.WIDTH/2, self.HEIGHT/2), self.COLOR_TEXT_SHADOW, center=True)
            
        # Game Over
        if self.game_over_reason:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            draw_text(self.game_over_reason, self.font_huge, (255, 50, 50), (self.WIDTH/2, self.HEIGHT/2), (0,0,0), center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources,
            "zombies_remaining": len(self.zombies),
        }
        
    def _grid_to_world(self, x, y):
        return (
            self.GRID_OFFSET_X + int((x + 0.5) * self.CELL_SIZE),
            self.GRID_OFFSET_Y + int((y + 0.5) * self.CELL_SIZE)
        )

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example usage for interactive play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for display ---
    pygame.display.set_caption("Zombie Siege")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    while not terminated:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Get player input ---
        keys = pygame.key.get_pressed()
        mov = 0 # no-op
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Reason: {env.game_over_reason}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()