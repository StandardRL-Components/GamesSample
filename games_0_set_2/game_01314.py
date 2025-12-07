
# Generated: 2025-08-27T16:44:41.039635
# Source Brief: brief_01314.md
# Brief Index: 1314

        
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
        "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Isometric tower defense. Strategically place towers to defend your base from waves of enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 18, 12
        self.TILE_WIDTH, self.TILE_HEIGHT = 32, 16
        self.MAX_STEPS = 3000 # Increased for longer games
        self.MAX_WAVES = 10
        
        # Colors (Bright & Contrasting)
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PATH = (60, 65, 70)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_CURSOR = (255, 255, 0, 150)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 150)
        self.TOWER_COLORS = [(50, 150, 255), (255, 200, 50), (200, 100, 255)]
        
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
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game path (grid coordinates)
        self.path_waypoints = [
            (-1, 5), (2, 5), (2, 2), (7, 2), (7, 8), (13, 8), (13, 3), (18, 3)
        ]
        
        # Tower definitions
        self.tower_specs = [
            {"cost": 50, "range": 2.5, "damage": 8, "cooldown": 15, "type": "single"}, # Fast, low damage
            {"cost": 100, "range": 3.5, "damage": 30, "cooldown": 60, "type": "single"}, # Slow, high damage
            {"cost": 150, "range": 2.0, "damage": 5, "cooldown": 45, "type": "aoe", "aoe_radius": 1.5} # AoE
        ]

        # Initialize state variables
        self.reset()

        # Validate implementation
        self.validate_implementation()

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = (self.WIDTH / 2) + (grid_x - grid_y) * self.TILE_WIDTH
        screen_y = 100 + (grid_x + grid_y) * self.TILE_HEIGHT
        return int(screen_x), int(screen_y)

    def _screen_to_iso(self, screen_x, screen_y):
        # This is an approximation for finding tile under mouse, not used in agent logic
        screen_y_norm = screen_y - 100
        screen_x_norm = screen_x - (self.WIDTH / 2)
        grid_y = (screen_y_norm / self.TILE_HEIGHT - screen_x_norm / self.TILE_WIDTH) / 2
        grid_x = screen_x_norm / self.TILE_WIDTH + grid_y
        return int(grid_x), int(grid_y)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.resources = 125 # Start with enough for a tower or two
        self.wave_number = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.wave_spawn_timer = 0
        self.wave_spawn_index = 0
        self.between_wave_timer = 90 # 3 seconds at 30fps
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return
        
        num_enemies = 5 + self.wave_number * 3
        base_health = 20 + self.wave_number * 15
        base_speed = 0.03 + self.wave_number * 0.002
        
        self.current_wave_enemies = []
        for i in range(num_enemies):
            health_mult = 1 + (i / num_enemies) * 0.5 # Stronger enemies at end of wave
            speed_mult = 1 + (i / num_enemies) * 0.2
            self.current_wave_enemies.append({
                "health": base_health * health_mult,
                "speed": base_speed * speed_mult,
                "spawn_delay": i * (30 // self.wave_number + 5)
            })

        self.wave_spawn_index = 0
        self.wave_spawn_timer = self.current_wave_enemies[0]['spawn_delay']

    def _spawn_enemy(self):
        if self.wave_spawn_index >= len(self.current_wave_enemies):
            return
            
        spec = self.current_wave_enemies[self.wave_spawn_index]
        start_pos = self.path_waypoints[0]
        
        self.enemies.append({
            "pos": [float(start_pos[0]), float(start_pos[1])],
            "max_health": spec["health"],
            "health": spec["health"],
            "speed": spec["speed"],
            "waypoint_index": 1
        })
        self.wave_spawn_index += 1
        if self.wave_spawn_index < len(self.current_wave_enemies):
            self.wave_spawn_timer = self.current_wave_enemies[self.wave_spawn_index]['spawn_delay']

    def step(self, action):
        reward = 0.0
        
        # --- 1. Handle Input ---
        self._handle_input(action)
        
        # --- 2. Update Game State ---
        reward_events = {"damage": 0, "kills": 0}
        self._update_towers(reward_events)
        self._update_projectiles(reward_events)
        base_damage = self._update_enemies()
        self._update_particles()
        
        # --- 3. Update Wave Logic ---
        if self.wave_spawn_index < len(self.current_wave_enemies):
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0:
                self._spawn_enemy()
        elif not self.enemies: # Wave cleared
            if self.wave_number >= self.MAX_WAVES:
                self.win = True
            else:
                self.between_wave_timer -= 1
                if self.between_wave_timer <= 0:
                    reward += 50.0 # Wave clear bonus
                    self.score += 50
                    self.between_wave_timer = 90
                    self._start_next_wave()
        
        # --- 4. Calculate Reward ---
        self.base_health -= base_damage
        if self.base_health < 100:
            reward -= 0.01
        
        reward += reward_events["damage"] * 0.1
        reward += reward_events["kills"] * 1.0
        self.score += reward_events["kills"] * 1.0
        
        # --- 5. Check Termination ---
        self.steps += 1
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            terminated = True
            reward -= 100.0
        elif self.win:
            terminated = True
            reward += 100.0
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)
        
        # Place tower on key press
        if space_held and not self.last_space_held:
            self._place_tower()
        
        # Cycle tower on key press
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_specs)
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _place_tower(self):
        spec = self.tower_specs[self.selected_tower_type]
        if self.resources >= spec["cost"]:
            # Check if location is valid (not on path, not on another tower)
            is_valid_location = True
            cx, cy = self.cursor_pos
            if 1 <= cy <= 3 or 4 <= cy <= 6 and cx >= 2 or 7 <= cy <= 9 and cx <= 7: # Simplified path check
                 is_valid_location = False
            for t in self.towers:
                if t["pos"] == self.cursor_pos:
                    is_valid_location = False
                    break
            
            if is_valid_location:
                self.resources -= spec["cost"]
                self.towers.append({
                    "pos": list(self.cursor_pos),
                    "spec": spec,
                    "cooldown": 0,
                    "type_index": self.selected_tower_type
                })
                # sfx: place_tower.wav
                for _ in range(20):
                    self._create_particle(self._iso_to_screen(*self.cursor_pos), self.TOWER_COLORS[self.selected_tower_type])

    def _update_towers(self, reward_events):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue
            
            target = None
            min_dist = tower["spec"]["range"] ** 2
            
            for enemy in self.enemies:
                dist_sq = (tower["pos"][0] - enemy["pos"][0])**2 + (tower["pos"][1] - enemy["pos"][1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                tower["cooldown"] = tower["spec"]["cooldown"]
                start_pos = self._iso_to_screen(tower["pos"][0], tower["pos"][1])
                self.projectiles.append({
                    "start_pos": start_pos,
                    "target": target,
                    "spec": tower["spec"],
                    "color": self.TOWER_COLORS[tower["type_index"]]
                })
                # sfx: fire_weapon.wav

    def _update_projectiles(self, reward_events):
        for proj in self.projectiles[:]:
            target_pos = self._iso_to_screen(proj["target"]["pos"][0], proj["target"]["pos"][1])
            proj_pos = proj.get("pos", proj["start_pos"])
            
            dx = target_pos[0] - proj_pos[0]
            dy = target_pos[1] - proj_pos[1]
            dist = math.hypot(dx, dy)
            
            if dist < 10: # Hit
                self._handle_projectile_hit(proj, reward_events)
                self.projectiles.remove(proj)
                continue

            speed = 15
            proj["pos"] = (proj_pos[0] + dx/dist * speed, proj_pos[1] + dy/dist * speed)

    def _handle_projectile_hit(self, proj, reward_events):
        spec = proj["spec"]
        target_pos = self._iso_to_screen(proj["target"]["pos"][0], proj["target"]["pos"][1])
        
        if spec["type"] == "single":
            if proj["target"] in self.enemies: # Check if target still exists
                proj["target"]["health"] -= spec["damage"]
                reward_events["damage"] += 1
                for _ in range(5): self._create_particle(target_pos, proj["color"], size=3)
                # sfx: hit_single.wav
        
        elif spec["type"] == "aoe":
            for _ in range(30): self._create_particle(target_pos, proj["color"], size=5, lifespan=20)
            # sfx: hit_aoe.wav
            for enemy in self.enemies:
                e_pos = enemy["pos"]
                t_pos = proj["target"]["pos"]
                dist_sq = (e_pos[0] - t_pos[0])**2 + (e_pos[1] - t_pos[1])**2
                if dist_sq <= spec["aoe_radius"]**2:
                    enemy["health"] -= spec["damage"]
                    reward_events["damage"] += 1

    def _update_enemies(self):
        base_damage = 0
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                # sfx: enemy_die.wav
                self.resources += int(enemy["max_health"] / 4)
                self.score += int(enemy["max_health"] / 4)
                e_screen_pos = self._iso_to_screen(*enemy["pos"])
                for _ in range(30): self._create_particle(e_screen_pos, self.COLOR_ENEMY, lifespan=25)
                self.enemies.remove(enemy)
                reward_events = {"kills": 1} # This seems to be missing from original call, but needed
                continue

            if enemy["waypoint_index"] >= len(self.path_waypoints):
                base_damage += 10 # Damage to base
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                continue

            target_wp = self.path_waypoints[enemy["waypoint_index"]]
            dx = target_wp[0] - enemy["pos"][0]
            dy = target_wp[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < 0.1:
                enemy["waypoint_index"] += 1
            else:
                enemy["pos"][0] += dx / dist * enemy["speed"]
                enemy["pos"][1] += dy / dist * enemy["speed"]
        return base_damage

    def _create_particle(self, pos, color, size=2, lifespan=15, speed=2):
        angle = random.uniform(0, 2 * math.pi)
        vel = (math.cos(angle) * speed * random.uniform(0.5, 1.5), 
               math.sin(angle) * speed * random.uniform(0.5, 1.5))
        self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color, "size": size})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_path()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_iso_rect(self, surface, color, grid_pos, height_offset=0):
        x, y = grid_pos
        points = [
            self._iso_to_screen(x, y),
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1),
            self._iso_to_screen(x, y + 1)
        ]
        points = [(px, py - height_offset) for px, py in points]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_grid(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                self._draw_iso_rect(self.screen, self.COLOR_GRID, (x, y))

    def _render_path(self):
        for i in range(len(self.path_waypoints) - 1):
            start_wp = self.path_waypoints[i]
            end_wp = self.path_waypoints[i+1]
            if start_wp[0] == end_wp[0]: # Vertical path
                for y in range(min(start_wp[1], end_wp[1]), max(start_wp[1], end_wp[1]) + 1):
                    self._draw_iso_rect(self.screen, self.COLOR_PATH, (start_wp[0], y))
            else: # Horizontal path
                for x in range(min(start_wp[0], end_wp[0]), max(start_wp[0], end_wp[0]) + 1):
                    self._draw_iso_rect(self.screen, self.COLOR_PATH, (x, start_wp[1]))

    def _render_towers(self):
        for tower in self.towers:
            color = self.TOWER_COLORS[tower["type_index"]]
            self._draw_iso_rect(self.screen, color, tower["pos"], height_offset=8)
            # Cooldown indicator
            if tower["cooldown"] > 0:
                cooldown_ratio = tower["cooldown"] / tower["spec"]["cooldown"]
                pos = self._iso_to_screen(*tower["pos"])
                pygame.draw.circle(self.screen, (255,255,255,50), (pos[0], pos[1]-12), 8, 1)
                arc_end = -math.pi/2 + (2*math.pi * cooldown_ratio)
                pygame.draw.arc(self.screen, (255,255,255), (pos[0]-8, pos[1]-20, 16, 16), -math.pi/2, arc_end, 2)


    def _render_enemies(self):
        for enemy in sorted(self.enemies, key=lambda e: e["pos"][0] + e["pos"][1]):
            screen_pos = self._iso_to_screen(*enemy["pos"])
            size = 8
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1] - size, size, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1] - size, size, self.COLOR_ENEMY)
            
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_width = 20
            bar_y = screen_pos[1] - 2 * size - 4
            pygame.draw.rect(self.screen, (50, 0, 0), (screen_pos[0] - bar_width/2, bar_y, bar_width, 4))
            pygame.draw.rect(self.screen, (0, 255, 0), (screen_pos[0] - bar_width/2, bar_y, bar_width * health_ratio, 4))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = proj.get("pos", proj["start_pos"])
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 3, proj["color"])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 3, proj["color"])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 15))
            color = (*p["color"], alpha)
            pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), int(p["size"]))

    def _render_cursor(self):
        spec = self.tower_specs[self.selected_tower_type]
        is_valid = self.resources >= spec["cost"] # Simplified validity check for color
        color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        
        # Draw cursor box
        self._draw_iso_rect(self.screen, color, self.cursor_pos, height_offset=2)
        
        # Draw range indicator
        center_screen = self._iso_to_screen(*self.cursor_pos)
        range_px = spec["range"] * math.sqrt(self.TILE_WIDTH**2 / 4 + self.TILE_HEIGHT**2)
        
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.TOWER_COLORS[self.selected_tower_type], 50), center_screen, int(range_px))
        self.screen.blit(s, (0,0))

    def _render_ui(self):
        # Base Health
        health_text = self.font_large.render(f"Base HP: {int(self.base_health)}", True, self.COLOR_BASE)
        self.screen.blit(health_text, (10, 10))
        
        # Resources
        resource_text = self.font_large.render(f"Resources: {self.resources}", True, self.TOWER_COLORS[1])
        self.screen.blit(resource_text, (10, 40))
        
        # Wave Info
        if not self.win and self.wave_number <= self.MAX_WAVES:
            wave_text = self.font_large.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        elif self.win:
            wave_text = self.font_large.render("VICTORY!", True, (0, 255, 0))
        else: # Should not happen, but for safety
            wave_text = self.font_large.render(f"Wave: {self.wave_number-1}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Selected Tower
        spec = self.tower_specs[self.selected_tower_type]
        color = self.TOWER_COLORS[self.selected_tower_type]
        tower_info = f"Tower {self.selected_tower_type+1}: Cost {spec['cost']}"
        tower_text = self.font_small.render(tower_info, True, color)
        self.screen.blit(tower_text, (self.WIDTH - tower_text.get_width() - 10, 45))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
            "towers_placed": len(self.towers),
            "enemies_on_screen": len(self.enemies),
        }

    def close(self):
        pygame.font.quit()
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a screen
    pygame.display.set_caption("Isometric Tower Defense")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # For human play, map keys to actions
        action = np.array([0, 0, 0]) # no-op
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Game Over! Score: {info['score']}, Survived {info['wave']-1} waves.")
            obs, info = env.reset() # Auto-restart
            pygame.time.wait(2000)

        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        env.clock.tick(30) # Run at 30 FPS

    env.close()