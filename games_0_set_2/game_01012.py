
# Generated: 2025-08-27T15:30:07.579895
# Source Brief: brief_01012.md
# Brief Index: 1012

        
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

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Space to build a Gun Tower, or Shift to build a Cannon Tower. "
        "Submit a 'no-op' action (no keys pressed) to start the next wave."
    )

    game_description = (
        "An isometric tower defense game. Place towers to defend your base from waves of enemies. "
        "Survive 10 waves to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 24)
        self.font_wave = pygame.font.Font(None, 48)

        # --- Colors ---
        self.COLOR_BG = (25, 35, 55)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PATH = (80, 70, 60)
        self.COLOR_PATH_BORDER = (60, 50, 40)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_BASE_DMG = (200, 50, 50)
        self.COLOR_PLACEABLE = (60, 80, 60, 100)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_GOLD = (255, 200, 0)
        self.COLOR_HEALTH = (0, 200, 100)
        
        self.TOWER_SPECS = {
            1: {"name": "Gun", "cost": 30, "range": 100, "fire_rate": 15, "projectile_speed": 8, "damage": 10, "color": (0, 200, 255)},
            2: {"name": "Cannon", "cost": 75, "range": 150, "fire_rate": 60, "projectile_speed": 5, "damage": 40, "color": (255, 100, 0)}
        }

        # --- Game World ---
        self.grid_size = (12, 12)
        self.tile_width_half = 22
        self.tile_height_half = 11
        self.world_offset = (self.screen_width // 2, 100)

        self.enemy_path = [
            (-1, 5), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), 
            (5, 5), (5, 4), (5, 3), (5, 2), 
            (6, 2), (7, 2), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
            (7, 7), (6, 7), (5, 7), (4, 7), (3, 7),
            (3, 8), (3, 9), (3, 10), (4, 10), (5, 10), (6, 10),
            (7, 10), (8, 10), (9, 10), (10, 10), (11, 10), (12, 10)
        ]
        
        self.placement_zones = [
            (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9),
            (4, 3), (4, 4), (4, 6), (4, 8), (4, 9),
            (6, 4), (6, 5), (6, 6), (6, 8),
            (7, 4), (7, 5), (7, 6), (7, 8),
            (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9),
        ]
        
        # --- State Variables ---
        self.rng = None
        self.steps = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 0
        self.gold = 0
        self.current_wave = 0
        self.max_waves = 10
        self.game_phase = "" # "placement" or "wave"
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = (0, 0)
        self.message = ""
        self.message_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.gold = 80
        self.current_wave = 0
        self.game_phase = "placement"
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = self.placement_zones[0]
        self.message = ""
        self.message_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        if self.message_timer > 0:
            self.message_timer -= 1
        else:
            self.message = ""

        if self.game_phase == "placement":
            reward += self._handle_placement_phase(action)
        elif self.game_phase == "wave":
            reward += self._handle_wave_phase()

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] * 0.95)

        terminated = self.base_health <= 0 or self.current_wave > self.max_waves or self.steps >= 2500
        
        if self.base_health <= 0 and not self.game_over:
            reward -= 100
            self.game_over = True
            self.win = False
        
        if self.current_wave > self.max_waves and not self.game_over:
            reward += 100
            self.game_over = True
            self.win = True
            
        if self.steps >= 2500 and not self.game_over:
            self.game_over = True
            self.win = False

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_placement_phase(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle cursor movement
        current_index = self.placement_zones.index(self.cursor_pos)
        if movement == 1: # Up
            next_index = max(0, current_index - 1)
        elif movement == 2: # Down
            next_index = min(len(self.placement_zones) - 1, current_index + 1)
        elif movement == 3: # Left
            # This is an abstract left, find closest with smaller x, then y
            x, y = self.cursor_pos
            candidates = [i for i, (px, py) in enumerate(self.placement_zones) if px < x or (px == x and py < y)]
            next_index = max(candidates) if candidates else current_index
        elif movement == 4: # Right
            # Abstract right
            x, y = self.cursor_pos
            candidates = [i for i, (px, py) in enumerate(self.placement_zones) if px > x or (px == x and py > y)]
            next_index = min(candidates) if candidates else current_index
        else:
            next_index = current_index
        
        self.cursor_pos = self.placement_zones[next_index]

        # Handle tower placement
        if space_held:
            self._place_tower(1)
        elif shift_held:
            self._place_tower(2)
        
        # Start wave on no-op
        if movement == 0 and not space_held and not shift_held:
            self.game_phase = "wave"
            self.current_wave += 1
            self._spawn_wave()
            return 1 # Reward for starting a wave
        return 0

    def _place_tower(self, tower_type):
        spec = self.TOWER_SPECS[tower_type]
        if self.gold >= spec["cost"]:
            is_occupied = any(t["pos"] == self.cursor_pos for t in self.towers)
            if not is_occupied:
                self.gold -= spec["cost"]
                screen_pos = self._grid_to_iso(self.cursor_pos)
                self.towers.append({
                    "pos": self.cursor_pos, "type": tower_type, "cooldown": 0, "target": None
                })
                self._create_particles(screen_pos, 20, spec["color"])
                # sfx: tower_place.wav
            else:
                self._set_message("Tile is occupied!")
        else:
            self._set_message("Not enough gold!")

    def _handle_wave_phase(self):
        reward = 0
        
        # Update towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            # Find target
            tower_screen_pos = self._grid_to_iso(tower["pos"])
            potential_targets = []
            for i, enemy in enumerate(self.enemies):
                dist = math.hypot(enemy["screen_pos"][0] - tower_screen_pos[0], enemy["screen_pos"][1] - tower_screen_pos[1])
                if dist <= spec["range"]:
                    potential_targets.append((dist, i))
            
            if potential_targets:
                potential_targets.sort()
                target_index = potential_targets[0][1]
                target_enemy = self.enemies[target_index]
                
                # Fire projectile
                self.projectiles.append({
                    "start_pos": list(tower_screen_pos),
                    "pos": list(tower_screen_pos),
                    "target_pos": list(target_enemy["screen_pos"]),
                    "speed": spec["projectile_speed"],
                    "damage": spec["damage"],
                    "type": tower["type"]
                })
                tower["cooldown"] = spec["fire_rate"]
                # sfx: shoot_gun.wav or shoot_cannon.wav

        # Update and move projectiles
        surviving_projectiles = []
        for proj in self.projectiles:
            hit = self._move_projectile(proj)
            if not hit:
                surviving_projectiles.append(proj)
            else:
                reward += self._handle_projectile_hit(proj)
        self.projectiles = surviving_projectiles

        # Update and move enemies
        surviving_enemies = []
        for enemy in self.enemies:
            if self._move_enemy(enemy):
                surviving_enemies.append(enemy)
        self.enemies = surviving_enemies

        # Check for wave completion
        if not self.enemies:
            self.game_phase = "placement"
            reward += 10 # Wave complete bonus
        
        return reward
    
    def _move_projectile(self, proj):
        target_vector = (proj["target_pos"][0] - proj["pos"][0], proj["target_pos"][1] - proj["pos"][1])
        dist = math.hypot(*target_vector)
        
        if dist < proj["speed"]:
            proj["pos"] = proj["target_pos"]
            return True # Hit
        
        norm_vector = (target_vector[0] / dist, target_vector[1] / dist)
        proj["pos"][0] += norm_vector[0] * proj["speed"]
        proj["pos"][1] += norm_vector[1] * proj["speed"]
        return False
        
    def _handle_projectile_hit(self, proj):
        reward = 0
        hit_pos = proj["pos"]
        
        self._create_particles(hit_pos, 15, self.TOWER_SPECS[proj["type"]]["color"], 0.8)
        # sfx: hit_small.wav
        
        for enemy in self.enemies:
            if math.hypot(enemy["screen_pos"][0] - hit_pos[0], enemy["screen_pos"][1] - hit_pos[1]) < 10:
                enemy["health"] -= proj["damage"]
                enemy["last_hit_timer"] = 10
                if enemy["health"] <= 0:
                    reward += 0.5 # Kill bonus
                    self.gold += enemy["gold_value"]
                    self._create_particles(enemy["screen_pos"], 40, (255, 255, 255), 1.5)
                    # sfx: enemy_explode.wav
        
        self.enemies = [e for e in self.enemies if e["health"] > 0]
        return reward

    def _move_enemy(self, enemy):
        if enemy["path_index"] >= len(self.enemy_path) - 1:
            self.base_health = max(0, self.base_health - enemy["damage"])
            self._create_particles(self._grid_to_iso((11.5, 9.5)), 50, self.COLOR_BASE_DMG, 2.0)
            # sfx: base_damage.wav
            return False # Reached end

        target_grid_pos = self.enemy_path[enemy["path_index"] + 1]
        target_screen_pos = self._grid_to_iso(target_grid_pos)
        
        move_vec = (target_screen_pos[0] - enemy["screen_pos"][0], target_screen_pos[1] - enemy["screen_pos"][1])
        dist = math.hypot(*move_vec)

        if dist < enemy["speed"]:
            enemy["path_index"] += 1
            remaining_speed = enemy["speed"] - dist
            # Recalculate for next segment if needed
            if enemy["path_index"] < len(self.enemy_path) - 1:
                new_target_grid = self.enemy_path[enemy["path_index"] + 1]
                new_target_screen = self._grid_to_iso(new_target_grid)
                new_move_vec = (new_target_screen[0] - enemy["screen_pos"][0], new_target_screen[1] - enemy["screen_pos"][1])
                new_dist = math.hypot(*new_move_vec)
                if new_dist > 0:
                    enemy["screen_pos"][0] += (new_move_vec[0] / new_dist) * remaining_speed
                    enemy["screen_pos"][1] += (new_move_vec[1] / new_dist) * remaining_speed
        else:
            enemy["screen_pos"][0] += (move_vec[0] / dist) * enemy["speed"]
            enemy["screen_pos"][1] += (move_vec[1] / dist) * enemy["speed"]
        
        if enemy["last_hit_timer"] > 0: enemy["last_hit_timer"] -= 1
        return True

    def _spawn_wave(self):
        num_enemies = 3 + self.current_wave
        base_health = 50 * (1.15 ** (self.current_wave - 1))
        base_speed = 1.0 + self.current_wave * 0.1
        
        for i in range(num_enemies):
            start_pos = self._grid_to_iso(self.enemy_path[0])
            offset_start_pos = [start_pos[0] - i * 20, start_pos[1] - i * 10]
            self.enemies.append({
                "screen_pos": offset_start_pos,
                "path_index": 0,
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed + self.rng.uniform(-0.1, 0.1),
                "damage": 10,
                "gold_value": 5 + self.current_wave,
                "last_hit_timer": 0
            })
    
    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(0.5, 2.0) * speed_mult
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.rng.integers(20, 40),
                "radius": self.rng.uniform(2, 5),
                "color": color
            })

    def _set_message(self, text):
        self.message = text
        self.message_timer = 90 # 3 seconds at 30fps

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.grid_size[1]):
            for c in range(self.grid_size[0]):
                p1 = self._grid_to_iso((c, r))
                p2 = self._grid_to_iso((c + 1, r))
                p3 = self._grid_to_iso((c + 1, r + 1))
                p4 = self._grid_to_iso((c, r + 1))
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # Draw path
        for i in range(len(self.enemy_path) - 1):
            p1_grid = self.enemy_path[i]
            p2_grid = self.enemy_path[i+1]
            # Draw wider path tiles
            self._draw_iso_rect(p1_grid, (1, 1), self.COLOR_PATH)
        self._draw_iso_rect(self.enemy_path[-1], (1, 1), self.COLOR_PATH)

        # Draw placement zones
        for zone in self.placement_zones:
            self._draw_iso_rect(zone, (1, 1), self.COLOR_PLACEABLE, filled=False, border_width=1)

        # Draw base
        self._draw_iso_cube((11, 9.5, 0), (1, 1, 0.5), self.COLOR_BASE)
        
        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            color = spec["color"]
            if tower["cooldown"] > 0:
                # Dim color when on cooldown
                color = tuple(c * 0.6 for c in color)
            self._draw_iso_cube((tower["pos"][0], tower["pos"][1], 0), (0.8, 0.8, 0.7), color)
            # Draw range indicator if cursor is on it
            if tower["pos"] == self.cursor_pos:
                pygame.gfxdraw.aacircle(self.screen, *map(int, self._grid_to_iso(tower["pos"])), int(spec["range"]), (255,255,255,100))

        # Draw cursor
        if self.game_phase == "placement":
            self._draw_iso_rect(self.cursor_pos, (1, 1), self.COLOR_CURSOR, filled=False, border_width=2)

        # Draw enemies
        for enemy in sorted(self.enemies, key=lambda e: e["screen_pos"][1]):
            pos = (int(enemy["screen_pos"][0]), int(enemy["screen_pos"][1]))
            color = (200, 40, 40) if enemy["last_hit_timer"] == 0 else (255, 150, 150)
            pygame.draw.circle(self.screen, color, pos, 6)
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (50,0,0), (pos[0] - 8, pos[1] - 12, 16, 4))
            pygame.draw.rect(self.screen, (0,255,0), (pos[0] - 8, pos[1] - 12, int(16 * health_pct), 4))
            
        # Draw projectiles
        for proj in self.projectiles:
            color = self.TOWER_SPECS[proj["type"]]["color"]
            pygame.draw.circle(self.screen, color, (int(proj["pos"][0]), int(proj["pos"][1])), 3)
            pygame.draw.circle(self.screen, (255,255,255), (int(proj["pos"][0]), int(proj["pos"][1])), 1)

        # Draw particles
        for p in self.particles:
            if p["radius"] > 1:
                pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(p["radius"]))

    def _render_ui(self):
        # Top Left: Gold
        gold_text = self.font_ui.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, 10))

        # Top Right: Base Health
        health_color = self.COLOR_HEALTH if self.base_health > 30 else self.COLOR_BASE_DMG
        health_text = self.font_ui.render(f"BASE HP: {self.base_health}", True, health_color)
        self.screen.blit(health_text, (self.screen_width - health_text.get_width() - 10, 10))

        # Bottom Center: Wave Info
        if self.game_phase == "placement" and self.current_wave < self.max_waves:
            wave_text = self.font_wave.render(f"WAVE {self.current_wave + 1}", True, self.COLOR_TEXT)
            text_rect = wave_text.get_rect(center=(self.screen_width / 2, self.screen_height - 30))
            self.screen.blit(wave_text, text_rect)
        elif self.game_phase == "wave":
            wave_text = self.font_ui.render(f"WAVE {self.current_wave} IN PROGRESS", True, self.COLOR_TEXT)
            text_rect = wave_text.get_rect(center=(self.screen_width / 2, self.screen_height - 20))
            self.screen.blit(wave_text, text_rect)

        # Message display
        if self.message_timer > 0:
            msg_surf = self.font_msg.render(self.message, True, self.COLOR_CURSOR)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_surf, msg_rect)

        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            end_text = self.font_wave.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.current_wave,
            "steps": self.steps,
            "enemies_left": len(self.enemies),
            "game_phase": self.game_phase,
        }

    def close(self):
        pygame.quit()

    # --- Drawing Helpers ---
    def _grid_to_iso(self, grid_pos):
        x, y = grid_pos
        iso_x = (x - y) * self.tile_width_half + self.world_offset[0]
        iso_y = (x + y) * self.tile_height_half + self.world_offset[1]
        return iso_x, iso_y

    def _draw_iso_rect(self, grid_pos, size, color, filled=True, border_width=0):
        x, y = grid_pos
        w, h = size
        p1 = self._grid_to_iso((x, y))
        p2 = self._grid_to_iso((x + w, y))
        p3 = self._grid_to_iso((x + w, y + h))
        p4 = self._grid_to_iso((x, y + h))
        points = [p1, p2, p3, p4]
        if filled:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        if border_width > 0:
            pygame.draw.lines(self.screen, color, True, points, border_width)

    def _draw_iso_cube(self, pos, size, color):
        x, y, z = pos
        w, d, h = size
        
        top_color = color
        side_color_1 = tuple(max(0, c - 40) for c in color)
        side_color_2 = tuple(max(0, c - 60) for c in color)
        
        # Points for the cube
        p = [self._grid_to_iso((x + dw, y + dd)) for dd in [0, d] for dw in [0, w]]
        p_top = [(px, py - z * self.tile_height_half * 2) for px, py in p]
        p_bottom = [(px, py - (z+h) * self.tile_height_half * 2) for px, py in p]
        
        # Draw sides first
        pygame.gfxdraw.filled_polygon(self.screen, [p_bottom[1], p_bottom[3], p_top[3], p_top[1]], side_color_1)
        pygame.gfxdraw.filled_polygon(self.screen, [p_bottom[2], p_bottom[3], p_top[3], p_top[2]], side_color_2)
        # Draw top
        pygame.gfxdraw.filled_polygon(self.screen, [p_bottom[0], p_bottom[1], p_bottom[3], p_bottom[2]], top_color)
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be executed when the environment is used by Gymnasium runners
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Setup ---
    pygame.display.set_caption(env.game_description)
    game_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    action = [0, 0, 0] # No-op
    
    while not done:
        # --- Pygame event handling for manual play ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key

        keys = pygame.key.get_pressed()
        
        # Map keys to action space
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        env.clock.tick(30)

    env.close()