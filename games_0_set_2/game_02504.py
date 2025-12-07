
# Generated: 2025-08-27T20:35:03.232400
# Source Brief: brief_02504.md
# Brief Index: 2504

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # --- User-facing strings ---
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press Space to place the selected tower. Hold Shift to cycle tower types."
    )
    game_description = (
        "Defend your base from waves of enemies by strategically placing musical towers in a top-down rhythm-based tower defense game."
    )

    # --- Frame advance setting ---
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 30 * 180 # 3 minutes at 30fps
    FPS = 30
    CURSOR_SPEED = 8

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (40, 50, 80)
    COLOR_PATH_BORDER = (60, 75, 120)
    COLOR_GRID = (30, 40, 60)
    COLOR_BASE = (0, 150, 255)
    COLOR_BASE_GLOW = (0, 100, 200)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (200, 40, 40)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 40, 60, 180)
    COLOR_CURSOR_VALID = (255, 255, 255)
    COLOR_CURSOR_INVALID = (255, 0, 0)
    
    # --- Tower Types ---
    TOWER_SPECS = {
        0: {"name": "Pulse Cannon", "cost": 100, "range": 70, "damage": 2, "fire_rate": 15, "color": (0, 255, 150), "proj_speed": 8},
        1: {"name": "Sonic Boom", "cost": 150, "range": 100, "damage": 5, "fire_rate": 45, "color": (255, 150, 0), "proj_speed": 6},
    }

    # --- Wave Config ---
    WAVE_CONFIG = [
        {"count": 5, "health": 10, "speed": 1.0, "bounty": 10},
        {"count": 7, "health": 12, "speed": 1.1, "bounty": 12},
        {"count": 10, "health": 15, "speed": 1.2, "bounty": 15},
        {"count": 13, "health": 18, "speed": 1.3, "bounty": 18},
        {"count": 16, "health": 22, "speed": 1.4, "bounty": 20},
    ]
    WAVE_PREP_TIME = 30 * 5 # 5 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- Game Path ---
        self.path_waypoints = [
            pygame.Vector2(-20, 100),
            pygame.Vector2(150, 100),
            pygame.Vector2(150, 300),
            pygame.Vector2(450, 300),
            pygame.Vector2(450, 100),
            pygame.Vector2(self.SCREEN_WIDTH + 20, 100)
        ]
        self.path_width = 40

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.base_health = 0
        self.resources = 0
        self.current_wave_index = 0
        self.wave_active = False
        self.wave_timer = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(0, 0)
        self.selected_tower_type = 0
        self.last_shift_press = False
        self.last_space_press = False
        self.base_hit_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.base_health = 100
        self.resources = 250
        
        self.current_wave_index = -1
        self.wave_active = False
        self.wave_timer = self.WAVE_PREP_TIME // 2

        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.selected_tower_type = 0
        self.last_shift_press = True # Prevent action on first frame
        self.last_space_press = True # Prevent action on first frame

        self.base_hit_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Time penalty

        self._handle_input(action)
        
        # --- Game Logic Update ---
        self.steps += 1
        
        if self.base_hit_timer > 0:
            self.base_hit_timer -= 1
            
        if not self.wave_active:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                reward += self._start_next_wave()
        
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # --- Check Wave Completion ---
        if self.wave_active and not self.enemies:
            self.wave_active = False
            self.wave_timer = self.WAVE_PREP_TIME
            reward += 50
            self.score += 250
            
            # --- Check Victory Condition ---
            if self.current_wave_index >= len(self.WAVE_CONFIG) - 1:
                self.victory = True
                self.game_over = True
                reward += 100
                self.score += 1000

        # --- Check Termination Conditions ---
        terminated = self.base_health <= 0 or self.victory or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.base_health <= 0:
                 self.score -= 500 # Penalty for losing

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # --- Cycle Tower (on press) ---
        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_press = shift_held

        # --- Place Tower (on press) ---
        if space_held and not self.last_space_press:
            self._place_tower()
        self.last_space_press = space_held

    def _start_next_wave(self):
        self.current_wave_index += 1
        if self.current_wave_index < len(self.WAVE_CONFIG):
            self.wave_active = True
            wave_info = self.WAVE_CONFIG[self.current_wave_index]
            for i in range(wave_info["count"]):
                enemy = {
                    "pos": pygame.Vector2(self.path_waypoints[0].x - i * 30, self.path_waypoints[0].y),
                    "health": wave_info["health"],
                    "max_health": wave_info["health"],
                    "speed": wave_info["speed"],
                    "waypoint_index": 1,
                    "bounty": wave_info["bounty"],
                    "id": self.np_random.integers(1, 1_000_000)
                }
                self.enemies.append(enemy)
            return 10 # Small reward for starting a wave
        return 0

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["waypoint_index"] >= len(self.path_waypoints):
                self.base_health -= 10
                self.base_hit_timer = 10 # Screen flash for 10 frames
                self.enemies.remove(enemy)
                reward -= 5
                self.score -= 50
                # sfx: base_damage.wav
                continue

            target_pos = self.path_waypoints[enemy["waypoint_index"]]
            direction = (target_pos - enemy["pos"]).normalize() if (target_pos - enemy["pos"]).length() > 0 else pygame.Vector2(0)
            enemy["pos"] += direction * enemy["speed"]

            if (enemy["pos"] - target_pos).length_squared() < (enemy["speed"] * enemy["speed"]):
                enemy["waypoint_index"] += 1
        return reward

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] == 0:
                # Find a target
                target = None
                min_dist_sq = tower["range"] ** 2
                for enemy in self.enemies:
                    dist_sq = (tower["pos"] - enemy["pos"]).length_squared()
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        target = enemy
                
                if target:
                    # Fire projectile
                    self.projectiles.append({
                        "pos": pygame.Vector2(tower["pos"]),
                        "target_id": target["id"],
                        "speed": tower["proj_speed"],
                        "damage": tower["damage"],
                        "color": tower["color"]
                    })
                    tower["cooldown"] = tower["fire_rate"]
                    # sfx: tower_fire.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e["id"] == proj["target_id"]), None)

            if not target_enemy:
                self.projectiles.remove(proj)
                continue
            
            direction = (target_enemy["pos"] - proj["pos"]).normalize() if (target_enemy["pos"] - proj["pos"]).length() > 0 else pygame.Vector2(0)
            proj["pos"] += direction * proj["speed"]

            if (proj["pos"] - target_enemy["pos"]).length_squared() < 100: # Hit radius
                target_enemy["health"] -= proj["damage"]
                reward += 0.1 # Reward for hitting
                self._create_particles(proj["pos"], proj["color"], 5, 2)
                self.projectiles.remove(proj)
                # sfx: enemy_hit.wav

                if target_enemy["health"] <= 0:
                    reward += 1
                    self.score += target_enemy["bounty"] * 2
                    self.resources += target_enemy["bounty"]
                    self._create_particles(target_enemy["pos"], self.COLOR_ENEMY, 20, 4)
                    self.enemies.remove(target_enemy)
                    # sfx: enemy_destroy.wav
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _is_valid_placement(self, pos):
        # Check resources
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources < spec["cost"]:
            return False

        # Check proximity to path
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            # Simple bounding box check for the segment
            rect = pygame.Rect(min(p1.x, p2.x) - self.path_width, min(p1.y, p2.y) - self.path_width,
                               abs(p1.x - p2.x) + 2 * self.path_width, abs(p1.y - p2.y) + 2 * self.path_width)
            if rect.collidepoint(pos):
                return False
        
        # Check proximity to other towers
        for tower in self.towers:
            if (pos - tower["pos"]).length_squared() < (self.path_width/2)**2:
                return False
        
        return True

    def _place_tower(self):
        if self._is_valid_placement(self.cursor_pos):
            spec = self.TOWER_SPECS[self.selected_tower_type]
            self.resources -= spec["cost"]
            self.score += spec["cost"]
            self.towers.append({
                "pos": pygame.Vector2(self.cursor_pos),
                "type": self.selected_tower_type,
                "range": spec["range"],
                "damage": spec["damage"],
                "fire_rate": spec["fire_rate"],
                "cooldown": 0,
                "color": spec["color"],
                "proj_speed": spec["proj_speed"]
            })
            # sfx: place_tower.wav
            
    def _create_particles(self, pos, color, count, speed_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "lifespan": self.np_random.integers(10, 20),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        self._render_grid()

        # --- Game Elements ---
        self._render_path()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()

        # --- UI Overlay ---
        self._render_ui()

        # --- Screen Effects ---
        if self.base_hit_timer > 0:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.base_hit_timer / 10))
            s.fill((255, 0, 0, alpha))
            self.screen.blit(s, (0, 0))
        
        if self.game_over:
            self._render_game_over()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        beat = (self.steps % self.FPS) / self.FPS
        glow = 5 + 5 * math.sin(beat * math.pi * 2)
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
    def _render_path(self):
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, p1, p2, self.path_width + 4)
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.path_width)

    def _render_base(self):
        base_pos = (int(self.path_waypoints[-1].x - 20), int(self.path_waypoints[-1].y))
        beat = (self.steps % (self.FPS * 2)) / (self.FPS * 2)
        radius = 20 + 5 * math.sin(beat * math.pi * 2)
        pygame.gfxdraw.filled_circle(self.screen, base_pos[0], base_pos[1], int(radius), self.COLOR_BASE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, base_pos[0], base_pos[1], 18, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, base_pos[0], base_pos[1], 18, self.COLOR_BASE)

    def _render_towers(self):
        beat = (self.steps % self.FPS) / self.FPS
        for tower in self.towers:
            pos_int = (int(tower["pos"].x), int(tower["pos"].y))
            # Draw range indicator
            if tower["cooldown"] > 0:
                s = pygame.Surface((tower["range"]*2, tower["range"]*2), pygame.SRCALPHA)
                cooldown_pct = tower["cooldown"] / tower["fire_rate"]
                pygame.draw.arc(s, tower["color"] + (50,), (0, 0, tower["range"]*2, tower["range"]*2), -math.pi/2, -math.pi/2 + (2*math.pi * cooldown_pct), 3)
                self.screen.blit(s, (pos_int[0] - tower["range"], pos_int[1] - tower["range"]))
            
            # Draw tower body
            glow_radius = 12 + 2 * math.sin(beat * math.pi * 2)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(glow_radius), tower["color"] + (100,))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 10, tower["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 10, tower["color"])

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"].x), int(enemy["pos"].y))
            
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 10, self.COLOR_ENEMY_GLOW)
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_ENEMY)

            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            bar_width = 16
            bar_height = 4
            bar_x = pos_int[0] - bar_width // 2
            bar_y = pos_int[1] - 15
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * health_pct), bar_height))

    def _render_projectiles(self):
        beat = (self.steps % (self.FPS // 2)) / (self.FPS // 2)
        for proj in self.projectiles:
            pos_int = (int(proj["pos"].x), int(proj["pos"].y))
            radius = 4 + 2 * abs(math.sin(beat * math.pi))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(radius), proj["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(radius), proj["color"])

    def _render_particles(self):
        for p in self.particles:
            pos_int = (int(p["pos"].x), int(p["pos"].y))
            alpha = int(255 * (p["lifespan"] / 20))
            color_with_alpha = p["color"] + (alpha,) if len(p["color"]) == 3 else (p["color"][0], p["color"][1], p["color"][2], alpha)
            temp_surf = pygame.Surface((int(p["radius"])*2, int(p["radius"])*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (int(p["radius"]), int(p["radius"])), int(p["radius"]))
            self.screen.blit(temp_surf, (pos_int[0] - int(p["radius"]), pos_int[1] - int(p["radius"])))

    def _render_cursor(self):
        pos_int = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        spec = self.TOWER_SPECS[self.selected_tower_type]
        is_valid = self._is_valid_placement(self.cursor_pos)
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        color_with_alpha = color + (100,)

        # Draw range indicator
        s = pygame.Surface((spec["range"]*2, spec["range"]*2), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(s, spec["range"], spec["range"], spec["range"] - 1, color_with_alpha)
        self.screen.blit(s, (pos_int[0] - spec["range"], pos_int[1] - spec["range"]))
        
        # Draw cursor
        pygame.draw.line(self.screen, color, (pos_int[0] - 10, pos_int[1]), (pos_int[0] + 10, pos_int[1]), 2)
        pygame.draw.line(self.screen, color, (pos_int[0], pos_int[1] - 10), (pos_int[0], pos_int[1] + 10), 2)
        
    def _render_ui(self):
        # UI Background Panel
        panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        panel.fill(self.COLOR_UI_BG)
        self.screen.blit(panel, (0,0))
        
        # Health
        health_text = self.font_small.render(f"Base: {max(0, self.base_health)}/100", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        # Resources
        res_text = self.font_small.render(f"Resources: ${self.resources}", True, (255, 223, 0))
        self.screen.blit(res_text, (150, 10))
        
        # Wave
        wave_str = f"Wave: {self.current_wave_index + 1}/{len(self.WAVE_CONFIG)}" if self.current_wave_index >= 0 else "Wave: 1/5"
        if not self.wave_active and not self.game_over:
            wave_str += f" (in {self.wave_timer // self.FPS + 1}s)"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (300, 10))
        
        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_text = self.font_small.render(f"Tower: {spec['name']} (${spec['cost']})", True, spec['color'])
        self.screen.blit(tower_text, (450, 10))

        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 50))
        self.screen.blit(score_text, score_rect)

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0,0))
        
        msg = "VICTORY!" if self.victory else "GAME OVER"
        color = (0, 255, 100) if self.victory else (255, 50, 50)
        
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        self.screen.blit(text, text_rect)
        
        final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
        self.screen.blit(final_score_text, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave_index + 1,
            "enemies_left": len(self.enemies)
        }
        
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythm Tower Defense")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling and frame rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    env.close()
    pygame.quit()