
# Generated: 2025-08-27T13:48:43.300244
# Source Brief: brief_00486.md
# Brief Index: 486

        
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
        "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from 10 waves of enemies by placing towers. Earn gold by defeating enemies and use it to build more defenses. New towers unlock as you survive longer."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_PATH = (45, 45, 55)
    COLOR_BASE = (0, 200, 100)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_INVALID = (255, 0, 0, 100)
    COLOR_VALID = (0, 255, 0, 100)
    
    TOWER_COLORS = [
        (80, 120, 255),  # Blue - Basic
        (255, 200, 50),  # Yellow - Sniper
        (200, 80, 255),  # Purple - Gatling
    ]

    TOWER_STATS = [
        {"cost": 50, "range": 80, "damage": 12, "fire_rate": 20, "proj_speed": 8},
        {"cost": 75, "range": 150, "damage": 50, "fire_rate": 60, "proj_speed": 12},
        {"cost": 100, "range": 60, "damage": 5, "fire_rate": 5, "proj_speed": 10},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Etc...        
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.gold = 150

        self.cursor_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.selected_tower_type_idx = 0
        self.available_tower_types = [0]
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.time_to_next_wave = 60  # Steps before first wave
        self.enemies_to_spawn_in_wave = 0
        self.enemy_spawn_timer = 0
        self.enemies_in_wave = 0

        self.last_shift_held = False
        self.last_space_held = False
        
        self.enemy_path = [
            pygame.Vector2(-20, 100), pygame.Vector2(100, 100),
            pygame.Vector2(100, 300), pygame.Vector2(300, 300),
            pygame.Vector2(300, 50), pygame.Vector2(540, 50),
            pygame.Vector2(540, 250), pygame.Vector2(self.SCREEN_WIDTH + 20, 250)
        ]
        self.base_pos = pygame.Vector2(self.SCREEN_WIDTH - 40, 250)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over and not self.game_won:
            self.steps += 1
            reward += 0.001 # Small survival reward

            # Unpack factorized action
            self._handle_input(action)
            
            wave_cleared_reward = self._update_waves()
            reward += wave_cleared_reward
            
            enemy_event_reward = self._update_entities()
            reward += enemy_event_reward

        terminated = self.base_health <= 0 or self.steps >= 1000 or self.game_won
        
        if self.base_health <= 0 and not self.game_over:
            self.game_over = True
            reward = -50.0

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        cursor_speed = 10
        if movement == 1: self.cursor_pos.y -= cursor_speed
        elif movement == 2: self.cursor_pos.y += cursor_speed
        elif movement == 3: self.cursor_pos.x -= cursor_speed
        elif movement == 4: self.cursor_pos.x += cursor_speed
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # --- Cycle Tower Type ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.available_tower_types)
        self.last_shift_held = shift_held

        # --- Place Tower ---
        if space_held and not self.last_space_held:
            tower_type = self.available_tower_types[self.selected_tower_type_idx]
            cost = self.TOWER_STATS[tower_type]["cost"]
            if self.gold >= cost and self._is_valid_placement(self.cursor_pos):
                self.gold -= cost
                self.towers.append({
                    "pos": self.cursor_pos.copy(),
                    "type": tower_type,
                    "cooldown": 0,
                    "target": None
                })
                # sfx: place_tower.wav
        self.last_space_held = space_held

    def _is_valid_placement(self, pos):
        # Check distance to path
        for i in range(len(self.enemy_path) - 1):
            p1 = self.enemy_path[i]
            p2 = self.enemy_path[i+1]
            rect = pygame.Rect(min(p1.x, p2.x) - 20, min(p1.y, p2.y) - 20, abs(p1.x - p2.x) + 40, abs(p1.y - p2.y) + 40)
            if rect.collidepoint(pos):
                return False
        # Check distance to other towers
        for tower in self.towers:
            if pos.distance_to(tower["pos"]) < 20:
                return False
        # Check distance to base
        if pos.distance_to(self.base_pos) < 30:
            return False
        return True

    def _update_waves(self):
        if self.wave_in_progress:
            # Spawning enemies
            if self.enemies_to_spawn_in_wave > 0:
                self.enemy_spawn_timer -= 1
                if self.enemy_spawn_timer <= 0:
                    self._spawn_enemy()
                    self.enemies_to_spawn_in_wave -= 1
                    self.enemy_spawn_timer = 8
            # Check for wave clear
            elif len(self.enemies) == 0:
                self.wave_in_progress = False
                self.time_to_next_wave = 150
                
                if self.wave_number >= 10:
                    self.game_won = True
                    return 100.0
                else:
                    return 50.0
        else: # Between waves
            self.time_to_next_wave -= 1
            if self.time_to_next_wave <= 0 and not self.game_won:
                self.wave_in_progress = True
                self.wave_number += 1
                
                if self.wave_number == 4: self.available_tower_types.append(1)
                if self.wave_number == 7: self.available_tower_types.append(2)
                
                self.enemies_in_wave = 5 + self.wave_number * 2
                self.enemies_to_spawn_in_wave = self.enemies_in_wave
        return 0.0

    def _spawn_enemy(self):
        health_scale = (1.05) ** (self.wave_number - 1)
        speed_scale = 1 + 0.05 * (self.wave_number - 1)
        
        enemy = {
            "pos": self.enemy_path[0].copy(),
            "path_idx": 0,
            "max_health": 50 * health_scale,
            "health": 50 * health_scale,
            "speed": 1.5 * speed_scale,
            "damage_flash": 0
        }
        self.enemies.append(enemy)

    def _update_entities(self):
        reward = 0
        
        # --- Update Towers ---
        for tower in self.towers:
            stats = self.TOWER_STATS[tower["type"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
            else:
                target = None
                min_dist = stats["range"]
                for enemy in self.enemies:
                    dist = tower["pos"].distance_to(enemy["pos"])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    tower["cooldown"] = stats["fire_rate"]
                    self.projectiles.append({
                        "pos": tower["pos"].copy(),
                        "target": target,
                        "type": tower["type"],
                        "speed": stats["proj_speed"],
                        "damage": stats["damage"]
                    })
                    # sfx: tower_shoot.wav

        # --- Update Projectiles ---
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            direction = (proj["target"]["pos"] - proj["pos"]).normalize()
            proj["pos"] += direction * proj["speed"]
            
            if proj["pos"].distance_to(proj["target"]["pos"]) < 8:
                proj["target"]["health"] -= proj["damage"]
                proj["target"]["damage_flash"] = 3
                self._create_particles(proj["pos"], self.TOWER_COLORS[proj["type"]], 5)
                self.projectiles.remove(proj)
                # sfx: projectile_hit.wav

        # --- Update Enemies ---
        for enemy in self.enemies[:]:
            if enemy["damage_flash"] > 0:
                enemy["damage_flash"] -= 1

            if enemy["health"] <= 0:
                self.gold += 5 + self.wave_number
                reward += 1.0
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 15)
                self.enemies.remove(enemy)
                # sfx: enemy_die.wav
                continue

            if enemy["path_idx"] < len(self.enemy_path) - 1:
                target_pos = self.enemy_path[enemy["path_idx"] + 1]
                direction = (target_pos - enemy["pos"])
                if direction.length() < enemy["speed"]:
                    enemy["pos"] = target_pos.copy()
                    enemy["path_idx"] += 1
                else:
                    enemy["pos"] += direction.normalize() * enemy["speed"]
            else:
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                reward -= 5.0
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
        
        # --- Update Particles ---
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                
        return reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                "life": self.np_random.integers(10, 20),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [tuple(p) for p in self.enemy_path], 30)

        base_rect = pygame.Rect(0, 0, 40, 40)
        base_rect.center = tuple(self.base_pos)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)

        for tower in self.towers:
            stats = self.TOWER_STATS[tower["type"]]
            color = self.TOWER_COLORS[tower["type"]]
            pos = (int(tower["pos"].x), int(tower["pos"].y))
            
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], stats["range"], (*color, 50))
            
            if tower["type"] == 0:
                points = [(pos[0], pos[1] - 8), (pos[0] - 7, pos[1] + 5), (pos[0] + 7, pos[1] + 5)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif tower["type"] == 1:
                rect = pygame.Rect(pos[0] - 6, pos[1] - 6, 12, 12)
                pygame.draw.rect(self.screen, color, rect)
            elif tower["type"] == 2:
                points = [(pos[0] + 8 * math.cos(math.radians(90 + i * 72)), pos[1] + 8 * math.sin(math.radians(90 + i * 72))) for i in range(5)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            color = (255, 255, 255) if enemy["damage_flash"] > 0 else self.COLOR_ENEMY
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, color)
            
            hb_len, hb_y = 14, pos[1] - 12
            health_ratio = enemy["health"] / enemy["max_health"]
            pygame.draw.line(self.screen, (100, 0, 0), (pos[0] - hb_len//2, hb_y), (pos[0] + hb_len//2, hb_y), 2)
            pygame.draw.line(self.screen, (0, 255, 0), (pos[0] - hb_len//2, hb_y), (pos[0] - hb_len//2 + int(hb_len * health_ratio), hb_y), 2)

        for proj in self.projectiles:
            pos = (int(proj["pos"].x), int(proj["pos"].y))
            color = self.TOWER_COLORS[proj["type"]]
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)
            
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            radius = int(p["radius"] * (p["life"] / 20.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p["color"])

    def _render_ui(self):
        cursor_pos_int = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        tower_type = self.available_tower_types[self.selected_tower_type_idx]
        stats = self.TOWER_STATS[tower_type]
        can_afford = self.gold >= stats["cost"]
        is_valid = self._is_valid_placement(self.cursor_pos)
        
        preview_color = self.COLOR_VALID if (can_afford and is_valid) else self.COLOR_INVALID
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], stats["range"], preview_color)
        
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos_int[0] - 10, cursor_pos_int[1]), (cursor_pos_int[0] + 10, cursor_pos_int[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_pos_int[0], cursor_pos_int[1] - 10), (cursor_pos_int[0], cursor_pos_int[1] + 10), 1)

        self._draw_text(f"💰 Gold: {self.gold}", (10, 10), self.font_m)
        self._draw_text(f"❤️ Base: {self.base_health}/{self.max_base_health}", (10, 40), self.font_m)
        
        wave_str = f"Wave: {self.wave_number}/10"
        if not self.wave_in_progress and not self.game_won:
            wave_str += f" (Next in {self.time_to_next_wave // 30 + 1}s)" if self.time_to_next_wave > 0 else ""
        self._draw_text(wave_str, (self.SCREEN_WIDTH - 10, 10), self.font_m, align="topright")

        tower_name = ["Basic", "Sniper", "Gatling"][tower_type]
        info_str = f"Selected: {tower_name} (Cost: {stats['cost']})"
        self._draw_text(info_str, (self.SCREEN_WIDTH - 10, self.SCREEN_HEIGHT - 10), self.font_s, align="bottomright")

        if self.game_over:
            self._draw_text("GAME OVER", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_l, align="center")
        elif self.game_won:
            self._draw_text("YOU WIN!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_l, align="center")

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, align, pos)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.wave_number,
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
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Tower Defense Gym Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement_action, space_action, shift_action = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        key_to_action = {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement_action = move_action
                break
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Gold: {info['gold']}, Health: {info['base_health']}, Wave: {info['wave']}")
            
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print("Game Over or Won!")
            pygame.time.wait(2000)
            running = False
            
        env.clock.tick(30)
        
    env.close()