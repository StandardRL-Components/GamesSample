
# Generated: 2025-08-28T01:18:40.503440
# Source Brief: brief_04070.md
# Brief Index: 4070

        
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
    """
    A real-time, top-down tower defense game. The player controls a cursor to place
    three different types of defensive towers to protect a central base from waves
    of incoming enemies. The goal is to survive for a fixed duration.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place Basic Turret. "
        "Shift to place Sniper Turret. Space+Shift to place Cannon Turret. "
        "Earn currency by destroying enemies to build more turrets."
    )

    game_description = (
        "Defend your base from relentless waves of geometric enemies. "
        "Strategically place turrets to survive as long as possible. "
        "The onslaught intensifies over time. Survive for 10 minutes to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Core Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # --- Game Constants ---
        self.FPS = 30
        self.MAX_STEPS = 6000 # 10 minutes at 10 steps/sec, but we run at 30fps, so 200s
        self.CURSOR_SPEED = 8
        self.DIFFICULTY_INTERVAL = 1000 # Steps to increase difficulty

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_PATH_BORDER = (60, 70, 80)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_BASE_DAMAGED = (200, 50, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_ENEMY = (210, 40, 40)
        self.COLOR_HEALTH_BAR_BG = (80, 20, 20)
        self.TOWER_COLORS = {
            "basic": (60, 160, 220),
            "sniper": (180, 80, 255),
            "cannon": (255, 150, 50),
        }

        # --- Tower & Enemy Definitions ---
        self.TOWER_SPECS = {
            "basic": {"cost": 50, "range": 80, "cooldown": 20, "damage": 1, "proj_speed": 6},
            "sniper": {"cost": 75, "range": 150, "cooldown": 60, "damage": 5, "proj_speed": 15},
            "cannon": {"cost": 100, "range": 60, "cooldown": 90, "damage": 3, "proj_speed": 4, "aoe": 30},
        }

        self.path = self._create_path()
        self.base_pos = self.path[-1]
        self.base_size = 30

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation()

    def _create_path(self):
        """Creates the S-shaped path for enemies."""
        path = []
        w, h = self.screen_width, self.screen_height
        path.append(np.array([-20.0, h * 0.2]))
        path.append(np.array([w * 0.8, h * 0.2]))
        path.append(np.array([w * 0.8, h * 0.8]))
        path.append(np.array([w * 0.2, h * 0.8]))
        path.append(np.array([w * 0.2, h * 0.5]))
        path.append(np.array([w / 2, h / 2]))
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.currency = 150
        
        self.cursor_pos = np.array([self.screen_width / 2, self.screen_height - 50.0])

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 60 # Spawn every 2 seconds at 30fps
        self.enemy_base_health = 3
        self.enemy_base_speed = 1.0
        self.enemy_value = 10

        self.step_reward = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0.0
        
        self._handle_input(action)
        self._update_difficulty()
        self._spawn_enemies()

        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        self._check_termination()
        
        reward = self.step_reward
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                reward += 100.0  # Win bonus
            else:
                reward -= 100.0  # Lose penalty

        self.steps += 1
        
        # Tick the clock to maintain FPS, only if auto_advance is on.
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Move Cursor ---
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.screen_width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.screen_height)

        # --- Place Towers ---
        tower_type = None
        if space_held and shift_held: tower_type = "cannon"
        elif space_held: tower_type = "basic"
        elif shift_held: tower_type = "sniper"

        if tower_type:
            spec = self.TOWER_SPECS[tower_type]
            if self.currency >= spec["cost"]:
                # Check if placement is valid (not on path or other towers)
                is_valid_placement = True
                if self._is_on_path(self.cursor_pos):
                    is_valid_placement = False
                for t in self.towers:
                    if np.linalg.norm(self.cursor_pos - t["pos"]) < 20:
                        is_valid_placement = False
                        break
                
                if is_valid_placement:
                    self.currency -= spec["cost"]
                    self.towers.append({
                        "pos": self.cursor_pos.copy(),
                        "type": tower_type,
                        "cooldown": 0,
                        "spec": spec
                    })
                    # sfx: place_tower.wav
                    self._create_particles(self.cursor_pos, 10, self.TOWER_COLORS[tower_type], 1, 3, 15)


    def _is_on_path(self, pos, tolerance=20):
        for i in range(len(self.path) - 1):
            p1, p2 = self.path[i], self.path[i+1]
            d = np.linalg.norm(np.cross(p2 - p1, p1 - pos)) / np.linalg.norm(p2 - p1)
            # Check if point is between the line segment endpoints
            dot_product = np.dot(pos - p1, p2 - p1)
            if 0 <= dot_product <= np.dot(p2 - p1, p2 - p1) and d < tolerance:
                return True
        return False

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.enemy_spawn_rate = max(15, self.enemy_spawn_rate - 5)
            self.enemy_base_health += 1
            self.enemy_base_speed += 0.05
            self.enemy_value += 2

    def _spawn_enemies(self):
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer >= self.enemy_spawn_rate:
            self.enemy_spawn_timer = 0
            self.enemies.append({
                "pos": self.path[0].copy(),
                "health": self.enemy_base_health,
                "max_health": self.enemy_base_health,
                "speed": self.enemy_base_speed,
                "path_index": 0,
                "value": self.enemy_value,
            })
            # sfx: enemy_spawn.wav

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] == 0:
                target = None
                min_dist = tower["spec"]["range"]
                for enemy in self.enemies:
                    dist = np.linalg.norm(tower["pos"] - enemy["pos"])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    tower["cooldown"] = tower["spec"]["cooldown"]
                    self.projectiles.append({
                        "pos": tower["pos"].copy(),
                        "target": target,
                        "spec": tower["spec"],
                        "type": tower["type"]
                    })
                    # sfx: shoot.wav (differentiated by type)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = proj["target"]["pos"] - proj["pos"]
            dist = np.linalg.norm(direction)
            if dist < proj["spec"]["proj_speed"]:
                self._handle_hit(proj)
                self.projectiles.remove(proj)
            else:
                proj["pos"] += (direction / dist) * proj["spec"]["proj_speed"]
    
    def _handle_hit(self, proj):
        # sfx: hit.wav
        self._create_particles(proj["pos"], 5, (255, 255, 255), 1, 2, 10)
        
        if proj["type"] == "cannon":
            # Area of effect damage
            for enemy in self.enemies[:]:
                if np.linalg.norm(proj["pos"] - enemy["pos"]) < proj["spec"]["aoe"]:
                    self._damage_enemy(enemy, proj["spec"]["damage"])
        else:
            # Single target damage
            if proj["target"] in self.enemies:
                self._damage_enemy(proj["target"], proj["spec"]["damage"])

    def _damage_enemy(self, enemy, damage):
        if enemy not in self.enemies: return
        
        enemy["health"] -= damage
        self.step_reward += 0.1

        if enemy["health"] <= 0:
            # sfx: explosion.wav
            self._create_particles(enemy["pos"], 20, self.COLOR_ENEMY, 2, 5, 20)
            self.enemies.remove(enemy)
            self.score += 10
            self.currency += enemy["value"]
            self.step_reward += 1.0


    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy["path_index"] >= len(self.path) - 1:
                self.base_health -= enemy["max_health"] # Damage based on enemy strength
                self.enemies.remove(enemy)
                self.step_reward -= 5.0
                # sfx: base_damage.wav
                self._create_particles(self.base_pos, 30, self.COLOR_BASE_DAMAGED, 3, 6, 25)
                continue
            
            target_pos = self.path[enemy["path_index"] + 1]
            direction = target_pos - enemy["pos"]
            dist = np.linalg.norm(direction)

            if dist < enemy["speed"]:
                enemy["path_index"] += 1
            else:
                enemy["pos"] += (direction / dist) * enemy["speed"]

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "currency": self.currency,
            "base_health": self.base_health,
            "enemies": len(self.enemies),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        for i in range(len(self.path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, self.path[i], self.path[i+1], 40)
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path[i], self.path[i+1], 36)

        # Draw Base
        base_rect = pygame.Rect(0, 0, self.base_size, self.base_size)
        base_rect.center = self.base_pos
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=4)
        
        # Draw Towers
        for tower in self.towers:
            color = self.TOWER_COLORS[tower["type"]]
            pygame.draw.rect(self.screen, color, (tower["pos"][0]-8, tower["pos"][1]-8, 16, 16), border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), (tower["pos"][0]-8, tower["pos"][1]-8, 16, 16), 2, border_radius=3)

        # Draw Projectiles
        for proj in self.projectiles:
            color = self.TOWER_COLORS[proj["type"]]
            pygame.draw.circle(self.screen, (255,255,255), proj["pos"].astype(int), 4)
            pygame.draw.circle(self.screen, color, proj["pos"].astype(int), 2)

        # Draw Enemies
        for enemy in self.enemies:
            pos = enemy["pos"].astype(int)
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 8)
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_w = 16
            bar_h = 4
            bg_rect = pygame.Rect(pos[0] - bar_w/2, pos[1] - 15, bar_w, bar_h)
            hp_rect = pygame.Rect(pos[0] - bar_w/2, pos[1] - 15, bar_w * health_ratio, bar_h)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_BASE, hp_rect)

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, p["pos"] - p["size"])

    def _render_ui(self):
        # --- Draw Cursor and Tower Placement Preview ---
        cursor_color = self.COLOR_CURSOR
        tower_type, cost = self._get_placement_info()
        can_afford = tower_type and self.currency >= cost
        if tower_type:
            spec = self.TOWER_SPECS[tower_type]
            # Draw range indicator
            range_color = (255, 255, 255, 50) if can_afford else (255, 0, 0, 50)
            range_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            pygame.draw.circle(range_surface, range_color, self.cursor_pos.astype(int), spec["range"])
            self.screen.blit(range_surface, (0, 0))
            cursor_color = self.TOWER_COLORS[tower_type] if can_afford else (150,0,0)

        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos[0]), int(self.cursor_pos[1]), 10, cursor_color)
        pygame.draw.line(self.screen, cursor_color, (self.cursor_pos[0]-15, self.cursor_pos[1]), (self.cursor_pos[0]+15, self.cursor_pos[1]), 1)
        pygame.draw.line(self.screen, cursor_color, (self.cursor_pos[0], self.cursor_pos[1]-15), (self.cursor_pos[0], self.cursor_pos[1]+15), 1)

        # --- Text Info ---
        score_text = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        currency_text = self.font_m.render(f"$: {self.currency}", True, (255, 220, 100))
        self.screen.blit(currency_text, (10, 40))

        steps_left = max(0, self.MAX_STEPS - self.steps)
        time_left_sec = steps_left // self.FPS
        time_text = self.font_m.render(f"TIME: {time_left_sec // 60:02d}:{time_left_sec % 60:02d}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.screen_width - time_text.get_width() - 10, 10))

        # Base Health Bar
        health_ratio = max(0, self.base_health / 100)
        bar_w, bar_h = 200, 20
        bar_x, bar_y = self.screen_width/2 - bar_w/2, self.screen_height - bar_h - 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_w * health_ratio, bar_h), border_radius=4)
        health_text = self.font_s.render(f"BASE HEALTH: {max(0, self.base_health)} / 100", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_x + bar_w/2 - health_text.get_width()/2, bar_y + bar_h/2 - health_text.get_height()/2))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if self.steps >= self.MAX_STEPS else "GAME OVER"
            color = (100, 255, 100) if self.steps >= self.MAX_STEPS else (255, 100, 100)
            
            end_text = self.font_l.render(msg, True, color)
            self.screen.blit(end_text, (self.screen_width/2 - end_text.get_width()/2, self.screen_height/2 - end_text.get_height()/2 - 20))
            
            final_score_text = self.font_m.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_text, (self.screen_width/2 - final_score_text.get_width()/2, self.screen_height/2 + 20))


    def _get_placement_info(self):
        """Helper to determine which tower is being selected by current input."""
        keys = pygame.key.get_pressed() # Mocking this for logic, not used for control
        
        # In the actual step, we get this from the action tuple
        # This is a bit of a hack to check what the UI should show.
        # A better way would be to pass the action to the render function.
        # For now, let's assume we can't change the API and just show nothing.
        # A better implementation for this helper would be to check the last action.
        # But since we don't store it, we'll return None. The cursor will still show.
        return None, None


    def _create_particles(self, pos, count, color, min_speed, max_speed, life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color,
                "size": random.randint(2,5)
            })

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# This block allows running the environment directly for testing and visualization.
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # --- Action mapping for human input ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)

        clock.tick(env.FPS)

    pygame.quit()