
# Generated: 2025-08-27T12:36:40.623270
# Source Brief: brief_00104.md
# Brief Index: 104

        
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
        "Controls: Use arrow keys to move the placement cursor. Press 'Space' to build the selected tower. "
        "Press 'Shift' to cycle between tower types."
    )

    game_description = (
        "A top-down tower defense game. Strategically place towers to defend your base from waves of "
        "invading enemies. Manage your resources and survive all 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000  # Extended to allow for 10 waves at 30fps
    WAVES_TO_WIN = 10

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_PATH = (40, 50, 60)
    COLOR_PATH_BORDER = (60, 70, 80)
    COLOR_BASE = (0, 150, 100)
    COLOR_BASE_GLOW = (0, 200, 150)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_ENEMY_GLOW = (255, 100, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_DIM = (150, 150, 150)
    COLOR_CURSOR_VALID = (255, 255, 255, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)

    # Tower Type Definitions
    TOWER_SPECS = [
        {
            "name": "Gatling",
            "cost": 25, "range": 80, "damage": 2, "fire_rate": 5, # frames per shot
            "color": (255, 200, 0), "proj_color": (255, 255, 100)
        },
        {
            "name": "Cannon",
            "cost": 75, "range": 120, "damage": 15, "fire_rate": 45,
            "color": (0, 150, 255), "proj_color": (100, 200, 255)
        }
    ]

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

        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_title = pygame.font.Font(None, 50)

        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.last_space_state = 0
        self.last_shift_state = 0

        self.reset()
        
        # This check is disabled for submission but useful for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.base_pos = np.array([self.SCREEN_WIDTH - 40, self.SCREEN_HEIGHT / 2])
        self.base_health = 100
        self.resources = 80

        self.path = self._generate_path()
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.wave_timer = 0
        self.inter_wave_timer = 150 # Time before first wave
        self.enemies_to_spawn = []

        self.cursor_pos = np.array([100.0, 100.0])
        self.selected_tower_type = 0

        self.last_space_state = 0
        self.last_shift_state = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.001 # Small time penalty

        if not self.game_over:
            reward += self._handle_input(movement, space_held, shift_held)
            self._update_waves()
            reward += self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_enemies()
        
        self._update_particles()
        
        self.score += reward
        self.steps += 1

        terminated = self._check_termination()
        if terminated:
            if self.victory:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # --- Cursor Movement ---
        cursor_speed = 5
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # --- Cycle Tower (on press) ---
        if shift_held and not self.last_shift_state:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_state = shift_held

        # --- Place Tower (on press) ---
        if space_held and not self.last_space_state:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.resources >= spec["cost"] and self._is_valid_placement(self.cursor_pos):
                self.resources -= spec["cost"]
                self.towers.append({
                    "pos": self.cursor_pos.copy(),
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "target": None
                })
                # Positive reward for a strategic action
                reward += 0.5 
                # sfx: place_tower.wav
        self.last_space_state = space_held
        return reward

    def _update_waves(self):
        if not self.enemies and not self.enemies_to_spawn:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0 and self.current_wave < self.WAVES_TO_WIN:
                self.current_wave += 1
                self._start_next_wave()
                self.inter_wave_timer = 300 # 10 seconds between waves
        
        if self.enemies_to_spawn:
            self.wave_timer -=1
            if self.wave_timer <= 0:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.wave_timer = 15 # 0.5 sec between enemies

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            
            if tower["cooldown"] == 0:
                closest_enemy = None
                min_dist = spec["range"] ** 2
                for enemy in self.enemies:
                    dist_sq = np.sum((tower["pos"] - enemy["pos"])**2)
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        closest_enemy = enemy
                
                if closest_enemy:
                    self.projectiles.append({
                        "pos": tower["pos"].copy(),
                        "target": closest_enemy,
                        "type": tower["type"],
                        "speed": 10
                    })
                    tower["cooldown"] = spec["fire_rate"]
                    # sfx: shoot_gatling.wav or shoot_cannon.wav
                    self._create_particles(tower["pos"], spec["color"], 3, count=3, speed=2, life=10)
        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            spec = self.TOWER_SPECS[proj["type"]]
            target_pos = proj["target"]["pos"]
            direction = target_pos - proj["pos"]
            dist = np.linalg.norm(direction)

            if dist < proj["speed"]:
                proj["target"]["health"] -= spec["damage"]
                reward += 0.1 # Reward for hitting
                self.projectiles.remove(proj)
                self._create_particles(target_pos, proj["target"]["color"], 5, count=5, speed=3, life=15)
                # sfx: enemy_hit.wav
                continue

            direction /= dist
            proj["pos"] += direction * proj["speed"]
            
            # Remove projectiles whose targets are already dead
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                self.resources += enemy["reward"]
                reward += 1.0 # Reward for defeating enemy
                self.enemies.remove(enemy)
                self._create_particles(enemy["pos"], (255, 255, 255), 10, count=15, speed=4, life=20)
                # sfx: enemy_explode.wav
                continue

            waypoint = self.path[enemy["waypoint_idx"]]
            direction = waypoint - enemy["pos"]
            dist = np.linalg.norm(direction)

            if dist < enemy["speed"]:
                enemy["pos"] = waypoint.copy()
                if enemy["waypoint_idx"] < len(self.path) - 1:
                    enemy["waypoint_idx"] += 1
                else: # Reached base
                    self.base_health = max(0, self.base_health - 10)
                    reward -= 10 # Penalty for base damage
                    self.enemies.remove(enemy)
                    self._create_particles(self.base_pos, (255, 0, 0), 20, count=20, speed=5, life=30)
                    # sfx: base_damage.wav
            else:
                enemy["pos"] += (direction / dist) * enemy["speed"]
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave
        }

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.current_wave >= self.WAVES_TO_WIN and not self.enemies and not self.enemies_to_spawn:
            self.game_over = True
            self.victory = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _render_game(self):
        # Path
        path_width = 40
        for i in range(len(self.path) - 1):
            p1 = tuple(self.path[i].astype(int))
            p2 = tuple(self.path[i+1].astype(int))
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, p1, p2, path_width + 4)
        for i in range(len(self.path) - 1):
            p1 = tuple(self.path[i].astype(int))
            p2 = tuple(self.path[i+1].astype(int))
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, path_width)

        # Base
        base_size = 20
        base_rect = pygame.Rect(self.base_pos[0] - base_size, self.base_pos[1] - base_size, base_size*2, base_size*2)
        glow_size = base_size + 5 + 5 * math.sin(self.steps * 0.1)
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), int(glow_size), (*self.COLOR_BASE_GLOW, 50))
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=3)
        
        # Towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pos = tower["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_TEXT)
            # Firing flash
            if spec["fire_rate"] - tower["cooldown"] < 3:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, (255,255,255,150))

        # Enemies
        for enemy in self.enemies:
            pos = enemy["pos"].astype(int)
            size = int(enemy["size"])
            glow_size = size + 2 + 2 * math.sin(self.steps * 0.2 + pos[0])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_size, (*self.COLOR_ENEMY_GLOW, 50))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0]-size, pos[1]-size, size*2, size*2))

        # Projectiles
        for proj in self.projectiles:
            spec = self.TOWER_SPECS[proj["type"]]
            pos = proj["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, spec["proj_color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, (255,255,255))

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
            color = (*p["color"], alpha)
            pos = p["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["size"]), color)

    def _render_ui(self):
        # Cursor
        spec = self.TOWER_SPECS[self.selected_tower_type]
        pos = self.cursor_pos.astype(int)
        is_valid = self._is_valid_placement(self.cursor_pos) and self.resources >= spec["cost"]
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], spec["range"], cursor_color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec["range"], (255,255,255,150))
        pygame.draw.line(self.screen, (255,255,255), (pos[0]-10, pos[1]), (pos[0]+10, pos[1]), 1)
        pygame.draw.line(self.screen, (255,255,255), (pos[0], pos[1]-10), (pos[0], pos[1]+10), 1)

        # Top Bar
        wave_text = self.font_main.render(f"Wave: {self.current_wave}/{self.WAVES_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        health_text = self.font_main.render(f"Base Health: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - health_text.get_width() - 10, 10))

        # Bottom Bar (Tower Info & Resources)
        res_text = self.font_main.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (self.SCREEN_WIDTH/2 - res_text.get_width()/2, self.SCREEN_HEIGHT - 35))
        
        tower_name = self.font_small.render(f"Selected: {spec['name']}", True, self.COLOR_TEXT)
        tower_cost = self.font_small.render(f"Cost: {spec['cost']}", True, self.COLOR_TEXT)
        self.screen.blit(tower_name, (10, self.SCREEN_HEIGHT - 40))
        self.screen.blit(tower_cost, (10, self.SCREEN_HEIGHT - 25))

        # Game Over / Victory Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            message = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_BASE_GLOW if self.victory else self.COLOR_ENEMY_GLOW
            end_text = self.font_title.render(message, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))

    def _generate_path(self):
        path = []
        path.append(np.array([-20.0, self.SCREEN_HEIGHT * 0.2]))
        path.append(np.array([self.SCREEN_WIDTH * 0.7, self.SCREEN_HEIGHT * 0.2]))
        path.append(np.array([self.SCREEN_WIDTH * 0.7, self.SCREEN_HEIGHT * 0.8]))
        path.append(np.array([self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT * 0.8]))
        path.append(np.array([self.SCREEN_WIDTH * 0.2, self.base_pos[1]]))
        path.append(self.base_pos.copy())
        return path

    def _start_next_wave(self):
        num_enemies = 5 + self.current_wave * 2
        base_health = 10 + self.current_wave * 5
        base_speed = 1.0 + self.current_wave * 0.1
        
        self.enemies_to_spawn = []
        for i in range(num_enemies):
            self.enemies_to_spawn.append({
                "pos": self.path[0].copy(),
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed + self.np_random.uniform(-0.1, 0.1),
                "size": 6,
                "reward": 5 + self.current_wave,
                "waypoint_idx": 1,
                "color": self.COLOR_ENEMY
            })
    
    def _is_valid_placement(self, pos):
        # Check distance to other towers
        for tower in self.towers:
            if np.linalg.norm(pos - tower["pos"]) < 30:
                return False
        
        # Check distance to path
        path_width = 40
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            # simplified point-to-line-segment distance check
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0.0:
                if np.linalg.norm(pos - p1) < path_width: return False
            t = max(0, min(1, np.dot(pos - p1, p2 - p1) / l2))
            projection = p1 + t * (p2 - p1)
            if np.linalg.norm(pos - projection) < path_width / 2 + 10: # 10 is tower radius
                return False
        return True
    
    def _create_particles(self, pos, color, size, count, speed, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 1.0) * speed
            vel = np.array([math.cos(angle), math.sin(angle)]) * vel_mag
            self.particles.append({
                "pos": pos.copy() + self.np_random.uniform(-2,2, size=2),
                "vel": vel,
                "life": self.np_random.integers(life // 2, life),
                "max_life": life,
                "color": color,
                "size": self.np_random.uniform(1, size)
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different screen for display if running directly
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    terminated = False
    clock = pygame.time.Clock()
    
    print("="*30)
    print("Tower Defense Gym Environment")
    print(GameEnv.user_guide)
    print("="*30)

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Display ---
        # The observation is (H, W, C), but pygame wants (W, H, C)
        # So we need to convert it back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
    pygame.quit()