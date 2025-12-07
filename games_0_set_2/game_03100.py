
# Generated: 2025-08-27T22:21:58.963624
# Source Brief: brief_03100.md
# Brief Index: 3100

        
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
        "Controls: Arrow keys to move cursor. Space to place a tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * 120 # 2 minutes max
        self.MAX_WAVES = 10
        self.STARTING_RESOURCES = 250
        self.STARTING_BASE_HEALTH = 200

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PATH = (45, 45, 55)
        self.COLOR_PATH_BORDER = (65, 65, 75)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_BASE_GLOW = (0, 200, 100)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 100, 100)
        self.COLOR_TOWER_1 = (50, 100, 220)
        self.COLOR_TOWER_1_GLOW = (100, 150, 255)
        self.COLOR_TOWER_2 = (220, 150, 50)
        self.COLOR_TOWER_2_GLOW = (255, 200, 100)
        self.COLOR_PROJECTILE_1 = (150, 200, 255)
        self.COLOR_PROJECTILE_2 = (255, 220, 150)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_SUCCESS = (100, 255, 100)
        self.COLOR_UI_DANGER = (255, 100, 100)
        self.COLOR_PLACEMENT_VALID = (100, 255, 100, 100)
        self.COLOR_PLACEMENT_INVALID = (255, 100, 100, 100)

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
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Tower definitions
        self.TOWER_TYPES = [
            {
                "name": "Gun Turret", "cost": 100, "range": 100, "damage": 5, "fire_rate": 0.5,
                "color": self.COLOR_TOWER_1, "glow": self.COLOR_TOWER_1_GLOW,
                "proj_color": self.COLOR_PROJECTILE_1, "proj_speed": 8, "proj_size": 2,
            },
            {
                "name": "Cannon", "cost": 150, "range": 120, "damage": 20, "fire_rate": 2.0,
                "color": self.COLOR_TOWER_2, "glow": self.COLOR_TOWER_2_GLOW,
                "proj_color": self.COLOR_PROJECTILE_2, "proj_speed": 5, "proj_size": 4,
            }
        ]
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_timer = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.enemy_path = []
        self.placement_zones = []
        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.selected_tower_type = 0
        self.last_shift_press = False
        self.last_space_press = False
        self.reward_this_step = 0

        # Run validation check
        self.validate_implementation()

    def _generate_path(self):
        path = []
        side = self.np_random.integers(4)
        if side == 0: x, y = 0, self.np_random.integers(50, self.HEIGHT - 50)
        elif side == 1: x, y = self.WIDTH, self.np_random.integers(50, self.HEIGHT - 50)
        elif side == 2: x, y = self.np_random.integers(50, self.WIDTH - 50), 0
        else: x, y = self.np_random.integers(50, self.WIDTH - 50), self.HEIGHT
        
        path.append((x, y))
        
        # Create a few intermediate points to snake towards the center
        for _ in range(self.np_random.integers(2, 5)):
            px, py = path[-1]
            nx = self.np_random.integers(int(self.WIDTH * 0.2), int(self.WIDTH * 0.8))
            ny = self.np_random.integers(int(self.HEIGHT * 0.2), int(self.HEIGHT * 0.8))
            path.append((nx, ny))

        path.append(self.base_pos)
        return path

    def _generate_placement_zones(self):
        zones = []
        for _ in range(15):
            valid = False
            while not valid:
                x = self.np_random.integers(50, self.WIDTH - 50)
                y = self.np_random.integers(50, self.HEIGHT - 50)
                pos = pygame.Vector2(x, y)
                
                # Check distance to path
                on_path = False
                for i in range(len(self.enemy_path) - 1):
                    p1 = pygame.Vector2(self.enemy_path[i])
                    p2 = pygame.Vector2(self.enemy_path[i+1])
                    if p1.distance_to(p2) > 0:
                        dist_to_segment = pos.distance_to(p1 + (p2 - p1).normalize() * pos.dot(p2 - p1) / p1.distance_to(p2))
                        if dist_to_segment < 40: # Not too close to path
                            on_path = True
                            break
                if on_path: continue

                # Check distance to other zones
                too_close = False
                for z in zones:
                    if pos.distance_to(z) < 60:
                        too_close = True
                        break
                if too_close: continue
                
                # Check distance to base
                if pos.distance_to(self.base_pos) < 70: continue
                
                valid = True
                zones.append(pos)
        return zones

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.STARTING_BASE_HEALTH
        self.resources = self.STARTING_RESOURCES
        self.current_wave = 0
        self.wave_timer = 150 # Time until first wave
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.enemy_path = self._generate_path()
        self.placement_zones = self._generate_placement_zones()
        
        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT - 30]
        self.selected_tower_type = 0
        self.last_shift_press = False
        self.last_space_press = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = -0.01 # Time penalty

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        cursor_speed = 5
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Cycle tower on shift press
        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        self.last_shift_press = shift_held
        
        # Place tower on space press
        if space_held and not self.last_space_press:
            self._place_tower()
        self.last_space_press = space_held

        # --- Update Game Logic ---
        self._update_waves()
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        self.steps += 1
        self.score += self.reward_this_step
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.win:
                self.reward_this_step += 100
                self.score += 100
            else:
                self.reward_this_step -= 100
                self.score -= 100
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _place_tower(self):
        tower_data = self.TOWER_TYPES[self.selected_tower_type]
        if self.resources >= tower_data["cost"]:
            valid_placement = False
            for zone in self.placement_zones:
                if pygame.Vector2(self.cursor_pos).distance_to(zone) < 20:
                    # Check if another tower is already here
                    is_occupied = any(pygame.Vector2(t['pos']).distance_to(zone) < 20 for t in self.towers)
                    if not is_occupied:
                        self.resources -= tower_data["cost"]
                        self.towers.append({
                            "pos": (zone.x, zone.y),
                            "type_id": self.selected_tower_type,
                            "cooldown": 0
                        })
                        # sfx: tower_place.wav
                        valid_placement = True
                        break
            # if not valid_placement: sfx: error.wav
    
    def _update_waves(self):
        if self.current_wave >= self.MAX_WAVES:
            if not self.enemies:
                self.win = True
            return

        self.wave_timer -= 1
        if self.wave_timer <= 0 and not self.enemies:
            self.current_wave += 1
            self.wave_timer = 30 * 10 # 10 seconds between waves
            self._spawn_wave()
            # sfx: wave_start.wav

    def _spawn_wave(self):
        num_enemies = 5 + self.current_wave * 2
        base_health = 10 + self.current_wave * 5
        base_speed = 0.8 + self.current_wave * 0.05
        
        for i in range(num_enemies):
            self.enemies.append({
                "pos": pygame.Vector2(self.enemy_path[0]),
                "waypoint_idx": 1,
                "health": base_health + self.np_random.uniform(-2, 2),
                "max_health": base_health,
                "speed": base_speed + self.np_random.uniform(-0.1, 0.1),
                "spawn_delay": i * 15 # Stagger spawns
            })

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            if enemy["spawn_delay"] > 0:
                enemy["spawn_delay"] -= 1
                continue

            if enemy["waypoint_idx"] < len(self.enemy_path):
                target_pos = pygame.Vector2(self.enemy_path[enemy["waypoint_idx"]])
                direction = (target_pos - enemy["pos"]).normalize()
                enemy["pos"] += direction * enemy["speed"]
                
                if enemy["pos"].distance_to(target_pos) < 5:
                    enemy["waypoint_idx"] += 1
            else: # Reached base
                self.base_health -= 10
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                self._create_particles(self.base_pos, self.COLOR_UI_DANGER, 20)

    def _update_towers(self):
        for tower in self.towers:
            tower_data = self.TOWER_TYPES[tower["type_id"]]
            tower["cooldown"] = max(0, tower["cooldown"] - 1 / self.FPS)
            if tower["cooldown"] <= 0:
                target = None
                min_dist = tower_data["range"]
                for enemy in self.enemies:
                    if enemy["spawn_delay"] > 0: continue
                    dist = pygame.Vector2(tower["pos"]).distance_to(enemy["pos"])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    tower["cooldown"] = tower_data["fire_rate"]
                    self.projectiles.append({
                        "pos": pygame.Vector2(tower["pos"]),
                        "target": target,
                        "type_id": tower["type_id"]
                    })
                    # sfx: shoot_1.wav or shoot_2.wav

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            proj_data = self.TOWER_TYPES[proj["type_id"]]
            
            if proj["target"] not in self.enemies: # Target already dead
                self.projectiles.remove(proj)
                continue

            target_pos = proj["target"]["pos"]
            direction = (target_pos - proj["pos"])
            
            if direction.length() < proj_data["proj_speed"]:
                # Hit
                proj["target"]["health"] -= proj_data["damage"]
                self.reward_this_step += 0.1
                # sfx: enemy_hit.wav
                self._create_particles(proj["pos"], proj_data["color"], 5)
                if proj["target"]["health"] <= 0:
                    self.resources += 10
                    self.reward_this_step += 1
                    # sfx: enemy_destroy.wav
                    self._create_particles(proj["target"]["pos"], self.COLOR_ENEMY, 30)
                    self.enemies.remove(proj["target"])
                    
                    if not self.enemies and self.wave_timer > 0 and self.current_wave > 0:
                        self.reward_this_step += 10 # Wave clear bonus
                
                self.projectiles.remove(proj)
            else:
                proj["pos"] += direction.normalize() * proj_data["proj_speed"]

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(10, 25),
                'color': color
            })

    def _check_termination(self):
        if self.game_over: return True
        if self.base_health <= 0:
            self.game_over = True
            self.win = False
            return True
        if self.win:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        if len(self.enemy_path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH_BORDER, False, self.enemy_path, 30)
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.enemy_path, 26)

        # Draw placement zones
        for zone in self.placement_zones:
            pygame.gfxdraw.filled_circle(self.screen, int(zone.x), int(zone.y), 15, (60, 60, 70, 50))
            pygame.gfxdraw.aacircle(self.screen, int(zone.x), int(zone.y), 15, (80, 80, 90, 100))

        # Draw base
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 20, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 20, self.COLOR_BASE_GLOW)
        
        # Draw towers
        for tower in self.towers:
            data = self.TOWER_TYPES[tower["type_id"]]
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            if data["name"] == "Gun Turret": # Triangle
                points = [(pos[0], pos[1] - 8), (pos[0] - 7, pos[1] + 5), (pos[0] + 7, pos[1] + 5)]
                pygame.gfxdraw.aapolygon(self.screen, points, data["glow"])
                pygame.gfxdraw.filled_polygon(self.screen, points, data["color"])
            else: # Square
                rect = pygame.Rect(pos[0] - 6, pos[1] - 6, 12, 12)
                pygame.draw.rect(self.screen, data["color"], rect)
                pygame.draw.rect(self.screen, data["glow"], rect, 1)

        # Draw projectiles
        for proj in self.projectiles:
            data = self.TOWER_TYPES[proj["type_id"]]
            pygame.gfxdraw.filled_circle(self.screen, int(proj["pos"].x), int(proj["pos"].y), data["proj_size"], data["proj_color"])

        # Draw enemies
        for enemy in self.enemies:
            if enemy["spawn_delay"] > 0: continue
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY_GLOW)
            # Health bar
            hp_ratio = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (50,50,50), (pos[0]-8, pos[1]-12, 16, 3))
            pygame.draw.rect(self.screen, self.COLOR_UI_SUCCESS, (pos[0]-8, pos[1]-12, int(16 * hp_ratio), 3))

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['life'] / 8), p['color'])

        # Draw cursor
        self._render_cursor()

    def _render_cursor(self):
        cursor_x, cursor_y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        tower_data = self.TOWER_TYPES[self.selected_tower_type]
        
        # Check if placement is valid
        can_afford = self.resources >= tower_data["cost"]
        is_on_zone = False
        is_occupied = True
        for zone in self.placement_zones:
            if pygame.Vector2(self.cursor_pos).distance_to(zone) < 20:
                is_on_zone = True
                is_occupied = any(pygame.Vector2(t['pos']).distance_to(zone) < 1 for t in self.towers)
                break
        
        is_valid = can_afford and is_on_zone and not is_occupied
        color = self.COLOR_PLACEMENT_VALID if is_valid else self.COLOR_PLACEMENT_INVALID

        # Draw range indicator
        pygame.gfxdraw.filled_circle(self.screen, cursor_x, cursor_y, tower_data["range"], color)
        pygame.gfxdraw.aacircle(self.screen, cursor_x, cursor_y, tower_data["range"], color)
        # Draw cursor crosshair
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (cursor_x - 10, cursor_y), (cursor_x + 10, cursor_y), 1)
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (cursor_x, cursor_y - 10), (cursor_x, cursor_y + 10), 1)

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.WIDTH, 30))
        
        # Base Health
        health_text = self.font_medium.render(f"Base: {max(0, int(self.base_health))}/{self.STARTING_BASE_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 5))
        
        # Resources
        res_text = self.font_medium.render(f"Resources: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (230, 5))
        
        # Wave Info
        wave_str = f"Wave: {self.current_wave}/{self.MAX_WAVES}"
        if not self.enemies and self.current_wave < self.MAX_WAVES and self.wave_timer > 0:
            wave_str += f" (Next in {int(self.wave_timer / self.FPS)}s)"
        wave_text = self.font_medium.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (420, 5))
        
        # Bottom bar for selected tower
        pygame.draw.rect(self.screen, (0,0,0,150), (0, self.HEIGHT - 30, self.WIDTH, 30))
        tower_data = self.TOWER_TYPES[self.selected_tower_type]
        tower_info = f"Selected: {tower_data['name']} | Cost: {tower_data['cost']} | Dmg: {tower_data['damage']} | Range: {tower_data['range']}"
        tower_text = self.font_small.render(tower_info, True, self.COLOR_UI_TEXT)
        text_rect = tower_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 15))
        self.screen.blit(tower_text, text_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_UI_SUCCESS if self.win else self.COLOR_UI_DANGER
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
            "enemies_remaining": len(self.enemies)
        }

    def close(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Tower Defense Siege")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space = 0    # 0=released, 1=held
        shift = 0    # 0=released, 1=held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Survived to Wave: {info['wave']}")
            # obs, info = env.reset() # Uncomment to auto-restart
            running = False # Comment to auto-restart
        
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()