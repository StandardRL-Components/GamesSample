import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:34:08.852426
# Source Brief: brief_00552.md
# Brief Index: 552
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Defend a geometric core from increasingly difficult waves of enemies 
    by strategically cloning and ricocheting dodecahedron projectiles with 
    varying properties.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend a geometric core from waves of incoming enemies. Aim your reticle and clone "
        "ricocheting projectiles to destroy them."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim the reticle. Press space to fire a projectile and "
        "use shift to switch between shield and pierce modes."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen dimensions
    WIDTH, HEIGHT = 640, 400
    
    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (20, 40, 60)
    COLOR_CORE = (50, 100, 255)
    COLOR_CORE_GLOW = (50, 100, 255, 50)
    COLOR_ENEMY_CUBE = (255, 180, 0)
    COLOR_ENEMY_PYRAMID = (255, 100, 0)
    COLOR_ENEMY_GLOW = (255, 180, 0, 50)
    COLOR_PROJ_SHIELD = (0, 255, 100)
    COLOR_PROJ_PIERCE = (255, 50, 50)
    COLOR_PROJ_SHIELD_GLOW = (0, 255, 100, 80)
    COLOR_PROJ_PIERCE_GLOW = (255, 50, 50, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (0, 255, 100)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    
    # Game parameters
    MAX_STEPS = 1000
    TOTAL_WAVES = 20
    INITIAL_CORE_HEALTH = 100
    MAX_PROJECTILES = 30
    
    PLAYER_RETICLE_ROT_SPEED = 0.1  # Radians per step
    PLAYER_RETICLE_MIN_DIST = 50
    PLAYER_RETICLE_MAX_DIST = 150
    PLAYER_RETICLE_MOVE_SPEED = 2
    
    PROJECTILE_SPEED = 8
    PROJECTILE_LIFESPAN = 180 # 6 seconds at 30fps
    PROJECTILE_COOLDOWN = 5 # 5 steps

    ENEMY_BASE_SPEED = 0.5
    ENEMY_SPEED_WAVE_INC = 0.05
    ENEMY_BASE_HEALTH = 3
    ENEMY_HEALTH_WAVE_INC_INTERVAL = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.core_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self._precompute_shapes()
        
        self.render_mode = render_mode
        
        self.reset()
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_clear_bonus_given = False

        self.core_health = self.INITIAL_CORE_HEALTH
        self.wave_number = 0

        self.projectiles = []
        self.enemies = []
        self.particles = []

        self.reticle_angle = 0.0
        self.reticle_dist = (self.PLAYER_RETICLE_MIN_DIST + self.PLAYER_RETICLE_MAX_DIST) / 2
        
        self.projectile_state = "shield" # "shield" or "pierce"
        self.last_shift_held = False
        self.projectile_cooldown_timer = 0
        
        self._start_new_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        step_reward = 0

        # --- Handle Actions ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game State ---
        if self.projectile_cooldown_timer > 0:
            self.projectile_cooldown_timer -= 1

        reward_events = self._update_game_entities()
        step_reward += reward_events["enemy_hit"] * 0.1
        step_reward += reward_events["enemy_destroyed"] * 1.0
        step_reward -= reward_events["core_hit"] * 1.0

        # --- Wave Progression ---
        if not self.enemies and self.wave_number <= self.TOTAL_WAVES and not self.wave_clear_bonus_given:
            step_reward += 100
            self.wave_clear_bonus_given = True
            if self.wave_number < self.TOTAL_WAVES:
                self._start_new_wave()
            else: # Victory
                self.game_over = True


        # --- Termination Conditions ---
        terminated = False
        truncated = False
        if self.core_health <= 0:
            step_reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        elif self.wave_number > self.TOTAL_WAVES: # Victory condition
            terminated = True
            self.game_over = True

        self.score += step_reward

        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Reticle movement
        if movement == 1: # Up
            self.reticle_dist = min(self.PLAYER_RETICLE_MAX_DIST, self.reticle_dist + self.PLAYER_RETICLE_MOVE_SPEED)
        elif movement == 2: # Down
            self.reticle_dist = max(self.PLAYER_RETICLE_MIN_DIST, self.reticle_dist - self.PLAYER_RETICLE_MOVE_SPEED)
        elif movement == 3: # Left
            self.reticle_angle -= self.PLAYER_RETICLE_ROT_SPEED
        elif movement == 4: # Right
            self.reticle_angle += self.PLAYER_RETICLE_ROT_SPEED

        # Clone projectile
        if space_held and self.projectile_cooldown_timer == 0 and len(self.projectiles) < self.MAX_PROJECTILES:
            # sfx: player_shoot.wav
            self.projectile_cooldown_timer = self.PROJECTILE_COOLDOWN
            reticle_pos = self._get_reticle_pos()
            direction = (reticle_pos - self.core_pos) / np.linalg.norm(reticle_pos - self.core_pos)
            
            self.projectiles.append({
                "pos": reticle_pos,
                "vel": direction * self.PROJECTILE_SPEED,
                "state": self.projectile_state,
                "lifespan": self.PROJECTILE_LIFESPAN,
                "rotation": random.uniform(0, 2 * math.pi),
                "trail": []
            })
            
        # Switch projectile state
        if shift_held and not self.last_shift_held:
            # sfx: switch_mode.wav
            self.projectile_state = "pierce" if self.projectile_state == "shield" else "shield"
        self.last_shift_held = shift_held

    def _update_game_entities(self):
        rewards = {"enemy_hit": 0, "enemy_destroyed": 0, "core_hit": 0}
        
        self._update_projectiles()
        self._handle_projectile_collisions()
        
        enemy_rewards = self._update_enemies()
        rewards["enemy_hit"] += enemy_rewards["enemy_hit"]
        rewards["enemy_destroyed"] += enemy_rewards["enemy_destroyed"]
        rewards["core_hit"] += enemy_rewards["core_hit"]
        
        self._update_particles()
        return rewards

    def _update_projectiles(self):
        for p in self.projectiles:
            p["trail"].append(p["pos"].copy())
            if len(p["trail"]) > 5:
                p["trail"].pop(0)
            
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["rotation"] += 0.1

            # Wall bounces
            if p["pos"][0] <= 0 or p["pos"][0] >= self.WIDTH:
                p["vel"][0] *= -1
                p["pos"][0] = np.clip(p["pos"][0], 0, self.WIDTH)
                # sfx: bounce.wav
            if p["pos"][1] <= 0 or p["pos"][1] >= self.HEIGHT:
                p["vel"][1] *= -1
                p["pos"][1] = np.clip(p["pos"][1], 0, self.HEIGHT)
                # sfx: bounce.wav

        # Remove old projectiles
        self.projectiles[:] = [p for p in self.projectiles if p["lifespan"] > 0]
        if len(self.projectiles) > self.MAX_PROJECTILES:
            self.projectiles.pop(0)

    def _handle_projectile_collisions(self):
        destroyed_indices = set()
        for i in range(len(self.projectiles)):
            for j in range(i + 1, len(self.projectiles)):
                p1 = self.projectiles[i]
                p2 = self.projectiles[j]
                
                dist = np.linalg.norm(p1["pos"] - p2["pos"])
                if dist < 12: # Collision radius
                    if p1["state"] == "pierce" and p2["state"] == "pierce":
                        destroyed_indices.add(i)
                        destroyed_indices.add(j)
                        self._spawn_explosion(p1["pos"], 15, self.COLOR_PROJ_PIERCE)
                        self._spawn_explosion(p2["pos"], 15, self.COLOR_PROJ_PIERCE)
                    elif p1["state"] == "pierce":
                        destroyed_indices.add(j)
                        self._spawn_explosion(p2["pos"], 15, self.COLOR_PROJ_SHIELD)
                    elif p2["state"] == "pierce":
                        destroyed_indices.add(i)
                        self._spawn_explosion(p1["pos"], 15, self.COLOR_PROJ_SHIELD)
                    else: # shield vs shield
                        # sfx: shield_reflect.wav
                        # Elastic collision approximation
                        v1, v2 = p1["vel"], p2["vel"]
                        p1["vel"], p2["vel"] = v2, v1
                        # Prevent sticking
                        p1["pos"] += p1["vel"]
                        p2["pos"] += p2["vel"]
        
        if destroyed_indices:
            # sfx: projectile_destroy.wav
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in destroyed_indices]

    def _update_enemies(self):
        rewards = {"enemy_hit": 0, "enemy_destroyed": 0, "core_hit": 0}
        projectiles_to_remove = set()
        enemies_to_remove = set()

        for i, enemy in enumerate(self.enemies):
            # Movement
            direction = (self.core_pos - enemy["pos"])
            dist_to_core = np.linalg.norm(direction)
            if dist_to_core > 1:
                enemy["pos"] += (direction / dist_to_core) * enemy["speed"]
            
            enemy["rotation"] += enemy["rot_speed"]

            # Collision with core
            if dist_to_core < 25: # Core radius
                # sfx: core_damage.wav
                self.core_health -= 10
                self.core_health = max(0, self.core_health)
                rewards["core_hit"] += 1
                enemies_to_remove.add(i)
                self._spawn_explosion(enemy["pos"], 20, self.COLOR_ENEMY_CUBE)
                continue

            # Collision with projectiles
            for j, p in enumerate(self.projectiles):
                if np.linalg.norm(enemy["pos"] - p["pos"]) < 15: # Enemy radius
                    # sfx: enemy_hit.wav
                    rewards["enemy_hit"] += 1
                    enemy["health"] -= 1
                    
                    if p["state"] == "pierce":
                         # Pierce projectiles are not destroyed on hit
                         pass
                    else: # Shield projectiles are destroyed
                        if j not in projectiles_to_remove:
                            projectiles_to_remove.add(j)
                            self._spawn_explosion(p["pos"], 10, self.COLOR_PROJ_SHIELD)

                    if enemy["health"] <= 0 and i not in enemies_to_remove:
                        # sfx: enemy_explode.wav
                        rewards["enemy_destroyed"] += 1
                        enemies_to_remove.add(i)
                        self._spawn_explosion(enemy["pos"], 30, self.COLOR_ENEMY_CUBE)
                        break

        # Remove destroyed entities
        if enemies_to_remove:
            self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]
        if projectiles_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
            
        return rewards

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles[:] = [p for p in self.particles if p["lifespan"] > 0]

    def _start_new_wave(self):
        self.wave_number += 1
        self.wave_clear_bonus_given = False
        if self.wave_number > self.TOTAL_WAVES:
            return

        num_enemies = 3 + self.wave_number
        enemy_speed = self.ENEMY_BASE_SPEED + self.wave_number * self.ENEMY_SPEED_WAVE_INC
        enemy_health = self.ENEMY_BASE_HEALTH + (self.wave_number // self.ENEMY_HEALTH_WAVE_INC_INTERVAL)

        for _ in range(num_enemies):
            edge = random.randint(0, 3)
            if edge == 0: # top
                pos = np.array([random.uniform(0, self.WIDTH), -20])
            elif edge == 1: # bottom
                pos = np.array([random.uniform(0, self.WIDTH), self.HEIGHT + 20])
            elif edge == 2: # left
                pos = np.array([-20, random.uniform(0, self.HEIGHT)])
            else: # right
                pos = np.array([self.WIDTH + 20, random.uniform(0, self.HEIGHT)])

            enemy_type = "cube" if self.wave_number < 10 or random.random() > 0.3 else "pyramid"
            
            self.enemies.append({
                "pos": pos,
                "speed": enemy_speed * (1.2 if enemy_type == "pyramid" else 1.0),
                "health": enemy_health,
                "type": enemy_type,
                "rotation": random.uniform(0, 2*math.pi),
                "rot_speed": random.uniform(-0.05, 0.05)
            })
        assert self.enemies[0]["speed"] > 0, "Enemy speed must be positive"
        assert self.enemies[0]["health"] > 0, "Enemy health must be positive"

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "core_health": self.core_health}

    # --- Rendering Methods ---
    def _render_game(self):
        self._draw_grid()
        self._draw_core()
        for enemy in self.enemies: self._draw_enemy(enemy)
        for p in self.particles: self._draw_particle(p)
        for proj in self.projectiles: self._draw_projectile(proj)
        self._draw_reticle()

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Wave
        wave_text = self.font_large.render(f"WAVE {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Core Health Bar
        bar_width = 100
        bar_height = 10
        health_pct = self.core_health / self.INITIAL_CORE_HEALTH
        
        bg_rect = pygame.Rect(self.core_pos[0] - bar_width / 2, self.core_pos[1] + 35, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect, border_radius=3)
        
        fill_rect = pygame.Rect(self.core_pos[0] - bar_width / 2, self.core_pos[1] + 35, bar_width * health_pct, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, fill_rect, border_radius=3)

        if self.game_over:
            if self.wave_number > self.TOTAL_WAVES:
                end_text = "VICTORY"
            else:
                end_text = "GAME OVER"
            
            text_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _precompute_shapes(self):
        # Simplified dodecahedron (decagon with inner lines)
        self.dodec_points = []
        for i in range(10):
            angle = i * (2 * math.pi / 10)
            self.dodec_points.append((math.cos(angle), math.sin(angle)))
            
        # Cube (3D projected)
        self.cube_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ])
        self.cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
            (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Pyramid
        self.pyramid_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1], [0, 1, 0]
        ])
        self.pyramid_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 4), (2, 4), (3, 4)
        ]

    def _get_rotated_points(self, points, angle_x, angle_y, angle_z):
        cx, sx = math.cos(angle_x), math.sin(angle_x)
        cy, sy = math.cos(angle_y), math.sin(angle_y)
        cz, sz = math.cos(angle_z), math.sin(angle_z)
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        rotated = points @ rot_z @ rot_y @ rot_x
        return rotated[:, :2] # Orthographic projection

    def _draw_grid(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _draw_core(self):
        self._draw_shape_2d(self.dodec_points, self.core_pos, 25, self.steps * 0.01, self.COLOR_CORE, self.COLOR_CORE_GLOW, 40)
        
    def _draw_enemy(self, enemy):
        size = 10 if enemy["type"] == "cube" else 12
        glow_size = 18
        if enemy["type"] == "cube":
            points = self._get_rotated_points(self.cube_verts, enemy["rotation"], enemy["rotation"]*0.5, enemy["rotation"]*0.2)
            self._draw_shape_3d_lines(points, enemy["pos"], size, self.cube_edges, self.COLOR_ENEMY_CUBE, self.COLOR_ENEMY_GLOW, glow_size)
        else: # pyramid
            points = self._get_rotated_points(self.pyramid_verts, enemy["rotation"], enemy["rotation"]*0.5, enemy["rotation"]*0.2)
            self._draw_shape_3d_lines(points, enemy["pos"], size, self.pyramid_edges, self.COLOR_ENEMY_PYRAMID, self.COLOR_ENEMY_GLOW, glow_size)

    def _draw_projectile(self, p):
        # Trail
        for i, pos in enumerate(p["trail"]):
            alpha = (i / len(p["trail"])) * 100
            color = p["state"] == "shield" and self.COLOR_PROJ_SHIELD or self.COLOR_PROJ_PIERCE
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 2, (*color, int(alpha)))
            
        color, glow_color = (self.COLOR_PROJ_SHIELD, self.COLOR_PROJ_SHIELD_GLOW) if p["state"] == "shield" else (self.COLOR_PROJ_PIERCE, self.COLOR_PROJ_PIERCE_GLOW)
        self._draw_shape_2d(self.dodec_points, p["pos"], 7, p["rotation"], color, glow_color, 15)

    def _draw_reticle(self):
        pos = self._get_reticle_pos()
        color = self.COLOR_PROJ_SHIELD if self.projectile_state == "shield" else self.COLOR_PROJ_PIERCE
        
        # Aura
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 12, (*color, 30))
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 12, (*color, 80))
        
        # Crosshair
        pygame.draw.line(self.screen, color, (pos[0] - 5, pos[1]), (pos[0] + 5, pos[1]), 1)
        pygame.draw.line(self.screen, color, (pos[0], pos[1] - 5), (pos[0], pos[1] + 5), 1)

    def _draw_particle(self, particle):
        size = int(particle["lifespan"] / particle["max_lifespan"] * 3)
        if size > 0:
            pygame.draw.circle(self.screen, particle["color"], particle["pos"], size)
            
    def _draw_shape_2d(self, base_points, center, size, rotation, color, glow_color, glow_size):
        points = [(size * math.cos(rotation + math.atan2(y, x)) * x_scale + center[0],
                   size * math.sin(rotation + math.atan2(y, x)) * y_scale + center[1])
                  for x, y in base_points for x_scale, y_scale in [(1,1)]]
        
        glow_points = [(glow_size * math.cos(rotation + math.atan2(y, x)) * x_scale + center[0],
                        glow_size * math.sin(rotation + math.atan2(y, x)) * y_scale + center[1])
                       for x, y in base_points for x_scale, y_scale in [(1,1)]]

        pygame.gfxdraw.filled_polygon(self.screen, glow_points, glow_color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Inner lines for 3D effect
        for i in range(len(points)):
            pygame.draw.aaline(self.screen, (255,255,255,50), (int(center[0]), int(center[1])), (int(points[i][0]), int(points[i][1])))

    def _draw_shape_3d_lines(self, points, center, size, edges, color, glow_color, glow_size):
        scaled_points = (points * size) + center
        
        # Glow effect
        for i in range(len(scaled_points)):
            pygame.gfxdraw.filled_circle(self.screen, int(scaled_points[i][0]), int(scaled_points[i][1]), 3, glow_color)

        for i1, i2 in edges:
            p1 = scaled_points[i1]
            p2 = scaled_points[i2]
            pygame.draw.aaline(self.screen, color, (p1[0], p1[1]), (p2[0], p2[1]))

    def _get_reticle_pos(self):
        return self.core_pos + np.array([math.cos(self.reticle_angle) * self.reticle_dist, 
                                         math.sin(self.reticle_angle) * self.reticle_dist])

    def _spawn_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            lifespan = random.randint(15, 30)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color
            })
            
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
        assert info["core_health"] == self.INITIAL_CORE_HEALTH
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        # Test termination logic by simulating core death
        self.core_health = 0
        _, _, term, _, _ = self.step(self.action_space.sample())
        assert term == True
        self.reset()
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, uncomment this section and run the file.
    # You need to have pygame installed.
    # Controls: Arrow keys to aim, Space to shoot, Shift to switch mode.
    
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    # pygame.display.set_caption("Geometric Core Defense")
    # clock = pygame.time.Clock()
    # done = False
    # total_reward = 0

    # while not done:
    #     movement = 0 # No-op
    #     space = 0
    #     shift = 0

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True

    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     if keys[pygame.K_DOWN]: movement = 2
    #     if keys[pygame.K_LEFT]: movement = 3
    #     if keys[pygame.K_RIGHT]: movement = 4
    #     if keys[pygame.K_SPACE]: space = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #     total_reward += reward

    #     # Display the observation from the environment
    #     frame = np.transpose(obs, (1, 0, 2))
    #     surf = pygame.surfarray.make_surface(frame)
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     clock.tick(30) # Limit to 30 FPS

    # print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Wave: {info['wave']}")
    # env.close()
    
    # --- Random Agent Test ---
    # This test runs a random agent for a few episodes.
    episodes = 3
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            if terminated or truncated:
                print(f"Episode {ep+1}: Finished in {step_count} steps. Total Reward: {total_reward:.2f}. Final Info: {info}")
                break
    env.close()