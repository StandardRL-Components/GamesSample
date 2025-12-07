import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


# Set the SDL video driver to "dummy" for headless execution, as required.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Import additional pygame modules that might be needed.
import pygame.gfxdraw
from gymnasium.spaces import MultiDiscrete

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player defends interconnected spheres from
    geometric attacks. The core gameplay involves strategically allocating power,
    placing shields, and firing projectiles to survive waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- Fixes Start ---
    game_description = (
        "Defend interconnected spheres from geometric attacks by allocating power, placing shields, "
        "and firing projectiles to survive waves of enemies."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select a sphere. Press space to allocate power. "
        "Press shift to fire projectiles (during a wave) or cycle shield types (between waves)."
    )
    auto_advance = True
    # --- Fixes End ---

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 22)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # --- Visual & Game Constants ---
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_SPHERE_HEALTHY = (0, 255, 128)
        self.COLOR_SPHERE_DAMAGED = (255, 255, 0)
        self.COLOR_SPHERE_CRITICAL = (255, 50, 50)
        self.COLOR_SHIELD_TRI = (0, 191, 255)
        self.COLOR_SHIELD_SQR = (255, 100, 255)
        self.COLOR_PROJECTILE = (255, 165, 0)
        self.COLOR_ENEMY_TRI = (255, 255, 255)
        self.COLOR_ENEMY_SQR = (255, 200, 200)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_CONNECT = (50, 50, 80)
        self.COLOR_SELECT = (0, 255, 255)

        self.MAX_STEPS = 2500
        self.MAX_WAVES = 20
        self.INTER_WAVE_DURATION = 120  # 4 seconds at 30 FPS

        # --- Initialize State Variables ---
        self.spheres = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.wave = 0
        self.inter_wave_timer = 0
        self.wave_in_progress = False
        self.selected_sphere_idx = 0
        self.action_cooldowns = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.wave = 0
        self.inter_wave_timer = self.INTER_WAVE_DURATION
        self.wave_in_progress = False
        self.selected_sphere_idx = 0

        self.action_cooldowns = {'select': 0, 'fire': 0, 'power': 0, 'shield': 0}

        self.spheres = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self._setup_level()
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1

        self._update_cooldowns()
        self._handle_player_input(action)

        if self.wave_in_progress:
            self._update_enemies()
            self._update_projectiles()
            collision_reward = self._handle_collisions()
            reward += collision_reward

            if not self.enemies and self.wave_in_progress:
                self.wave_in_progress = False
                self.inter_wave_timer = self.INTER_WAVE_DURATION
                reward += 1.0  # Wave clear bonus
        else:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0 and self.wave < self.MAX_WAVES:
                self.wave += 1
                self._start_new_wave()
                self._unlock_content()

        self._update_particles()
        self._update_spheres()
        self.score += reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            active_spheres = sum(1 for s in self.spheres if s["is_active"])
            if self.wave > self.MAX_WAVES:
                reward += 100.0  # Win bonus
            elif active_spheres == 0:
                reward -= 100.0  # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "wave": self.wave,
            "enemies_remaining": len(self.enemies),
        }

    def close(self):
        pygame.quit()

    # --- Game Logic Sub-modules ---

    def _setup_level(self):
        self.spheres.append(self._create_sphere(pos=(self.WIDTH // 4, self.HEIGHT // 2)))

    def _start_new_wave(self):
        self.wave_in_progress = True
        num_enemies = 2 + self.wave
        enemy_speed = 0.8 + (self.wave // 5) * 0.2
        enemy_health = 1 + (self.wave // 4)

        active_indices = [i for i, s in enumerate(self.spheres) if s["is_active"]]
        if not active_indices:
            return

        for _ in range(num_enemies):
            spawn_pos = self._get_spawn_pos()
            target_sphere_idx = self.np_random.choice(active_indices)
            enemy_type = 0  # Triangle
            if self.wave >= 5 and self.np_random.random() > 0.5:
                enemy_type = 1  # Square
            
            self.enemies.append(self._create_enemy(spawn_pos, target_sphere_idx, enemy_speed, enemy_health, enemy_type))

    def _unlock_content(self):
        if self.wave == 3:
            self.spheres.append(self._create_sphere(pos=(self.WIDTH * 3 // 4, self.HEIGHT // 3)))
        if self.wave == 6:
            self.spheres.append(self._create_sphere(pos=(self.WIDTH * 3 // 4, self.HEIGHT * 2 // 3)))

    def _handle_player_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        active_sphere_indices = [i for i, s in enumerate(self.spheres) if s["is_active"]]
        if not active_sphere_indices:
            return

        # 1. Sphere Selection (Movement)
        if movement != 0 and self.action_cooldowns['select'] == 0:
            direction = 1 if movement in [2, 4] else -1  # Down/Right or Up/Left
            if self.selected_sphere_idx not in active_sphere_indices:
                self.selected_sphere_idx = active_sphere_indices[0]
            else:
                current_list_idx = active_sphere_indices.index(self.selected_sphere_idx)
                new_list_idx = (current_list_idx + direction) % len(active_sphere_indices)
                self.selected_sphere_idx = active_sphere_indices[new_list_idx]
            self.action_cooldowns['select'] = 5

        selected_sphere = self.spheres[self.selected_sphere_idx]

        # 2. Allocate Power (Space)
        if space_held and self.action_cooldowns['power'] == 0:
            if selected_sphere['power'] < selected_sphere['max_power']:
                selected_sphere['power'] += 1
                self.action_cooldowns['power'] = 3

        # 3. Fire/Cycle Shield (Shift)
        if shift_held:
            if self.wave_in_progress:  # In-Wave: Fire Projectile
                if self.action_cooldowns['fire'] == 0 and selected_sphere['power'] > 0:
                    self._fire_projectile(self.selected_sphere_idx)
                    selected_sphere['power'] -= 1
                    self.action_cooldowns['fire'] = 15
            else:  # Inter-Wave: Cycle Shield
                if self.action_cooldowns['shield'] == 0:
                    max_shield_type = 0
                    if self.wave >= 5: max_shield_type = 1
                    selected_sphere['shield_type'] = (selected_sphere['shield_type'] + 1) % (max_shield_type + 1)
                    self.action_cooldowns['shield'] = 10

    def _fire_projectile(self, sphere_idx):
        sphere = self.spheres[sphere_idx]
        closest_enemy, min_dist = None, float('inf')
        for enemy in self.enemies:
            dist = np.linalg.norm(enemy['pos'] - sphere['pos'])
            if dist < min_dist:
                min_dist, closest_enemy = dist, enemy

        if closest_enemy:
            direction = closest_enemy['pos'] - sphere['pos']
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
                self.projectiles.append(self._create_projectile(sphere['pos'], direction))

    def _handle_collisions(self):
        reward = 0.0
        
        projectiles_to_remove = set()
        enemies_to_remove = set()

        for i, proj in enumerate(self.projectiles):
            for j, enemy in enumerate(self.enemies):
                if i in projectiles_to_remove or j in enemies_to_remove: continue
                if np.linalg.norm(proj['pos'] - enemy['pos']) < enemy['radius']:
                    enemy['health'] -= 1
                    self._create_particle_burst(proj['pos'], 5, self.COLOR_PROJECTILE)
                    projectiles_to_remove.add(i)
                    if enemy['health'] <= 0:
                        reward += 0.1
                        self._create_particle_burst(enemy['pos'], 20, self.COLOR_ENEMY_TRI if enemy['type'] == 0 else self.COLOR_ENEMY_SQR, 2.0)
                        enemies_to_remove.add(j)
                    break
        
        for j, enemy in enumerate(self.enemies):
            if j in enemies_to_remove: continue
            for sphere in self.spheres:
                if not sphere["is_active"]: continue
                if np.linalg.norm(enemy['pos'] - sphere['pos']) < sphere['radius']:
                    if enemy['type'] == sphere['shield_type']:
                        self._create_particle_burst(enemy['pos'], 10, self.COLOR_SHIELD_TRI if enemy['type'] == 0 else self.COLOR_SHIELD_SQR)
                    else:
                        damage = 10
                        sphere['health'] -= damage
                        reward -= 0.1 * damage
                        self._create_particle_burst(enemy['pos'], 15, self.COLOR_SPHERE_CRITICAL)
                    enemies_to_remove.add(j)
                    break
        
        if projectiles_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        if enemies_to_remove:
            self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]
        
        return reward

    def _check_termination(self):
        if self.wave > self.MAX_WAVES: return True
        active_spheres = sum(1 for s in self.spheres if s["is_active"])
        if active_spheres == 0 and self.wave > 0: return True
        return False

    def _update_cooldowns(self):
        for k in self.action_cooldowns:
            if self.action_cooldowns[k] > 0:
                self.action_cooldowns[k] -= 1
    
    def _update_spheres(self):
        for sphere in self.spheres:
            if sphere["is_active"] and sphere["health"] <= 0:
                sphere["is_active"] = False
                self._create_particle_burst(sphere["pos"], 50, self.COLOR_SPHERE_CRITICAL, 3.0)
            if self.steps % 30 == 0 and sphere["power"] < sphere["max_power"]:
                sphere["power"] += 1

    def _update_enemies(self):
        active_indices = [i for i, s in enumerate(self.spheres) if s["is_active"]]
        if not active_indices: return

        for enemy in self.enemies:
            if not self.spheres[enemy['target_sphere_idx']]['is_active']:
                enemy['target_sphere_idx'] = self.np_random.choice(active_indices)
            
            target_sphere = self.spheres[enemy['target_sphere_idx']]
            direction = target_sphere['pos'] - enemy['pos']
            norm = np.linalg.norm(direction)
            if norm > 0:
                enemy['pos'] += direction / norm * enemy['speed']

    def _update_projectiles(self):
        self.projectiles[:] = [p for p in self.projectiles if 0 <= p['pos'][0] <= self.WIDTH and 0 <= p['pos'][1] <= self.HEIGHT]
        for proj in self.projectiles:
            proj['pos'] += proj['vel'] * 8.0
            if self.np_random.random() > 0.5:
                self._create_particle(proj['pos'], self.COLOR_PROJECTILE, lifespan=10, size=1)

    def _update_particles(self):
        self.particles[:] = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _create_sphere(self, pos):
        return {"pos": np.array(pos, dtype=float), "radius": 25.0, "max_radius": 25.0, "health": 100.0, "max_health": 100.0, "power": 10, "max_power": 20, "shield_type": 0, "is_active": True}

    def _create_enemy(self, pos, target_idx, speed, health, type):
        return {"pos": np.array(pos, dtype=float), "target_sphere_idx": target_idx, "speed": speed, "health": health, "type": type, "radius": 8.0}

    def _create_projectile(self, pos, vel):
        return {"pos": np.array(pos, dtype=float), "vel": np.array(vel, dtype=float)}

    def _create_particle(self, pos, color, lifespan=20, size=2, vel=None):
        if vel is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.0)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        self.particles.append({"pos": np.array(pos, dtype=float), "vel": vel, "lifespan": lifespan, "max_lifespan": lifespan, "color": color, "size": size})

    def _create_particle_burst(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.5) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(15, 30)
            self._create_particle(pos, color, lifespan, vel=vel)

    def _get_spawn_pos(self):
        side = self.np_random.integers(0, 4)
        if side == 0: return np.array([self.np_random.uniform(0, self.WIDTH), -20.0])
        if side == 1: return np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20.0])
        if side == 2: return np.array([-20.0, self.np_random.uniform(0, self.HEIGHT)])
        return np.array([self.WIDTH + 20.0, self.np_random.uniform(0, self.HEIGHT)])

    def _render_game(self):
        active_spheres = [s for s in self.spheres if s["is_active"]]
        if len(active_spheres) > 1:
            for i in range(len(active_spheres)):
                for j in range(i + 1, len(active_spheres)):
                    p1 = tuple(active_spheres[i]['pos'].astype(int))
                    p2 = tuple(active_spheres[j]['pos'].astype(int))
                    pygame.draw.aaline(self.screen, self.COLOR_CONNECT, p1, p2, 2)
        
        for i, sphere in enumerate(self.spheres):
            if not sphere["is_active"]: continue
            pos = sphere['pos']
            health_ratio = max(0, sphere['health'] / sphere['max_health'])
            color = self._interpolate_color(self.COLOR_SPHERE_CRITICAL, self.COLOR_SPHERE_HEALTHY, health_ratio)
            self._draw_glowing_circle(self.screen, pos, sphere['radius'], color, tuple(c//2 for c in color))
            if sphere['shield_type'] > 0:
                shield_color = self.COLOR_SHIELD_TRI if sphere['shield_type'] == 0 else self.COLOR_SHIELD_SQR
                self._draw_shield(pos, sphere['radius'] + 8, shield_color, sphere['shield_type'])
            if i == self.selected_sphere_idx:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                radius = sphere['radius'] + 12 + pulse * 3
                alpha = int(100 + pulse * 100)
                self._draw_glowing_circle(self.screen, pos, radius, self.COLOR_SELECT, self.COLOR_SELECT, filled=False, alpha=alpha)

        for proj in self.projectiles:
            p1 = proj['pos'] - proj['vel'] * 4
            p2 = proj['pos'] + proj['vel'] * 4
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, tuple(p1.astype(int)), tuple(p2.astype(int)))
        
        for enemy in self.enemies:
            color = self.COLOR_ENEMY_TRI if enemy['type'] == 0 else self.COLOR_ENEMY_SQR
            if enemy['type'] == 0: self._draw_aa_triangle(self.screen, enemy['pos'], enemy['radius'], color)
            else: self._draw_aa_square(self.screen, enemy['pos'], enemy['radius'], color)
                
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

    def _render_ui(self):
        for sphere in self.spheres:
            if not sphere["is_active"]: continue
            power_text = self.font_small.render(str(sphere['power']), True, self.COLOR_UI)
            text_rect = power_text.get_rect(center=tuple(sphere['pos'].astype(int)))
            self.screen.blit(power_text, text_rect)

        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        wave_text = self.font_medium.render(f"Wave: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_UI)
        wave_rect = wave_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(wave_text, wave_rect)

        if self.wave_in_progress:
            enemies_text = self.font_medium.render(f"Enemies: {len(self.enemies)}", True, self.COLOR_UI)
            enemies_rect = enemies_text.get_rect(topright=(self.WIDTH - 10, 40))
            self.screen.blit(enemies_text, enemies_rect)
        elif self.wave < self.MAX_WAVES:
            timer_sec = math.ceil(self.inter_wave_timer / 30)
            msg = f"Wave {self.wave + 1} starting in {timer_sec}"
            inter_text = self.font_large.render(msg, True, self.COLOR_UI)
            inter_rect = inter_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(inter_text, inter_rect)
        elif self.wave >= self.MAX_WAVES:
            inter_text = self.font_large.render("YOU WIN!", True, self.COLOR_UI)
            inter_rect = inter_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(inter_text, inter_rect)

    def _draw_glowing_circle(self, surf, pos, radius, color, glow_color, filled=True, alpha=255):
        pos_int = tuple(pos.astype(int))
        for i in range(4):
            glow_radius = int(radius + i * 2)
            glow_alpha = int(alpha / (3 + i*2))
            pygame.gfxdraw.aacircle(surf, pos_int[0], pos_int[1], glow_radius, (*glow_color, glow_alpha))
        if filled:
            pygame.gfxdraw.filled_circle(surf, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.aacircle(surf, pos_int[0], pos_int[1], int(radius), color)

    def _draw_shield(self, pos, radius, color, type):
        if type == 0:
            points = [(pos[0], pos[1] - radius), (pos[0] - radius * math.cos(math.pi/6), pos[1] + radius * math.sin(math.pi/6)), (pos[0] + radius * math.cos(math.pi/6), pos[1] + radius * math.sin(math.pi/6))]
        else:
            half_rad = radius / math.sqrt(2)
            points = [(pos[0] - half_rad, pos[1] - half_rad), (pos[0] + half_rad, pos[1] - half_rad), (pos[0] + half_rad, pos[1] + half_rad), (pos[0] - half_rad, pos[1] + half_rad)]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, color)

    def _draw_aa_triangle(self, surf, pos, radius, color):
        angle_offset = self.steps * 0.02
        points = [(pos[0] + radius * math.cos(2 * math.pi * i / 3 + angle_offset), pos[1] + radius * math.sin(2 * math.pi * i / 3 + angle_offset)) for i in range(3)]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(surf, int_points, color)
        pygame.gfxdraw.filled_polygon(surf, int_points, color)

    def _draw_aa_square(self, surf, pos, radius, color):
        angle_offset = -self.steps * 0.02
        points = [(pos[0] + radius * math.cos(math.pi / 4 + math.pi / 2 * i + angle_offset), pos[1] + radius * math.sin(math.pi / 4 + math.pi / 2 * i + angle_offset)) for i in range(4)]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(surf, int_points, color)
        pygame.gfxdraw.filled_polygon(surf, int_points, color)

    def _interpolate_color(self, color1, color2, factor):
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))


if __name__ == '__main__':
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen_width, screen_height = 960, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Sphere Defense")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset | Q: Quit")
    print("="*30 + "\n")

    while not terminated and not truncated:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset(seed=42)
                total_reward = 0
                terminated, truncated = False, False

        if terminated:
            break

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        truncated = trunc

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Info: {info}")
            # Render one last time
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (screen_width, screen_height))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000) # Wait 2 seconds before closing
            break

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (screen_width, screen_height))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    env.close()