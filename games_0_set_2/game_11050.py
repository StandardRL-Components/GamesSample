import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
import random
from collections import namedtuple

# Helper data structures for clarity
Particle = namedtuple("Particle", ["pos", "vel", "radius", "color", "lifetime"])
Projectile = namedtuple("Projectile", ["pos", "vel", "color", "damage", "radius"])
Enemy = namedtuple("Enemy", ["pos", "vel", "health", "max_health", "radius", "cooldown"])
Shield = namedtuple("Shield", ["pos", "health", "max_health", "radius"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Defend your fractal fortress core from waves of incoming enemies by aiming a crosshair and firing projectiles. Deploy shields to protect the core."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the crosshair. Press space to fire. "
        "Press shift to enter/cycle deploy mode, then press space to build a shield at the crosshair."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.W, self.H = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 40, bold=True)
        self.font_deploy = pygame.font.SysFont("Consolas", 16, bold=True)
        self.np_random = None

        # --- Visual & Game Constants ---
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 75, 128)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (128, 25, 25)
        self.COLOR_HEALTH = (50, 220, 100)
        self.COLOR_RESOURCE = (255, 200, 0)
        self.COLOR_FRACTAL = (40, 50, 70)
        self.COLOR_SHIELD = (100, 200, 255)
        self.COLOR_SHIELD_GLOW = (50, 100, 128)
        self.COLOR_WHITE = (220, 220, 220)

        self.MAX_STEPS = 2500
        self.TOTAL_WAVES = 10

        # Initialize state variables
        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0.0

        # Fortress
        self.core_pos = np.array([self.W / 2, self.H / 2], dtype=float)
        self.core_radius = 20
        self.core_max_health = 100
        self.core_health = self.core_max_health
        self.fractal_lines = self._generate_fractal_fortress(
            start_pos=self.core_pos, angle=90, length=70, depth=3
        )

        # Player
        self.crosshair_pos = np.array([self.W / 2, self.H / 4], dtype=float)
        self.crosshair_speed = 8.0
        self.fire_cooldown = 0
        self.fire_rate = 8  # steps

        # Upgrade System
        self.deploy_mode = False
        self.resources = 1
        self.available_upgrades = []
        self.unlocked_upgrade_types = ["shield"]
        self.selected_upgrade_index = 0
        self.deployed_shields = []
        self.shield_cost = 1

        # Action Handling
        self.prev_action = np.array([0, 0, 0])

        # Entities
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        # Wave Management
        self.wave = 0
        self.wave_transition_timer = 120 # 4s @ 30fps
        self.enemies_killed_this_wave = 0
        self.enemies_to_spawn_this_wave = 0
        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0.0
        
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        self.game_over = terminated or truncated

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            self._handle_collisions()
            self._cleanup_entities()
            self.steps += 1

        self.prev_action = action
        
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        self.game_over = terminated or truncated

        if terminated and self.core_health > 0: # Win condition
            self.reward_this_step += 100.0
            self.score += 100

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action
        space_just_pressed = space_action == 1 and self.prev_action[1] == 0
        shift_just_pressed = shift_action == 1 and self.prev_action[2] == 0

        # --- Movement ---
        if movement != 0:
            self.deploy_mode = False # Cancel deploy mode on movement
            if movement == 1: self.crosshair_pos[1] -= self.crosshair_speed
            elif movement == 2: self.crosshair_pos[1] += self.crosshair_speed
            elif movement == 3: self.crosshair_pos[0] -= self.crosshair_speed
            elif movement == 4: self.crosshair_pos[0] += self.crosshair_speed
        
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.W)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.H)

        # --- Shift: Cycle Upgrades ---
        if shift_just_pressed and self.available_upgrades:
            self.deploy_mode = True
            self.selected_upgrade_index = (self.selected_upgrade_index + 1) % len(self.available_upgrades)

        # --- Space: Fire or Deploy ---
        if space_just_pressed:
            if self.deploy_mode and self.available_upgrades:
                upgrade_type = self.available_upgrades[self.selected_upgrade_index]
                if upgrade_type == "shield" and self.resources >= self.shield_cost:
                    self.resources -= self.shield_cost
                    new_shield = Shield(pos=self.crosshair_pos.copy(), health=50, max_health=50, radius=30)
                    self.deployed_shields.append(new_shield)
                    self.deploy_mode = False
            elif not self.deploy_mode and self.fire_cooldown <= 0:
                self.fire_cooldown = self.fire_rate
                direction = self.crosshair_pos - self.core_pos
                distance = np.linalg.norm(direction)
                if distance > 0:
                    vel = (direction / distance) * 12.0
                    new_proj = Projectile(pos=self.core_pos.copy(), vel=vel, color=self.COLOR_PLAYER, damage=10, radius=5)
                    self.player_projectiles.append(new_proj)

    def _update_game_state(self):
        # Cooldowns
        if self.fire_cooldown > 0: self.fire_cooldown -= 1

        # Wave management
        if self.enemies_killed_this_wave >= self.enemies_to_spawn_this_wave and self.wave <= self.TOTAL_WAVES:
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
            else:
                self._start_new_wave()

        # Update enemies
        for i, enemy in enumerate(self.enemies):
            direction = self.core_pos - enemy.pos
            distance = np.linalg.norm(direction)
            if distance > 0:
                new_vel = (direction / distance) * (0.8 + self.wave * 0.05)
            else:
                new_vel = np.array([0.0, 0.0])
            
            new_pos = enemy.pos + new_vel
            new_cooldown = enemy.cooldown - 1
            
            if new_cooldown <= 0:
                if distance > 0: # Only fire if not on top of the core
                    proj_vel = (direction / distance) * 5.0
                    enemy_proj = Projectile(pos=enemy.pos.copy(), vel=proj_vel, color=self.COLOR_ENEMY, damage= (1 + self.wave // 2), radius=4)
                    self.enemy_projectiles.append(enemy_proj)
                new_cooldown = self.np_random.integers(80, 121) - self.wave * 2

            self.enemies[i] = enemy._replace(pos=new_pos, vel=new_vel, cooldown=new_cooldown)

        # Update projectiles
        self.player_projectiles = [p._replace(pos=p.pos + p.vel) for p in self.player_projectiles]
        self.enemy_projectiles = [p._replace(pos=p.pos + p.vel) for p in self.enemy_projectiles]

        # Update particles
        new_particles = []
        for p in self.particles:
            if p.lifetime > 0:
                new_particles.append(p._replace(pos=p.pos + p.vel, lifetime=p.lifetime - 1, radius=max(0, p.radius * 0.95)))
        self.particles = new_particles

    def _handle_collisions(self):
        # This function is refactored to avoid list.remove() on namedtuples containing numpy arrays,
        # which causes a ValueError due to ambiguous boolean evaluation of element-wise array comparison.
        # The new logic builds new lists of surviving entities, which is safer and avoids mutation-during-iteration bugs.

        # --- Player projectiles vs Enemies ---
        surviving_player_projectiles = []
        enemy_damage_map = {i: 0 for i in range(len(self.enemies))}

        for proj in self.player_projectiles:
            hit = False
            for i, enemy in enumerate(self.enemies):
                if enemy_damage_map.get(i, 0) + self.enemies[i].health <= 0: continue # Skip already dead enemies
                if np.linalg.norm(proj.pos - enemy.pos) < enemy.radius + proj.radius:
                    hit = True
                    enemy_damage_map[i] += proj.damage
                    self._create_explosion(proj.pos, self.COLOR_ENEMY, 10)
                    self.reward_this_step += 0.1
                    break
            if not hit:
                surviving_player_projectiles.append(proj)
        
        self.player_projectiles = surviving_player_projectiles

        new_enemies = []
        for i, enemy in enumerate(self.enemies):
            total_damage = enemy_damage_map[i]
            if enemy.health - total_damage <= 0:
                self._create_explosion(enemy.pos, self.COLOR_ENEMY, 30)
                self.reward_this_step += 1.0
                self.score += 1
                self.resources += 1
                self.enemies_killed_this_wave += 1
            else:
                new_enemies.append(enemy._replace(health=enemy.health - total_damage))
        self.enemies = new_enemies

        # --- Enemy projectiles vs Shields & Core ---
        surviving_enemy_projectiles = []
        shield_damage_map = {i: 0 for i in range(len(self.deployed_shields))}
        core_damage = 0

        for proj in self.enemy_projectiles:
            hit = False
            # Check shields
            for i, shield in enumerate(self.deployed_shields):
                if shield_damage_map.get(i, 0) + self.deployed_shields[i].health <= 0: continue
                if np.linalg.norm(proj.pos - shield.pos) < shield.radius + proj.radius:
                    hit = True
                    shield_damage_map[i] += proj.damage
                    self._create_explosion(proj.pos, self.COLOR_SHIELD, 5)
                    break
            if hit:
                continue

            # Check core
            if np.linalg.norm(proj.pos - self.core_pos) < self.core_radius + proj.radius:
                hit = True
                core_damage += proj.damage
                self.reward_this_step -= 0.5
                self._create_explosion(proj.pos, self.COLOR_HEALTH, 15)
            
            if not hit:
                surviving_enemy_projectiles.append(proj)
        
        self.enemy_projectiles = surviving_enemy_projectiles

        # Apply shield damage
        new_shields = []
        for i, shield in enumerate(self.deployed_shields):
            total_damage = shield_damage_map[i]
            if shield.health - total_damage <= 0:
                self._create_explosion(shield.pos, self.COLOR_SHIELD, 20)
            else:
                new_shields.append(shield._replace(health=shield.health - total_damage))
        self.deployed_shields = new_shields

        # Apply core damage
        self.core_health -= core_damage

    def _cleanup_entities(self):
        is_on_screen = lambda pos: 0 <= pos[0] <= self.W and 0 <= pos[1] <= self.H
        self.player_projectiles = [p for p in self.player_projectiles if is_on_screen(p.pos)]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if is_on_screen(p.pos)]

    def _start_new_wave(self):
        if self.wave > 0:
            self.reward_this_step += 5.0
            self.score += 5
        
        self.wave += 1
        if self.wave > self.TOTAL_WAVES: return

        self.enemies_killed_this_wave = 0
        self.enemies_to_spawn_this_wave = 5 + self.wave * 2
        self.wave_transition_timer = 120
        
        for _ in range(self.enemies_to_spawn_this_wave):
            self._spawn_enemy()
        
        if self.wave % 3 == 0 and "shield" not in self.available_upgrades:
            self.available_upgrades.append("shield")

    def _spawn_enemy(self):
        side = self.np_random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top': pos = np.array([self.np_random.uniform(0, self.W), -20.0])
        elif side == 'bottom': pos = np.array([self.np_random.uniform(0, self.W), self.H + 20.0])
        elif side == 'left': pos = np.array([-20.0, self.np_random.uniform(0, self.H)])
        else: pos = np.array([self.W + 20.0, self.np_random.uniform(0, self.H)])
        
        health = 20 + 5 * (self.wave - 1)
        new_enemy = Enemy(
            pos=pos, vel=np.zeros(2), health=health, max_health=health,
            radius=12, cooldown=self.np_random.integers(100, 151)
        )
        self.enemies.append(new_enemy)

    def _is_terminated(self):
        if self.core_health <= 0:
            return True
        if self.wave > self.TOTAL_WAVES and not self.enemies:
            return True
        return False

    def _is_truncated(self):
        return self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "resources": self.resources}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Fractal Fortress
        for start, end in self.fractal_lines:
            pygame.draw.line(self.screen, self.COLOR_FRACTAL, start, end, 2)
        
        # Deployed Shields
        for shield in self.deployed_shields:
            alpha = int(100 + 155 * (shield.health / shield.max_health))
            self._draw_glowing_circle(shield.pos, shield.radius, self.COLOR_SHIELD, self.COLOR_SHIELD_GLOW, alpha)

        # Fortress Core
        self._draw_glowing_circle(self.core_pos, self.core_radius, self.COLOR_HEALTH, self.COLOR_HEALTH, 255)

        # Player Projectiles
        for p in self.player_projectiles:
            self._draw_glowing_circle(p.pos, p.radius, p.color, self.COLOR_PLAYER_GLOW)

        # Enemy Projectiles
        for p in self.enemy_projectiles:
            self._draw_glowing_circle(p.pos, p.radius, p.color, self.COLOR_ENEMY_GLOW)

        # Enemies
        for enemy in self.enemies:
            self._draw_glowing_circle(enemy.pos, enemy.radius, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW)
            # Health bar for enemy
            bar_w = enemy.radius * 2 * (enemy.health / enemy.max_health)
            bar_h = 4
            bar_pos = (enemy.pos[0] - enemy.radius, enemy.pos[1] - enemy.radius - bar_h - 2)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (*bar_pos, bar_w, bar_h))

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.lifetime / 30.0))))
            color = (*p.color, alpha)
            pos_int = (int(p.pos[0]), int(p.pos[1]))
            radius_int = max(1, int(p.radius))
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, radius_int, color)

        # Crosshair
        pos_int = (int(self.crosshair_pos[0]), int(self.crosshair_pos[1]))
        color = self.COLOR_SHIELD if self.deploy_mode else self.COLOR_PLAYER
        glow = self.COLOR_SHIELD_GLOW if self.deploy_mode else self.COLOR_PLAYER_GLOW
        self._draw_glowing_circle(self.crosshair_pos, 8, color, glow)
        pygame.draw.line(self.screen, color, (pos_int[0] - 12, pos_int[1]), (pos_int[0] - 4, pos_int[1]), 2)
        pygame.draw.line(self.screen, color, (pos_int[0] + 4, pos_int[1]), (pos_int[0] + 12, pos_int[1]), 2)
        pygame.draw.line(self.screen, color, (pos_int[0], pos_int[1] - 12), (pos_int[0], pos_int[1] - 4), 2)
        pygame.draw.line(self.screen, color, (pos_int[0], pos_int[1] + 4), (pos_int[0], pos_int[1] + 12), 2)

    def _render_ui(self):
        # Core Health Bar
        health_ratio = max(0, self.core_health / self.core_max_health)
        bar_width = (self.W - 40) * health_ratio
        pygame.draw.rect(self.screen, self.COLOR_FRACTAL, (20, self.H - 30, self.W - 40, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (20, self.H - 30, bar_width, 20))
        health_text = self.font_ui.render(f"CORE: {int(self.core_health)}/{self.core_max_health}", True, self.COLOR_WHITE)
        self.screen.blit(health_text, (25, self.H - 29))

        # Wave Info
        wave_text = self.font_ui.render(f"WAVE: {self.wave}/{self.TOTAL_WAVES}", True, self.COLOR_WHITE)
        self.screen.blit(wave_text, (20, 10))

        # Resource Info
        res_text = self.font_ui.render(f"RESOURCES: {self.resources}", True, self.COLOR_RESOURCE)
        self.screen.blit(res_text, (20, 35))

        # Score Info
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        score_rect = score_text.get_rect(topright=(self.W - 20, 10))
        self.screen.blit(score_text, score_rect)
        
        # Deploy Mode Indicator
        if self.deploy_mode and self.available_upgrades:
            upgrade_name = self.available_upgrades[self.selected_upgrade_index].upper()
            deploy_text = self.font_deploy.render(f"DEPLOY: {upgrade_name}", True, self.COLOR_SHIELD)
            text_rect = deploy_text.get_rect(center=(self.crosshair_pos[0], self.crosshair_pos[1] - 30))
            self.screen.blit(deploy_text, text_rect)
            # Draw placement radius
            pygame.gfxdraw.aacircle(self.screen, int(self.crosshair_pos[0]), int(self.crosshair_pos[1]), 30, self.COLOR_SHIELD)

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            if self.core_health <= 0:
                msg = "FORTRESS DESTROYED"
                color = self.COLOR_ENEMY
            else:
                msg = "VICTORY"
                color = self.COLOR_PLAYER
            end_text = self.font_big.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(end_text, end_rect)

    def _generate_fractal_fortress(self, start_pos, angle, length, depth):
        lines = []
        q = [(start_pos, angle, length, depth)]
        while q:
            pos, ang, lng, d = q.pop(0)
            if d <= 0 or lng < 2:
                continue
            
            rad = math.radians(ang)
            end_pos = pos + np.array([math.cos(rad), -math.sin(rad)]) * lng
            lines.append((pos, end_pos))

            q.append((end_pos, ang + 35, lng * 0.7, d - 1))
            q.append((end_pos, ang - 35, lng * 0.7, d - 1))
        return lines

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifetime = self.np_random.integers(15, 31)
            radius = self.np_random.uniform(2, 5)
            self.particles.append(Particle(pos.copy(), vel, radius, color, lifetime))

    def _draw_glowing_circle(self, pos, radius, color, glow_color, alpha=255):
        pos_int = (int(pos[0]), int(pos[1]))
        radius_int = max(1, int(radius))
        
        # Glow
        glow_radius = int(radius_int * 1.8)
        if glow_radius > 0:
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*glow_color, int(alpha/4)), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main circle
        if alpha == 255:
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, radius_int, color)
            pygame.gfxdraw.aacircle(self.screen, *pos_int, radius_int, color)
        else:
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, radius_int, (*color, alpha))
            pygame.gfxdraw.aacircle(self.screen, *pos_int, radius_int, (*color, alpha))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Fractal Fortress")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    # To track "just pressed" state for manual play
    prev_keys = pygame.key.get_pressed()

    while running:
        movement = 0 # 0: none
        space = 0 # 0: released
        shift = 0 # 0: released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset(seed=42)
                total_reward = 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        # The environment's "just pressed" logic needs a stream of 0s and 1s
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        prev_keys = keys

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset(seed=42)
            total_reward = 0.0
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS

    env.close()