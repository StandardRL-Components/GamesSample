import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:58:47.311325
# Source Brief: brief_00127.md
# Brief Index: 127
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for game entities
class Quad:
    def __init__(self, pos, angle, color, glow_color, base_speed, health=100):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = angle
        self.color = color
        self.glow_color = glow_color
        self.size = 15
        self.health = health
        self.max_health = health
        self.base_speed = base_speed
        self.current_speed_modifier = 1.0
        self.steer_speed = 3.5
        self.acceleration = 0.4
        self.braking = 0.6
        self.drag = 0.95  # friction
        self.max_vel = base_speed
        self.blast_cooldown = 0
        self.blast_damage_modifier = 1.0
        self.powerup = None
        self.powerup_timer = 0
        self.shield_active = False
        # For AI
        self.target_waypoint_index = 1
        self.ai_fire_rate = 0.1 # Fix for latent bug in difficulty scaling

    def update(self, movement_action=0):
        # Movement
        if movement_action == 1:  # Accelerate
            self.vel += pygame.math.Vector2(self.acceleration, 0).rotate(-self.angle)
        if movement_action == 2:  # Brake
            self.vel *= (1 - self.braking * 0.1) # Softer brake
        if movement_action == 3:  # Left
            self.angle -= self.steer_speed
        if movement_action == 4:  # Right
            self.angle += self.steer_speed

        self.max_vel = self.base_speed * self.current_speed_modifier
        if self.vel.length() > self.max_vel:
            self.vel.scale_to_length(self.max_vel)
        
        self.pos += self.vel
        self.vel *= self.drag

        if self.blast_cooldown > 0:
            self.blast_cooldown -= 1
            
        if self.powerup_timer > 0:
            self.powerup_timer -= 1
            if self.powerup_timer == 0:
                self.current_speed_modifier = 1.0
                self.blast_damage_modifier = 1.0
                self.shield_active = False
                self.powerup = None

    def get_corners(self):
        points = []
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # Quad shape: wider at back, pointy at front
        front = self.pos + pygame.math.Vector2(self.size, 0).rotate(-self.angle)
        back_l = self.pos + pygame.math.Vector2(-self.size * 0.7, self.size * 0.6).rotate(-self.angle)
        back_r = self.pos + pygame.math.Vector2(-self.size * 0.7, -self.size * 0.6).rotate(-self.angle)
        
        return [front, back_r, back_l]


class Projectile:
    def __init__(self, pos, angle, owner, damage=10):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(20, 0).rotate(-angle)
        self.owner = owner
        self.color = (100, 255, 100) # Green
        self.radius = 5
        self.damage = damage
        self.lifetime = 60 # frames

class Particle:
    def __init__(self, pos, vel, color, radius, lifetime):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.color = color
        self.radius = radius
        self.lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.radius *= 0.97
        return self.lifetime > 0 and self.radius > 0.5

class PowerUp:
    def __init__(self, pos, type):
        self.pos = pygame.math.Vector2(pos)
        self.type = type # "speed", "shield", "damage"
        self.color = (255, 223, 0) # Gold
        self.radius = 12
        self.pulse = 0

    def update(self):
        self.pulse = (self.pulse + 3) % 360

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Futuristic anti-gravity racing game. Collect power-ups, blast opponents, "
        "and upgrade your vehicle to win."
    )
    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to steer. "
        "Press space to fire a blast and shift to activate a collected power-up."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 15, 30)
        self.COLOR_TRACK = (50, 40, 70)
        self.COLOR_TRACK_BORDER = (90, 80, 110)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 150)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (150, 25, 25)
        self.COLOR_UI_TEXT = (220, 220, 255)

        # Persistent upgrades
        self.player_base_speed = 10.0
        self.player_blast_damage = 1.0
        self.player_blast_capacity = 3
        
        self.reset()
        # self.validate_implementation() # This is a non-standard developer tool, can be removed.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False

        self._create_track()

        self.player = Quad(
            pos=self.track_points[0] + pygame.math.Vector2(0, 20),
            angle= -90,
            color=self.COLOR_PLAYER,
            glow_color=self.COLOR_PLAYER_GLOW,
            base_speed=self.player_base_speed
        )
        self.player.blast_inventory = self.player_blast_capacity

        self.enemies = []
        enemy_colors = [(255, 100, 0), (255, 150, 0), (200, 50, 100)]
        for i in range(3):
            offset = pygame.math.Vector2(random.uniform(-40, 40), random.uniform(-60, -20))
            self.enemies.append(Quad(
                pos=self.track_points[0] + offset,
                angle=-90,
                color=enemy_colors[i],
                glow_color=tuple(c*0.6 for c in enemy_colors[i]),
                base_speed=8.0
            ))
        
        self.projectiles = []
        self.particles = []
        self._spawn_powerups()
        
        self.player_progress_index = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0

        # --- Handle Input ---
        fire_pressed = space_held and not self.last_space_held
        powerup_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        if fire_pressed and self.player.blast_cooldown == 0 and self.player.blast_inventory > 0:
            # sfx: Player Laser Shot
            self.player.blast_inventory -= 1
            self.player.blast_cooldown = 15 # 0.5s cooldown at 30fps
            proj_pos = self.player.pos + pygame.math.Vector2(self.player.size, 0).rotate(-self.player.angle)
            self.projectiles.append(Projectile(proj_pos, self.player.angle, 'player', 10 * self.player.blast_damage_modifier * self.player_blast_damage))
            self._create_particles(proj_pos, 5, (200, 255, 200), 2, 1.5)

        if powerup_pressed and self.player.powerup:
            # sfx: Powerup Activate
            if self.player.powerup == "speed":
                self.player.current_speed_modifier = 1.5
                self.player.powerup_timer = 150 # 5 seconds
            elif self.player.powerup == "shield":
                self.player.shield_active = True
                self.player.powerup_timer = 210 # 7 seconds
            elif self.player.powerup == "damage":
                self.player.blast_damage_modifier = 2.0
                self.player.powerup_timer = 300 # 10 seconds
            self.player.powerup = "active"


        # --- Update Game State ---
        self.player.update(movement)
        for enemy in self.enemies:
            self._update_enemy_ai(enemy)
        
        self.projectiles = [p for p in self.projectiles if self._update_projectile(p)]
        self.particles = [p for p in self.particles if p.update()]
        for p in self.powerups: p.update()

        # --- Collisions & Rewards ---
        reward += self._handle_collisions()

        # Progress reward
        old_progress = self.player_progress_index
        self.player_progress_index = self._get_closest_waypoint(self.player.pos, self.player_progress_index)
        if self.player_progress_index > old_progress or (self.player_progress_index == 0 and old_progress == len(self.track_points) - 2):
             reward += 0.1 * (self.player_progress_index - old_progress)
        
        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 500 == 0:
            for e in self.enemies: e.base_speed += 0.1
        if self.steps > 0 and self.steps % 1000 == 0:
            for e in self.enemies: e.ai_fire_rate += 0.05

        # --- Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.player.health <= 0 or self.steps >= 5000:
                reward = -100 # Loss
                # sfx: Player Explosion
            elif self.player_progress_index >= len(self.track_points) - 2:
                reward = 100 # Win
                # sfx: Race Win
                self._apply_upgrades()
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _apply_upgrades(self):
        # Simple upgrade path
        choice = random.choice(["speed", "damage", "capacity"])
        if choice == "speed": self.player_base_speed = min(15.0, self.player_base_speed + 0.5)
        elif choice == "damage": self.player_blast_damage = min(3.0, self.player_blast_damage + 0.2)
        elif choice == "capacity": self.player_blast_capacity = min(6, self.player_blast_capacity + 1)
    
    def _create_track(self):
        self.track_points = []
        center_y = self.height / 2
        num_points = 20
        for i in range(num_points):
            x = (self.width / (num_points - 1)) * i
            y = center_y + math.sin(i / 3) * 100 + random.uniform(-20, 20)
            self.track_points.append(pygame.math.Vector2(x, y))
        self.track_width = 80

    def _spawn_powerups(self):
        self.powerups = []
        for _ in range(3):
            segment = random.randint(2, len(self.track_points) - 3)
            pos = self.track_points[segment] + pygame.math.Vector2(random.uniform(-20, 20), random.uniform(-20, 20))
            type = random.choice(["speed", "shield", "damage"])
            self.powerups.append(PowerUp(pos, type))

    def _update_enemy_ai(self, enemy):
        # Navigation
        target_pos = self.track_points[enemy.target_waypoint_index]
        if enemy.pos.distance_to(target_pos) < 100:
            enemy.target_waypoint_index = (enemy.target_waypoint_index + 1) % len(self.track_points)
        
        vec_to_target = target_pos - enemy.pos
        target_angle = vec_to_target.angle_to(pygame.math.Vector2(1, 0))
        
        angle_diff = (target_angle - enemy.angle + 180) % 360 - 180
        
        # Accelerate and steer
        enemy_movement = 1
        if abs(angle_diff) > 15:
            enemy_movement = 2 # Brake if turning sharply
        
        if angle_diff < -2: enemy.angle -= enemy.steer_speed * 0.8
        elif angle_diff > 2: enemy.angle += enemy.steer_speed * 0.8
        
        enemy.update(enemy_movement)

        # Shooting
        if enemy.blast_cooldown == 0 and enemy.pos.distance_to(self.player.pos) < 250:
            vec_to_player = self.player.pos - enemy.pos
            player_angle = vec_to_player.angle_to(pygame.math.Vector2(1,0))
            angle_diff_player = (player_angle - enemy.angle + 180) % 360 - 180
            if abs(angle_diff_player) < 10: # If aiming at player
                # sfx: Enemy Laser Shot
                enemy.blast_cooldown = 90 # 3s cooldown
                proj_pos = enemy.pos + pygame.math.Vector2(enemy.size, 0).rotate(-enemy.angle)
                self.projectiles.append(Projectile(proj_pos, enemy.angle, 'enemy'))

    def _update_projectile(self, p):
        p.pos += p.vel
        p.lifetime -= 1
        if p.lifetime <= 0: return False
        if not (0 < p.pos.x < self.width and 0 < p.pos.y < self.height): return False
        return True

    def _handle_collisions(self):
        reward = 0
        
        # Projectile collisions
        for p in self.projectiles[:]:
            if p.owner == 'player':
                for e in self.enemies:
                    if p.pos.distance_to(e.pos) < e.size:
                        # sfx: Hit Success
                        e.health -= p.damage
                        self._create_particles(e.pos, 15, self.COLOR_ENEMY, 4, 3)
                        self.projectiles.remove(p)
                        reward += 1
                        if e.health <= 0: e.pos = pygame.math.Vector2(-1000, -1000) # "Kill"
                        break
            elif p.owner == 'enemy':
                if p.pos.distance_to(self.player.pos) < self.player.size:
                    if self.player.shield_active:
                        # sfx: Shield Block
                        self._create_particles(self.player.pos, 10, (100, 100, 255), 5, 2)
                    else:
                        # sfx: Player Hit
                        self.player.health -= p.damage
                        self._create_particles(self.player.pos, 15, self.COLOR_PLAYER, 4, 3)
                        reward -= 1
                    self.projectiles.remove(p)

        # Quad-wall collisions
        if not self._is_on_track(self.player.pos):
            self.player.vel *= 0.8 # heavy speed loss
            self.player.health -= 0.1
            reward -= 0.01
        for e in self.enemies:
            if not self._is_on_track(e.pos):
                e.vel *= 0.8
        
        # Powerup collection
        for pu in self.powerups[:]:
            if self.player.pos.distance_to(pu.pos) < self.player.size + pu.radius:
                # sfx: Powerup Collect
                if self.player.powerup is None:
                    self.player.powerup = pu.type
                self.powerups.remove(pu)
                reward += 5

        return reward

    def _is_on_track(self, pos):
        closest_dist_sq = float('inf')
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            l2 = p1.distance_squared_to(p2)
            if l2 == 0: continue
            t = max(0, min(1, (pos - p1).dot(p2 - p1) / l2))
            projection = p1 + t * (p2 - p1)
            closest_dist_sq = min(closest_dist_sq, pos.distance_squared_to(projection))
        return closest_dist_sq < (self.track_width / 2) ** 2

    def _get_closest_waypoint(self, pos, start_index=0):
        closest_dist_sq = float('inf')
        closest_index = start_index
        # Search a few waypoints ahead and behind for efficiency
        for i in range(max(0, start_index - 2), min(len(self.track_points), start_index + 5)):
            dist_sq = pos.distance_squared_to(self.track_points[i])
            if dist_sq < closest_dist_sq:
                closest_dist_sq = dist_sq
                closest_index = i
        return closest_index

    def _check_termination(self):
        win = self.player_progress_index >= len(self.track_points) - 2
        lose_health = self.player.health <= 0
        lose_steps = self.steps >= 5000
        return win or lose_health or lose_steps

    def _create_particles(self, pos, count, color, speed, size):
        for _ in range(count):
            angle = random.uniform(0, 360)
            vel = pygame.math.Vector2(random.uniform(0.5, 1.5) * speed, 0).rotate(angle)
            lifetime = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, color, size, lifetime))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Track
        pygame.draw.lines(self.screen, self.COLOR_TRACK_BORDER, False, [p.xy for p in self.track_points], self.track_width + 10)
        pygame.draw.lines(self.screen, self.COLOR_TRACK, False, [p.xy for p in self.track_points], self.track_width)
        pygame.draw.line(self.screen, (255,255,255), self.track_points[-2].xy, self.track_points[-1].xy, 3) # Finish line

        # Powerups
        for pu in self.powerups:
            r = pu.radius + math.sin(math.radians(pu.pulse)) * 3
            pygame.gfxdraw.filled_circle(self.screen, int(pu.pos.x), int(pu.pos.y), int(r), pu.color)
            pygame.gfxdraw.aacircle(self.screen, int(pu.pos.x), int(pu.pos.y), int(r), pu.color)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, (int(p.pos.x), int(p.pos.y)), int(p.radius))

        # Projectiles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), p.radius, p.color)
            pygame.gfxdraw.aacircle(self.screen, int(p.pos.x), int(p.pos.y), p.radius, p.color)

        # Quads
        for e in self.enemies:
            if e.health > 0: self._render_quad(e)
        self._render_quad(self.player)

    def _render_quad(self, quad):
        # Thruster particles
        if quad.vel.length() > 2:
            for _ in range(2):
                offset = pygame.math.Vector2(-quad.size*0.8, random.uniform(-quad.size*0.4, quad.size*0.4)).rotate(-quad.angle)
                vel = (quad.vel * -0.5).rotate(random.uniform(-15, 15))
                self.particles.append(Particle(quad.pos + offset, vel, (255, 180, 50), random.uniform(2,4), 10))

        # Glow effect
        for i in range(4):
            glow_color = (*quad.glow_color, 40 - i * 10)
            s = pygame.Surface((quad.size * 4, quad.size * 4), pygame.SRCALPHA)
            points = [(p[0] - quad.pos.x + quad.size*2, p[1] - quad.pos.y + quad.size*2) for p in quad.get_corners()]
            pygame.draw.polygon(s, glow_color, points)
            self.screen.blit(s, (int(quad.pos.x - quad.size*2), int(quad.pos.y - quad.size*2)), special_flags=pygame.BLEND_RGBA_ADD)

        # Main body
        pygame.draw.polygon(self.screen, quad.color, [p.xy for p in quad.get_corners()])
        pygame.draw.aalines(self.screen, tuple(min(255, c+50) for c in quad.color), True, [p.xy for p in quad.get_corners()])

        # Shield effect
        if quad.shield_active:
            alpha = 100 + math.sin(self.steps * 0.2) * 50
            pygame.gfxdraw.filled_circle(self.screen, int(quad.pos.x), int(quad.pos.y), int(quad.size * 1.5), (150, 150, 255, int(alpha)))

    def _render_ui(self):
        # Health Bar
        health_frac = max(0, self.player.health / self.player.max_health)
        health_color = (int(255 * (1 - health_frac)), int(255 * health_frac), 0)
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, health_color, (10, 10, 200 * health_frac, 20))
        health_text = self.font_small.render(f"HP", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Energy Blasts
        blast_text = self.font_small.render(f"BLASTS: {self.player.blast_inventory}/{self.player_blast_capacity}", True, self.COLOR_UI_TEXT)
        self.screen.blit(blast_text, (10, 35))

        # Progress
        progress_frac = self.player_progress_index / (len(self.track_points) - 2)
        progress_text = self.font_large.render(f"PROGRESS: {int(progress_frac * 100)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(progress_text, (self.width/2 - progress_text.get_width()/2, 10))

        # Powerup
        if self.player.powerup:
            pu_text_str = f"POWERUP: {self.player.powerup.upper()}"
            if self.player.powerup == "active":
                pu_text_str = f"ACTIVE: {int(self.player.powerup_timer/30)+1}s"
            pu_text = self.font_small.render(pu_text_str, True, (255,223,0))
            self.screen.blit(pu_text, (self.width - pu_text.get_width() - 10, self.height - 30))

        # Minimap
        map_rect = pygame.Rect(self.width - 110, 10, 100, 80)
        pygame.draw.rect(self.screen, (0,0,0,150), map_rect)
        pygame.draw.rect(self.screen, self.COLOR_TRACK_BORDER, map_rect, 1)
        map_points = [((p.x / self.width) * 96 + map_rect.x + 2, (p.y / self.height) * 76 + map_rect.y + 2) for p in self.track_points]
        pygame.draw.lines(self.screen, self.COLOR_TRACK, False, map_points, 2)
        # Player on map
        px = (self.player.pos.x / self.width) * 96 + map_rect.x + 2
        py = (self.player.pos.y / self.height) * 76 + map_rect.y + 2
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(px), int(py)), 3)
        # Enemies on map
        for e in self.enemies:
            if e.health > 0:
                ex = (e.pos.x / self.width) * 96 + map_rect.x + 2
                ey = (e.pos.y / self.height) * 76 + map_rect.y + 2
                pygame.draw.circle(self.screen, e.color, (int(ex), int(ey)), 2)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player.health,
            "progress": self.player_progress_index / (len(self.track_points) - 2),
            "upgrades": {
                "speed": self.player_base_speed,
                "damage": self.player_blast_damage,
                "capacity": self.player_blast_capacity
            }
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running local implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Quad Racer")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        # This is a bit tricky for manual play, as step() expects simultaneous actions
        # We'll just check if they are held down in this frame
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        # Combine into a single action
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
        if done:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            print(f"Final Upgrades: {info['upgrades']}")
            obs, info = env.reset()
            total_reward = 0
            # To prevent auto-closing, let's wait for a key press to restart
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        done = True # to exit the main loop
                    if event.type == pygame.KEYDOWN:
                        waiting = False

    env.close()