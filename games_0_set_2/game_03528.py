
# Generated: 2025-08-27T23:39:14.665433
# Source Brief: brief_03528.md
# Brief Index: 3528

        
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


# Helper class for isometric transformations and drawing
class IsoHelper:
    def __init__(self, surface, TILE_WIDTH=32, TILE_HEIGHT=16, offset_x=320, offset_y=100):
        self.surface = surface
        self.TILE_WIDTH_HALF = TILE_WIDTH / 2
        self.TILE_HEIGHT_HALF = TILE_HEIGHT / 2
        self.offset_x = offset_x
        self.offset_y = offset_y

    def cart_to_iso(self, x, y):
        iso_x = (x - y) * self.TILE_WIDTH_HALF + self.offset_x
        iso_y = (x + y) * self.TILE_HEIGHT_HALF + self.offset_y
        return int(iso_x), int(iso_y)

    def draw_iso_poly(self, points, color, z=0):
        iso_points = [self.cart_to_iso(p[0], p[1]) for p in points]
        iso_points = [(p[0], p[1] - z) for p in iso_points]
        if len(iso_points) > 2:
            pygame.gfxdraw.aapolygon(self.surface, iso_points, color)
            pygame.gfxdraw.filled_polygon(self.surface, iso_points, color)

    def draw_iso_circle(self, x, y, radius, color, z=0):
        iso_x, iso_y = self.cart_to_iso(x, y)
        pygame.gfxdraw.aacircle(self.surface, iso_x, int(iso_y - z), int(radius), color)
        pygame.gfxdraw.filled_circle(self.surface, iso_x, int(iso_y - z), int(radius), color)

# --- Game Entity Classes ---
class Player:
    def __init__(self, np_random):
        self.np_random = np_random
        self.pos = np.array([0.0, 0.0])
        self.speed = 0.2
        self.radius = 0.4
        self.health = 100
        self.max_health = 100
        self.ammo = 30
        self.max_ammo = 50
        self.aim_angle = -math.pi / 2
        self.shoot_cooldown = 0
        self.shove_cooldown = 0
        self.damage_flash = 0

    def move(self, direction_vector):
        if np.linalg.norm(direction_vector) > 0:
            self.pos += direction_vector * self.speed
            self.aim_angle = math.atan2(direction_vector[1], direction_vector[0])

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)
        self.damage_flash = 10

class Zombie:
    def __init__(self, pos, speed, np_random):
        self.np_random = np_random
        self.pos = np.array(pos, dtype=float)
        self.base_speed = speed
        self.speed = speed
        self.radius = 0.3 + self.np_random.random() * 0.2
        self.health = 100
        self.hit_flash = 0
        self.color_variation = self.np_random.integers(0, 50)

    def update(self, player_pos):
        direction = player_pos - self.pos
        dist = np.linalg.norm(direction)
        if dist > 0.1:
            direction /= dist
            self.pos += direction * self.speed
        if self.hit_flash > 0:
            self.hit_flash -= 1

    def take_damage(self, amount):
        self.health -= amount
        self.hit_flash = 5
        return self.health <= 0

class Projectile:
    def __init__(self, pos, angle):
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array([math.cos(angle), math.sin(angle)]) * 0.8
        self.lifespan = 30

    def update(self):
        self.pos += self.velocity
        self.lifespan -= 1

class Particle:
    def __init__(self, pos, vel, size, lifespan, color, gravity=0):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.size = size
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.gravity = gravity

    def update(self):
        self.pos += self.vel
        self.vel[1] += self.gravity
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

# --- Main Game Environment ---
class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    user_guide = "Controls: Arrows to move and aim. Hold Space to shoot. Press Shift for a defensive shove."
    game_description = "Survive waves of zombies in a gritty, isometric arena. Collect ammo and stay alive!"
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_BOUNDS = np.array([12, 12])
    MAX_STEPS = 2500
    MAX_WAVES = 10
    
    # Colors
    COLOR_BG = (25, 25, 30)
    COLOR_GRID = (40, 40, 45)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150, 50)
    COLOR_ZOMBIE_BASE = (200, 50, 50)
    COLOR_AMMO = (255, 220, 0)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.iso_helper = IsoHelper(self.screen, offset_x=self.SCREEN_WIDTH // 2, offset_y=120)

        # State variables are initialized in reset()
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player = Player(self.np_random)
        self.zombies = []
        self.projectiles = []
        self.ammo_packs = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.wave = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._start_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        self.reward_this_step = 0.1  # Survival reward

        if not self.game_over and not self.game_won:
            self._handle_input(action)
            self._update_game_state()
            self._cleanup_entities()

        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                self.reward_this_step += 100
            elif self.game_over:
                self.reward_this_step -= 100
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, self.reward_this_step, terminated, False, info

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1.0 # Up
        elif movement == 2: move_vec[1] = 1.0  # Down
        elif movement == 3: move_vec[0] = -1.0 # Left
        elif movement == 4: move_vec[0] = 1.0  # Right
        self.player.move(move_vec)

        if space_held and self.player.shoot_cooldown <= 0 and self.player.ammo > 0:
            self._shoot()
        
        if shift_held and not self.prev_shift_held and self.player.shove_cooldown <= 0:
            self._shove()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        # Update cooldowns
        if self.player.shoot_cooldown > 0: self.player.shoot_cooldown -= 1
        if self.player.shove_cooldown > 0: self.player.shove_cooldown -= 1
        if self.player.damage_flash > 0: self.player.damage_flash -= 1

        # Player bounds
        self.player.pos = np.clip(self.player.pos, -self.WORLD_BOUNDS, self.WORLD_BOUNDS)

        # Update projectiles and check hits
        for p in self.projectiles:
            p.update()
            for z in self.zombies:
                if np.linalg.norm(p.pos - z.pos) < z.radius + 0.1:
                    if z.take_damage(50):
                        self._create_particles(z.pos, 20, (200, 0, 0), 10, 1.0)
                        self.zombies.remove(z)
                        self.score += 10
                        self.reward_this_step += 1.0
                        if self.np_random.random() < 0.2: # 20% chance to drop ammo
                           self._spawn_ammo_pack(z.pos)
                    else:
                        self._create_particles(p.pos, 5, self.COLOR_WHITE, 4, 0.2)
                    p.lifespan = 0 # Remove projectile on hit
                    # SFX: bullet_hit.wav
                    break
        
        # Update zombies and check player collision
        for z in self.zombies:
            z.update(self.player.pos)
            if np.linalg.norm(z.pos - self.player.pos) < z.radius + self.player.radius:
                damage = 5
                self.player.take_damage(damage)
                self.reward_this_step -= 0.1 * damage
                # SFX: player_hurt.wav

        # Check ammo pack collection
        for pack in self.ammo_packs:
            if np.linalg.norm(pack - self.player.pos) < self.player.radius + 0.3:
                ammo_gain = 15
                self.player.ammo = min(self.player.max_ammo, self.player.ammo + ammo_gain)
                self.ammo_packs.remove(pack)
                self.score += 5
                self.reward_this_step += 0.5
                self._create_particles(pack, 10, self.COLOR_AMMO, 5, 0.5)
                # SFX: ammo_pickup.wav
                break

        # Update particles
        for particle in self.particles:
            particle.update()

        # Check for wave completion
        if not self.zombies and not self.game_over:
            if self.wave >= self.MAX_WAVES:
                self.game_won = True
            else:
                self._start_wave()

    def _shoot(self):
        self.player.ammo -= 1
        self.player.shoot_cooldown = 5 # Fire rate
        start_pos = self.player.pos + np.array([math.cos(self.player.aim_angle), math.sin(self.player.aim_angle)]) * 0.5
        self.projectiles.append(Projectile(start_pos, self.player.aim_angle))
        self._create_particles(start_pos, 3, self.COLOR_WHITE, 3, 0.1, gravity=0)
        # SFX: shoot.wav

    def _shove(self):
        self.player.shove_cooldown = 60 # 2 second cooldown
        self._create_particles(self.player.pos, 30, (200, 200, 255), 8, 0.3, is_ring=True)
        shove_radius = 2.0
        for z in self.zombies:
            dist_vec = z.pos - self.player.pos
            dist = np.linalg.norm(dist_vec)
            if 0 < dist < shove_radius:
                shove_force = (shove_radius - dist) * 0.5
                z.pos += (dist_vec / dist) * shove_force
        # SFX: shove.wav

    def _start_wave(self):
        self.wave += 1
        num_zombies = 10 + (self.wave - 1) * 5
        zombie_speed = 0.02 + (self.wave - 1) * 0.005
        for _ in range(num_zombies):
            self._spawn_zombie(zombie_speed)

    def _spawn_zombie(self, speed):
        angle = self.np_random.random() * 2 * math.pi
        dist = self.np_random.uniform(self.WORLD_BOUNDS[0] - 2, self.WORLD_BOUNDS[0])
        pos = np.array([math.cos(angle) * dist, math.sin(angle) * dist])
        self.zombies.append(Zombie(pos, speed, self.np_random))

    def _spawn_ammo_pack(self, pos):
        self.ammo_packs.append(pos)
        
    def _cleanup_entities(self):
        self.projectiles = [p for p in self.projectiles if p.lifespan > 0]
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _check_termination(self):
        if self.player.health <= 0:
            self.game_over = True
            return True
        if self.game_won:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "ammo": self.player.ammo, "health": self.player.health}

    def _render_game(self):
        # Ground grid
        for i in range(-15, 16):
            self.iso_helper.draw_iso_poly([(-15, i), (15, i)], self.COLOR_GRID)
            self.iso_helper.draw_iso_poly([(i, -15), (i, 15)], self.COLOR_GRID)

        # Draw entities in Z-order (bottom to top)
        entities = []
        for pack in self.ammo_packs:
            entities.append({'type': 'ammo', 'pos': pack, 'y': pack[1]})
        for z in self.zombies:
            entities.append({'type': 'zombie', 'obj': z, 'y': z.pos[1]})
        entities.append({'type': 'player', 'obj': self.player, 'y': self.player.pos[1]})

        entities.sort(key=lambda e: e['y'])
        
        for entity in entities:
            if entity['type'] == 'ammo':
                self.iso_helper.draw_iso_poly([
                    entity['pos'] + np.array([-0.2, -0.2]), entity['pos'] + np.array([0.2, -0.2]),
                    entity['pos'] + np.array([0.2, 0.2]), entity['pos'] + np.array([-0.2, 0.2])
                ], self.COLOR_AMMO, z=0.5)
            elif entity['type'] == 'zombie':
                self._render_zombie(entity['obj'])
            elif entity['type'] == 'player':
                self._render_player(entity['obj'])

        # Render projectiles and particles on top
        for p in self.projectiles:
            self.iso_helper.draw_iso_circle(p.pos[0], p.pos[1], 2, self.COLOR_PROJECTILE, z=5)
        
        for part in self.particles:
            alpha = int(255 * (part.lifespan / part.max_lifespan))
            color = (*part.color, alpha)
            self.iso_helper.draw_iso_circle(part.pos[0], part.pos[1], part.size, color, z=part.pos[1] * 0.1 + 5)

    def _render_player(self, p):
        # Player shadow
        self.iso_helper.draw_iso_circle(p.pos[0], p.pos[1], 10, (0,0,0,100))
        
        # Player body
        player_z = 10
        self.iso_helper.draw_iso_circle(p.pos[0], p.pos[1], 12, self.COLOR_PLAYER, z=player_z)
        
        # Glow effect
        glow_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (15, 15), 15)
        iso_x, iso_y = self.iso_helper.cart_to_iso(p.pos[0], p.pos[1])
        self.screen.blit(glow_surf, (iso_x - 15, iso_y - 15 - player_z))

        # Aiming indicator
        aim_end_x = p.pos[0] + math.cos(p.aim_angle) * 1.5
        aim_end_y = p.pos[1] + math.sin(p.aim_angle) * 1.5
        start_iso = self.iso_helper.cart_to_iso(p.pos[0], p.pos[1])
        end_iso = self.iso_helper.cart_to_iso(aim_end_x, aim_end_y)
        pygame.draw.aaline(self.screen, self.COLOR_WHITE, (start_iso[0], start_iso[1] - player_z), (end_iso[0], end_iso[1] - player_z), 1)

    def _render_zombie(self, z):
        # Zombie shadow
        self.iso_helper.draw_iso_circle(z.pos[0], z.pos[1], z.radius * 25, (0,0,0,100))

        # Zombie body
        zombie_z = z.radius * 20
        color = self.COLOR_ZOMBIE_BASE
        if z.hit_flash > 0:
            color = self.COLOR_WHITE
        else:
            color = (max(0, self.COLOR_ZOMBIE_BASE[0] - z.color_variation), 
                     self.COLOR_ZOMBIE_BASE[1], self.COLOR_ZOMBIE_BASE[2])
        
        self.iso_helper.draw_iso_circle(z.pos[0], z.pos[1], z.radius * 30, color, z=zombie_z)

    def _render_ui(self):
        # Health Bar
        health_pct = self.player.health / self.player.max_health
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_RED, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_GREEN, (10, 10, int(bar_width * health_pct), 20))
        health_text = self.font_small.render(f"HP: {self.player.health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 35))

        # Wave Counter
        wave_text = self.font_small.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Ammo Counter (near player)
        px, py = self.iso_helper.cart_to_iso(self.player.pos[0], self.player.pos[1])
        ammo_text = self.font_small.render(f"{self.player.ammo}", True, self.COLOR_AMMO)
        text_rect = ammo_text.get_rect(center=(px, py - 40))
        self.screen.blit(ammo_text, text_rect)

        # Game Over / Win Message
        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_RED
        elif self.game_won:
            msg = "YOU SURVIVED!"
            color = self.COLOR_GREEN
        else:
            return

        end_text = self.font_large.render(msg, True, color)
        text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(end_text, text_rect)
        
        # Red overlay on damage
        if self.player.damage_flash > 0:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.player.damage_flash / 10.0))
            s.fill((255, 0, 0, alpha))
            self.screen.blit(s, (0, 0))

    def _create_particles(self, pos, count, color, size_base, speed, gravity=0.01, is_ring=False):
        for _ in range(count):
            if is_ring:
                angle = self.np_random.random() * 2 * math.pi
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed * (0.5 + self.np_random.random() * 0.5)
            else:
                vel = self.np_random.standard_normal(2) * speed
            
            lifespan = self.np_random.integers(10, 20)
            size = size_base * (0.5 + self.np_random.random() * 0.5)
            self.particles.append(Particle(pos, vel, size, lifespan, color, gravity))

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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Zombie Survival")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    # Display controls
    print("\n" + "="*30)
    print(f"GAME: {GameEnv.game_description}")
    print(f"CONTROLS: {GameEnv.user_guide}")
    print("="*30 + "\n")
    
    total_reward = 0
    
    while running:
        # --- Action mapping for human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode Finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            running = False # End after one episode for simple run
            pygame.time.wait(3000) # Pause to see end screen

    env.close()
    pygame.quit()