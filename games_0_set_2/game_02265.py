import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to jump, ↓ to duck. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a powerful robot in a side-scrolling environment to destroy waves of enemy drones."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.ground_y = self.HEIGHT - 50
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.WIN_CONDITION = 20

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_GROUND = (25, 25, 45)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_TURRET = (40, 200, 120)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_PLAYER_PROJ = (100, 200, 255)
        self.COLOR_ENEMY_PROJ = (255, 150, 50)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (50, 255, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 50, 50)

        # EXACT spaces:
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Background cityscape (pre-rendered for efficiency)
        self.background_surface = self._create_background()

        # State variables are initialized in reset()
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.enemies_destroyed = 0
        self.game_over = False
        self.enemy_spawn_timer = 0
        self.enemy_projectile_speed = 0
        self.screen_shake = 0

        # Initialize state
        # self.reset() is not called in __init__ as per Gymnasium standard practice.
        # The user of the env is expected to call it.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Player state
        self.player = {
            "x": self.WIDTH / 4,
            "y": self.ground_y,
            "w": 30, "h": 50,
            "vx": 0, "vy": 0,
            "speed": 5,
            "gravity": 0.8,
            "jump_power": -15,
            "on_ground": True,
            "is_ducking": False,
            "health": 100,
            "max_health": 100,
            "fire_cooldown": 0,
            "max_fire_cooldown": 8,  # 4 shots per second approx
            "facing_right": True
        }

        # Game state
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.enemies_destroyed = 0
        self.game_over = False

        self.enemy_spawn_timer = 0
        self.enemy_projectile_speed = 4.0
        self.screen_shake = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        # shift_held is unused in this design

        # --- Handle Input & Player Logic ---
        self._handle_player_input(movement, space_held)
        self._update_player()

        # --- Update Game Objects ---
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()

        # --- Handle Collisions & Rewards ---
        reward += self._handle_collisions()

        # --- Spawn new enemies ---
        self._spawn_enemies()

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_projectile_speed += 0.5

        # --- Step-based Rewards/Penalties ---
        enemy_in_range = any(abs(e['x'] - self.player['x']) < self.WIDTH / 2 for e in self.enemies)
        if not space_held and enemy_in_range:
            reward -= 0.01

        # --- Check Termination Conditions ---
        terminated = False
        if self.player['health'] <= 0:
            reward -= 100
            terminated = True
            # sfx: player_death
        elif self.enemies_destroyed >= self.WIN_CONDITION:
            reward += 100
            terminated = True
            # sfx: victory
        
        truncated = self.steps >= self.MAX_STEPS

        self.game_over = terminated or truncated

        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_pressed):
        # Horizontal Movement
        if movement == 3:  # Left
            self.player['vx'] = -self.player['speed']
            self.player['facing_right'] = False
        elif movement == 4:  # Right
            self.player['vx'] = self.player['speed']
            self.player['facing_right'] = True
        else:
            self.player['vx'] = 0

        # Ducking
        self.player['is_ducking'] = (movement == 2 and self.player['on_ground'])
        if self.player['is_ducking']:
            self.player['vx'] *= 0.5  # Slower when ducking

        # Jumping
        if movement == 1 and self.player['on_ground']:
            self.player['vy'] = self.player['jump_power']
            self.player['on_ground'] = False
            # sfx: jump

        # Firing
        if self.player['fire_cooldown'] > 0:
            self.player['fire_cooldown'] -= 1

        if space_pressed and self.player['fire_cooldown'] == 0:
            self._fire_player_projectile()
            self.player['fire_cooldown'] = self.player['max_fire_cooldown']

    def _fire_player_projectile(self):
        # sfx: player_shoot
        direction = 1 if self.player['facing_right'] else -1
        turret_offset_y = -self.player['h'] * 0.6
        proj_x = self.player['x'] + (self.player['w'] / 2 + 5) * direction
        proj_y = self.player['y'] + turret_offset_y

        self.player_projectiles.append({
            "x": proj_x, "y": proj_y,
            "vx": 15 * direction,
            "w": 10, "h": 4,
            "trail": []
        })
        # Muzzle flash
        self.particles.extend(self._create_particle_burst(proj_x, proj_y, self.COLOR_PLAYER_PROJ, 10, 2, 5))

    def _update_player(self):
        # Apply physics
        self.player['x'] += self.player['vx']
        if not self.player['on_ground']:
            self.player['vy'] += self.player['gravity']
            self.player['y'] += self.player['vy']

        # Ground collision
        player_h = self.player['h'] / 2 if self.player['is_ducking'] else self.player['h']
        if self.player['y'] > self.ground_y - player_h / 2:
            self.player['y'] = self.ground_y - player_h / 2
            self.player['vy'] = 0
            if not self.player['on_ground']:
                # sfx: land
                self.screen_shake = 5
            self.player['on_ground'] = True

        # Screen boundaries
        self.player['x'] = np.clip(self.player['x'], self.player['w'] / 2, self.WIDTH - self.player['w'] / 2)

    def _update_projectiles(self):
        # Player projectiles
        for p in self.player_projectiles[:]:
            p['trail'].append((p['x'], p['y']))
            if len(p['trail']) > 5: p['trail'].pop(0)
            p['x'] += p['vx']
            if p['x'] < 0 or p['x'] > self.WIDTH:
                self.player_projectiles.remove(p)

        # Enemy projectiles
        for p in self.enemy_projectiles[:]:
            p['trail'].append((p['x'], p['y']))
            if len(p['trail']) > 5: p['trail'].pop(0)
            p['x'] += p['vx']
            p['y'] += p['vy']
            if not (0 < p['x'] < self.WIDTH and 0 < p['y'] < self.HEIGHT):
                self.enemy_projectiles.remove(p)

    def _update_enemies(self):
        for e in self.enemies:
            # Movement
            target_x = self.player['x']
            if abs(e['x'] - target_x) > e['w']:
                e['x'] += 1 if target_x > e['x'] else -1

            # Firing
            e['fire_cooldown'] -= 1
            if e['fire_cooldown'] <= 0:
                self._fire_enemy_projectile(e)
                e['fire_cooldown'] = e['max_fire_cooldown'] + self.np_random.integers(-20, 20)

    def _fire_enemy_projectile(self, enemy):
        # sfx: enemy_shoot
        dx = self.player['x'] - enemy['x']
        dy = (self.player['y'] - self.player['h'] / 2) - enemy['y']
        dist = math.hypot(dx, dy)
        if dist == 0: return

        vx = (dx / dist) * self.enemy_projectile_speed
        vy = (dy / dist) * self.enemy_projectile_speed

        self.enemy_projectiles.append({
            "x": enemy['x'], "y": enemy['y'],
            "vx": vx, "vy": vy,
            "w": 8, "h": 8,
            "trail": []
        })

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if len(self.enemies) < 5 and self.enemy_spawn_timer <= 0:
            side = self.np_random.choice([-1, 1])
            x = -30 if side == 1 else self.WIDTH + 30
            y = self.np_random.integers(50, self.ground_y - 80)

            self.enemies.append({
                "x": x, "y": y,
                "w": 40, "h": 30,
                "health": 3, "max_health": 3,
                "fire_cooldown": self.np_random.integers(50, 100),
                "max_fire_cooldown": 100
            })
            self.enemy_spawn_timer = 100  # Cooldown between spawns

    def _handle_collisions(self):
        reward = 0

        # Player projectiles vs Enemies
        for p in self.player_projectiles[:]:
            proj_rect = pygame.Rect(p['x'] - p['w'] / 2, p['y'] - p['h'] / 2, p['w'], p['h'])
            for e in self.enemies[:]:
                enemy_rect = pygame.Rect(e['x'] - e['w'] / 2, e['y'] - e['h'] / 2, e['w'], e['h'])
                if proj_rect.colliderect(enemy_rect):
                    # sfx: enemy_hit
                    self.player_projectiles.remove(p)
                    e['health'] -= 1
                    reward += 0.1
                    self.particles.extend(self._create_particle_burst(p['x'], p['y'], self.COLOR_ENEMY, 15, 1, 3))

                    if e['health'] <= 0:
                        # sfx: enemy_explode
                        self.enemies.remove(e)
                        self.enemies_destroyed += 1
                        reward += 1
                        self.particles.extend(self._create_particle_burst(e['x'], e['y'], (255, 180, 50), 50, 2, 8))
                    break

        # Enemy projectiles vs Player
        player_h = self.player['h'] / 2 if self.player['is_ducking'] else self.player['h']
        player_rect = pygame.Rect(self.player['x'] - self.player['w'] / 2, self.player['y'] - player_h, self.player['w'],
                                  player_h)
        for p in self.enemy_projectiles[:]:
            proj_rect = pygame.Rect(p['x'] - p['w'] / 2, p['y'] - p['h'] / 2, p['w'], p['h'])
            if proj_rect.colliderect(player_rect):
                # sfx: player_hit
                self.enemy_projectiles.remove(p)
                self.player['health'] -= 10
                self.screen_shake = 10
                self.particles.extend(self._create_particle_burst(p['x'], p['y'], self.COLOR_PLAYER, 20, 1, 4))
                break

        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Apply screen shake
        render_offset_x, render_offset_y = 0, 0
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset_x = self.np_random.integers(-5, 6)
            render_offset_y = self.np_random.integers(-5, 6)

        # Clear screen with background
        self.screen.blit(self.background_surface, (render_offset_x, render_offset_y))

        # Render all game elements
        self._render_game(render_offset_x, render_offset_y)

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, ox, oy):
        # Draw ground
        ground_rect = pygame.Rect(0, self.ground_y, self.WIDTH, self.HEIGHT - self.ground_y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect.move(ox, oy))

        # Draw all game elements
        self._render_particles(ox, oy)
        self._render_projectiles(ox, oy)
        self._render_enemies(ox, oy)
        self._render_player(ox, oy)

    def _create_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        bg.fill(self.COLOR_BG)
        # Far buildings
        for _ in range(20):
            x = random.randint(0, self.WIDTH)
            w = random.randint(20, 50)
            h = random.randint(50, 150)
            y = self.ground_y - h
            color = tuple(c + random.randint(0, 10) for c in self.COLOR_GROUND)
            pygame.draw.rect(bg, color, (x, y, w, h))
        # Near buildings
        for _ in range(10):
            x = random.randint(0, self.WIDTH)
            w = random.randint(40, 80)
            h = random.randint(100, 250)
            y = self.ground_y - h
            color = tuple(c + random.randint(10, 20) for c in self.COLOR_GROUND)
            pygame.draw.rect(bg, color, (x, y, w, h))
        return bg

    def _render_player(self, ox, oy):
        p = self.player

        # Body animation
        player_h = p['h']
        if p['is_ducking']:
            player_h /= 2
        elif not p['on_ground']:
            # Stretch/squash when jumping/falling
            squash = 1 - abs(p['vy']) / (p['jump_power'] * -1.5)
            player_h *= max(0.8, squash)

        body_rect = pygame.Rect(p['x'] - p['w'] / 2, p['y'] - player_h, p['w'], player_h)

        # Turret
        turret_w, turret_h = p['w'] * 1.2, p['h'] * 0.25
        turret_y_offset = -player_h + turret_h / 2
        if p['fire_cooldown'] > p['max_fire_cooldown'] - 3:  # Recoil
            turret_x_offset = -5 if p['facing_right'] else 5
        else:
            turret_x_offset = 0

        turret_rect = pygame.Rect(p['x'] - turret_w / 2 + turret_x_offset,
                                  p['y'] + turret_y_offset - turret_h / 2, turret_w, turret_h)

        # Draw
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect.move(ox, oy), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_TURRET, turret_rect.move(ox, oy), border_radius=2)

        # Glow effect
        glow_surf = pygame.Surface((p['w'] + 20, player_h + 20), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER, 30), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, (body_rect.x - 10 + ox, body_rect.y - 10 + oy))

    def _render_enemies(self, ox, oy):
        for e in self.enemies:
            rect = pygame.Rect(e['x'] - e['w'] / 2, e['y'] - e['h'] / 2, e['w'], e['h'])
            pygame.draw.ellipse(self.screen, self.COLOR_ENEMY, rect.move(ox, oy))
            # "Wings"
            wing_y = e['y']
            pygame.draw.line(self.screen, self.COLOR_ENEMY, (e['x'] - e['w'] + ox, wing_y + oy),
                             (e['x'] + e['w'] + ox, wing_y + oy), 3)

            # Health bar
            health_pct = e['health'] / e['max_health']
            bar_w = e['w'] * health_pct
            bar_rect = pygame.Rect(e['x'] - e['w'] / 2, e['y'] - e['h'] / 2 - 8, bar_w, 4)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, bar_rect.move(ox, oy))

    def _render_projectiles(self, ox, oy):
        # Player projectiles
        for p in self.player_projectiles:
            # Trail
            if len(p['trail']) > 1:
                trail_points = [(int(tx + ox), int(ty + oy)) for tx, ty in p['trail']]
                pygame.draw.lines(self.screen, self.COLOR_PLAYER_PROJ, False, trail_points, width=int(p['h']))
            # Main projectile
            rect = pygame.Rect(p['x'] - p['w'] / 2, p['y'] - p['h'] / 2, p['w'], p['h'])
            pygame.draw.rect(self.screen, (255, 255, 255), rect.move(ox, oy), border_radius=2)

        # Enemy projectiles
        for p in self.enemy_projectiles:
            # Trail
            if len(p['trail']) > 1:
                trail_points = [(int(tx + ox), int(ty + oy)) for tx, ty in p['trail']]
                pygame.draw.aalines(self.screen, self.COLOR_ENEMY_PROJ, False, trail_points)
            # Main projectile
            pos = (int(p['x'] + ox), int(p['y'] + oy))
            pygame.gfxdraw.filled_circle(self.screen, *pos, int(p['w'] / 2), self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.filled_circle(self.screen, *pos, int(p['w'] / 4), (255, 255, 255))

    def _render_particles(self, ox, oy):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect())
            self.screen.blit(temp_surf, (p['x'] - p['size'] + ox, p['y'] - p['size'] + oy),
                             special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Player Health Bar
        health_pct = max(0, self.player.get('health', 0) / self.player.get('max_health', 1))
        bar_w = 200
        bar_h = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_w * health_pct, bar_h))
        health_text = self.font_small.render(f"HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Enemy Kill Count
        kill_text = self.font_large.render(f"KILLS: {self.enemies_destroyed} / {self.WIN_CONDITION}", True,
                                           self.COLOR_UI_TEXT)
        self.screen.blit(kill_text, (self.WIDTH / 2 - kill_text.get_width() / 2, self.HEIGHT - 35))

        # Game Over / Win Text
        if self.game_over:
            msg = "MISSION FAILED"
            if self.enemies_destroyed >= self.WIN_CONDITION:
                msg = "MISSION ACCOMPLISHED"

            end_text = self.font_large.render(msg, True, (255, 255, 100))
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player.get('health', 0),
            "enemies_destroyed": self.enemies_destroyed
        }

    def _create_particle_burst(self, x, y, color, count, min_speed, max_speed):
        particles = []
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            life = self.np_random.integers(10, 20)
            particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': life, 'max_life': life,
                'size': self.np_random.integers(2, 5),
                'color': color
            })
        return particles

    def close(self):
        pygame.quit()


# Example of how to run the environment for human play
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")

    # --- Manual Play Loop ---
    # This loop allows a human to play the game.
    # It will not run in a typical RL training setup.

    # Re-initialize pygame for display
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("Mech Annihilation")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    terminated = False
    truncated = False

    # Action buffer
    action = env.action_space.sample()
    action.fill(0)

    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Get key presses
        keys = pygame.key.get_pressed()

        # Map keys to MultiDiscrete action space
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0  # No-op

        # Space button
        action[1] = 1 if keys[pygame.K_SPACE] else 0

        # Shift button
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control frame rate
        env.clock.tick(env.FPS)

    print(f"Game Over. Final Score: {info['score']}, Steps: {info['steps']}")

    # Keep the window open for a bit to see the final screen
    pygame.time.wait(3000)

    env.close()