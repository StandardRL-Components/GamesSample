import os
import os
import pygame


# This must be set before pygame is imported to ensure the correct video driver is used.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Shift to dash. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mech through a hostile city. Destroy red targets for points while evading fire from enemy turrets. Destroy 15 targets to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 200
        self.WORLD_HEIGHT = 200

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        # For headless rendering, a display mode must be set for functions like
        # pygame.font.Font() or pygame.surfarray.array3d() to work.
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (34, 40, 49) # Dark blue-grey
        self.COLOR_BUILDING = (57, 62, 70)
        self.COLOR_PLAYER = (118, 247, 76)
        self.COLOR_PLAYER_PROJECTILE = (0, 255, 255)
        self.COLOR_TARGET = (214, 40, 40)
        self.COLOR_TURRET = (247, 127, 0)
        self.COLOR_ENEMY_PROJECTILE = (252, 191, 73)
        self.COLOR_TEXT = (238, 238, 238)
        self.COLOR_HEALTH_BAR = (0, 200, 0)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)

        # Game constants
        self.MAX_STEPS = 1000
        self.WIN_CONDITION = 15
        self.SCALE = 2.5 # Visual scale from world to screen
        self.PLAYER_FIRE_COOLDOWN_MAX = 10 # frames
        self.TURRET_HEALTH_MAX = 5
        self.TURRET_RESPAWN_TIME = 150 # 5 seconds
        self.INITIAL_TURRET_COUNT = 1
        self.MAX_TURRETS = 5

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_facing_dir = None
        self.player_fire_cooldown = None
        self.projectiles = []
        self.targets = []
        self.turrets = []
        self.particles = []
        self.buildings = []
        self.steps = 0
        self.score = 0
        self.targets_destroyed = 0
        self.turret_current_fire_rate = 0
        self.game_over = False
        self.game_won = False
        self.last_dist_to_target = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.targets_destroyed = 0
        self.game_over = False
        self.game_won = False

        self.player_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=np.float32)
        self.player_health = 100
        self.player_facing_dir = np.array([0, -1], dtype=np.float32) # Up
        self.player_fire_cooldown = 0

        self.projectiles.clear()
        self.targets.clear()
        self.turrets.clear()
        self.particles.clear()
        self.buildings.clear()

        self.turret_current_fire_rate = 60 # 2 seconds at 30fps

        self._generate_buildings()
        for _ in range(3): # Start with 3 targets
            self._spawn_target()

        for _ in range(self.INITIAL_TURRET_COUNT):
            self._spawn_turret()

        self.last_dist_to_target = self._get_dist_to_nearest_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Store distance to nearest target before moving
        dist_before_move = self._get_dist_to_nearest_target()

        # 1. Movement
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1 # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1 # Right

        if np.any(move_vec):
            speed = 3.0 if shift_held else 1.5
            self.player_pos += move_vec * speed
            self.player_facing_dir = move_vec
            # Clamp player position to world boundaries
            self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WORLD_WIDTH)
            self.player_pos[1] = np.clip(self.player_pos[1], 0, self.WORLD_HEIGHT)

        # Movement reward
        if dist_before_move is not None:
            dist_after_move = self._get_dist_to_nearest_target()
            if dist_after_move < dist_before_move:
                reward += 0.1 # Moved closer
            else:
                reward -= 0.02 # Moved away or stood still

        # 2. Firing
        self.player_fire_cooldown = max(0, self.player_fire_cooldown - 1)
        if space_held and self.player_fire_cooldown == 0:
            self._spawn_projectile(self.player_pos.copy(), self.player_facing_dir.copy(), 'player')
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX
            # Sound: Player fire

        # --- Game Logic Update ---
        self._update_projectiles()
        reward += self._update_turrets()
        self._update_particles()
        self._update_difficulty()

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        truncated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward -= 100 # Penalty for dying
        elif self.targets_destroyed >= self.WIN_CONDITION:
            terminated = True
            self.game_over = True
            self.game_won = True
            reward += 100 # Bonus for winning
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Can also be truncation
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Update Methods ---
    def _update_projectiles(self):
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            p['pos'] += p['vel'] * 4.0

            # Check world boundaries
            if not (0 < p['pos'][0] < self.WORLD_WIDTH and 0 < p['pos'][1] < self.WORLD_HEIGHT):
                projectiles_to_remove.append(i)
                continue

            # Check collisions
            if p['owner'] == 'player':
                # Player projectile vs Target
                targets_to_remove = []
                for j, target_pos in enumerate(self.targets):
                    if np.linalg.norm(p['pos'] - target_pos) < 5:
                        targets_to_remove.append(j)
                        self.score += 10
                        self.targets_destroyed += 1
                        self._create_explosion(target_pos, self.COLOR_TARGET)
                        if i not in projectiles_to_remove: projectiles_to_remove.append(i)
                        # Sound: Target destroyed
                        break
                for j in sorted(targets_to_remove, reverse=True):
                    del self.targets[j]
                    self._spawn_target()

                # Player projectile vs Turret
                for j, turret in enumerate(self.turrets):
                    if not turret['is_spawning'] and np.linalg.norm(p['pos'] - turret['pos']) < 6:
                        turret['health'] -= 1
                        self._create_explosion(turret['pos'], self.COLOR_TURRET, num_particles=5, size=2)
                        if i not in projectiles_to_remove: projectiles_to_remove.append(i)
                        break

            elif p['owner'] == 'enemy':
                # Enemy projectile vs Player
                if np.linalg.norm(p['pos'] - self.player_pos) < 5:
                    self.player_health -= 10 # Harder hits from turrets
                    self.score -= 1
                    self._create_explosion(self.player_pos, self.COLOR_PLAYER, num_particles=10, size=3)
                    if i not in projectiles_to_remove: projectiles_to_remove.append(i)
                    # Sound: Player hit

        for i in sorted(list(set(projectiles_to_remove)), reverse=True):
            del self.projectiles[i]

    def _update_turrets(self):
        reward = 0
        turrets_to_remove = []
        for i, turret in enumerate(self.turrets):
            if turret['is_spawning']:
                turret['spawn_timer'] -= 1
                if turret['spawn_timer'] <= 0:
                    turret['is_spawning'] = False
                continue

            if turret['health'] <= 0:
                turrets_to_remove.append(i)
                self._create_explosion(turret['pos'], self.COLOR_TURRET, num_particles=30, duration=40)
                reward += 5 # Smaller reward for destroying turrets
                self.score += 5
                # Sound: Turret destroyed
                continue

            turret['fire_cooldown'] -= 1
            if turret['fire_cooldown'] <= 0:
                direction = self.player_pos - turret['pos']
                dist = np.linalg.norm(direction)
                if dist > 0:
                    direction = direction / dist
                    self._spawn_projectile(turret['pos'].copy(), direction, 'enemy')
                    turret['fire_cooldown'] = self.turret_current_fire_rate + self.np_random.integers(-10, 10)
                    # Sound: Enemy fire

        for i in sorted(turrets_to_remove, reverse=True):
            del self.turrets[i]
            self._spawn_turret(is_respawn=True) # Respawn a new one

        return reward

    def _update_particles(self):
        particles_to_remove = []
        for i, particle in enumerate(self.particles):
            particle['pos'] += particle['vel']
            particle['life'] -= 1
            if particle['life'] <= 0:
                particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _update_difficulty(self):
        # Every 500 steps, increase turret fire rate
        if self.steps > 0 and self.steps % 500 == 0:
            self.turret_current_fire_rate = max(15, self.turret_current_fire_rate * 0.95) # 5% faster, min 0.5s
            if len(self.turrets) < self.MAX_TURRETS:
                self._spawn_turret()

    # --- Spawning Methods ---
    def _spawn_projectile(self, pos, vel, owner):
        projectile = {
            'pos': pos,
            'vel': vel,
            'owner': owner,
            'color': self.COLOR_PLAYER_PROJECTILE if owner == 'player' else self.COLOR_ENEMY_PROJECTILE
        }
        self.projectiles.append(projectile)

    def _spawn_target(self):
        # Spawn away from the player
        while True:
            pos = self.np_random.uniform(low=10, high=self.WORLD_WIDTH-10, size=2)
            if np.linalg.norm(pos - self.player_pos) > 40:
                self.targets.append(pos)
                break

    def _spawn_turret(self, is_respawn=False):
        while True:
            pos = self.np_random.uniform(low=10, high=self.WORLD_WIDTH-10, size=2)
            if np.linalg.norm(pos - self.player_pos) > 50:
                self.turrets.append({
                    'pos': pos,
                    'health': self.TURRET_HEALTH_MAX,
                    'fire_cooldown': self.turret_current_fire_rate,
                    'is_spawning': is_respawn,
                    'spawn_timer': self.TURRET_RESPAWN_TIME if is_respawn else 0
                })
                break

    def _generate_buildings(self):
        for _ in range(20):
            w = self.np_random.uniform(10, 30)
            h = self.np_random.uniform(10, 30)
            x = self.np_random.uniform(0, self.WORLD_WIDTH - w)
            y = self.np_random.uniform(0, self.WORLD_HEIGHT - h)
            height = self.np_random.uniform(10, 40)
            self.buildings.append({'rect': pygame.Rect(x, y, w, h), 'height': height})

    def _create_explosion(self, pos, color, num_particles=20, duration=20, size=2):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(duration // 2, duration),
                'color': color,
                'size': size
            })

    # --- Helper Methods ---
    def _get_dist_to_nearest_target(self):
        if not self.targets:
            return None
        distances = [np.linalg.norm(self.player_pos - t_pos) for t_pos in self.targets]
        return min(distances)

    def _iso_transform(self, x, y, z=0):
        iso_x = (x - y) * self.SCALE
        iso_y = (x + y) * 0.5 * self.SCALE - z * self.SCALE
        return iso_x, iso_y

    def _world_to_screen(self, pos, z=0):
        rel_x = pos[0] - self.player_pos[0]
        rel_y = pos[1] - self.player_pos[1]
        iso_x, iso_y = self._iso_transform(rel_x, rel_y, z)
        screen_x = self.SCREEN_WIDTH / 2 + iso_x
        screen_y = self.SCREEN_HEIGHT / 2.5 + iso_y # Push horizon up
        return int(screen_x), int(screen_y)

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Create render list and sort by depth (Painter's Algorithm) ---
        renderables = []

        # Targets (ground level)
        for t_pos in self.targets:
            renderables.append((t_pos[0] + t_pos[1], self._render_target, (t_pos,)))

        # Buildings
        for b in self.buildings:
            sort_key = (b['rect'].x + b['rect'].width / 2) + (b['rect'].y + b['rect'].height / 2)
            renderables.append((sort_key, self._render_building, (b,)))

        # Turrets
        for t in self.turrets:
            renderables.append((t['pos'][0] + t['pos'][1], self._render_turret, (t,)))

        # Player
        renderables.append((self.player_pos[0] + self.player_pos[1], self._render_player, ()))

        renderables.sort(key=lambda item: item[0])

        # --- Execute render calls ---
        for _, render_func, args in renderables:
            render_func(*args)

        # --- Render elements on top ---
        # Projectiles
        for p in self.projectiles:
            self._render_projectile(p)

        # Particles
        for part in self.particles:
            self._render_particle(part)

    def _render_building(self, building):
        rect = building['rect']
        height = building['height']

        points = [
            (rect.x, rect.y), (rect.right, rect.y),
            (rect.right, rect.bottom), (rect.x, rect.bottom)
        ]

        screen_points_base = [self._world_to_screen(p) for p in points]
        screen_points_top = [self._world_to_screen(p, height) for p in points]

        # Draw sides
        darker_color = tuple(int(c * 0.7) for c in self.COLOR_BUILDING)
        pygame.gfxdraw.filled_polygon(self.screen, [screen_points_base[0], screen_points_top[0], screen_points_top[1], screen_points_base[1]], darker_color)
        pygame.gfxdraw.filled_polygon(self.screen, [screen_points_base[1], screen_points_top[1], screen_points_top[2], screen_points_base[2]], darker_color)

        # Draw top
        pygame.gfxdraw.filled_polygon(self.screen, screen_points_top, self.COLOR_BUILDING)
        pygame.gfxdraw.aapolygon(self.screen, screen_points_top, self.COLOR_BUILDING)

    def _render_target(self, pos):
        sx, sy = self._world_to_screen(pos)
        radius = int(5 * self.SCALE)
        pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, self.COLOR_TARGET)
        pygame.gfxdraw.aacircle(self.screen, sx, sy, radius-2, self.COLOR_TARGET)
        pygame.draw.line(self.screen, self.COLOR_TARGET, (sx - radius, sy), (sx + radius, sy), 1)
        pygame.draw.line(self.screen, self.COLOR_TARGET, (sx, sy - radius), (sx, sy + radius), 1)

    def _render_turret(self, turret):
        pos, health = turret['pos'], turret['health']
        sx, sy = self._world_to_screen(pos, z=2)
        radius = int(4 * self.SCALE)

        if turret['is_spawning']:
            # Pulsing spawn effect
            alpha = int(128 + 127 * math.sin(self.steps * 0.3))
            temp_surface = self.screen.convert_alpha()
            temp_surface.fill((0,0,0,0))
            pygame.gfxdraw.filled_circle(temp_surface, sx, sy, radius, (*self.COLOR_TURRET, alpha))
            self.screen.blit(temp_surface, (0,0))
        else:
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, self.COLOR_TURRET)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, (0,0,0))

            # Health bar for turret
            health_percent = health / self.TURRET_HEALTH_MAX
            bar_w = int(radius * 2)
            bar_h = 4
            bar_x = sx - radius
            bar_y = sy - radius - 8
            pygame.draw.rect(self.screen, (100,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0,200,0), (bar_x, bar_y, int(bar_w * health_percent), bar_h))

    def _render_player(self):
        # Bobbing animation
        z_offset = 2 + math.sin(self.steps * 0.2)
        sx, sy = self._world_to_screen(self.player_pos, z=z_offset)

        # Simple rhombus shape for player
        size = int(5 * self.SCALE)
        points = [
            (sx, sy - size//2),
            (sx + size//2, sy),
            (sx, sy + size//2),
            (sx - size//2, sy)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, (255,255,255))

        # Aiming reticle
        aim_end = self.player_pos + self.player_facing_dir * 15
        ax, ay = self._world_to_screen(aim_end, z=z_offset)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (sx, sy), (ax, ay), 1)

    def _render_projectile(self, p):
        start_pos = self._world_to_screen(p['pos'], z=4)
        end_pos_world = p['pos'] - p['vel'] * 1.5
        end_pos = self._world_to_screen(end_pos_world, z=4)
        pygame.draw.line(self.screen, p['color'], start_pos, end_pos, 3)

    def _render_particle(self, particle):
        sx, sy = self._world_to_screen(particle['pos'], z=2)
        alpha = max(0, min(255, int(255 * (particle['life'] / 20.0))))
        radius = int(particle['size'] * (particle['life'] / 20.0))
        if radius > 0:
            temp_surface = self.screen.convert_alpha()
            temp_surface.fill((0,0,0,0))
            pygame.gfxdraw.filled_circle(temp_surface, sx, sy, max(0, radius), (*particle['color'], alpha))
            self.screen.blit(temp_surface, (0,0))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Targets Destroyed
        target_text = self.font_small.render(f"TARGETS: {self.targets_destroyed} / {self.WIN_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (10, 35))

        # Health Bar
        health_percent = max(0, self.player_health / 100)
        bar_w, bar_h = 200, 20
        bar_x, bar_y = self.SCREEN_WIDTH - bar_w - 10, 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_w * health_percent), bar_h))
        health_text = self.font_small.render("HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_x + bar_w/2 - health_text.get_width()/2, bar_y + bar_h/2 - health_text.get_height()/2))

        # Game Over / Win message
        if self.game_over:
            message = "MISSION FAILED"
            color = self.COLOR_TARGET
            if self.game_won:
                message = "MISSION ACCOMPLISHED"
                color = self.COLOR_PLAYER

            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))

            s = pygame.Surface((text_rect.width + 20, text_rect.height + 20), pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, (text_rect.x-10, text_rect.y-10))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "targets_destroyed": self.targets_destroyed
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly.
    # Note: Because SDL_VIDEODRIVER is set to "dummy" globally, this will
    # run without a window. To play interactively, you would need to
    # comment out the os.environ line at the top of the file.
    env = GameEnv()
    obs, info = env.reset()

    running = True
    total_reward = 0

    # This will create a dummy display surf, not a visible window.
    pygame.display.set_caption("Mech City Rampage")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    action = np.array([0, 0, 0]) # No-op, no fire, no dash

    while running:
        # --- Human Input Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Reset action
        action.fill(0)

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
            action[0] = 0

        # Fire
        if keys[pygame.K_SPACE]:
            action[1] = 1

        # Dash
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(30) # Limit to 30 FPS

    env.close()