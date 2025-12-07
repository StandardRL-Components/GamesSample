import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:29:47.517156
# Source Brief: brief_01097.md
# Brief Index: 1097
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Pilot a shrinking spaceship through vibrant nebulae, matching portal colors
    to gather resources and strategically resizing to survive cosmic storms and
    pirate attacks.
    """
    game_description = (
        "Pilot a shrinking spaceship through vibrant nebulae, matching portal colors to gather resources "
        "and strategically resizing to survive cosmic storms and pirate attacks."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Hold space to increase size, and hold shift to decrease size."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TARGET_RESOURCES = 1000
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_STORM = (180, 0, 255)
        self.COLOR_PIRATE = (100, 100, 110)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BAR = (40, 40, 60)
        self.PORTAL_COLORS = {
            "red": (255, 50, 50),
            "green": (50, 255, 50),
            "blue": (50, 50, 255)
        }

        # Player settings
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_ACCELERATION = 0.8
        self.PLAYER_MAX_SPEED = 8.0
        self.PLAYER_DRAG = 0.95
        self.PLAYER_MIN_SIZE = 8
        self.PLAYER_MAX_SIZE = 30
        self.PLAYER_RESIZE_SPEED = 0.5

        # Entity settings
        self.STORM_DAMAGE_THRESHOLD = 20 # Player size > this -> damage
        self.PIRATE_ATTACK_THRESHOLD = 15 # Player size < this -> attack
        self.STORM_DAMAGE = 0.5
        self.PIRATE_BULLET_DAMAGE = 10
        self.PIRATE_SPEED = 1.5
        self.PIRATE_SHOOT_COOLDOWN = 60 # steps
        self.BULLET_SPEED = 5

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.game_over = False

        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.player_health = None
        self.resources = None

        self.portals = None
        self.storms = None
        self.pirates = None
        self.bullets = None
        self.particles = None
        self.nebula_clouds = None

        self.storm_spawn_timer = None
        self.pirate_spawn_timer = None
        self.portal_spawn_timer = None

        self.difficulty_modifier = None
        self.screen_shake_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.game_over = False

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_size = (self.PLAYER_MIN_SIZE + self.PLAYER_MAX_SIZE) / 2
        self.player_health = self.PLAYER_MAX_HEALTH

        self.resources = {"red": 0, "green": 0, "blue": 0}

        self.portals = []
        self.storms = []
        self.pirates = []
        self.bullets = []
        self.particles = deque(maxlen=500) # Use deque for performance

        self.difficulty_modifier = 1.0
        self.storm_spawn_timer = 200
        self.pirate_spawn_timer = 300
        self.portal_spawn_timer = 0

        self._generate_nebula()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            is_truncated = self.steps >= self.MAX_STEPS
            return obs, 0, True, is_truncated, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Input & Update Player ---
        reward += self._handle_input_and_update_player(action)

        # --- 2. Update Game Entities ---
        self._update_spawners()
        reward += self._update_portals()
        reward += self._update_storms()
        reward += self._update_pirates()
        reward += self._update_bullets()
        self._update_particles()

        # --- 3. Check Termination Conditions ---
        terminated = (self.player_health <= 0 or
                      self.get_total_resources() >= self.TARGET_RESOURCES)
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            if self.player_health <= 0:
                reward -= 100 # Ship destruction penalty
            elif self.get_total_resources() >= self.TARGET_RESOURCES:
                reward += 100 # Victory bonus
            self.game_over = True
        
        if truncated:
            self.game_over = True

        if self.screen_shake_timer > 0:
            self.screen_shake_timer -= 1

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Update Logic Sub-functions ---

    def _handle_input_and_update_player(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        accel = pygame.math.Vector2(0, 0)
        if movement == 1: accel.y = -1 # Up
        elif movement == 2: accel.y = 1  # Down
        elif movement == 3: accel.x = -1 # Left
        elif movement == 4: accel.x = 1  # Right

        if accel.length() > 0:
            accel.scale_to_length(self.PLAYER_ACCELERATION)
            # Thruster particles
            if self.steps % 2 == 0:
                p_vel = -accel * random.uniform(2, 4) + (self.player_vel * 0.5)
                self._create_particle(self.player_pos, p_vel, 15, (255, 180, 50), 4)

        self.player_vel += accel
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)

        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_DRAG

        # --- Screen Wrap ---
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT

        # --- Resizing ---
        if space_held:
            self.player_size += self.PLAYER_RESIZE_SPEED
        if shift_held:
            self.player_size -= self.PLAYER_RESIZE_SPEED
        self.player_size = np.clip(self.player_size, self.PLAYER_MIN_SIZE, self.PLAYER_MAX_SIZE)

        return 0 # No intrinsic reward for moving/resizing

    def _update_spawners(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 10 == 0:
            self.difficulty_modifier += 0.001

        # Portals
        self.portal_spawn_timer -= 1
        if self.portal_spawn_timer <= 0 and len(self.portals) < 5:
            self._spawn_portal()
            self.portal_spawn_timer = random.randint(60, 120)

        # Storms
        self.storm_spawn_timer -= 1
        if self.storm_spawn_timer <= 0:
            self._spawn_storm()
            self.storm_spawn_timer = int(random.randint(400, 600) / self.difficulty_modifier)

        # Pirates
        self.pirate_spawn_timer -= 1
        if self.pirate_spawn_timer <= 0 and len(self.pirates) < 3:
            self._spawn_pirate()
            self.pirate_spawn_timer = int(random.randint(500, 700) / self.difficulty_modifier)

    def _update_portals(self):
        reward = 0
        for portal in self.portals[:]:
            dist = self.player_pos.distance_to(portal['pos'])
            if dist < self.player_size + portal['size']:
                # Collect resource
                self.resources[portal['type']] += 25
                reward += 1.0 # Event-based reward for collection
                self.portals.remove(portal)
                # sfx: portal_collect.wav
                for _ in range(30):
                    self._create_particle(self.player_pos, None, 25, portal['color'], random.uniform(1,4))
        return reward

    def _update_storms(self):
        reward = 0
        for storm in self.storms[:]:
            storm['lifespan'] -= 1
            if storm['lifespan'] <= 0:
                self.storms.remove(storm)
                continue

            storm['pos'] += storm['vel']
            storm['pos'].x %= self.WIDTH
            storm['pos'].y %= self.HEIGHT

            # Add particles to storm
            if self.steps % 3 == 0:
                p_pos = storm['pos'] + pygame.math.Vector2(random.uniform(-storm['size'], storm['size']), random.uniform(-storm['size'], storm['size']))
                self._create_particle(p_pos, None, 40, self.COLOR_STORM, random.uniform(1,3), is_storm=True)

            # Check for player damage
            dist = self.player_pos.distance_to(storm['pos'])
            if dist < self.player_size + storm['size'] and self.player_size > self.STORM_DAMAGE_THRESHOLD:
                self.player_health -= self.STORM_DAMAGE
                reward -= 0.1 # Continuous penalty for taking damage
                self.screen_shake_timer = 5
                if random.random() < 0.1: # Spark effect
                    # sfx: electricity_crackle.wav
                    p_vel = (storm['pos'] - self.player_pos).normalize() * 3
                    self._create_particle(self.player_pos, p_vel, 10, (255,255,255), 2)
        return reward

    def _update_pirates(self):
        reward = 0
        for pirate in self.pirates[:]:
            # Move towards player
            direction = (self.player_pos - pirate['pos'])
            if direction.length() > 0:
                direction.normalize_ip()
            pirate['pos'] += direction * self.PIRATE_SPEED

            # Screen wrap
            pirate['pos'].x %= self.WIDTH
            pirate['pos'].y %= self.HEIGHT

            # Shoot if player is small
            pirate['cooldown'] -= 1
            if self.player_size < self.PIRATE_ATTACK_THRESHOLD and pirate['cooldown'] <= 0:
                self._spawn_bullet(pirate['pos'], direction)
                pirate['cooldown'] = self.PIRATE_SHOOT_COOLDOWN
                # sfx: laser_shoot.wav
        return reward

    def _update_bullets(self):
        reward = 0
        for bullet in self.bullets[:]:
            bullet['pos'] += bullet['vel']

            # Check for collision with player
            if self.player_pos.distance_to(bullet['pos']) < self.player_size:
                self.player_health -= self.PIRATE_BULLET_DAMAGE
                reward -= 0.1 # Damage penalty
                self.bullets.remove(bullet)
                self.screen_shake_timer = 10
                # sfx: player_hit.wav
                for _ in range(20):
                    self._create_particle(self.player_pos, None, 15, (255, 100, 0), random.uniform(1,3))
                continue

            # Remove if off-screen (though world wraps, bullets shouldn't)
            if not (0 <= bullet['pos'].x < self.WIDTH and 0 <= bullet['pos'].y < self.HEIGHT):
                self.bullets.remove(bullet)
        return reward

    def _update_particles(self):
        for p in list(self.particles):
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                p['pos'] += p['vel']
                p['size'] = max(0, p['size'] - 0.1)

    # --- Spawning Logic ---

    def _get_spawn_pos(self):
        # Spawn away from player
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top': return pygame.math.Vector2(random.uniform(0, self.WIDTH), -50)
        if edge == 'bottom': return pygame.math.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + 50)
        if edge == 'left': return pygame.math.Vector2(-50, random.uniform(0, self.HEIGHT))
        if edge == 'right': return pygame.math.Vector2(self.WIDTH + 50, random.uniform(0, self.HEIGHT))

    def _spawn_portal(self):
        ptype = random.choice(list(self.PORTAL_COLORS.keys()))
        pos = pygame.math.Vector2(random.uniform(50, self.WIDTH-50), random.uniform(50, self.HEIGHT-50))
        self.portals.append({'pos': pos, 'type': ptype, 'color': self.PORTAL_COLORS[ptype], 'size': 12})

    def _spawn_storm(self):
        pos = self._get_spawn_pos()
        vel = (pygame.math.Vector2(self.WIDTH/2, self.HEIGHT/2) - pos).normalize() * random.uniform(0.5, 1.5)
        self.storms.append({
            'pos': pos, 'vel': vel, 'size': random.uniform(40, 70),
            'lifespan': random.randint(300, 500)
        })

    def _spawn_pirate(self):
        pos = self._get_spawn_pos()
        self.pirates.append({'pos': pos, 'cooldown': random.randint(0, self.PIRATE_SHOOT_COOLDOWN)})

    def _spawn_bullet(self, pos, direction):
        self.bullets.append({'pos': pos.copy(), 'vel': direction * self.BULLET_SPEED})

    def _create_particle(self, pos, vel, lifespan, color, size, is_storm=False):
        if vel is None:
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed

        if is_storm:
            # Storm particles swirl
            vel = vel.rotate(random.uniform(-90, 90))

        self.particles.append({
            'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan,
            'color': color, 'size': size
        })

    # --- State & Info ---

    def get_total_resources(self):
        return sum(self.resources.values())

    def _get_info(self):
        return {
            "score": self.get_total_resources(),
            "steps": self.steps,
            "health": self.player_health,
            "player_size": self.player_size,
            "difficulty": self.difficulty_modifier
        }

    # --- Rendering ---

    def _get_observation(self):
        # Determine screen offset for shake effect
        shake_offset = pygame.math.Vector2(0, 0)
        if self.screen_shake_timer > 0:
            shake_offset.x = random.randint(-4, 4)
            shake_offset.y = random.randint(-4, 4)

        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render game elements with potential shake
        self._render_background(shake_offset)
        self._render_particles(shake_offset)
        self._render_portals(shake_offset)
        self._render_storms(shake_offset)
        self._render_bullets(shake_offset)
        self._render_pirates(shake_offset)
        self._render_player(shake_offset)

        # Render UI on top, without shake
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, offset):
        for cloud in self.nebula_clouds:
            # Parallax effect
            pos = (cloud['pos'] + offset * cloud['depth']).copy()
            pos.x %= self.WIDTH
            pos.y %= self.HEIGHT

            # Use a pre-rendered surface for alpha blending
            s = pygame.Surface((cloud['size']*2, cloud['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, cloud['color'], (cloud['size'], cloud['size']), cloud['size'])
            self.screen.blit(s, (int(pos.x - cloud['size']), int(pos.y - cloud['size'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_particles(self, offset):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 20))
            # Clamp alpha to the valid [0, 255] range to prevent TypeError
            alpha = max(0, min(255, alpha))
            if alpha > 0:
                r, g, b = p['color']
                pos = p['pos'] + offset
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(p['size']), (r, g, b, alpha))

    def _render_portals(self, offset):
        for portal in self.portals:
            pos = portal['pos'] + offset
            size = portal['size']
            color = portal['color']
            # Pulsing effect
            pulse = (math.sin(self.steps * 0.1) + 1) / 2

            # Outer glow
            glow_size = int(size * (1.5 + pulse * 0.5))
            glow_color = (*color, 50)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), glow_size, glow_color)

            # Inner circle
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), size, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), size, color)

    def _render_storms(self, offset):
        # Storms are rendered via their particles in _render_particles
        pass

    def _render_bullets(self, offset):
        for bullet in self.bullets:
            pos = bullet['pos'] + offset
            end_pos = pos - bullet['vel'].normalize() * 8
            pygame.draw.line(self.screen, (255,100,100), (int(pos.x), int(pos.y)), (int(end_pos.x), int(end_pos.y)), 3)

    def _render_pirates(self, offset):
        for pirate in self.pirates:
            pos = pirate['pos'] + offset
            direction = (self.player_pos - pos)
            if direction.length() > 0:
                direction.normalize_ip()
            else: # Avoid division by zero if pirate is on top of player
                direction = pygame.math.Vector2(1, 0)
            angle = direction.angle_to(pygame.math.Vector2(1, 0))

            points = [
                pygame.math.Vector2(12, 0),
                pygame.math.Vector2(-8, 7),
                pygame.math.Vector2(-5, 0),
                pygame.math.Vector2(-8, -7),
            ]

            rotated_points = [p.rotate(-angle) + pos for p in points]
            int_points = [(int(p.x), int(p.y)) for p in rotated_points]

            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PIRATE)
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PIRATE)

    def _render_player(self, offset):
        pos = self.player_pos + offset
        size = self.player_size

        # Glow effect
        for i in range(4, 0, -1):
            alpha = 60 - i * 15
            glow_size = int(size * (1 + i * 0.2))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), glow_size, (*self.COLOR_PLAYER, alpha))

        # Main ship body
        angle = self.player_vel.angle_to(pygame.math.Vector2(1,0)) if self.player_vel.length() > 0 else 0
        points = [
            pygame.math.Vector2(size, 0),
            pygame.math.Vector2(-size * 0.5, size * 0.7),
            pygame.math.Vector2(-size * 0.2, 0),
            pygame.math.Vector2(-size * 0.5, -size * 0.7),
        ]
        rotated_points = [p.rotate(-angle) + pos for p in points]
        int_points = [(int(p.x), int(p.y)) for p in rotated_points]
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # --- Health Bar ---
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        health_color = (int(255 * (1 - health_pct)), int(255 * health_pct), 0)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, 10, 150, 20))
        if health_pct > 0:
            pygame.draw.rect(self.screen, health_color, (10, 10, 150 * health_pct, 20))

        # --- Size Bar ---
        size_pct = (self.player_size - self.PLAYER_MIN_SIZE) / (self.PLAYER_MAX_SIZE - self.PLAYER_MIN_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (170, 10, 100, 20))
        pygame.draw.rect(self.screen, (200, 200, 255), (170, 10, 100 * size_pct, 20))
        # Add markers for thresholds
        storm_thresh_x = 170 + 100 * (self.STORM_DAMAGE_THRESHOLD - self.PLAYER_MIN_SIZE) / (self.PLAYER_MAX_SIZE - self.PLAYER_MIN_SIZE)
        pirate_thresh_x = 170 + 100 * (self.PIRATE_ATTACK_THRESHOLD - self.PLAYER_MIN_SIZE) / (self.PLAYER_MAX_SIZE - self.PLAYER_MIN_SIZE)
        pygame.draw.line(self.screen, self.COLOR_STORM, (storm_thresh_x, 8), (storm_thresh_x, 32), 2)
        pygame.draw.line(self.screen, self.COLOR_PIRATE, (pirate_thresh_x, 8), (pirate_thresh_x, 32), 2)

        # --- Resource Bar ---
        total_res = self.get_total_resources()
        bar_width = self.WIDTH - 290
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (280, 10, bar_width, 20))

        current_x = 280
        if total_res > 0:
            for r_type, r_val in self.resources.items():
                r_pct_total = r_val / self.TARGET_RESOURCES
                r_width = bar_width * r_pct_total
                pygame.draw.rect(self.screen, self.PORTAL_COLORS[r_type], (current_x, 10, r_width, 20))
                current_x += r_width

        res_text = self.font_ui.render(f"{total_res}/{self.TARGET_RESOURCES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (285, 12))

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))

        message = "MISSION COMPLETE" if self.get_total_resources() >= self.TARGET_RESOURCES else "SHIP DESTROYED"
        color = (100, 255, 100) if self.get_total_resources() >= self.TARGET_RESOURCES else (255, 100, 100)

        text = self.font_game_over.render(message, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        s.blit(text, text_rect)
        self.screen.blit(s, (0,0))

    def _generate_nebula(self):
        self.nebula_clouds = []
        for _ in range(50):
            depth = random.uniform(0.1, 0.6)
            self.nebula_clouds.append({
                'pos': pygame.math.Vector2(random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)),
                'size': random.randint(50, 200),
                'color': (random.randint(15, 40), random.randint(10, 25), random.randint(20, 50), random.randint(50, 100)),
                'depth': depth
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # To run interactively, you might need to comment out the os.environ line at the top
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Nebula Drifter")
        run_interactive = True
    except pygame.error:
        print("pygame.display.set_mode failed. Running in headless mode. No interactive controls.")
        run_interactive = False

    clock = pygame.time.Clock()

    done = False
    total_reward = 0

    print("\n--- Controls ---")
    print("Arrows: Move")
    print("Space: Increase Size")
    print("Shift: Decrease Size")
    print("R: Reset")
    print("Q: Quit")

    while not done:
        action = env.action_space.sample() # Default to random action
        if run_interactive:
            # --- Human Input ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]

            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        done = True
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print("--- Game Reset ---")

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if run_interactive:
            # --- Rendering ---
            # The observation is already the rendered screen
            # We just need to convert it back to a Pygame surface to display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(env.metadata["render_fps"])

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            if not run_interactive:
                # In headless test mode, auto-reset
                obs, info = env.reset()
                total_reward = 0
            else:
                # In interactive mode, wait for user to reset or quit
                pass

    env.close()