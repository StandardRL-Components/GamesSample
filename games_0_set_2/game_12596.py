import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:43:57.344200
# Source Brief: brief_02596.md
# Brief Index: 2596
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Navigate a treacherous asteroid field as a space pirate. Evade patrols and use your "
        "cloaking device by matching asteroid colors to reach the target sector."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move left and right. Press space to activate your cloak "
        "when near a group of same-colored asteroids to become temporarily invincible."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TARGET_SECTOR_X = 5000
        self.MAX_STEPS = 2000
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_ASTEROID_RED = (255, 80, 80)
        self.COLOR_ASTEROID_GREEN = (80, 255, 80)
        self.COLOR_ASTEROID_BLUE = (80, 80, 255)
        self.ASTEROID_COLORS = [self.COLOR_ASTEROID_RED, self.COLOR_ASTEROID_GREEN, self.COLOR_ASTEROID_BLUE]
        self.COLOR_PATROL_LIGHT = (255, 255, 200)
        self.COLOR_TARGET = (255, 220, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_CLOAK_BAR = (100, 100, 255)
        
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_speed = 8
        self.world_x = 0.0
        self.asteroids = []
        self.patrols = []
        self.particles = []
        self.stars = []
        self.cloak_timer = 0
        self.space_pressed_last_frame = False
        self.last_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.world_x = 0.0
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.cloak_timer = 0
        self.last_reward = 0
        self.space_pressed_last_frame = False

        self.asteroids = []
        self.patrols = []
        self.particles = []
        self.stars = [{'pos': pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)), 
                       'depth': self.np_random.uniform(0.1, 0.6)} for _ in range(150)]
        
        self._initial_spawn()
        
        return self._get_observation(), self._get_info()

    def _initial_spawn(self):
        # Spawn initial asteroids and patrols across the starting area
        for _ in range(20):
            self._spawn_asteroid(random_x=True)
        for _ in range(3):
            self._spawn_patrol(random_x=True)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), self.last_reward, True, False, self._get_info()

        reward = 0.1  # Survival reward
        
        self._handle_input(action)
        self._update_world()
        
        # Cloak activation logic
        space_held = action[1] == 1
        if space_held and not self.space_pressed_last_frame:
            cloak_reward, combo_size, matched_color = self._try_cloak()
            if combo_size > 0:
                reward += cloak_reward
                # SFX: Cloak activate sound
                self._create_particle_burst(self.player_pos, matched_color, 30 + combo_size * 10)
        self.space_pressed_last_frame = space_held

        # Check for game end conditions
        terminated, terminal_reward = self._check_termination_conditions()
        if terminated:
            self.game_over = True
            reward = terminal_reward
        
        self.score += reward
        self.steps += 1
        self.last_reward = reward

        truncated = False
        if self.steps >= self.MAX_STEPS and not self.game_over:
            terminated = True
            truncated = True
            self.game_over = True
            # SFX: Time up failure sound

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 3:  # Left
            self.world_x -= self.player_speed
        elif movement == 4:  # Right
            self.world_x += self.player_speed
        
        self.world_x = max(0, self.world_x)

    def _update_world(self):
        # Update cloak
        self.cloak_timer = max(0, self.cloak_timer - 1)
        
        # Update asteroids
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
        self.asteroids = [a for a in self.asteroids if a['pos'].y < self.HEIGHT + 50 and a['pos'].y > -50]
        
        # Difficulty scaling
        if self.steps % 500 == 0 and self.steps > 0:
            self._spawn_asteroid() # Increase density
        if self.steps % 250 == 0 and self.steps > 0:
            for p in self.patrols:
                p['speed'] += 0.001 # Increase patrol speed

        # Respawn asteroids
        if len(self.asteroids) < 20 + (self.steps // 500):
            self._spawn_asteroid()

        # Update patrols
        for patrol in self.patrols:
            patrol['angle'] += patrol['dir'] * patrol['speed']
            if not patrol['sweep_min'] < patrol['angle'] < patrol['sweep_max']:
                patrol['dir'] *= -1
        
        # Update particles
        for particle in self.particles:
            particle['pos'] += particle['vel']
            particle['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _try_cloak(self):
        CLOAK_RADIUS = 60
        MIN_COMBO = 2
        
        nearby_asteroids = [a for a in self.asteroids if self.player_pos.distance_to(self._world_to_screen(a['pos'])) < CLOAK_RADIUS]
        
        color_counts = {i: [] for i in range(len(self.ASTEROID_COLORS))}
        for asteroid in nearby_asteroids:
            color_counts[asteroid['color_idx']].append(asteroid)
        
        best_color_idx = -1
        max_count = 0
        for i, group in color_counts.items():
            if len(group) > max_count:
                max_count = len(group)
                best_color_idx = i

        if max_count >= MIN_COMBO:
            matched_asteroids = color_counts[best_color_idx]
            self.asteroids = [a for a in self.asteroids if a not in matched_asteroids]
            
            combo_size = len(matched_asteroids)
            self.cloak_timer += 60 + (combo_size - MIN_COMBO) * 30 # 2s base + 1s per extra
            
            reward = 1.0 + 5.0 * (combo_size - MIN_COMBO)
            return reward, combo_size, self.ASTEROID_COLORS[best_color_idx]
        
        return 0, 0, None

    def _check_termination_conditions(self):
        # Reached target
        if self.world_x >= self.TARGET_SECTOR_X:
            # SFX: Victory fanfare
            return True, 100.0

        is_cloaked = self.cloak_timer > 0
        if not is_cloaked:
            # Collision with asteroid
            for asteroid in self.asteroids:
                screen_pos = self._world_to_screen(asteroid['pos'])
                if self.player_pos.distance_to(screen_pos) < asteroid['radius'] + 10:
                    # SFX: Explosion sound
                    self._create_particle_burst(self.player_pos, self.COLOR_PLAYER, 100)
                    return True, -100.0

            # Detected by patrol
            for patrol in self.patrols:
                if self._is_point_in_cone(self.player_pos, self._world_to_screen(patrol['pos']), patrol['angle'], patrol['radius'], patrol['fov']):
                    # SFX: Detection alarm
                    self._create_particle_burst(self.player_pos, self.COLOR_PATROL_LIGHT, 50)
                    return True, -100.0

        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_target()
        self._render_asteroids()
        self._render_patrols()
        self._render_player()
        self._render_particles()

    def _render_background(self):
        for star in self.stars:
            screen_x = (star['pos'].x - self.world_x * star['depth']) % self.WIDTH
            screen_y = star['pos'].y
            brightness = int(100 * star['depth'])
            color = (brightness, brightness, brightness + 50)
            self.screen.set_at((int(screen_x), int(screen_y)), color)

    def _render_target(self):
        target_screen_x = self.TARGET_SECTOR_X - self.world_x + self.WIDTH / 2
        if 0 < target_screen_x < self.WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_TARGET, (target_screen_x, 0, 100, self.HEIGHT), 2)
            for i in range(10):
                alpha = 100 - i * 10
                color = (*self.COLOR_TARGET, alpha)
                s = pygame.Surface((100, self.HEIGHT), pygame.SRCALPHA)
                pygame.draw.rect(s, color, (0, 0, 100, self.HEIGHT))
                self.screen.blit(s, (target_screen_x - i*10, 0))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            screen_pos = self._world_to_screen(asteroid['pos'])
            if -50 < screen_pos.x < self.WIDTH + 50 and -50 < screen_pos.y < self.HEIGHT + 50:
                color = self.ASTEROID_COLORS[asteroid['color_idx']]
                self._draw_glow_circle(screen_pos, asteroid['radius'], color)

    def _render_patrols(self):
        for patrol in self.patrols:
            screen_pos = self._world_to_screen(patrol['pos'])
            if -patrol['radius'] < screen_pos.x < self.WIDTH + patrol['radius']:
                self._draw_cone(screen_pos, patrol['angle'], patrol['radius'], patrol['fov'], self.COLOR_PATROL_LIGHT)

    def _render_player(self):
        is_cloaked = self.cloak_timer > 0
        alpha = 100 if is_cloaked else 255
        
        # Player ship shape
        p1 = self.player_pos + pygame.Vector2(0, -15)
        p2 = self.player_pos + pygame.Vector2(-10, 10)
        p3 = self.player_pos + pygame.Vector2(10, 10)
        points = [p1, p2, p3]

        # Glow effect
        glow_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
        glow_points = [(p.x - self.player_pos.x + 20, p.y - self.player_pos.y + 20) for p in points]
        pygame.draw.polygon(glow_surface, (*self.COLOR_PLAYER_GLOW, alpha // 4), glow_points)
        glow_surface = pygame.transform.smoothscale(glow_surface, (80, 80))
        self.screen.blit(glow_surface, self.player_pos - pygame.Vector2(40, 40), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship body
        pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], (*self.COLOR_PLAYER, alpha))
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], (*self.COLOR_PLAYER, alpha))

        # Cloak duration bar
        if is_cloaked:
            bar_width = 60
            bar_height = 8
            bar_x = self.player_pos.x - bar_width / 2
            bar_y = self.player_pos.y + 20
            fill_ratio = self.cloak_timer / (60 + (5-2)*30) # Max possible duration approx
            pygame.draw.rect(self.screen, (50,50,100), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_CLOAK_BAR, (bar_x, bar_y, bar_width * fill_ratio, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

    def _render_particles(self):
        for p in self.particles:
            screen_pos = self._world_to_screen(p['pos'])
            size = max(1, p['size'] * (p['life'] / p['max_life']))
            color = (*p['color'], int(255 * (p['life'] / p['max_life'])))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (screen_pos.x - size, screen_pos.y - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        progress = self.world_x / self.TARGET_SECTOR_X
        progress_text = self.font_ui.render(f"PROGRESS: {int(progress*100)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(progress_text, (self.WIDTH - progress_text.get_width() - 10, 10))

        if self.game_over:
            if self.world_x >= self.TARGET_SECTOR_X:
                end_text = self.font_big.render("TARGET REACHED", True, self.COLOR_TARGET)
            else:
                end_text = self.font_big.render("MISSION FAILED", True, self.COLOR_ASTEROID_RED)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "world_x": self.world_x,
            "cloak_timer": self.cloak_timer,
        }

    def _world_to_screen(self, world_pos):
        return pygame.Vector2(world_pos.x - self.world_x + self.WIDTH / 2, world_pos.y)

    def _spawn_asteroid(self, random_x=False):
        if random_x:
            x = self.np_random.uniform(-self.WIDTH/2, self.WIDTH/2)
        else:
            x = self.np_random.choice([-self.WIDTH/2 - 50, self.WIDTH/2 + 50])
        
        world_pos_x = self.world_x + x
        world_pos_y = self.np_random.uniform(-50, -20)
        
        vel_x = self.np_random.uniform(-0.5, 0.5)
        vel_y = self.np_random.uniform(1, 3)
        
        self.asteroids.append({
            'pos': pygame.Vector2(world_pos_x, world_pos_y),
            'vel': pygame.Vector2(vel_x, vel_y),
            'radius': self.np_random.uniform(10, 25),
            'color_idx': self.np_random.integers(0, len(self.ASTEROID_COLORS))
        })
        
    def _spawn_patrol(self, random_x=False):
        if random_x:
            x = self.np_random.uniform(0, self.WIDTH)
        else:
            x = self.world_x + self.np_random.choice([-self.WIDTH/2, self.WIDTH/2])
        
        y = self.np_random.uniform(50, self.HEIGHT - 150)
        
        self.patrols.append({
            'pos': pygame.Vector2(x, y),
            'angle': self.np_random.uniform(0, 2 * math.pi),
            'speed': self.np_random.uniform(0.01, 0.02),
            'dir': 1,
            'radius': self.np_random.uniform(150, 250),
            'fov': self.np_random.uniform(math.pi / 6, math.pi / 4), # 30-45 degrees
            'sweep_min': self.np_random.uniform(math.pi * 1.25, math.pi * 1.5),
            'sweep_max': self.np_random.uniform(math.pi * 1.75, math.pi * 2.0)
        })

    def _create_particle_burst(self, pos, color, count):
        # SFX: Particle burst sound
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': self._world_to_screen(pos).copy() if isinstance(pos, pygame.Vector2) and pos.y < self.HEIGHT else pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(15, 40),
                'max_life': 40,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _draw_glow_circle(self, pos, radius, color):
        glow_color = (*[min(255, c + 50) for c in color], 50)
        s = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color, (radius*2, radius*2), radius*1.5)
        s = pygame.transform.smoothscale(s, (int(radius*3), int(radius*3)))
        self.screen.blit(s, (pos.x - radius*1.5, pos.y - radius*1.5), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)

    def _draw_cone(self, center, angle, radius, fov, color):
        points = [center]
        num_segments = 20
        for i in range(num_segments + 1):
            theta = angle - fov / 2 + (fov * i / num_segments)
            points.append(center + pygame.Vector2(math.cos(theta), math.sin(theta)) * radius)
        
        if len(points) > 2:
            pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], (*color, 50))
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], (*color, 50))

    def _is_point_in_cone(self, point, cone_center, cone_angle, cone_radius, cone_fov):
        dist = point.distance_to(cone_center)
        if dist > cone_radius:
            return False
        
        angle_to_point = math.atan2(point.y - cone_center.y, point.x - cone_center.x)
        
        # Normalize angles to be in [0, 2*pi]
        cone_angle = cone_angle % (2 * math.pi)
        angle_to_point = angle_to_point % (2 * math.pi)

        # Check if angle_to_point is within the cone's fov
        min_angle = (cone_angle - cone_fov / 2) % (2 * math.pi)
        max_angle = (cone_angle + cone_fov / 2) % (2 * math.pi)
        
        if min_angle > max_angle: # Wraps around 0
            return angle_to_point >= min_angle or angle_to_point <= max_angle
        else:
            return min_angle <= angle_to_point <= max_angle

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and is not part of the environment's core logic.
    # It will re-enable the display for human interaction.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    import pygame # Re-import to use the new video driver setting

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Pirate Cloaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # No-op
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_r]:
            obs, info = env.reset()
            total_reward = 0
            done = False
            
        if not done:
            action = [movement_action, space_action, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.metadata['render_fps'])
        
    env.close()