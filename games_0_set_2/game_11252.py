import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:40:52.055289
# Source Brief: brief_01252.md
# Brief Index: 1252
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a futuristic cityscape in this cyberpunk racer. Dodge obstacles and police, "
        "collect energy, and use your nitro boost on the beat to maximize your score."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to control speed and ←→ to steer. "
        "Press space to activate your nitro boost."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30  # Assumed FPS for smooth interpolation

    # Colors (Synthwave/Cyberpunk Palette)
    COLOR_BG = (15, 10, 40)
    COLOR_GRID = (50, 40, 90)
    COLOR_BUILDING = (30, 20, 60)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_POLICE = (255, 50, 50)
    COLOR_POLICE_GLOW = (255, 0, 100)
    COLOR_ENERGY = (50, 255, 50)
    COLOR_ENERGY_GLOW = (100, 255, 100)
    COLOR_OBSTACLE = (100, 100, 120)
    COLOR_NITRO = (255, 0, 180)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_METER = (40, 30, 80)
    COLOR_BEAT_GOOD = (0, 255, 0)
    COLOR_BEAT_BAD = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- GYM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_score = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- INTERNAL STATE ---
        # These will be properly initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        self.player_pos = [0, 0]
        self.player_vel_x = 0
        self.world_scroll_speed = 0
        self.nitro_energy = 0
        self.is_nitro_active = False

        self.beat_timer = 0
        self.beat_period = 45 # 80 BPM @ 30 FPS
        self.on_beat_window = 4

        self.obstacles = []
        self.police = []
        self.pickups = []
        self.particles = []
        self.road_lines = []
        self.buildings = []
        
        # self.reset() is called by the wrapper/runner, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- PLAYER STATE ---
        self.player_pos = [self.WIDTH / 2, self.HEIGHT * 0.8]
        self.player_vel_x = 0
        self.world_scroll_speed = 4.0

        # --- GAME STATE ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.nitro_energy = 100.0

        # --- RHYTHM MECHANIC ---
        self.beat_timer = 0
        
        # --- ENTITY LISTS ---
        self.obstacles = []
        self.police = []
        self.pickups = []
        self.particles = []
        self.road_lines = []
        self.buildings = []

        # --- POPULATE WORLD ---
        for i in range(15):
            self.road_lines.append([random.uniform(0.1, 0.9) * self.WIDTH, i * self.HEIGHT / 10])
        for _ in range(10):
            self.buildings.append(self._create_building(random.uniform(0, self.HEIGHT)))
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.is_nitro_active = False

        if not self.game_over:
            # Unpack action
            movement = action[0]
            space_held = action[1] == 1
            # shift_held is unused for now

            # Update logic
            self._handle_input(movement, space_held)
            self._update_player()
            self._update_world()
            self._spawn_entities()
            self._check_collisions()
            
            # Survival reward
            self.reward_this_step += 0.1

        # Update counters
        self.steps += 1
        self.beat_timer = (self.beat_timer + 1) % self.beat_period
        self.score += self.reward_this_step
        
        # Check termination conditions
        terminated = self.game_over
        truncated = self.steps >= 2000
        if truncated and not self.game_over: # Max steps reached
             # No penalty for timeout
             pass

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # --- MOVEMENT ---
        steer_speed = 6.0
        if movement == 3: # Left
            self.player_vel_x = -steer_speed
        elif movement == 4: # Right
            self.player_vel_x = steer_speed
        else:
            self.player_vel_x = 0

        # --- SPEED CONTROL ---
        base_speed = 4.0
        accel_speed = 6.0
        brake_speed = 2.0
        
        if movement == 1: # Up (Accelerate)
            self.world_scroll_speed = accel_speed
        elif movement == 2: # Down (Brake)
            self.world_scroll_speed = brake_speed
        else:
            self.world_scroll_speed = base_speed

        # --- NITRO ---
        is_on_beat = abs(self.beat_timer - self.beat_period // 2) <= self.on_beat_window
        if space_held and self.nitro_energy > 0.1:
            # Sfx: Nitro boost
            self.is_nitro_active = True
            self.world_scroll_speed += 10.0
            self.nitro_energy = max(0, self.nitro_energy - 1.5)
            if is_on_beat:
                self.reward_this_step += 1.0
                # Sfx: Perfect beat!
            else:
                self.reward_this_step += 0.2

    def _update_player(self):
        self.player_pos[0] += self.player_vel_x
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        
        # Engine trail particles
        if self.steps % 2 == 0:
            color = self.COLOR_NITRO if self.is_nitro_active else self.COLOR_PLAYER_GLOW
            self._create_particles(
                (self.player_pos[0], self.player_pos[1] + 15),
                2, color, 3, 0.8, (0, -0.5)
            )

    def _update_world(self):
        # Update all entities and effects based on scroll speed
        for entity_list in [self.obstacles, self.police, self.pickups, self.road_lines]:
            for entity in entity_list:
                entity[1] += self.world_scroll_speed
        
        for building in self.buildings:
            building['y'] += self.world_scroll_speed * 0.5 # Parallax
        
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1] + self.world_scroll_speed
            p['life'] -= 1

        # Police AI
        police_speed_multiplier = min(2.0, 1.0 + 0.05 * (self.steps // 200))
        for p in self.police:
            dist_to_player = abs(p[0] - self.player_pos[0])
            if dist_to_player < 150: # Pursuit range
                p[0] += np.sign(self.player_pos[0] - p[0]) * 2.0 * police_speed_multiplier
            p[0] = np.clip(p[0], 20, self.WIDTH - 20)

        # Filter out old entities
        self.obstacles = [o for o in self.obstacles if o[1] < self.HEIGHT + 20]
        self.police = [p for p in self.police if p[1] < self.HEIGHT + 50]
        self.pickups = [p for p in self.pickups if p[1] < self.HEIGHT + 20]
        self.particles = [p for p in self.particles if p['life'] > 0]
        self.road_lines = [r for r in self.road_lines if r[1] < self.HEIGHT + 10]
        self.buildings = [b for b in self.buildings if b['y'] < self.HEIGHT + b['h']]

    def _spawn_entities(self):
        # Spawn road lines
        if len(self.road_lines) < 15:
            self.road_lines.append([random.uniform(0.1, 0.9) * self.WIDTH, -10])
        
        # Spawn buildings
        if len(self.buildings) < 10:
            self.buildings.append(self._create_building(-100))

        # Spawn obstacles, police, pickups
        if self.np_random.random() < 0.02:
            x = self.np_random.uniform(50, self.WIDTH - 50)
            self.obstacles.append([x, -20])
        
        if self.np_random.random() < 0.005 and len(self.police) < 3:
            x = self.np_random.choice([self.WIDTH * 0.25, self.WIDTH * 0.75])
            self.police.append([x, -50])
            
        if self.np_random.random() < 0.01:
            x = self.np_random.uniform(50, self.WIDTH - 50)
            self.pickups.append([x, -20])

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 15, 20, 30)

        # Player vs Obstacles
        for o in self.obstacles:
            if player_rect.colliderect(pygame.Rect(o[0]-15, o[1]-15, 30, 30)):
                self.game_over = True
                self.reward_this_step -= 100
                self._create_particles(self.player_pos, 50, self.COLOR_OBSTACLE, 8, 3.0)
                # Sfx: Crash
                return

        # Player vs Police
        for p in self.police:
            if player_rect.colliderect(pygame.Rect(p[0]-12, p[1]-20, 24, 40)):
                self.game_over = True
                self.reward_this_step -= 100
                self._create_particles(self.player_pos, 50, self.COLOR_POLICE, 8, 3.0)
                # Sfx: Busted
                return

        # Player vs Pickups
        for i, pk in enumerate(self.pickups):
            if player_rect.colliderect(pygame.Rect(pk[0]-10, pk[1]-10, 20, 20)):
                self.nitro_energy = min(100.0, self.nitro_energy + 25)
                self.reward_this_step += 5.0
                self._create_particles(pk, 20, self.COLOR_ENERGY, 5, 2.0)
                self.pickups.pop(i)
                # Sfx: Pickup
                break

    def _get_observation(self):
        # --- RENDER ---
        # Background
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        # Game elements
        self._render_entities()
        if not self.game_over:
            self._render_player()
        self._render_particles()

        # UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "nitro": self.nitro_energy,
        }

    # --- RENDERING HELPERS ---
    
    def _render_background(self):
        # Distant buildings
        for b in self.buildings:
            pygame.draw.rect(self.screen, self.COLOR_BUILDING, (b['x'], b['y'], b['w'], b['h']))

        # Scrolling grid lines
        for r in self.road_lines:
            pygame.draw.rect(self.screen, self.COLOR_GRID, (r[0] - 1, r[1], 2, 20))

    def _render_entities(self):
        # Obstacles
        for o in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (o[0]-15, o[1]-15, 30, 30))

        # Pickups
        for pk in self.pickups:
            self._draw_glow_circle(int(pk[0]), int(pk[1]), 10, self.COLOR_ENERGY, self.COLOR_ENERGY_GLOW)

        # Police
        for p in self.police:
            self._draw_vehicle(p, self.COLOR_POLICE, self.COLOR_POLICE_GLOW, is_police=True)

    def _render_player(self):
        self._draw_vehicle(self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), (*color, alpha))
    
    def _render_ui(self):
        # Score
        score_text = self.font_score.render(f"{int(self.score):05d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 10))

        # Nitro Gauge
        bar_x, bar_y, bar_w, bar_h = 20, 20, 200, 20
        pygame.draw.rect(self.screen, self.COLOR_UI_METER, (bar_x, bar_y, bar_w, bar_h))
        fill_w = (self.nitro_energy / 100.0) * bar_w
        pygame.draw.rect(self.screen, self.COLOR_NITRO, (bar_x, bar_y, fill_w, bar_h))
        nitro_text = self.font_ui.render("NITRO", True, self.COLOR_UI_TEXT)
        self.screen.blit(nitro_text, (bar_x + 5, bar_y))
        
        # Rhythm Meter
        beat_x, beat_y, beat_w, beat_h = 20, 50, 200, 10
        pygame.draw.rect(self.screen, self.COLOR_UI_METER, (beat_x, beat_y, beat_w, beat_h))
        
        # Beat indicator
        progress = self.beat_timer / self.beat_period
        indicator_x = beat_x + progress * beat_w
        pygame.draw.rect(self.screen, (255,255,255), (indicator_x - 1, beat_y, 2, beat_h))
        
        # "On beat" zone
        zone_center = beat_x + 0.5 * beat_w
        zone_width = (self.on_beat_window / self.beat_period) * beat_w * 2
        pygame.draw.rect(self.screen, self.COLOR_BEAT_GOOD, (zone_center - zone_width / 2, beat_y, zone_width, beat_h), 2)
        
    # --- UTILITY HELPERS ---

    def _create_particles(self, pos, count, color, max_size, speed_mult, base_vel=(0,0)):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_mult
            vel = [math.cos(angle) * speed + base_vel[0], math.sin(angle) * speed + base_vel[1]]
            life = self.np_random.integers(10, 26)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(1, max_size)
            })

    def _draw_glow_circle(self, x, y, radius, color, glow_color):
        for i in range(radius, 0, -2):
            alpha = 100 * (1 - (i / radius))
            pygame.gfxdraw.filled_circle(self.screen, x, y, i + 3, (*glow_color, int(alpha)))
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _draw_vehicle(self, pos, color, glow_color, is_police=False):
        x, y = int(pos[0]), int(pos[1])
        if is_police:
            points = [(x, y - 20), (x - 12, y + 20), (x + 12, y + 20)]
            light_bar = [(x-10, y-5), (x-5, y-5), (x,y-5), (x+5, y-5), (x+10, y-5)]
        else: # Player
            points = [(x, y - 15), (x - 10, y + 15), (x + 10, y + 15)]

        # Glow
        for i in range(15, 0, -3):
            alpha = 60 * (1 - (i / 15))
            pygame.gfxdraw.aapolygon(self.screen, points, (*glow_color, int(alpha)))
        
        # Body
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Police lights
        if is_police:
            light_color = self.COLOR_POLICE if self.steps % 10 < 5 else self.COLOR_PLAYER
            pygame.draw.rect(self.screen, light_color, (x-8, y-8, 16, 4))
            
    def _create_building(self, y_pos):
        side = self.np_random.choice(['left', 'right'])
        width = self.np_random.uniform(40, 100)
        height = self.np_random.uniform(100, 300)
        if side == 'left':
            x = self.np_random.uniform(-width, 0)
        else:
            x = self.np_random.uniform(self.WIDTH, self.WIDTH + width)
        return {'x': x, 'y': y_pos, 'w': width, 'h': height}

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # For human play
    pygame.display.set_caption("Cyberpunk Racer")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(env.FPS)

    env.close()
    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")