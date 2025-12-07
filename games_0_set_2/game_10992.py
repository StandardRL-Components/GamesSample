import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:16:30.057200
# Source Brief: brief_00992.md
# Brief Index: 992
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Portal Gravity Environment

    A physics-based puzzle game where the agent navigates a shifting gravity
    environment. The goal is to activate colored constellations by creating
    matching portals near them, which unlocks an exit rift to escape through.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A physics-based puzzle game where you control a celestial body. Flip gravity and create portals to activate constellations and unlock the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) for thrust. Press 'space' to create portals and 'shift' to flip gravity."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 2000
    FPS = 30

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_STARS = (200, 200, 220)
    COLOR_NEBULA_1 = (40, 20, 60, 50)
    COLOR_NEBULA_2 = (20, 40, 60, 40)
    COLOR_PLAYER = (255, 255, 100)
    COLOR_PLAYER_GLOW = (255, 200, 0)
    COLOR_EXIT_INACTIVE = (100, 100, 120)
    COLOR_EXIT_ACTIVE = (200, 0, 255)
    COLOR_PORTAL_RED = (255, 50, 50)
    COLOR_PORTAL_GREEN = (50, 255, 50)
    COLOR_PORTAL_BLUE = (50, 100, 255)
    PORTAL_COLORS = [COLOR_PORTAL_RED, COLOR_PORTAL_GREEN, COLOR_PORTAL_BLUE]
    COLOR_GRAVITY_NORMAL = (0, 255, 150)
    COLOR_GRAVITY_INVERTED = (255, 50, 100)
    COLOR_UI_TEXT = (220, 220, 240)

    # Physics
    GRAVITY = 0.3
    PLAYER_THRUST = 0.5
    PLAYER_DRAG = 0.985
    MAX_VELOCITY = 8
    BOUNCE_DAMPENING = 0.7

    # Game Object Sizes
    PLAYER_RADIUS = 10
    PORTAL_RADIUS = 15
    CONSTELLATION_RADIUS = 40
    CONSTELLATION_ACTIVATION_RADIUS = 50
    EXIT_RADIUS = 25
    PORTAL_PLACEMENT_OFFSET = 40

    # Rewards
    REWARD_WIN = 100.0
    REWARD_CONSTELLATION_ACTIVATED = 25.0
    REWARD_PER_STEP = -0.01
    REWARD_DISTANCE_FACTOR = 0.1

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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self._generate_starfield()

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.gravity_dir = None
        self.portals = None
        self.constellations = None
        self.placing_portal_color_idx = None
        self.placing_portal_entry_pos = None
        self.exit_pos = None
        self.exit_unlocked = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.last_dist_to_target = None
        self.particles = None

        self._setup_level()
        # self.reset() is called by the wrapper or runner, no need to call it here.


    def _setup_level(self):
        """Defines the layout of a level."""
        self.constellations_config = [
            {'pos': pygame.Vector2(100, 100), 'color': self.COLOR_PORTAL_RED},
            {'pos': pygame.Vector2(self.WIDTH - 100, 100), 'color': self.COLOR_PORTAL_GREEN},
            {'pos': pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 80), 'color': self.COLOR_PORTAL_BLUE}
        ]
        self.exit_pos_config = pygame.Vector2(self.WIDTH / 2, 50)
        self.player_start_pos_config = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)


    def _generate_starfield(self):
        """Creates a static starfield for the background."""
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.uniform(0.5, 1.5)
            self.stars.append((x, y, size))
        
        self.nebulae = []
        for _ in range(5):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            radius = random.randint(100, 200)
            color = random.choice([self.COLOR_NEBULA_1, self.COLOR_NEBULA_2])
            self.nebulae.append({'pos': (x, y), 'radius': radius, 'color': color})


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Note: Pygame's randomness is not controlled by this seed.
            # For full determinism, one might need to seed `random` module globally.
            pass

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = self.player_start_pos_config.copy()
        self.player_vel = pygame.Vector2(0, 0)
        self.gravity_dir = 1  # 1 for down, -1 for up

        self.portals = []
        self.placing_portal_color_idx = 0
        self.placing_portal_entry_pos = None
        
        self.constellations = []
        for config in self.constellations_config:
            self.constellations.append({
                'pos': config['pos'].copy(),
                'color': config['color'],
                'activated': False,
                'flash_timer': 0
            })

        self.exit_pos = self.exit_pos_config.copy()
        self.exit_unlocked = False

        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles = []

        self.last_dist_to_target = self._get_dist_to_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = self.REWARD_PER_STEP
        
        # --- 1. Handle Input ---
        # Gravity Flip (on press)
        if shift_held and not self.prev_shift_held:
            self.gravity_dir *= -1
            # sfx: gravity_shift.wav

        # Portal Creation (on press)
        if space_held and not self.prev_space_held:
            portal_reward = self._create_portal()
            reward += portal_reward
            self.score += portal_reward
            # sfx: portal_create.wav

        # Movement
        self._apply_movement(movement)

        # --- 2. Update Physics & Game Logic ---
        self._update_physics()
        self._check_portal_traversal()
        self._update_particles()
        
        # Check if all constellations are activated to unlock the exit
        if not self.exit_unlocked and all(c['activated'] for c in self.constellations):
            self.exit_unlocked = True
            # sfx: exit_unlocked.wav

        # --- 3. Calculate Rewards & Termination ---
        terminated = False
        truncated = False
        if self.exit_unlocked and self.player_pos.distance_to(self.exit_pos) < self.EXIT_RADIUS + self.PLAYER_RADIUS:
            terminated = True
            reward += self.REWARD_WIN
            self.score += self.REWARD_WIN
            self.game_over = True
            # sfx: win_level.wav
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        # Distance-based reward
        new_dist = self._get_dist_to_target()
        dist_reward = (self.last_dist_to_target - new_dist) * self.REWARD_DISTANCE_FACTOR
        reward += dist_reward
        self.score += dist_reward
        self.last_dist_to_target = new_dist

        # --- 4. Update State for Next Step ---
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        self.steps += 1

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _apply_movement(self, movement):
        force = pygame.Vector2(0, 0)
        if movement == 1: force.y = -self.PLAYER_THRUST  # Up
        elif movement == 2: force.y = self.PLAYER_THRUST # Down
        elif movement == 3: force.x = -self.PLAYER_THRUST # Left
        elif movement == 4: force.x = self.PLAYER_THRUST # Right
        self.player_vel += force

    def _update_physics(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY * self.gravity_dir
        # Clamp velocity
        if self.player_vel.length() > self.MAX_VELOCITY:
            self.player_vel.scale_to_length(self.MAX_VELOCITY)
        # Apply drag
        self.player_vel *= self.PLAYER_DRAG
        # Update position
        self.player_pos += self.player_vel

        # Wall bouncing
        if self.player_pos.x < self.PLAYER_RADIUS:
            self.player_pos.x = self.PLAYER_RADIUS
            self.player_vel.x *= -self.BOUNCE_DAMPENING
        if self.player_pos.x > self.WIDTH - self.PLAYER_RADIUS:
            self.player_pos.x = self.WIDTH - self.PLAYER_RADIUS
            self.player_vel.x *= -self.BOUNCE_DAMPENING
        if self.player_pos.y < self.PLAYER_RADIUS:
            self.player_pos.y = self.PLAYER_RADIUS
            self.player_vel.y *= -self.BOUNCE_DAMPENING
        if self.player_pos.y > self.HEIGHT - self.PLAYER_RADIUS:
            self.player_pos.y = self.HEIGHT - self.PLAYER_RADIUS
            self.player_vel.y *= -self.BOUNCE_DAMPENING

    def _create_portal(self):
        # Determine placement position
        if self.player_vel.length() > 0.5:
            direction = self.player_vel.normalize()
        else:
            direction = pygame.Vector2(0, -self.gravity_dir) # Place "up" relative to gravity
        
        placement_pos = self.player_pos + direction * self.PORTAL_PLACEMENT_OFFSET

        # Clamp to screen bounds
        placement_pos.x = np.clip(placement_pos.x, self.PORTAL_RADIUS, self.WIDTH - self.PORTAL_RADIUS)
        placement_pos.y = np.clip(placement_pos.y, self.PORTAL_RADIUS, self.HEIGHT - self.PORTAL_RADIUS)

        color = self.PORTAL_COLORS[self.placing_portal_color_idx]

        if self.placing_portal_entry_pos is None:
            # Placing the entry portal
            self.placing_portal_entry_pos = placement_pos
            self._create_particles(placement_pos, color, 20)
        else:
            # Placing the exit portal, completing the pair
            entry_pos = self.placing_portal_entry_pos
            exit_pos = placement_pos
            
            portal_pair = {
                'entry_pos': entry_pos,
                'exit_pos': exit_pos,
                'color': color,
                'last_use': -100 # Cooldown timer
            }
            self.portals.append(portal_pair)
            self._create_particles(exit_pos, color, 20)

            # Reset for next portal pair
            self.placing_portal_entry_pos = None
            self.placing_portal_color_idx = (self.placing_portal_color_idx + 1) % len(self.PORTAL_COLORS)
            
            # Check for constellation activation
            return self._check_constellation_activation(portal_pair)
        return 0.0

    def _check_constellation_activation(self, portal_pair):
        reward = 0.0
        for const in self.constellations:
            if not const['activated'] and const['color'] == portal_pair['color']:
                dist_to_entry = const['pos'].distance_to(portal_pair['entry_pos'])
                dist_to_exit = const['pos'].distance_to(portal_pair['exit_pos'])

                if min(dist_to_entry, dist_to_exit) < self.CONSTELLATION_ACTIVATION_RADIUS:
                    const['activated'] = True
                    const['flash_timer'] = self.FPS // 2 # Flash for 0.5 seconds
                    reward += self.REWARD_CONSTELLATION_ACTIVATED
                    self._create_particles(const['pos'], const['color'], 50, 3.0)
                    # sfx: constellation_activated.wav
                    break # Activate only one constellation per pair
        return reward

    def _check_portal_traversal(self):
        for p in self.portals:
            # Check cooldown
            if self.steps < p['last_use'] + self.FPS // 2: # 0.5s cooldown
                continue

            # Check entry portal
            if self.player_pos.distance_to(p['entry_pos']) < self.PLAYER_RADIUS + self.PORTAL_RADIUS:
                self.player_pos = p['exit_pos'].copy()
                self.player_vel = self.player_vel.rotate(180) # Keep momentum but flip direction
                p['last_use'] = self.steps
                self._create_particles(p['entry_pos'], p['color'], 30, 2.0)
                self._create_particles(p['exit_pos'], p['color'], 30, 2.0)
                # sfx: portal_travel.wav
                return

            # Check exit portal
            if self.player_pos.distance_to(p['exit_pos']) < self.PLAYER_RADIUS + self.PORTAL_RADIUS:
                self.player_pos = p['entry_pos'].copy()
                self.player_vel = self.player_vel.rotate(180)
                p['last_use'] = self.steps
                self._create_particles(p['exit_pos'], p['color'], 30, 2.0)
                self._create_particles(p['entry_pos'], p['color'], 30, 2.0)
                # sfx: portal_travel.wav
                return

    def _get_dist_to_target(self):
        unactivated = [c for c in self.constellations if not c['activated']]
        if unactivated:
            # Target the closest unactivated constellation
            return min(self.player_pos.distance_to(c['pos']) for c in unactivated)
        elif self.exit_unlocked:
            # Target the exit
            return self.player_pos.distance_to(self.exit_pos)
        else:
            # Should not happen if logic is correct, but as a fallback
            return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_constellations()
        self._render_exit()
        self._render_portals()
        self._render_particles()
        self._render_player()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "exit_unlocked": self.exit_unlocked,
            "constellations_activated": sum(1 for c in self.constellations if c['activated']),
        }

    def _render_background(self):
        for nebula in self.nebulae:
            surf = pygame.Surface((nebula['radius']*2, nebula['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, nebula['color'], (nebula['radius'], nebula['radius']), nebula['radius'])
            self.screen.blit(surf, (nebula['pos'][0] - nebula['radius'], nebula['pos'][1] - nebula['radius']), special_flags=pygame.BLEND_RGBA_ADD)

        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STARS, (x, y), size)

    def _draw_glowing_circle(self, pos, color, glow_color, radius, glow_radius_factor=2.0, steps=5):
        x, y = int(pos.x), int(pos.y)
        for i in range(steps, 0, -1):
            alpha = int(255 * (i / steps)**2 * 0.2)
            current_radius = int(radius + (radius * (glow_radius_factor - 1)) * (1 - i / steps))
            c = glow_color + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, x, y, current_radius, c)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)

    def _render_player(self):
        self._draw_glowing_circle(self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, self.PLAYER_RADIUS)

    def _render_constellations(self):
        for const in self.constellations:
            color = const['color']
            if const['activated']:
                if const['flash_timer'] > 0:
                    # Pulsing flash effect
                    t = (self.FPS // 2 - const['flash_timer']) / (self.FPS // 2)
                    pulse = abs(math.sin(t * math.pi * 4))
                    glow_color = tuple(min(255, int(c * (1 + pulse))) for c in color)
                    const['flash_timer'] -= 1
                else:
                    glow_color = color
                self._draw_glowing_circle(const['pos'], color, glow_color, self.CONSTELLATION_RADIUS, 1.5)
            else:
                # Dimmed, inactive look
                dim_color = tuple(int(c * 0.4) for c in color)
                pygame.gfxdraw.filled_circle(self.screen, int(const['pos'].x), int(const['pos'].y), self.CONSTELLATION_RADIUS, (*dim_color, 100))
                pygame.gfxdraw.aacircle(self.screen, int(const['pos'].x), int(const['pos'].y), self.CONSTELLATION_RADIUS, dim_color)

    def _render_portals(self):
        # Render the portal being placed
        if self.placing_portal_entry_pos:
            color = self.PORTAL_COLORS[self.placing_portal_color_idx]
            self._render_single_portal(self.placing_portal_entry_pos, color, True)

        # Render completed portal pairs
        for p in self.portals:
            self._render_single_portal(p['entry_pos'], p['color'])
            self._render_single_portal(p['exit_pos'], p['color'])
            # Draw a faint line connecting paired portals
            pygame.draw.aaline(self.screen, p['color'] + (50,), p['entry_pos'], p['exit_pos'])

    def _render_single_portal(self, pos, color, is_placing=False):
        x, y = int(pos.x), int(pos.y)
        radius = self.PORTAL_RADIUS
        
        # Shimmer effect
        shimmer_radius = radius + math.sin(self.steps * 0.2 + pos.x) * 2
        
        # Glow
        glow_color = tuple(min(255, c + 50) for c in color)
        self._draw_glowing_circle(pos, color, glow_color, radius, 1.8)

        # Inner black hole
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius - 2, (0,0,0))
        pygame.gfxdraw.aacircle(self.screen, x, y, int(shimmer_radius), color)
        
        if is_placing: # Add a pulsing effect to indicate it's temporary
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            pulse_color = (255,255,255, int(pulse * 100))
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, pulse_color)


    def _render_exit(self):
        color = self.COLOR_EXIT_ACTIVE if self.exit_unlocked else self.COLOR_EXIT_INACTIVE
        glow = self.COLOR_EXIT_ACTIVE if self.exit_unlocked else self.COLOR_BG
        
        if self.exit_unlocked:
            # Animated, swirling effect
            for i in range(5):
                angle = (self.steps * 2 + i * 72) % 360
                offset = self.EXIT_RADIUS * 0.7 * math.sin(self.steps * 0.05 + i)
                r = self.EXIT_RADIUS + offset
                pygame.gfxdraw.arc(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), int(r), angle, angle + 120, color)
        
        self._draw_glowing_circle(self.exit_pos, color, glow, self.EXIT_RADIUS, 1.5)
        pygame.gfxdraw.filled_circle(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), self.EXIT_RADIUS - 2, (10, 5, 20))


    def _create_particles(self, pos, color, count, max_speed=1.5):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(20, 40)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = p['color'] + (int(alpha),)
            size = 2 * (p['life'] / p['max_life'])
            pygame.draw.circle(self.screen, color, p['pos'], size)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Portal Status
        if self.placing_portal_entry_pos:
            status_text = "Place Exit Portal"
        else:
            status_text = "Place Entry Portal"
        
        portal_text = self.font_small.render(status_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(portal_text, (10, 10))
        
        next_color = self.PORTAL_COLORS[self.placing_portal_color_idx]
        pygame.draw.rect(self.screen, next_color, (15, 30, 20, 10))
        pygame.draw.rect(self.screen, tuple(c*0.5 for c in next_color), (15, 30, 20, 10), 1)

        # Gravity Indicator
        grav_color = self.COLOR_GRAVITY_NORMAL if self.gravity_dir == 1 else self.COLOR_GRAVITY_INVERTED
        if self.gravity_dir == 1: # Down
            points = [(self.WIDTH - 25, self.HEIGHT - 30), (self.WIDTH - 15, self.HEIGHT - 30), (self.WIDTH - 20, self.HEIGHT - 15)]
        else: # Up
            points = [(self.WIDTH - 25, 15), (self.WIDTH - 15, 15), (self.WIDTH - 20, 30)]
        pygame.gfxdraw.aapolygon(self.screen, points, grav_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, grav_color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the test suite.
    # To run, you'll need to unset the SDL_VIDEODRIVER dummy variable, e.g., by commenting out the line at the top.
    
    # To enable rendering, comment out the following line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # And uncomment this one:
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Portal Gravity")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()